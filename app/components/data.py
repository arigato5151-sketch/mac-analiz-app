"""Read-only data access for the match statistics screen."""

from __future__ import annotations

from datetime import date, datetime, time, timezone
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st

from config.settings import get_public_supabase_settings
from db.db_client import DatabaseError, PublicSupabaseRestClient


@st.cache_resource(show_spinner=False)
def get_db() -> PublicSupabaseRestClient:
    """Create a cached client that cannot write to Supabase."""
    settings = get_public_supabase_settings()
    return PublicSupabaseRestClient(settings.supabase_url, settings.supabase_anon_key)


@st.cache_data(ttl=900, show_spinner="Maç istatistikleri yükleniyor…")
def load_completed_match_history() -> list[dict[str, Any]]:
    """Read all completed history for the statistics archive."""
    return get_db().select_all(
        "matches",
        columns=(
            "id,league_id,home_team_id,away_team_id,match_date,status,"
            "home_score,away_score,home_halftime_score,away_halftime_score,home_xg,away_xg,home_xa,away_xa,"
            "home_corners,away_corners"
        ),
        filters={
            "status": "eq.finished",
            "home_score": "not.is.null",
            "away_score": "not.is.null",
        },
        order="match_date.desc,id.desc",
    )


@st.cache_data(ttl=900, show_spinner=False)
def load_recent_team_matches(
    league_id: int, team_ids: tuple[int, ...], before: str, *, limit: int = 5
) -> list[dict[str, Any]]:
    """Read only each requested team's latest completed league matches."""
    before_utc = pd.to_datetime(before, utc=True, errors="coerce")
    if pd.isna(before_utc):
        return []
    before_filter = before_utc.isoformat()
    columns = "id,home_team_id,away_team_id,match_date,home_score,away_score,home_xg,away_xg"
    team_matches: dict[int, dict[int, dict[str, Any]]] = {}
    for team_id in sorted({int(value) for value in team_ids}):
        rows = get_db().select(
            "matches",
            columns=columns,
            filters={
                "status": "eq.finished",
                "home_score": "not.is.null",
                "away_score": "not.is.null",
                "league_id": f"eq.{int(league_id)}",
                "match_date": f"lt.{before_filter}",
                "or": f"(home_team_id.eq.{team_id},away_team_id.eq.{team_id})",
            },
            limit=limit,
            order="match_date.desc,id.desc",
        )
        team_matches[team_id] = {int(row["id"]): row for row in rows}
    return sorted(
        (row for rows in team_matches.values() for row in rows.values()),
        key=lambda row: (str(row.get("match_date", "")), int(row.get("id", 0))),
        reverse=True,
    )


@st.cache_data(ttl=900, show_spinner=False)
def load_league_match_history(league_id: int) -> list[dict[str, Any]]:
    """Load only the score fields needed by league-level forecast baselines."""
    return get_db().select_all(
        "matches",
        columns=(
            "id,league_id,home_team_id,away_team_id,match_date,status,"
            "home_score,away_score,home_halftime_score,away_halftime_score,"
            "home_corners,away_corners"
        ),
        filters={
            "status": "eq.finished",
            "home_score": "not.is.null",
            "away_score": "not.is.null",
            "league_id": f"eq.{int(league_id)}",
        },
        order="match_date.desc,id.desc",
    )


@st.cache_data(ttl=900, show_spinner=False)
def load_reference_catalog() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read team and league names used to label stored match statistics."""
    db = get_db()
    return (
        pd.DataFrame(db.select_all("teams", columns="id,name")),
        pd.DataFrame(db.select_all("leagues", columns="id,name,country")),
    )


@st.cache_data(ttl=300, show_spinner="Günün maçları ve oranları yükleniyor…")
def load_match_day(
    match_day: date,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Read fixtures and IdDaa quotes for one Istanbul calendar day."""
    istanbul = ZoneInfo("Europe/Istanbul")
    start = datetime.combine(match_day, time.min, tzinfo=istanbul).astimezone(timezone.utc)
    end = datetime.combine(date.fromordinal(match_day.toordinal() + 1), time.min, tzinfo=istanbul).astimezone(timezone.utc)
    db = get_db()
    matches = db.select_all(
        "matches",
        columns="id,league_id,home_team_id,away_team_id,match_date,status",
        filters={"match_date": f"gte.{start.isoformat()}", "and": f"(match_date.lt.{end.isoformat()})"},
        order="match_date.asc,id.asc",
    )
    # The odds collector may run before kickoff day (and retry later), so quote
    # capture time is not a reliable way to find quotes for a selected fixture.
    # Query by the fixture IDs and keep the latest capture in the UI instead.
    now = datetime.now(timezone.utc)
    match_ids = [
        int(row["id"])
        for row in matches
        if row.get("id") is not None
        and str(row.get("status") or "").casefold() not in {"live", "finished"}
        and pd.to_datetime(row.get("match_date"), utc=True, errors="coerce") > pd.Timestamp(now)
    ]
    quotes: list[dict[str, Any]] = []
    for offset in range(0, len(match_ids), 100):
        chunk = match_ids[offset : offset + 100]
        if not chunk:
            continue
        quotes.extend(
            db.select_all(
                "odds_quote_history",
                columns="id,match_id,bookmaker,odds,captured_at,source_updated_at",
                filters={
                    "bookmaker": "eq.iddaa.com",
                    "match_id": f"in.({','.join(map(str, chunk))})",
                },
                order="captured_at.asc,id.asc",
            )
        )
    analyses: list[dict[str, Any]] = []
    for offset in range(0, len(match_ids), 100):
        chunk = match_ids[offset : offset + 100]
        if not chunk:
            continue
        try:
            analyses.extend(
                db.select_all(
                    "match_analysis_cache",
                    columns=(
                        "match_id,league_id,home_team_id,away_team_id,match_date,"
                        "model_version,calculated_at,analysis_json"
                    ),
                    filters={"match_id": f"in.({','.join(map(str, chunk))})"},
                )
            )
        except DatabaseError:
            # Keep the app usable while the cache migration is rolling out.
            analyses = []
            break
    return matches, quotes, analyses


@st.cache_data(ttl=86_400, show_spinner=False)
def load_iddaa_market_labels() -> dict[str, str]:
    """Read IdDaa's current names for stored sport-type/market-code pairs."""
    session = requests.Session()
    # Match the odds collector: local/host proxy settings can block this source.
    session.trust_env = False
    response = session.get(
        "https://sportsbookv2.iddaa.com/sportsbook/get_market_config",
        headers={"Accept": "application/json", "User-Agent": "mac-analiz-app/1.0"},
        timeout=15,
    )
    response.raise_for_status()
    payload = response.json()
    if not payload.get("isSuccess"):
        return {}

    groups = payload.get("data", {}).get("dmsg", {})
    labels: dict[str, str] = {}
    labels_by_code: dict[int, set[str]] = {}
    if not isinstance(groups, dict):
        return labels
    for group in groups.values():
        if not isinstance(group, dict) or not group.get("n"):
            continue
        for key in group.get("m", []):
            try:
                sport_type, market_code = (int(part) for part in str(key).split("_", 1))
            except (TypeError, ValueError):
                continue
            labels[f"{sport_type}:{market_code}"] = str(group["n"])
            labels_by_code.setdefault(market_code, set()).add(str(group["n"]))
    # Older stored quotes do not include sport_type. Use a code-only label only
    # when IdDaa assigns that code the same category across sport types.
    for market_code, names in labels_by_code.items():
        if len(names) == 1:
            labels[str(market_code)] = next(iter(names))
    return labels
