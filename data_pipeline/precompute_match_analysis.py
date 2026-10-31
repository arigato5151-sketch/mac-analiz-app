"""Persist nightly forecast and recent-form data for fast match-card rendering."""

from __future__ import annotations

import json
import math
from datetime import date, datetime, timezone
from typing import Any

import pandas as pd

from app.main import (
    ANALYSIS_MODEL_VERSION,
    build_precomputed_match_analysis,
    tune_recent_form_prior,
)
from app.ai_match_analysis import (
    MatchAnalysisAIError,
    build_match_analysis_context,
    generate_match_ai_packages,
)
from config.settings import get_optional_setting, get_supabase_admin_settings
from db.db_client import SupabaseRestClient

MATCH_COLUMNS = (
    "id,league_id,home_team_id,away_team_id,match_date,status"
)
HISTORY_COLUMNS = (
    "id,league_id,home_team_id,away_team_id,match_date,status,home_score,away_score,"
    "home_halftime_score,away_halftime_score,home_xg,away_xg,home_corners,away_corners"
)
NUMERIC_HISTORY_COLUMNS = (
    "id",
    "league_id",
    "home_team_id",
    "away_team_id",
    "home_score",
    "away_score",
    "home_halftime_score",
    "away_halftime_score",
    "home_corners",
    "away_corners",
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    if hasattr(value, "item"):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is pd.NA or value is pd.NaT:
        return None
    return value


def _load_league_history(db: SupabaseRestClient, league_id: int) -> pd.DataFrame:
    records = db.select_all(
        "matches",
        columns=HISTORY_COLUMNS,
        filters={
            "status": "eq.finished",
            "home_score": "not.is.null",
            "away_score": "not.is.null",
            "league_id": f"eq.{league_id}",
        },
        order="match_date.desc,id.desc",
    )
    history = pd.DataFrame(records)
    if history.empty:
        return history
    history["match_date"] = pd.to_datetime(
        history["match_date"], utc=True, errors="coerce"
    )
    for column in NUMERIC_HISTORY_COLUMNS:
        history[column] = pd.to_numeric(history[column], errors="coerce")
    return history.dropna(subset=["match_date"])


def collect_match_analysis_cache(
    db: SupabaseRestClient, *, now: datetime | None = None
) -> list[dict[str, Any]]:
    """Build analysis payloads for all upcoming scheduled fixtures."""
    reference = now or datetime.now(timezone.utc)
    if reference.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    reference = reference.astimezone(timezone.utc)
    cache_until = reference + pd.Timedelta(hours=36)
    fixtures = db.select_all(
        "matches",
        columns=MATCH_COLUMNS,
        filters={
            "status": "eq.scheduled",
            "match_date": f"gt.{reference.isoformat()}",
            "and": f"(match_date.lt.{cache_until.isoformat()})",
        },
        order="match_date.asc,id.asc",
    )
    cached_ai: dict[int, dict[str, Any]] = {}
    fixture_ids = [str(item["id"]) for item in fixtures if item.get("id") is not None]
    for offset in range(0, len(fixture_ids), 100):
        cached_rows = db.select_all(
            "match_analysis_cache",
            columns="match_id,model_version,analysis_json",
            filters={"match_id": f"in.({','.join(fixture_ids[offset:offset + 100])})"},
            order="match_id.asc",
        )
        for cached in cached_rows:
            cached_analysis = cached.get("analysis_json") or {}
            if (
                cached.get("model_version") == ANALYSIS_MODEL_VERSION
                and isinstance(cached_analysis, dict)
                and cached_analysis.get("ai_analysis")
                and cached_analysis.get("ai_risk_selections")
            ):
                cached_ai[int(cached["match_id"])] = {
                    "analysis": cached_analysis["ai_analysis"],
                    "risk_selections": cached_analysis["ai_risk_selections"],
                }
    team_names = {
        int(item["id"]): str(item["name"])
        for item in db.select_all("teams", columns="id,name")
        if item.get("id") is not None and item.get("name")
    }
    league_names = {
        int(item["id"]): str(item["name"])
        for item in db.select_all("leagues", columns="id,name")
        if item.get("id") is not None and item.get("name")
    }
    google_api_key = get_optional_setting("GOOGLE_API_KEY")
    google_model = get_optional_setting("GOOGLE_MODEL", "gemini-3.8-flash")
    ai_failures = 0
    if not google_api_key:
        print("GOOGLE_API_KEY is missing; nightly AI commentary will not be generated.")
    histories: dict[int, pd.DataFrame] = {}
    tuning_by_league: dict[int, dict[str, Any] | None] = {}
    rows: list[dict[str, Any]] = []
    cache_row_by_match: dict[int, dict[str, Any]] = {}
    ai_contexts: list[dict[str, Any]] = []
    analysis_by_match: dict[int, dict[str, Any]] = {}
    for fixture in fixtures:
        try:
            league_id = int(fixture["league_id"])
            match_id = int(fixture["id"])
            home_team_id = int(fixture["home_team_id"])
            away_team_id = int(fixture["away_team_id"])
            match_date = pd.to_datetime(fixture["match_date"], utc=True, errors="coerce")
            if pd.isna(match_date):
                continue
        except (KeyError, TypeError, ValueError):
            continue

        if league_id not in histories:
            histories[league_id] = _load_league_history(db, league_id)
            tuning_by_league[league_id] = (
                tune_recent_form_prior(histories[league_id])
                if not histories[league_id].empty
                else None
            )
        row = {
            **fixture,
            "id": match_id,
            "league_id": league_id,
            "home_team_id": home_team_id,
            "away_team_id": away_team_id,
            "match_date": match_date,
            "kickoff": match_date,
            "home_name": team_names.get(home_team_id, f"Ev sahibi {home_team_id}"),
            "away_name": team_names.get(away_team_id, f"Deplasman {away_team_id}"),
            "league_name": league_names.get(league_id, ""),
            "league_label": league_names.get(league_id, ""),
        }
        analysis = build_precomputed_match_analysis(
            row, histories[league_id], tuning_by_league[league_id]
        )
        if google_api_key and analysis.get("forecast"):
            analysis_by_match[match_id] = analysis
            if match_id in cached_ai:
                ai_package = cached_ai[match_id]
                analysis["ai_analysis"] = ai_package["analysis"]
                analysis["ai_risk_selections"] = ai_package["risk_selections"]
            else:
                ai_context = build_match_analysis_context(
                    row, histories[league_id], analysis["forecast"]
                )
                ai_context["match_id"] = match_id
                ai_context["model"]["candidate_markets"] = {
                    str(name): float(probability)
                    for name, probability in ai_context["model"]["candidate_markets"].items()
                }
                ai_contexts.append(ai_context)
        cache_row = {
                "match_id": match_id,
                "league_id": league_id,
                "home_team_id": home_team_id,
                "away_team_id": away_team_id,
                "match_date": match_date.isoformat(),
                "model_version": ANALYSIS_MODEL_VERSION,
                "calculated_at": reference.isoformat(),
                "analysis_json": _json_safe(analysis),
            }
        rows.append(cache_row)
        cache_row_by_match[match_id] = cache_row
    # One structured request can analyze several fixtures, keeping a full slate
    # within the free-tier daily request budget. Existing AI results are reused
    # by the second scheduled refresh instead of consuming the quota twice.
    for offset in range(0, len(ai_contexts), 10):
        batch = ai_contexts[offset : offset + 10]
        try:
            packages = generate_match_ai_packages(
                batch, api_key=google_api_key or "", model=google_model
            )
            for match_id, package in packages.items():
                analysis = analysis_by_match.get(match_id)
                if analysis is not None:
                    analysis["ai_analysis"] = package["analysis"]
                    analysis["ai_risk_selections"] = package["risk_selections"]
                    cache_row = cache_row_by_match.get(match_id)
                    if cache_row is not None:
                        cache_row["analysis_json"]["ai_analysis"] = package["analysis"]
                        cache_row["analysis_json"]["ai_risk_selections"] = package[
                            "risk_selections"
                        ]
            missing = {int(context["match_id"]) for context in batch} - packages.keys()
            ai_failures += len(missing)
            if missing:
                print(f"AI precompute returned no package for matches: {sorted(missing)}")
        except MatchAnalysisAIError as exc:
            ai_failures += len(batch)
            print(
                f"AI precompute failed for match batch "
                f"{offset // 10 + 1} ({len(batch)} matches): {exc}"
            )
    return rows


def main() -> None:
    settings = get_supabase_admin_settings()
    db = SupabaseRestClient(settings.supabase_url, settings.supabase_service_role_key)
    rows = collect_match_analysis_cache(db)
    db.delete(
        "match_analysis_cache",
        filters={"match_date": f"lt.{datetime.now(timezone.utc).isoformat()}"},
    )
    for offset in range(0, len(rows), 100):
        db.upsert(
            "match_analysis_cache",
            rows[offset : offset + 100],
            on_conflict="match_id",
        )
    print(
        json.dumps(
            {
                "fixtures_cached": len(rows),
                "ai_analyses_cached": sum(
                    1 for row in rows if row["analysis_json"].get("ai_analysis")
                ),
                "ai_failures": sum(
                    1
                    for row in rows
                    if row["analysis_json"].get("forecast")
                    and not row["analysis_json"].get("ai_analysis")
                ),
                "model_version": ANALYSIS_MODEL_VERSION,
            }
        )
    )


if __name__ == "__main__":
    main()
