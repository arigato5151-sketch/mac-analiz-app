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
from config.settings import get_supabase_admin_settings
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
    fixtures = db.select_all(
        "matches",
        columns=MATCH_COLUMNS,
        filters={
            "status": "eq.scheduled",
            "match_date": f"gt.{reference.isoformat()}",
        },
        order="match_date.asc,id.asc",
    )
    histories: dict[int, pd.DataFrame] = {}
    tuning_by_league: dict[int, dict[str, Any] | None] = {}
    rows: list[dict[str, Any]] = []
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
        }
        analysis = build_precomputed_match_analysis(
            row, histories[league_id], tuning_by_league[league_id]
        )
        rows.append(
            {
                "match_id": match_id,
                "league_id": league_id,
                "home_team_id": home_team_id,
                "away_team_id": away_team_id,
                "match_date": match_date.isoformat(),
                "model_version": ANALYSIS_MODEL_VERSION,
                "calculated_at": reference.isoformat(),
                "analysis_json": _json_safe(analysis),
            }
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
                "model_version": ANALYSIS_MODEL_VERSION,
            }
        )
    )


if __name__ == "__main__":
    main()
