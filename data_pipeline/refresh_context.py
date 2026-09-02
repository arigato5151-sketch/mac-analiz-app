"""Refresh pre-match form, Elo, and availability for upcoming teams."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

from config.leagues import LEAGUES_BY_ID
from config.settings import get_settings
from data_pipeline.api_client import ApiFootballClient
from data_pipeline.fetch_injuries import sync_injuries
from data_pipeline.fetch_team_stats import sync_team_form
from db.db_client import SupabaseRestClient
from monitoring.operational_events import record_api_diagnostics, record_event, record_exception
from models.feature_engineering import CausalFeatureState
from models.train_model import load_completed_matches


def upcoming_team_targets(
    db: SupabaseRestClient, *, now: datetime, horizon_days: int
) -> dict[int, set[int]]:
    """Return upcoming team IDs grouped by configured league."""
    if horizon_days < 1 or horizon_days > 14:
        raise ValueError("horizon_days must be between 1 and 14")
    rows = db.select_all(
        "matches",
        columns="league_id,home_team_id,away_team_id,match_date,status",
        filters={
            "status": "eq.scheduled",
            "and": (
                f"(match_date.gte.{now.isoformat()},"
                f"match_date.lte.{(now + timedelta(days=horizon_days)).isoformat()})"
            ),
        },
    )
    grouped: defaultdict[int, set[int]] = defaultdict(set)
    for row in rows:
        league_id = int(row["league_id"])
        if league_id not in LEAGUES_BY_ID:
            continue
        grouped[league_id].update(
            (int(row["home_team_id"]), int(row["away_team_id"]))
        )
    return dict(grouped)


def current_elo_ratings(
    completed_matches: list[dict[str, Any]], team_ids: set[int]
) -> dict[int, float]:
    """Rebuild causal Elo once and return ratings for requested teams."""
    state = CausalFeatureState()
    for row in sorted(
        completed_matches, key=lambda item: (item["match_date"], int(item["id"]))
    ):
        state.update(row)
    return {
        team_id: float(state.states[team_id].elo)
        for team_id in team_ids
    }


def refresh_upcoming_context(
    api: ApiFootballClient,
    db: SupabaseRestClient,
    *,
    now: datetime,
    horizon_days: int = 3,
) -> dict[str, Any]:
    """Refresh context with partial-failure reporting and safe retry semantics."""
    targets = upcoming_team_targets(db, now=now, horizon_days=horizon_days)
    all_team_ids = {team_id for teams in targets.values() for team_id in teams}
    if not all_team_ids:
        return {
            "leagues": 0,
            "teams": 0,
            "team_forms_written": 0,
            "availability_rows_written": 0,
            "failures": [],
        }

    elo_by_team = current_elo_ratings(load_completed_matches(db), all_team_ids)
    failures: list[dict[str, Any]] = []
    form_count = 0
    availability_count = 0

    for league_id, team_ids in sorted(targets.items()):
        season = LEAGUES_BY_ID[league_id].season
        for team_id in sorted(team_ids):
            try:
                sync_team_form(
                    api,
                    db,
                    team_id=team_id,
                    season=season,
                    elo_rating=elo_by_team[team_id],
                )
                form_count += 1
            except Exception as exc:  # One unavailable team must not block all leagues.
                failures.append(
                    {"scope": "team_form", "id": team_id, "error": str(exc)[:200]}
                )

        try:
            availability_count += sync_injuries(
                api, db, league_id=league_id, team_ids=team_ids
            )
        except Exception as exc:
            failures.append(
                {"scope": "injuries", "id": league_id, "error": str(exc)[:200]}
            )

    attempted = len(all_team_ids) + len(targets)
    if attempted and len(failures) / attempted > 0.30:
        raise RuntimeError(
            f"Context refresh failure ratio exceeded 30%: {len(failures)}/{attempted}"
        )
    return {
        "leagues": len(targets),
        "teams": len(all_team_ids),
        "team_forms_written": form_count,
        "availability_rows_written": availability_count,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=3)
    args = parser.parse_args()
    settings = get_settings()
    api = ApiFootballClient(settings.api_football_key)
    db = SupabaseRestClient(settings.supabase_url, settings.supabase_service_role_key)
    try:
        summary = refresh_upcoming_context(
            api, db, now=datetime.now(timezone.utc), horizon_days=args.days
        )
    except Exception as error:
        record_exception(db, component="refresh_context", operation="context refresh", error=error)
        raise
    if summary["failures"]:
        record_event(
            db,
            severity="warning",
            component="refresh_context",
            event_type="partial_failure",
            message="Upcoming context refresh completed with partial failures",
            context={"failure_count": len(summary["failures"]), "team_count": summary["teams"]},
        )
    diagnostics = api.diagnostics()
    record_api_diagnostics(db, component="refresh_context", diagnostics=diagnostics)
    print(json.dumps({**summary, "api": diagnostics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
