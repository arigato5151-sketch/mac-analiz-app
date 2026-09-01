"""Fetch a date's fixtures and persist completed results."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from config.settings import get_settings
from data_pipeline.api_client import ApiFootballClient
from data_pipeline.fetch_fixtures import SyncSummary, sync_fixtures
from db.db_client import SupabaseRestClient


def sync_results(
    api: ApiFootballClient, db: SupabaseRestClient, *, match_date: date
) -> SyncSummary:
    # The fixture upsert is idempotent and updates scores/status atomically.
    return sync_fixtures(
        api, db, start_date=match_date, end_date=match_date
    )


def sync_recent_results(
    api: ApiFootballClient,
    db: SupabaseRestClient,
    *,
    today: date,
    lookback_days: int,
) -> SyncSummary:
    """Reconcile recent days so a missed run cannot strand stale fixtures."""
    if lookback_days < 0 or lookback_days > 14:
        raise ValueError("lookback_days must be between 0 and 14")
    summary = sync_fixtures(
        api,
        db,
        start_date=today - timedelta(days=lookback_days),
        end_date=today,
    )
    sync_recent_xg(api, db, today=today, lookback_days=lookback_days)
    return summary


def extract_expected_goals(
    payload: list[dict[str, Any]], *, home_team_id: int, away_team_id: int,
) -> tuple[float, float] | None:
    """Read API-Football xG when the plan/provider exposes the statistic."""
    values: dict[int, float] = {}
    for team_stats in payload:
        team_id = (team_stats.get("team") or {}).get("id")
        if team_id not in {home_team_id, away_team_id}:
            continue
        for stat in team_stats.get("statistics", []):
            kind = str(stat.get("type", "")).lower().replace(" ", "_")
            if kind not in {"expected_goals", "xg"}:
                continue
            try:
                value = float(str(stat.get("value")).replace(",", "."))
            except (TypeError, ValueError):
                continue
            if value >= 0:
                values[int(team_id)] = value
    if home_team_id not in values or away_team_id not in values:
        return None
    return values[home_team_id], values[away_team_id]


def sync_recent_xg(
    api: ApiFootballClient, db: SupabaseRestClient, *, today: date, lookback_days: int,
) -> int:
    """Persist available xG for recently finished fixtures; missing API data is harmless."""
    start = datetime.combine(today - timedelta(days=lookback_days), datetime.min.time()).isoformat()
    end = datetime.combine(today + timedelta(days=1), datetime.min.time()).isoformat()
    matches = db.select_all(
        "matches", columns="id,home_team_id,away_team_id,home_xg,away_xg",
        filters={"status": "eq.finished", "and": f"(match_date.gte.{start},match_date.lt.{end})"},
    )
    updates: list[dict[str, Any]] = []
    for match in matches:
        if match.get("home_xg") is not None and match.get("away_xg") is not None:
            continue
        xg = extract_expected_goals(
            api.get("fixtures/statistics", {"fixture": int(match["id"])}),
            home_team_id=int(match["home_team_id"]), away_team_id=int(match["away_team_id"]),
        )
        if xg is not None:
            updates.append({"id": int(match["id"]), "home_xg": xg[0], "away_xg": xg[1]})
    if updates:
        db.upsert("matches", updates, on_conflict="id")
    return len(updates)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--date",
        type=date.fromisoformat,
        help="Reconcile one date instead of the rolling lookback window.",
    )
    parser.add_argument("--lookback-days", type=int, default=7)
    args = parser.parse_args()
    settings = get_settings()
    api = ApiFootballClient(settings.api_football_key)
    db = SupabaseRestClient(settings.supabase_url, settings.supabase_service_role_key)
    if args.date:
        summary = sync_results(api, db, match_date=args.date)
    else:
        summary = sync_recent_results(
            api,
            db,
            today=datetime.now(ZoneInfo("Europe/Istanbul")).date(),
            lookback_days=args.lookback_days,
        )
    print({"sync": asdict(summary), "api": api.diagnostics()})


if __name__ == "__main__":
    main()
