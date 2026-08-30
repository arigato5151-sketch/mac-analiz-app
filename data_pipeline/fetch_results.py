"""Fetch a date's fixtures and persist completed results."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import date, datetime, timedelta
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
    return sync_fixtures(
        api,
        db,
        start_date=today - timedelta(days=lookback_days),
        end_date=today,
    )


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
