"""Fetch a date's fixtures and persist completed results."""

from __future__ import annotations

import argparse
from datetime import date

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=date.fromisoformat, default=date.today())
    args = parser.parse_args()
    settings = get_settings()
    summary = sync_results(
        ApiFootballClient(settings.api_football_key),
        SupabaseRestClient(
            settings.supabase_url, settings.supabase_service_role_key
        ),
        match_date=args.date,
    )
    print(summary)


if __name__ == "__main__":
    main()
