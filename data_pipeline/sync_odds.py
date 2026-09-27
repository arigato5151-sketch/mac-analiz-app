"""Collect current bookmaker odds for upcoming fixtures."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone

from config.settings import get_settings
from data_pipeline.api_client import ApiFootballClient
from data_pipeline.odds import fetch_match_odds, record_odds_quote
from db.db_client import SupabaseRestClient


def sync_upcoming_odds(*, days: int = 7, max_matches: int = 120) -> dict[str, int]:
    if days < 1 or max_matches < 1:
        raise ValueError("days and max_matches must be positive")
    settings = get_settings()
    db = SupabaseRestClient(settings.supabase_url, settings.supabase_service_role_key)
    api = ApiFootballClient(settings.api_football_key)
    now = datetime.now(timezone.utc)
    matches = db.select_all(
        "matches",
        columns="id,match_date",
        filters={
            "status": "eq.scheduled",
            "and": f"(match_date.gte.{now.isoformat()},match_date.lte.{(now + timedelta(days=days)).isoformat()})",
        },
        order="match_date.asc",
    )[:max_matches]
    fetched = stored = failed = 0
    for match in matches:
        try:
            odds = fetch_match_odds(api, fixture_id=int(match["id"]))
            fetched += 1
            if odds and odds.has_any_market and record_odds_quote(
                db, match_id=int(match["id"]), odds=odds, captured_at=now.isoformat()
            ):
                stored += 1
        except Exception as error:
            failed += 1
            print(f"Odds sync failed for fixture {match['id']}: {type(error).__name__}")
    return {"matches": len(matches), "fetched": fetched, "stored": stored, "failed": failed}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--max-matches", type=int, default=120)
    args = parser.parse_args()
    print(json.dumps(sync_upcoming_odds(days=args.days, max_matches=args.max_matches)))


if __name__ == "__main__":
    main()
