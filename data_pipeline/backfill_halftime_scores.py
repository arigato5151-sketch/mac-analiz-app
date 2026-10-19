"""Backfill halftime scores for already-stored fixtures from API-Football."""

from __future__ import annotations

import argparse
from typing import Any

from config.leagues import TRACKED_LEAGUE_IDS
from config.settings import get_settings
from data_pipeline.api_client import ApiFootballClient
from db.db_client import SupabaseRestClient


def backfill_halftime_scores(
    api: ApiFootballClient,
    db: SupabaseRestClient,
    *,
    from_season: int,
    to_season: int,
    league_ids: tuple[int, ...] = TRACKED_LEAGUE_IDS,
) -> dict[str, Any]:
    if from_season < 2000 or to_season < from_season:
        raise ValueError("Invalid season range")
    stored_matches = {
        int(row["id"]): row
        for row in db.select_all("matches", columns="*")
        if row.get("id") is not None
    }
    updates: dict[int, dict[str, Any]] = {}
    scanned = 0
    for league_id in league_ids:
        for season in range(from_season, to_season + 1):
            fixtures = api.get("fixtures", {"league": league_id, "season": season})
            scanned += len(fixtures)
            for fixture in fixtures:
                fixture_id = (fixture.get("fixture") or {}).get("id")
                if fixture_id is None or int(fixture_id) not in stored_matches:
                    continue
                score = (fixture.get("score") or {}).get("halftime") or {}
                home, away = score.get("home"), score.get("away")
                if home is None or away is None:
                    continue
                updates[int(fixture_id)] = {
                    **stored_matches[int(fixture_id)],
                    "home_halftime_score": int(home),
                    "away_halftime_score": int(away),
                }
    records = list(updates.values())
    for start in range(0, len(records), 500):
        db.upsert("matches", records[start : start + 500], on_conflict="id")
    return {
        "seasons": [from_season, to_season],
        "leagues": len(league_ids),
        "fixtures_scanned": scanned,
        "halftime_scores_updated": len(records),
        "api": api.diagnostics(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-season", type=int, default=2025)
    parser.add_argument("--to-season", type=int, default=2026)
    parser.add_argument("--league-ids", type=int, nargs="*")
    args = parser.parse_args()
    settings = get_settings()
    api = ApiFootballClient(settings.api_football_key)
    db = SupabaseRestClient(settings.supabase_url, settings.supabase_service_role_key)
    summary = backfill_halftime_scores(
        api,
        db,
        from_season=args.from_season,
        to_season=args.to_season,
        league_ids=tuple(args.league_ids) if args.league_ids else TRACKED_LEAGUE_IDS,
    )
    print(summary)


if __name__ == "__main__":
    main()
