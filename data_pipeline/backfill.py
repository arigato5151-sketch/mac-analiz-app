"""Backfill historical fixtures for model training."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from config.leagues import LEAGUES_BY_ID, TRACKED_LEAGUE_IDS
from config.settings import get_settings
from data_pipeline.api_client import ApiFootballClient, ApiFootballError
from data_pipeline.fetch_fixtures import transform_fixtures
from db.db_client import SupabaseRestClient


@dataclass(slots=True)
class BackfillSummary:
    requested_pairs: int = 0
    completed_pairs: int = 0
    api_fixtures: int = 0
    finished_matches: int = 0
    teams_upserted: int = 0
    errors: list[str] = field(default_factory=list)


def batched(rows: Sequence[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    if size < 1:
        raise ValueError("Batch size must be positive")
    for start in range(0, len(rows), size):
        yield list(rows[start : start + size])


def _configured_league_rows(league_ids: Sequence[int]) -> list[dict[str, Any]]:
    return [
        {
            "id": LEAGUES_BY_ID[league_id].id,
            "name": LEAGUES_BY_ID[league_id].name,
            "country": LEAGUES_BY_ID[league_id].country,
            "season": LEAGUES_BY_ID[league_id].season,
            "is_active": True,
        }
        for league_id in league_ids
    ]


def backfill_history(
    api: ApiFootballClient,
    db: SupabaseRestClient,
    *,
    league_ids: Sequence[int],
    seasons: Sequence[int],
    batch_size: int = 500,
    fail_fast: bool = False,
) -> BackfillSummary:
    if not league_ids or not seasons:
        raise ValueError("At least one league and season are required")
    unknown = sorted(set(league_ids) - LEAGUES_BY_ID.keys())
    if unknown:
        raise ValueError(f"Unconfigured league IDs: {unknown}")

    summary = BackfillSummary(requested_pairs=len(league_ids) * len(seasons))
    db.upsert("leagues", _configured_league_rows(league_ids), on_conflict="id")

    for league_id in league_ids:
        for season in seasons:
            try:
                fixtures = api.get(
                    "fixtures",
                    {"league": league_id, "season": season, "timezone": "UTC"},
                )
                summary.api_fixtures += len(fixtures)
                teams, matches = transform_fixtures(fixtures)
                finished = [match for match in matches if match["status"] == "finished"]

                for batch in batched(teams, batch_size):
                    db.upsert("teams", batch, on_conflict="id")
                for batch in batched(finished, batch_size):
                    db.upsert("matches", batch, on_conflict="id")

                summary.completed_pairs += 1
                summary.teams_upserted += len(teams)
                summary.finished_matches += len(finished)
            except (ApiFootballError, RuntimeError, ValueError, KeyError) as exc:
                message = f"league={league_id}, season={season}: {exc}"
                summary.errors.append(message)
                if fail_fast:
                    raise

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--league-id",
        type=int,
        action="append",
        dest="league_ids",
        help="Repeat to limit leagues; defaults to all configured leagues",
    )
    parser.add_argument(
        "--seasons", type=int, nargs="+", default=[2023, 2024, 2025]
    )
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    summary = backfill_history(
        ApiFootballClient(settings.api_football_key),
        SupabaseRestClient(
            settings.supabase_url, settings.supabase_service_role_key
        ),
        league_ids=args.league_ids or TRACKED_LEAGUE_IDS,
        seasons=args.seasons,
        batch_size=args.batch_size,
        fail_fast=args.fail_fast,
    )
    print(summary)
    if summary.errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
