"""Fetch upcoming fixtures and upsert leagues, teams, and matches."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from typing import Any

from config.leagues import LEAGUES_BY_ID, TRACKED_LEAGUE_IDS
from config.settings import get_settings
from data_pipeline.api_client import ApiFootballClient
from db.db_client import SupabaseRestClient
from monitoring.operational_events import record_api_diagnostics, record_exception


FINISHED_STATUSES = {"FT", "AET", "PEN"}
LIVE_STATUSES = {"1H", "HT", "2H", "ET", "BT", "P", "SUSP", "INT", "LIVE"}


@dataclass(frozen=True, slots=True)
class SyncSummary:
    api_fixtures: int
    tracked_fixtures: int
    leagues_upserted: int
    teams_upserted: int
    matches_upserted: int


def normalize_status(short_status: str) -> str:
    if short_status in FINISHED_STATUSES:
        return "finished"
    if short_status in LIVE_STATUSES:
        return "live"
    if short_status in {"PST", "CANC", "ABD", "AWD", "WO"}:
        return "postponed" if short_status == "PST" else "cancelled"
    return "scheduled"


def _league_rows(league_ids: set[int]) -> list[dict[str, Any]]:
    return [
        {
            "id": league.id,
            "name": league.name,
            "country": league.country,
            "season": league.season,
            "is_active": True,
        }
        for league_id, league in LEAGUES_BY_ID.items()
        if league_id in league_ids
    ]


def transform_fixtures(
    fixtures: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    teams_by_id: dict[int, dict[str, Any]] = {}
    matches_by_id: dict[int, dict[str, Any]] = {}

    for item in fixtures:
        league_id = int(item["league"]["id"])
        home = item["teams"]["home"]
        away = item["teams"]["away"]
        for team in (home, away):
            teams_by_id[int(team["id"])] = {
                "id": int(team["id"]),
                "name": team["name"],
                "league_id": league_id,
                "logo_url": team.get("logo"),
            }

        status = item["fixture"]["status"]["short"]
        matches_by_id[int(item["fixture"]["id"])] = {
            "id": int(item["fixture"]["id"]),
            "league_id": league_id,
            "home_team_id": int(home["id"]),
            "away_team_id": int(away["id"]),
            "match_date": item["fixture"]["date"],
            "status": normalize_status(status),
            "home_score": item.get("goals", {}).get("home"),
            "away_score": item.get("goals", {}).get("away"),
        }

    return list(teams_by_id.values()), list(matches_by_id.values())


def sync_fixtures(
    api: ApiFootballClient,
    db: SupabaseRestClient,
    *,
    start_date: date,
    end_date: date,
    league_ids: tuple[int, ...] = TRACKED_LEAGUE_IDS,
) -> SyncSummary:
    if end_date < start_date:
        raise ValueError("end_date cannot be earlier than start_date")

    tracked_ids = set(league_ids)
    all_fixtures: list[dict[str, Any]] = []
    cursor = start_date
    while cursor <= end_date:
        all_fixtures.extend(
            api.get(
                "fixtures",
                {"date": cursor.isoformat(), "timezone": "Europe/Istanbul"},
            )
        )
        cursor += timedelta(days=1)

    tracked = [
        item for item in all_fixtures if int(item["league"]["id"]) in tracked_ids
    ]
    teams, matches = transform_fixtures(tracked)
    used_league_ids = {int(item["league"]["id"]) for item in tracked}
    leagues = _league_rows(used_league_ids)

    # Foreign-key order is intentional: leagues -> teams -> matches.
    db.upsert("leagues", leagues, on_conflict="id")
    db.upsert("teams", teams, on_conflict="id")
    db.upsert("matches", matches, on_conflict="id")

    return SyncSummary(
        api_fixtures=len(all_fixtures),
        tracked_fixtures=len(tracked),
        leagues_upserted=len(leagues),
        teams_upserted=len(teams),
        matches_upserted=len(matches),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=3, help="Days including today")
    args = parser.parse_args()
    if args.days < 1 or args.days > 14:
        parser.error("--days must be between 1 and 14")

    settings = get_settings()
    api = ApiFootballClient(settings.api_football_key)
    db = SupabaseRestClient(
        settings.supabase_url, settings.supabase_service_role_key
    )
    today = date.today()
    try:
        summary = sync_fixtures(
            api, db, start_date=today, end_date=today + timedelta(days=args.days - 1)
        )
    except Exception as error:
        record_exception(db, component="fetch_fixtures", operation="fixture sync", error=error)
        raise
    diagnostics = api.diagnostics()
    record_api_diagnostics(db, component="fetch_fixtures", diagnostics=diagnostics)
    print({"sync": asdict(summary), "api": diagnostics})


if __name__ == "__main__":
    main()
