"""Fetch upcoming fixtures and upsert leagues, teams, and matches."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from config.leagues import LEAGUES_BY_ID, TRACKED_LEAGUE_IDS
from config.settings import UPCOMING_HORIZON_DAYS, get_settings
from data_pipeline.api_client import ApiFootballClient, ApiFootballError
from db.db_client import SupabaseRestClient

FINISHED_STATUSES = {"FT", "AET", "PEN"}
LIVE_STATUSES = {"1H", "HT", "2H", "ET", "BT", "P", "SUSP", "INT", "LIVE"}


@dataclass(frozen=True, slots=True)
class SyncSummary:
    requested_start: str
    requested_end: str
    completed_start: str | None
    completed_end: str | None
    skipped_start: str | None
    skipped_end: str | None
    is_complete: bool
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
        halftime_score = (item.get("score") or {}).get("halftime") or {}
        matches_by_id[int(item["fixture"]["id"])] = {
            "id": int(item["fixture"]["id"]),
            "league_id": league_id,
            "home_team_id": int(home["id"]),
            "away_team_id": int(away["id"]),
            "match_date": item["fixture"]["date"],
            "status": normalize_status(status),
            "home_score": item.get("goals", {}).get("home"),
            "away_score": item.get("goals", {}).get("away"),
            "home_halftime_score": halftime_score.get("home"),
            "away_halftime_score": halftime_score.get("away"),
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
    skipped_dates: list[date] = []
    completed_dates: list[date] = []
    while cursor <= end_date:
        try:
            day_fixtures = api.get(
                "fixtures",
                {"date": cursor.isoformat(), "timezone": "Europe/Istanbul"},
            )
        except ApiFootballError as exc:
            # Free API-Football plans expose a moving date window. Skip dates
            # outside that advertised range but preserve all accessible dates
            # already fetched so the whole daily pipeline can keep running.
            message = str(exc)
            plan_window = re.search(
                r"Free plans do not have access to this date, try from "
                r"(\d{4}-\d{2}-\d{2}) to (\d{4}-\d{2}-\d{2})",
                message,
            )
            if plan_window:
                allowed_start = date.fromisoformat(plan_window.group(1))
                allowed_end = date.fromisoformat(plan_window.group(2))
                if cursor < allowed_start:
                    skipped_dates.extend(
                        current
                        for current in _date_range(cursor, min(end_date, allowed_start - timedelta(days=1)))
                    )
                    print(
                        f"API-Football plan excludes {cursor} through "
                        f"{allowed_start - timedelta(days=1)}; skipping to {allowed_start}.",
                        file=sys.stderr,
                    )
                    cursor = allowed_start
                    continue
                if cursor > allowed_end:
                    skipped_dates.extend(_date_range(cursor, end_date))
                    print(
                        f"API-Football plan excludes future fixture date {cursor}; "
                        "stopping at the last accessible date.",
                        file=sys.stderr,
                    )
                    break
            raise
        all_fixtures.extend(day_fixtures)
        completed_dates.append(cursor)
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
        requested_start=start_date.isoformat(),
        requested_end=end_date.isoformat(),
        completed_start=min(completed_dates).isoformat() if completed_dates else None,
        completed_end=max(completed_dates).isoformat() if completed_dates else None,
        skipped_start=min(skipped_dates).isoformat() if skipped_dates else None,
        skipped_end=max(skipped_dates).isoformat() if skipped_dates else None,
        is_complete=not skipped_dates,
        api_fixtures=len(all_fixtures),
        tracked_fixtures=len(tracked),
        leagues_upserted=len(leagues),
        teams_upserted=len(teams),
        matches_upserted=len(matches),
    )


def _date_range(start: date, end: date) -> list[date]:
    if end < start:
        return []
    return [start + timedelta(days=offset) for offset in range((end - start).days + 1)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--days",
        type=int,
        default=UPCOMING_HORIZON_DAYS,
        help="Days including today",
    )
    args = parser.parse_args()
    if args.days < 1 or args.days > 14:
        parser.error("--days must be between 1 and 14")

    settings = get_settings()
    api = ApiFootballClient(settings.api_football_key)
    db = SupabaseRestClient(
        settings.supabase_url, settings.supabase_service_role_key
    )
    today = datetime.now(ZoneInfo("Europe/Istanbul")).date()
    summary = sync_fixtures(
        api, db, start_date=today, end_date=today + timedelta(days=args.days - 1)
    )
    diagnostics = api.diagnostics()
    print(json.dumps({"sync": asdict(summary), "api": diagnostics}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
