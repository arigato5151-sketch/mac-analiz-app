"""Fetch current injuries/suspensions for a configured league."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from typing import Any

from config.leagues import LEAGUES_BY_ID
from config.settings import get_settings
from data_pipeline.api_client import ApiFootballClient
from db.db_client import SupabaseRestClient


def _availability_status(reason: str) -> str:
    normalized = reason.lower()
    if "suspend" in normalized or "card" in normalized:
        return "suspended"
    if "doubt" in normalized or "questionable" in normalized:
        return "doubtful"
    return "injured"


def transform_injuries(
    injuries: list[dict[str, Any]], league_id: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    teams: dict[int, dict[str, Any]] = {}
    availability: dict[tuple[int, str, str], dict[str, Any]] = {}
    now = datetime.now(timezone.utc).isoformat()
    for item in injuries:
        team = item["team"]
        player = item["player"]
        reason = player.get("reason") or player.get("type") or "injury"
        team_id = int(team["id"])
        teams[team_id] = {
            "id": team_id,
            "name": team["name"],
            "league_id": league_id,
            "logo_url": team.get("logo"),
        }
        status = _availability_status(reason)
        key = (team_id, player["name"], status)
        availability[key] = {
            "team_id": team_id,
            "player_name": player["name"],
            "status": status,
            "expected_return": None,
            "updated_at": now,
        }
    return list(teams.values()), list(availability.values())


def sync_injuries(
    api: ApiFootballClient,
    db: SupabaseRestClient,
    *,
    league_id: int,
) -> int:
    league = LEAGUES_BY_ID.get(league_id)
    if league is None:
        raise ValueError(f"League {league_id} is not configured")
    injuries = api.get("injuries", {"league": league_id, "season": league.season})
    teams, rows = transform_injuries(injuries, league_id)
    db.upsert("teams", teams, on_conflict="id")

    # Replace each affected team's snapshot to avoid stale availability rows.
    for team in teams:
        db.delete("player_availability", filters={"team_id": f"eq.{team['id']}"})
    if rows:
        db.upsert("player_availability", rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--league", type=int, required=True)
    args = parser.parse_args()
    settings = get_settings()
    count = sync_injuries(
        ApiFootballClient(settings.api_football_key),
        SupabaseRestClient(
            settings.supabase_url, settings.supabase_service_role_key
        ),
        league_id=args.league,
    )
    print({"availability_rows": count})


if __name__ == "__main__":
    main()
