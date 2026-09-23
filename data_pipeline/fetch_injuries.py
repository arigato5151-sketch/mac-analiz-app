"""Fetch current injuries/suspensions for a configured league."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from typing import Any

from config.leagues import LEAGUES_BY_ID
from config.settings import get_settings
from data_pipeline.api_client import ApiFootballClient
from db.db_client import SupabaseRestClient


MAX_PLAUSIBLE_UNAVAILABLE = 15


class AvailabilityDataQualityError(RuntimeError):
    """Raised before persistence when provider availability data is implausible."""


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
    availability: dict[tuple[int | None, int, str], dict[str, Any]] = {}
    now = datetime.now(timezone.utc).isoformat()
    for item in injuries:
        team = item["team"]
        player = item["player"]
        reason = player.get("reason") or player.get("type") or "injury"
        team_id = int(team["id"])
        fixture_id = (
            int(item["fixture"]["id"])
            if item.get("fixture", {}).get("id") is not None
            else None
        )
        teams[team_id] = {
            "id": team_id,
            "name": team["name"],
            "league_id": league_id,
            "logo_url": team.get("logo"),
        }
        status = _availability_status(reason)
        player_identity = str(player.get("id") or player["name"]).casefold()
        key = (fixture_id, team_id, player_identity)
        availability[key] = {
            "match_id": fixture_id,
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
    team_ids: set[int] | None = None,
    fixture_ids: set[int] | None = None,
) -> int:
    league = LEAGUES_BY_ID.get(league_id)
    if league is None:
        raise ValueError(f"League {league_id} is not configured")
    if fixture_ids:
        injuries = []
        for fixture_id in sorted(fixture_ids):
            fixture_rows = api.get("injuries", {"fixture": fixture_id})
            unexpected_fixtures = {
                int(item.get("fixture", {}).get("id") or 0)
                for item in fixture_rows
                if int(item.get("fixture", {}).get("id") or 0) != fixture_id
            }
            if unexpected_fixtures:
                raise AvailabilityDataQualityError(
                    f"Fixture {fixture_id} returned unrelated injury rows"
                )
            injuries.extend(fixture_rows)
    else:
        # A season-only request returns historical incidents and must never be
        # persisted as the current squad state.
        injuries = api.get(
            "injuries",
            {
                "league": league_id,
                "season": league.season,
                "date": datetime.now(timezone.utc).date().isoformat(),
            },
        )
    teams, rows = transform_injuries(injuries, league_id)
    if team_ids is not None:
        unexpected_teams = {
            int(row["team_id"]) for row in rows if int(row["team_id"]) not in team_ids
        }
        if fixture_ids and unexpected_teams:
            raise AvailabilityDataQualityError(
                "Fixture injury response contains teams outside the requested scope"
            )
        teams = [team for team in teams if int(team["id"]) in team_ids]
        rows = [row for row in rows if int(row["team_id"]) in team_ids]

    player_names_by_team: dict[int, set[str]] = {
        team_id: set() for team_id in (team_ids or set())
    }
    for row in rows:
        player_names_by_team.setdefault(int(row["team_id"]), set()).add(
            str(row["player_name"]).casefold()
        )
    implausible = {
        team_id: len(player_names)
        for team_id, player_names in player_names_by_team.items()
        if len(player_names) > MAX_PLAUSIBLE_UNAVAILABLE
    }
    if implausible:
        raise AvailabilityDataQualityError(
            f"Implausible unavailable-player totals: {implausible}"
        )
    db.upsert("teams", teams, on_conflict="id")

    # Clear every requested team, including teams that are no longer injured.
    refreshed_team_ids = team_ids if team_ids is not None else {
        int(team["id"]) for team in teams
    }
    refreshed_at = datetime.now(timezone.utc).isoformat()
    for team_id in refreshed_team_ids:
        db.delete("player_availability", filters={"team_id": f"eq.{team_id}"})
    if rows:
        db.upsert("player_availability", rows)
    counts_by_team = {
        team_id: len(player_names_by_team.get(team_id, set()))
        for team_id in refreshed_team_ids
    }
    available_by_team = {
        team_id: max(0, 22 - unavailable_count)
        for team_id, unavailable_count in counts_by_team.items()
    }
    db.upsert(
        "team_availability_status",
        [
            {
                "team_id": team_id,
                "match_id": None,
                "refreshed_at": refreshed_at,
                "ingested_at": refreshed_at,
                "observed_at": None,
                "available_count": available_by_team[team_id],
                "unavailable_count": count,
            }
            for team_id, count in counts_by_team.items()
        ],
        on_conflict="team_id",
    )
    # Append-only history keeps future training snapshots causal and auditable.
    history_rows: list[dict[str, Any]] = []
    if fixture_ids:
        counts_by_fixture_team: dict[tuple[int, int], int] = {}
        for row in rows:
            fixture_id = row.get("match_id")
            if fixture_id is not None:
                key = (int(fixture_id), int(row["team_id"]))
                counts_by_fixture_team[key] = counts_by_fixture_team.get(key, 0) + 1
        for (fixture_id, team_id), count in counts_by_fixture_team.items():
            history_rows.append(
                {
                    "team_id": team_id,
                    "match_id": fixture_id,
                    "refreshed_at": refreshed_at,
                    "observed_at": None,
                    "ingested_at": refreshed_at,
                    "available_count": max(0, 22 - count),
                    "unavailable_count": count,
                }
            )
    else:
        history_rows = [
            {
                "team_id": team_id,
                "match_id": None,
                "refreshed_at": refreshed_at,
                "observed_at": None,
                "ingested_at": refreshed_at,
                "available_count": available_by_team[team_id],
                "unavailable_count": count,
            }
            for team_id, count in counts_by_team.items()
        ]
    db.insert(
        "team_availability_history",
        history_rows,
    )
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
