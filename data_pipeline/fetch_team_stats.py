"""Calculate and persist recent team form from API-Football fixtures."""

from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
from typing import Any

from config.settings import get_settings
from data_pipeline.api_client import ApiFootballClient
from db.db_client import SupabaseRestClient


def build_team_form(
    fixtures: list[dict[str, Any]], team_id: int
) -> dict[str, Any]:
    finished = [
        item
        for item in fixtures
        if item["fixture"]["status"]["short"] in {"FT", "AET", "PEN"}
    ][:5]
    if not finished:
        raise ValueError(f"No finished fixtures found for team {team_id}")

    scored: list[int] = []
    conceded: list[int] = []
    wins = 0
    home_games = 0
    home_wins = 0
    away_games = 0
    away_wins = 0
    for item in finished:
        is_home = int(item["teams"]["home"]["id"]) == team_id
        goals_for = item["goals"]["home" if is_home else "away"] or 0
        goals_against = item["goals"]["away" if is_home else "home"] or 0
        won = goals_for > goals_against
        scored.append(goals_for)
        conceded.append(goals_against)
        wins += int(won)
        if is_home:
            home_games += 1
            home_wins += int(won)
        else:
            away_games += 1
            away_wins += int(won)

    count = len(finished)
    return {
        "team_id": team_id,
        "calculated_at": datetime.now(timezone.utc).isoformat(),
        "elo_rating": 1500.0,
        "avg_goals_scored_last5": sum(scored) / count,
        "avg_goals_conceded_last5": sum(conceded) / count,
        "win_rate_last5": wins / count,
        "home_away_split": {
            "home_win_rate": home_wins / home_games if home_games else None,
            "away_win_rate": away_wins / away_games if away_games else None,
            "sample_size": count,
        },
    }


def sync_team_form(
    api: ApiFootballClient,
    db: SupabaseRestClient,
    *,
    team_id: int,
    season: int,
) -> dict[str, Any]:
    fixtures = api.get("fixtures", {"team": team_id, "season": season, "last": 5})
    row = build_team_form(fixtures, team_id)
    db.upsert("team_form", [row], on_conflict="team_id,calculated_at")
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--team", type=int, required=True)
    parser.add_argument("--season", type=int, default=date.today().year)
    args = parser.parse_args()
    settings = get_settings()
    row = sync_team_form(
        ApiFootballClient(settings.api_football_key),
        SupabaseRestClient(
            settings.supabase_url, settings.supabase_service_role_key
        ),
        team_id=args.team,
        season=args.season,
    )
    print(row)


if __name__ == "__main__":
    main()
