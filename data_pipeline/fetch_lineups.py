"""Fetch confirmed API-Football lineups for uniquely matched imminent fixtures."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from config.leagues import LEAGUES_BY_ID
from config.settings import get_settings
from data_pipeline.api_client import ApiFootballClient
from data_pipeline.fetch_iddaa_odds import team_names_match
from db.db_client import DatabaseError, SupabaseRestClient

ISTANBUL = ZoneInfo("Europe/Istanbul")
MATCH_LOOKBACK = timedelta(minutes=15)
MATCH_LOOKAHEAD = timedelta(minutes=55)
MAX_KICKOFF_DIFFERENCE = timedelta(hours=2)
NON_PLAYABLE_STATUSES = {"PST", "CANC", "ABD", "AWD", "WO"}


def _as_utc(value: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def _fixture_teams(fixture: dict[str, Any]) -> tuple[str, str]:
    teams = fixture.get("teams") or {}
    home = str((teams.get("home") or {}).get("name") or "")
    away = str((teams.get("away") or {}).get("name") or "")
    return home, away


def _lineup_players(entries: Any) -> list[dict[str, Any]]:
    if not isinstance(entries, list):
        return []
    players = []
    for entry in entries:
        player = entry.get("player") if isinstance(entry, dict) else None
        if not isinstance(player, dict) or not player.get("name"):
            continue
        players.append(
            {
                "id": player.get("id"),
                "name": str(player["name"]),
                "number": player.get("number"),
                "position": player.get("pos"),
                "grid": player.get("grid"),
            }
        )
    return players


def _extract_lineup_rows(
    payload: list[dict[str, Any]], match: dict[str, Any], team_names: dict[int, str],
    *, confirmed_at: str,
) -> list[dict[str, Any]]:
    home_id, away_id = int(match["home_team_id"]), int(match["away_team_id"])
    home_name, away_name = team_names.get(home_id, ""), team_names.get(away_id, "")
    rows = []
    seen_team_ids: set[int] = set()
    for item in payload:
        api_name = str((item.get("team") or {}).get("name") or "")
        matched_ids = [
            team_id for team_id, name in ((home_id, home_name), (away_id, away_name))
            if name and team_names_match(name, api_name)
        ]
        if len(matched_ids) != 1 or matched_ids[0] in seen_team_ids:
            continue
        starters = _lineup_players(item.get("startXI"))
        if len(starters) < 11:
            continue
        team_id = matched_ids[0]
        seen_team_ids.add(team_id)
        rows.append(
            {
                "match_id": int(match["id"]),
                "team_id": team_id,
                "formation": item.get("formation"),
                "coach_name": (item.get("coach") or {}).get("name"),
                "starters": starters[:20],
                "substitutes": _lineup_players(item.get("substitutes"))[:30],
                "confirmed_at": confirmed_at,
            }
        )
    return rows


def _unique_fixture_matches(
    matches: list[dict[str, Any]],
    fixtures_by_competition_day: dict[tuple[int, str], list[dict[str, Any]]],
    team_names: dict[int, str],
) -> dict[int, dict[str, Any]]:
    candidates_by_match: dict[int, list[dict[str, Any]]] = {}
    reverse_candidates: dict[int, set[int]] = defaultdict(set)
    for match in matches:
        match_id = int(match["id"])
        kickoff = _as_utc(match.get("match_date"))
        league_id = int(match["league_id"])
        if kickoff is None or league_id not in LEAGUES_BY_ID:
            continue
        day = kickoff.astimezone(ISTANBUL).date().isoformat()
        home_name = team_names.get(int(match["home_team_id"]), "")
        away_name = team_names.get(int(match["away_team_id"]), "")
        candidates = []
        for fixture in fixtures_by_competition_day.get((league_id, day), []):
            fixture_id = (fixture.get("fixture") or {}).get("id")
            fixture_kickoff = _as_utc((fixture.get("fixture") or {}).get("date"))
            if fixture_id is None or fixture_kickoff is None:
                continue
            if abs(fixture_kickoff - kickoff) > MAX_KICKOFF_DIFFERENCE:
                continue
            if (fixture.get("league") or {}).get("id") != league_id:
                continue
            api_home, api_away = _fixture_teams(fixture)
            if not team_names_match(home_name, api_home) or not team_names_match(away_name, api_away):
                continue
            if str((fixture.get("fixture") or {}).get("status", {}).get("short", "")) in NON_PLAYABLE_STATUSES:
                continue
            candidates.append(fixture)
            reverse_candidates[int(fixture_id)].add(match_id)
        candidates_by_match[match_id] = candidates

    resolved: dict[int, dict[str, Any]] = {}
    for match_id, candidates in candidates_by_match.items():
        if len(candidates) != 1:
            if candidates:
                print(f"Ambiguous API fixture match skipped: match_id={match_id}")
            continue
        candidate = candidates[0]
        fixture_id = int(candidate["fixture"]["id"])
        if len(reverse_candidates[fixture_id]) != 1:
            print(f"Non-unique API fixture mapping skipped: match_id={match_id}")
            continue
        resolved[match_id] = candidate
    return resolved


def sync_upcoming_lineups(
    api: ApiFootballClient,
    db: SupabaseRestClient,
    *,
    now: datetime | None = None,
) -> dict[str, int]:
    """Match imminent IdDaa fixtures to API fixture IDs by unique teams/time."""
    reference = now or datetime.now(timezone.utc)
    if reference.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    reference = reference.astimezone(timezone.utc)
    lower = (reference - MATCH_LOOKBACK).isoformat()
    upper = (reference + MATCH_LOOKAHEAD).isoformat()
    matches = db.select_all(
        "matches",
        columns="id,league_id,home_team_id,away_team_id,match_date",
        filters={
            "status": "eq.scheduled",
            "and": f"(match_date.gte.{lower},match_date.lte.{upper})",
        },
        order="match_date.asc,id.asc",
    )
    if not matches:
        return {"matches_in_window": 0, "mapped": 0, "lineups_saved": 0}

    match_ids = [int(match["id"]) for match in matches]
    existing_rows: list[dict[str, Any]] = []
    for offset in range(0, len(match_ids), 100):
        chunk = match_ids[offset : offset + 100]
        existing_rows.extend(
            db.select_all(
                "fixture_lineups",
                columns="match_id,team_id,starters",
                filters={"match_id": f"in.({','.join(map(str, chunk))})"},
            )
        )
    completed_by_match: dict[int, set[int]] = defaultdict(set)
    for lineup in existing_rows:
        starters = lineup.get("starters") or []
        if isinstance(starters, list) and len(starters) >= 11:
            completed_by_match[int(lineup["match_id"])].add(int(lineup["team_id"]))
    matches = [
        match for match in matches
        if len(completed_by_match[int(match["id"])]) < 2
    ]
    if not matches:
        return {"matches_in_window": len(match_ids), "mapped": 0, "lineups_saved": 0}

    team_ids = sorted(
        {
            int(team_id)
            for match in matches
            for team_id in (match["home_team_id"], match["away_team_id"])
        }
    )
    team_names = {
        int(team["id"]): str(team["name"])
        for team in db.select_all(
            "teams",
            columns="id,name",
            filters={"id": f"in.({','.join(map(str, team_ids))})"},
        )
    }
    competition_days: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for match in matches:
        kickoff = _as_utc(match.get("match_date"))
        if kickoff is None:
            continue
        league_id = int(match["league_id"])
        league = LEAGUES_BY_ID.get(league_id)
        if league is None:
            continue
        local_day = kickoff.astimezone(ISTANBUL).date().isoformat()
        key = (league_id, local_day)
        if key in competition_days:
            continue
        try:
            competition_days[key] = api.get(
                "fixtures",
                {
                    "league": league_id,
                    "season": league.season,
                    "date": local_day,
                    "timezone": "Europe/Istanbul",
                },
            )
        except Exception as exc:
            print(f"Fixture ID lookup skipped for league={league_id}: {type(exc).__name__}")
            competition_days[key] = []
    resolved = _unique_fixture_matches(matches, competition_days, team_names)
    saved = 0
    for match in matches:
        match_id = int(match["id"])
        fixture = resolved.get(match_id)
        if fixture is None:
            continue
        kickoff = _as_utc(match.get("match_date"))
        if kickoff is None:
            continue
        remaining_team_ids = {
            int(match["home_team_id"]), int(match["away_team_id"])
        } - completed_by_match[match_id]
        try:
            lineup_payload = api.get(
                "fixtures/lineups", {"fixture": int(fixture["fixture"]["id"])}
            )
        except Exception as exc:
            print(f"Lineup request skipped for match_id={match_id}: {type(exc).__name__}")
            continue
        rows = _extract_lineup_rows(
            lineup_payload,
            match,
            team_names,
            confirmed_at=reference.isoformat(),
        )
        rows = [row for row in rows if int(row["team_id"]) in remaining_team_ids]
        if not rows:
            continue
        try:
            db.upsert("fixture_lineups", rows, on_conflict="match_id,team_id")
        except DatabaseError as exc:
            print(f"Lineup save skipped for match_id={match_id}: {type(exc).__name__}")
            continue
        saved += len(rows)
    return {
        "matches_in_window": len(match_ids),
        "mapped": len(resolved),
        "lineups_saved": saved,
    }


def main() -> None:
    settings = get_settings()
    api = ApiFootballClient(settings.api_football_key)
    db = SupabaseRestClient(settings.supabase_url, settings.supabase_service_role_key)
    summary = sync_upcoming_lineups(api, db)
    print({**summary, **api.diagnostics()})


if __name__ == "__main__":
    main()
