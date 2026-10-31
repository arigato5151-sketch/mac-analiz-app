"""Sync the app's upcoming football fixtures from the IdDaa bulletin."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

import requests

from config.leagues import LEAGUES_BY_ID, TRACKED_LEAGUE_IDS
from config.settings import get_settings
from data_pipeline.fetch_iddaa_odds import (
    competition_league_ids,
    fetch_bulletin,
    normalize_team,
    team_names_match,
)
from db.db_client import SupabaseRestClient

ISTANBUL = ZoneInfo("Europe/Istanbul")
TEAM_ID_BASE = 1_500_000_000
TEAM_ID_RANGE = 500_000_000


def _new_team_id(name: str, occupied: set[int]) -> int:
    key = normalize_team(name)
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    candidate = TEAM_ID_BASE + int.from_bytes(digest[:8], "big") % TEAM_ID_RANGE
    while candidate in occupied:
        candidate = TEAM_ID_BASE + (candidate - TEAM_ID_BASE + 1) % TEAM_ID_RANGE
    occupied.add(candidate)
    return candidate


def _team_ids(
    db_teams: list[dict[str, Any]], events: list[tuple[dict[str, Any], int]],
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    occupied = set()
    for team in db_teams:
        occupied.add(int(team["id"]))
        by_name[normalize_team(str(team["name"]))].append(team)

    ids: dict[str, int] = {}
    rows: dict[int, dict[str, Any]] = {}
    for event, league_id in events:
        for field in ("hn", "an"):
            name = str(event.get(field) or "").strip()
            normalized = normalize_team(name)
            if not normalized:
                raise ValueError(f"IdDaa event {event.get('i')} has an empty team name")
            if normalized not in ids:
                candidates = [
                    row for row in db_teams
                    if team_names_match(name, str(row["name"]))
                ]
                established = [row for row in candidates if int(row["id"]) < TEAM_ID_BASE]
                if established:
                    candidates = established
                else:
                    candidates = by_name.get(normalized, candidates)
                same_league = [row for row in candidates if int(row["league_id"]) == league_id]
                if len(same_league) == 1:
                    ids[normalized] = int(same_league[0]["id"])
                elif len(candidates) == 1:
                    ids[normalized] = int(candidates[0]["id"])
                else:
                    ids[normalized] = _new_team_id(name, occupied)
            team_id = ids[normalized]
            if team_id >= TEAM_ID_BASE:
                rows[team_id] = {
                    "id": team_id,
                    "name": name,
                    "league_id": league_id,
                    "logo_url": None,
                }
    return ids, list(rows.values())


def sync_iddaa_fixtures(
    db: SupabaseRestClient,
    *,
    session: requests.Session | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Upsert listed fixtures and hide stale fixtures absent from the bulletin."""
    http = session or requests.Session()
    http.headers.update({"Accept": "application/json", "User-Agent": "mac-analiz-app/1.0"})
    competitions, bulletin_events, _ = fetch_bulletin(http)
    competition_map = competition_league_ids(competitions)
    tracked = set(TRACKED_LEAGUE_IDS)
    events: list[tuple[dict[str, Any], int]] = []
    for event in bulletin_events:
        league_id = competition_map.get(int(event.get("ci", -1)))
        if league_id not in tracked:
            continue
        if not event.get("hn") or not event.get("an") or not event.get("d"):
            continue
        events.append((event, league_id))
    if not events:
        raise RuntimeError("IdDaa bulletin returned no fixtures for the tracked competitions")

    event_days = [
        datetime.fromtimestamp(int(event["d"]), timezone.utc).astimezone(ISTANBUL).date()
        for event, _ in events
    ]
    start_day, end_day = min(event_days), max(event_days)
    current_time = now or datetime.now(timezone.utc)
    if current_time.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    start = datetime.combine(start_day, datetime.min.time(), tzinfo=ISTANBUL).astimezone(timezone.utc)
    end = datetime.combine(end_day + timedelta(days=1), datetime.min.time(), tzinfo=ISTANBUL).astimezone(timezone.utc)
    league_ids = sorted({league_id for _, league_id in events})
    filters = {
        "league_id": f"in.({','.join(map(str, league_ids))})",
        "match_date": f"gte.{start.isoformat()}",
        "and": f"(match_date.lt.{end.isoformat()})",
    }
    matches_in_window = db.select_all(
        "matches",
        columns="id,league_id,home_team_id,away_team_id,match_date,status,home_score,away_score,home_halftime_score,away_halftime_score",
        filters=filters,
        order="match_date.asc,id.asc",
    )
    upcoming_matches = db.select_all(
        "matches",
        columns="id,league_id,home_team_id,away_team_id,match_date,status,home_score,away_score,home_halftime_score,away_halftime_score",
        filters={
            "league_id": f"in.({','.join(map(str, league_ids))})",
            "status": "in.(scheduled,live,unlisted)",
            "match_date": f"gte.{current_time.isoformat()}",
        },
        order="match_date.asc,id.asc",
    )
    matches_by_id = {int(row["id"]): row for row in matches_in_window}
    matches_by_id.update({int(row["id"]): row for row in upcoming_matches})
    matches = list(matches_by_id.values())
    teams = db.select_all("teams", columns="id,league_id,name", order="id.asc")
    team_names = {int(row["id"]): str(row["name"]) for row in teams}
    team_ids, new_teams = _team_ids(teams, events)

    by_pair: dict[tuple[int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for match in matches:
        key = (
            int(match["league_id"]),
            int(match["home_team_id"]),
            int(match["away_team_id"]),
        )
        by_pair[key].append(match)

    fixture_rows: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    seen_existing_ids: set[int] = set()
    new_fixture_ids: set[int] = set()
    for event, league_id in events:
        home_id = team_ids[normalize_team(str(event["hn"]))]
        away_id = team_ids[normalize_team(str(event["an"]))]
        kickoff = datetime.fromtimestamp(int(event["d"]), timezone.utc)
        candidates = by_pair.get((league_id, home_id, away_id), [])
        candidates = [row for row in candidates if int(row["id"]) not in seen_existing_ids]
        target = min(
            candidates,
            key=lambda row: abs(
                (datetime.fromisoformat(str(row["match_date"]).replace("Z", "+00:00")) - kickoff).total_seconds()
            ),
            default=None,
        )
        if target:
            match_id = int(target["id"])
            seen_existing_ids.add(match_id)
            # A finished result is historical data and must not be cleared by a
            # still-listed event; otherwise the bulletin controls fixture state.
            status = str(target.get("status") or "")
            is_finished = status == "finished"
        else:
            match_id = int(event["i"])
            new_fixture_ids.add(match_id)
            is_finished = False
        if match_id in seen_ids:
            raise RuntimeError(f"IdDaa bulletin contains duplicate fixture ID {match_id}")
        seen_ids.add(match_id)
        fixture_rows.append({
            "id": match_id,
            "league_id": league_id,
            "home_team_id": home_id,
            "away_team_id": away_id,
            "match_date": kickoff.isoformat(),
            "status": "finished" if is_finished else "scheduled",
            "home_score": target.get("home_score") if is_finished and target else None,
            "away_score": target.get("away_score") if is_finished and target else None,
            "home_halftime_score": target.get("home_halftime_score") if is_finished and target else None,
            "away_halftime_score": target.get("away_halftime_score") if is_finished and target else None,
        })

    # Use only event IDs that are not already occupied by unrelated matches.
    occupied_matches = db.select_all(
        "matches",
        columns="id,league_id,home_team_id,away_team_id",
        filters={"id": f"in.({','.join(map(str, sorted(new_fixture_ids)))})"},
    ) if new_fixture_ids else []
    events_by_id = {int(event["i"]): (event, league_id) for event, league_id in events}
    collision_ids: set[int] = set()
    for row in occupied_matches:
        fixture_id = int(row["id"])
        if fixture_id in seen_existing_ids:
            continue
        source = events_by_id.get(fixture_id)
        if source is None:
            collision_ids.add(fixture_id)
            continue
        event, league_id = source
        db_home = team_names.get(int(row["home_team_id"]), "")
        db_away = team_names.get(int(row["away_team_id"]), "")
        same_pair = (
            team_names_match(str(event["hn"]), db_home)
            and team_names_match(str(event["an"]), db_away)
        )
        if int(row["league_id"]) != league_id or not same_pair:
            collision_ids.add(fixture_id)
    if collision_ids:
        raise RuntimeError(f"IdDaa fixture IDs conflict with existing matches: {sorted(collision_ids)}")

    league_rows = [
        {
            "id": league_id,
            "name": LEAGUES_BY_ID[league_id].name,
            "country": LEAGUES_BY_ID[league_id].country,
            "season": LEAGUES_BY_ID[league_id].season,
            "is_active": True,
        }
        for league_id in league_ids
        if league_id in LEAGUES_BY_ID
    ]
    db.upsert("leagues", league_rows, on_conflict="id")
    db.upsert("teams", new_teams, on_conflict="id")
    db.upsert("matches", fixture_rows, on_conflict="id")

    # The app should reflect the bulletin: a future fixture no longer listed
    # for a covered day is hidden until IdDaa publishes it again.
    hidden_ids: list[int] = []
    current_utc = current_time.astimezone(timezone.utc)
    for match in matches:
        match_id = int(match["id"])
        kickoff = datetime.fromisoformat(str(match["match_date"]).replace("Z", "+00:00"))
        if (
            match_id not in seen_existing_ids
            and match_id not in seen_ids
            and match.get("status") in {"scheduled", "live"}
            and kickoff > current_utc
        ):
            hidden_ids.append(match_id)
    if hidden_ids:
        db.update(
            "matches",
            {"status": "unlisted"},
            filters={"id": f"in.({','.join(map(str, sorted(hidden_ids)))})"},
        )

    return {
        "source": "iddaa.com",
        "events_seen": len(events),
        "matches_upserted": len(fixture_rows),
        "teams_added": len(new_teams),
        "stale_fixtures_hidden": len(hidden_ids),
        "covered_start": start_day.isoformat(),
        "covered_end": end_day.isoformat(),
        "events_excluded_from_tracked_competitions": len(bulletin_events) - len(events),
    }


def main() -> None:
    settings = get_settings()
    db = SupabaseRestClient(settings.supabase_url, settings.supabase_service_role_key)
    print(json.dumps(sync_iddaa_fixtures(db), ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
