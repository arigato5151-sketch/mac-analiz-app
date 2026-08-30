"""Fetch confirmed starting elevens only when API-Football has published them."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from data_pipeline.api_client import ApiFootballClient
from db.db_client import SupabaseRestClient


def transform_lineups(match_id: int, lineups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize only complete XI payloads; empty API responses mean not announced."""
    confirmed_at = datetime.now(timezone.utc).isoformat()
    rows: list[dict[str, Any]] = []
    for lineup in lineups:
        starters = [entry.get("player", {}) for entry in lineup.get("startXI", [])]
        if len(starters) != 11:
            continue
        team = lineup.get("team", {})
        team_id = team.get("id")
        if team_id is None:
            continue
        rows.append(
            {
                "match_id": match_id,
                "team_id": int(team_id),
                "formation": lineup.get("formation"),
                "coach_name": (lineup.get("coach") or {}).get("name"),
                "starters": [
                    {"id": player.get("id"), "name": player.get("name"), "number": player.get("number"), "pos": player.get("pos"), "grid": player.get("grid")}
                    for player in starters
                ],
                "substitutes": [
                    {"id": (entry.get("player") or {}).get("id"), "name": (entry.get("player") or {}).get("name")}
                    for entry in lineup.get("substitutes", [])
                ],
                "confirmed_at": confirmed_at,
            }
        )
    return rows


def sync_fixture_lineups(
    api: ApiFootballClient, db: SupabaseRestClient, *, match_id: int
) -> int:
    rows = transform_lineups(match_id, api.get("fixtures/lineups", {"fixture": match_id}))
    if rows:
        db.upsert("fixture_lineups", rows, on_conflict="match_id,team_id")
    return len(rows)
