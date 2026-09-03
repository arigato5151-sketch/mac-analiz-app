from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from data_pipeline.fetch_injuries import sync_injuries
from data_pipeline.refresh_context import current_elo_ratings, upcoming_team_targets


class SelectDb:
    def __init__(self, matches: list[dict[str, Any]]) -> None:
        self.matches = matches

    def select_all(self, table: str, **_: Any) -> list[dict[str, Any]]:
        assert table == "matches"
        return self.matches


class EmptyApi:
    def get(self, endpoint: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        assert endpoint == "injuries"
        return []


class DeleteTrackingDb:
    def __init__(self) -> None:
        self.deleted_team_ids: list[str] = []
        self.upserts: list[tuple[str, list[dict[str, Any]]]] = []

    def upsert(self, table: str, rows: list[dict[str, Any]], **__: Any) -> list[dict[str, Any]]:
        self.upserts.append((table, rows))
        return []

    def delete(self, table: str, *, filters: dict[str, str]) -> None:
        assert table == "player_availability"
        self.deleted_team_ids.append(filters["team_id"])

    def insert(self, table: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        self.upserts.append((table, rows))
        return []


def test_upcoming_team_targets_groups_and_filters_leagues() -> None:
    db = SelectDb(
        [
            {"league_id": 39, "home_team_id": 1, "away_team_id": 2},
            {"league_id": 39, "home_team_id": 2, "away_team_id": 3},
            {"league_id": 999, "home_team_id": 4, "away_team_id": 5},
        ]
    )

    targets = upcoming_team_targets(
        db, now=datetime(2026, 8, 26, tzinfo=timezone.utc), horizon_days=3
    )

    assert targets == {39: {1, 2, 3}}


def test_current_elo_ratings_reflect_completed_result() -> None:
    ratings = current_elo_ratings(
        [
            {
                "id": 1,
                "league_id": 39,
                "home_team_id": 1,
                "away_team_id": 2,
                "match_date": "2026-01-01T12:00:00+00:00",
                "home_score": 2,
                "away_score": 0,
            }
        ],
        {1, 2},
    )

    assert ratings[1] > 1500
    assert ratings[2] < 1500


def test_injury_sync_clears_teams_with_no_current_injuries() -> None:
    db = DeleteTrackingDb()

    written = sync_injuries(EmptyApi(), db, league_id=39, team_ids={10, 20})

    assert written == 0
    assert set(db.deleted_team_ids) == {"eq.10", "eq.20"}
    snapshots = next(rows for table, rows in db.upserts if table == "team_availability_status")
    assert {row["team_id"] for row in snapshots} == {10, 20}
    assert {row["available_count"] for row in snapshots} == {22}
    assert {row["unavailable_count"] for row in snapshots} == {0}
    history = next(rows for table, rows in db.upserts if table == "team_availability_history")
    assert {row["team_id"] for row in history} == {10, 20}
