from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from data_pipeline.fetch_injuries import (
    AvailabilityDataQualityError,
    MAX_PLAUSIBLE_UNAVAILABLE,
    sync_injuries,
    transform_injuries,
)
from data_pipeline.refresh_context import (
    current_elo_ratings,
    upcoming_context_targets,
    upcoming_team_targets,
)


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
            {"id": 11, "league_id": 39, "home_team_id": 1, "away_team_id": 2},
            {"id": 12, "league_id": 39, "home_team_id": 2, "away_team_id": 3},
            {"id": 13, "league_id": 999, "home_team_id": 4, "away_team_id": 5},
        ]
    )

    targets = upcoming_team_targets(
        db, now=datetime(2026, 8, 26, tzinfo=timezone.utc), horizon_days=3
    )

    assert targets == {39: {1, 2, 3}}


def test_upcoming_context_targets_keep_exact_fixture_scope() -> None:
    targets = upcoming_context_targets(
        SelectDb(
            [
                {"id": 11, "league_id": 39, "home_team_id": 1, "away_team_id": 2},
                {"id": 12, "league_id": 39, "home_team_id": 2, "away_team_id": 3},
            ]
        ),
        now=datetime(2026, 8, 26, tzinfo=timezone.utc),
        horizon_days=3,
    )

    assert targets[39].team_ids == frozenset({1, 2, 3})
    assert targets[39].fixture_ids == frozenset({11, 12})


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


def test_transform_injuries_keeps_only_past_timezone_aware_provider_time() -> None:
    _, rows = transform_injuries(
        [
            {
                "fixture": {"id": 101},
                "team": {"id": 10, "name": "Team 10"},
                "player": {
                    "id": 7,
                    "name": "Player",
                    "reason": "Injury",
                    "updated_at": "2026-09-23T12:00:00+00:00",
                },
            },
            {
                "fixture": {"id": 101},
                "team": {"id": 10, "name": "Team 10"},
                "player": {
                    "id": 8,
                    "name": "Player 2",
                    "reason": "Injury",
                    "updated_at": "not-a-timestamp",
                },
            },
        ],
        39,
    )

    assert rows[0]["observed_at"] == "2026-09-23T12:00:00+00:00"
    assert rows[1]["observed_at"] is None


def test_injury_sync_queries_exact_fixtures_and_counts_unique_players() -> None:
    class FixtureApi:
        def __init__(self) -> None:
            self.fixtures: list[int] = []

        def get(self, endpoint: str, params: dict[str, Any]) -> list[dict[str, Any]]:
            assert endpoint == "injuries"
            fixture_id = int(params["fixture"])
            self.fixtures.append(fixture_id)
            return [
                {
                    "fixture": {"id": fixture_id},
                    "team": {"id": 10, "name": "Team 10"},
                    "player": {"id": 7, "name": "Player", "reason": "Injury"},
                }
            ]

    api = FixtureApi()
    db = DeleteTrackingDb()

    written = sync_injuries(
        api,
        db,
        league_id=39,
        team_ids={10},
        fixture_ids={101, 102},
    )

    assert api.fixtures == [101, 102]
    assert written == 2
    player_rows = next(rows for table, rows in db.upserts if table == "player_availability")
    assert {row["match_id"] for row in player_rows} == {101, 102}
    snapshots = next(rows for table, rows in db.upserts if table == "team_availability_status")
    assert snapshots[0]["unavailable_count"] == 1


def test_injury_sync_rejects_implausible_totals_before_delete() -> None:
    class ImplausibleApi:
        def get(self, _endpoint: str, params: dict[str, Any]) -> list[dict[str, Any]]:
            fixture_id = int(params["fixture"])
            return [
                {
                    "fixture": {"id": fixture_id},
                    "team": {"id": 10, "name": "Team 10"},
                    "player": {
                        "id": index,
                        "name": f"Player {index}",
                        "reason": "Injury",
                    },
                }
                for index in range(MAX_PLAUSIBLE_UNAVAILABLE + 1)
            ]

    db = DeleteTrackingDb()

    with pytest.raises(AvailabilityDataQualityError, match="Implausible"):
        sync_injuries(
            ImplausibleApi(),
            db,
            league_id=39,
            team_ids={10},
            fixture_ids={101},
        )

    assert db.deleted_team_ids == []
