from __future__ import annotations

from datetime import date
from typing import Any

import pytest

from data_pipeline.fetch_fixtures import (
    normalize_status,
    sync_fixtures,
    transform_fixtures,
)
from data_pipeline.fetch_injuries import transform_injuries
from data_pipeline.fetch_lineups import transform_lineups
from data_pipeline.api_client import ApiFootballClient
from data_pipeline.fetch_team_stats import build_team_form
from data_pipeline.fetch_results import extract_expected_goals, sync_recent_results
from data_pipeline.backfill import backfill_history


def fixture(
    fixture_id: int = 100,
    *,
    league_id: int = 39,
    home_id: int = 1,
    away_id: int = 2,
    home_goals: int | None = None,
    away_goals: int | None = None,
    status: str = "NS",
) -> dict[str, Any]:
    return {
        "fixture": {
            "id": fixture_id,
            "date": "2026-08-25T20:00:00+03:00",
            "status": {"short": status},
        },
        "league": {"id": league_id},
        "teams": {
            "home": {"id": home_id, "name": f"Home {home_id}", "logo": "h.png"},
            "away": {"id": away_id, "name": f"Away {away_id}", "logo": "a.png"},
        },
        "goals": {"home": home_goals, "away": away_goals},
    }


class FakeApi:
    def __init__(self, response: list[dict[str, Any]]) -> None:
        self.response = response
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get(self, endpoint: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        self.calls.append((endpoint, params))
        return self.response


class FakeDb:
    def __init__(self) -> None:
        self.upserts: list[tuple[str, list[dict[str, Any]], str | None]] = []

    def upsert(
        self, table: str, rows: list[dict[str, Any]], *, on_conflict: str | None = None
    ) -> list[dict[str, Any]]:
        self.upserts.append((table, rows, on_conflict))
        return rows

    def select_all(self, *_: Any, **__: Any) -> list[dict[str, Any]]:
        return []


@pytest.mark.parametrize(
    ("short", "expected"),
    [("NS", "scheduled"), ("1H", "live"), ("FT", "finished"), ("PST", "postponed")],
)
def test_normalize_status(short: str, expected: str) -> None:
    assert normalize_status(short) == expected


def test_transform_fixtures_deduplicates_teams() -> None:
    teams, matches = transform_fixtures(
        [fixture(), fixture(101, home_id=1, away_id=3)]
    )

    assert {team["id"] for team in teams} == {1, 2, 3}
    assert [match["id"] for match in matches] == [100, 101]


def test_sync_fixtures_filters_untracked_and_preserves_fk_order() -> None:
    api = FakeApi([fixture(), fixture(200, league_id=999)])
    db = FakeDb()

    summary = sync_fixtures(
        api, db, start_date=date(2026, 8, 25), end_date=date(2026, 8, 25)
    )

    assert summary.tracked_fixtures == 1
    assert [call[0] for call in db.upserts] == ["leagues", "teams", "matches"]
    assert api.calls == [
        ("fixtures", {"date": "2026-08-25", "timezone": "Europe/Istanbul"})
    ]


def test_sync_recent_results_reconciles_the_configured_lookback() -> None:
    api = FakeApi([])
    db = FakeDb()

    sync_recent_results(api, db, today=date(2026, 8, 30), lookback_days=2)

    assert api.calls == [
        ("fixtures", {"date": "2026-08-28", "timezone": "Europe/Istanbul"}),
        ("fixtures", {"date": "2026-08-29", "timezone": "Europe/Istanbul"}),
        ("fixtures", {"date": "2026-08-30", "timezone": "Europe/Istanbul"}),
    ]


def test_extract_expected_goals_accepts_provider_statistic() -> None:
    payload = [
        {"team": {"id": 1}, "statistics": [{"type": "expected_goals", "value": "1.45"}]},
        {"team": {"id": 2}, "statistics": [{"type": "Expected Goals", "value": "0.80"}]},
    ]
    assert extract_expected_goals(payload, home_team_id=1, away_team_id=2) == (1.45, 0.8)


def test_build_team_form_uses_team_perspective() -> None:
    fixtures = [
        fixture(status="FT", home_goals=2, away_goals=0),
        fixture(101, home_id=3, away_id=1, status="FT", home_goals=1, away_goals=1),
    ]

    row = build_team_form(fixtures, team_id=1)

    assert row["avg_goals_scored_last5"] == 1.5
    assert row["avg_goals_conceded_last5"] == 0.5
    assert row["win_rate_last5"] == 0.5
    assert row["home_away_split"]["sample_size"] == 2


def test_transform_injuries_maps_status_and_deduplicates() -> None:
    item = {
        "team": {"id": 1, "name": "Team", "logo": "team.png"},
        "player": {"id": 7, "name": "Player", "reason": "Red Card suspension"},
    }

    teams, rows = transform_injuries([item, item], league_id=39)

    assert len(teams) == 1
    assert len(rows) == 1
    assert rows[0]["status"] == "suspended"


def test_transform_lineups_accepts_only_complete_official_xis() -> None:
    full_xi = [
        {"player": {"id": index, "name": f"Player {index}", "pos": "M"}}
        for index in range(11)
    ]
    rows = transform_lineups(
        99,
        [
            {"team": {"id": 1}, "formation": "4-3-3", "coach": {"name": "Coach"}, "startXI": full_xi, "substitutes": []},
            {"team": {"id": 2}, "startXI": full_xi[:10]},
        ],
    )

    assert len(rows) == 1
    assert rows[0]["match_id"] == 99
    assert rows[0]["formation"] == "4-3-3"
    assert len(rows[0]["starters"]) == 11


def test_api_diagnostics_warns_when_remaining_quota_is_low() -> None:
    client = ApiFootballClient("test-key")
    client.request_count = 12
    client.last_rate_limit = {"x-ratelimit-requests-remaining": "50"}

    diagnostics = client.diagnostics()

    assert diagnostics["requests"] == 12
    assert diagnostics["warning"] == "API-Football remaining quota is low"


def test_backfill_only_writes_finished_matches() -> None:
    api = FakeApi(
        [
            fixture(status="FT", home_goals=2, away_goals=1),
            fixture(101, status="NS"),
        ]
    )
    db = FakeDb()

    summary = backfill_history(
        api, db, league_ids=[39], seasons=[2025], batch_size=1
    )

    assert summary.completed_pairs == 1
    assert summary.finished_matches == 1
    match_batches = [rows for table, rows, _ in db.upserts if table == "matches"]
    assert [[row["id"] for row in batch] for batch in match_batches] == [[100]]
