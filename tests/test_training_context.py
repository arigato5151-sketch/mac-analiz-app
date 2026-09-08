from __future__ import annotations

from models.train_model import attach_historical_context


def test_historical_context_uses_only_pre_kickoff_observations() -> None:
    matches = [{
        "id": 10,
        "home_team_id": 1,
        "away_team_id": 2,
        "match_date": "2026-01-10T12:00:00+00:00",
    }]
    availability = [
        {"team_id": 1, "refreshed_at": "2026-01-09T10:00:00+00:00", "available_count": 19},
        {"team_id": 1, "refreshed_at": "2026-01-11T10:00:00+00:00", "available_count": 22},
        {"team_id": 2, "refreshed_at": "2026-01-09T10:00:00+00:00", "available_count": 21},
    ]
    lineups = [
        {"match_id": 10, "team_id": 1, "confirmed_at": "2026-01-10T11:00:00+00:00"},
        {"match_id": 10, "team_id": 2, "confirmed_at": "2026-01-10T11:50:00+00:00"},
    ]

    row = attach_historical_context(
        matches, availability_history=availability, lineups=lineups
    )[0]

    assert row["home_available_count"] == 19
    assert row["home_unavailable_count"] == 3
    assert row["away_available_count"] == 21
    assert row["home_lineup_confirmed"] is True
    assert row["away_lineup_confirmed"] is False
