from datetime import datetime, timezone

from models.predict import _availability_at


def test_live_availability_prefers_matching_fixture_snapshot() -> None:
    selected = _availability_at(
        [
            {
                "team_id": 1,
                "match_id": None,
                "ingested_at": "2026-09-23T10:00:00+00:00",
                "available_count": 22,
                "unavailable_count": 0,
            },
            {
                "team_id": 1,
                "match_id": 101,
                "ingested_at": "2026-09-23T09:00:00+00:00",
                "available_count": 18,
                "unavailable_count": 4,
            },
            {
                "team_id": 1,
                "match_id": 102,
                "ingested_at": "2026-09-23T11:00:00+00:00",
                "available_count": 17,
                "unavailable_count": 5,
            },
        ],
        match_id=101,
        team_id=1,
        observed_at=datetime(2026, 9, 23, 12, tzinfo=timezone.utc),
    )

    assert selected == {"available_count": 18, "unavailable_count": 4}


def test_live_availability_ignores_future_snapshots() -> None:
    selected = _availability_at(
        [{
            "team_id": 1,
            "match_id": None,
            "ingested_at": "2026-09-23T18:00:00+00:00",
            "available_count": 10,
            "unavailable_count": 12,
        }],
        match_id=101,
        team_id=1,
        observed_at=datetime(2026, 9, 23, 17, tzinfo=timezone.utc),
    )

    assert selected is None
