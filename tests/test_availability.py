from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.components.availability import summarize_availability


def test_availability_is_current_only_with_a_recent_snapshot() -> None:
    now = datetime(2026, 8, 29, 12, tzinfo=timezone.utc)
    summary = summarize_availability(
        [{"team_id": 1, "status": "injured"}, {"team_id": 1, "status": "suspended"}],
        {"team_id": 1, "refreshed_at": (now - timedelta(hours=2)).isoformat()},
        team_id=1,
        now=now,
    )

    assert summary.status == "Güncel"
    assert (summary.injured, summary.suspended, summary.doubtful) == (1, 1, 0)


def test_availability_never_treats_missing_or_stale_data_as_zero_absences() -> None:
    now = datetime(2026, 8, 29, 12, tzinfo=timezone.utc)
    stale = summarize_availability(
        [],
        {"team_id": 1, "refreshed_at": (now - timedelta(hours=31)).isoformat()},
        team_id=1,
        now=now,
    )
    missing = summarize_availability([], None, team_id=2, now=now)

    assert stale.status == "Güncel değil"
    assert missing.status == "Bilinmiyor"


def test_suspiciously_high_totals_carry_a_quality_warning() -> None:
    now = datetime(2026, 8, 29, 12, tzinfo=timezone.utc)
    rows = [{"team_id": 1, "status": "injured"} for _ in range(20)]

    summary = summarize_availability(
        rows,
        {"team_id": 1, "refreshed_at": (now - timedelta(hours=2)).isoformat()},
        team_id=1,
        now=now,
    )

    assert summary.status == "Bilinmiyor"
    assert summary.injured == 0
    assert summary.quality_warning is not None


def test_snapshot_count_without_fixture_players_is_not_reported_as_zero() -> None:
    now = datetime(2026, 8, 29, 12, tzinfo=timezone.utc)

    summary = summarize_availability(
        [],
        {
            "team_id": 1,
            "refreshed_at": (now - timedelta(hours=2)).isoformat(),
            "unavailable_count": 4,
        },
        team_id=1,
        now=now,
    )

    assert summary.status == "Bilinmiyor"
    assert summary.quality_warning is not None


def test_plausible_totals_carry_no_quality_warning() -> None:
    now = datetime(2026, 8, 29, 12, tzinfo=timezone.utc)
    rows = [
        {"team_id": 1, "status": "injured"},
        {"team_id": 1, "status": "suspended"},
        {"team_id": 1, "status": "doubtful"},
    ]

    summary = summarize_availability(
        rows,
        {"team_id": 1, "refreshed_at": (now - timedelta(hours=2)).isoformat()},
        team_id=1,
        now=now,
    )

    assert summary.quality_warning is None
