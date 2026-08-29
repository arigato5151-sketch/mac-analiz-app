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
