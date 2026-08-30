from __future__ import annotations

from datetime import datetime, timedelta, timezone

from data_pipeline.data_quality import assess_morning_quality


def test_quality_report_detects_missing_prediction_and_stale_context() -> None:
    now = datetime(2026, 8, 29, 12, tzinfo=timezone.utc)
    report = assess_morning_quality(
        [{"id": 10, "home_team_id": 1, "away_team_id": 2}],
        [],
        [{"team_id": 1, "calculated_at": now.isoformat()}],
        [{"team_id": 2, "refreshed_at": (now - timedelta(hours=37)).isoformat()}],
        now=now,
    )

    assert report["healthy"] is False
    assert report["missing_prediction_ids"] == [10]
    assert report["stale_form_team_ids"] == [2]
    assert report["stale_availability_team_ids"] == [1, 2]


def test_quality_report_is_healthy_when_every_target_is_fresh() -> None:
    now = datetime(2026, 8, 29, 12, tzinfo=timezone.utc)
    report = assess_morning_quality(
        [{"id": 10, "home_team_id": 1, "away_team_id": 2}],
        [{"match_id": 10}],
        [
            {"team_id": 1, "calculated_at": now.isoformat()},
            {"team_id": 2, "calculated_at": now.isoformat()},
        ],
        [
            {"team_id": 1, "refreshed_at": now.isoformat()},
            {"team_id": 2, "refreshed_at": now.isoformat()},
        ],
        now=now,
    )

    assert report["healthy"] is True


def test_quality_report_rejects_stale_active_matches() -> None:
    now = datetime(2026, 8, 30, 12, tzinfo=timezone.utc)
    report = assess_morning_quality(
        [], [], [], [],
        active_matches=[{"id": 9, "match_date": "2026-08-30T06:00:00+00:00"}],
        now=now,
    )

    assert report["stale_active_match_ids"] == [9]
    assert report["healthy"] is False
