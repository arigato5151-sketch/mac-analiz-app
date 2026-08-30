from __future__ import annotations

from datetime import datetime, timedelta, timezone

from notifications.operational_alerts import build_health_report, format_alert


NOW = datetime(2026, 8, 30, 12, tzinfo=timezone.utc)


def test_health_report_detects_only_actionable_gaps() -> None:
    report = build_health_report(
        finished_matches=[{"id": 1, "match_date": (NOW - timedelta(hours=7)).isoformat()}],
        predictions=[{"match_id": 1, "predicted_at": (NOW - timedelta(hours=8)).isoformat(), "match_date": (NOW - timedelta(hours=7)).isoformat()}],
        evaluated_match_ids=set(),
        pending_result_notifications=[{"match_id": 2, "available_at": (NOW - timedelta(hours=4)).isoformat()}],
        active_matches=[{"id": 3, "match_date": (NOW - timedelta(hours=5)).isoformat()}],
        now=NOW,
    )

    assert report["healthy"] is False
    assert report["overdue_evaluation_count"] == 1
    assert report["stuck_result_notification_count"] == 1
    assert report["stale_active_match_count"] == 1


def test_alert_contains_counts_but_never_fixture_identifiers() -> None:
    report = {
        "healthy": False,
        "overdue_evaluation_count": 2,
        "stuck_result_notification_count": 1,
        "stale_active_match_count": 0,
    }

    alert = format_alert(report)

    assert "Gecikmiş değerlendirme: 2" in alert
    assert "Takılı sonuç bildirimi: 1" in alert
    assert "samples" not in alert
