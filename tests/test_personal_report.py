from datetime import datetime, timezone

from app.components.personal_report import summarize_decisions


def test_summarize_decisions_reports_recent_decision_quality() -> None:
    now = datetime(2026, 9, 27, tzinfo=timezone.utc)
    rows = [
        {"created_at": "2026-09-26T10:00:00+00:00", "decision": "played", "was_correct": True},
        {"created_at": "2026-09-25T10:00:00+00:00", "decision": "played", "was_correct": False},
        {"created_at": "2026-09-24T10:00:00+00:00", "decision": "passed", "was_correct": None},
        {"created_at": "2026-08-01T10:00:00+00:00", "decision": "played", "was_correct": True},
    ]
    summary = summarize_decisions(rows, now=now)
    assert summary["decision_count"] == 3
    assert summary["played_count"] == 2
    assert summary["passed_count"] == 1
    assert summary["accuracy"] == 0.5
