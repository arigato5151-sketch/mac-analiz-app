from __future__ import annotations

from typing import Any

from monitoring.operational_events import (
    record_api_diagnostics,
    record_event,
    sanitize_context,
)
from monitoring.operational_report import build_report


class FakeDb:
    def __init__(self) -> None:
        self.inserts: list[tuple[str, list[dict[str, Any]]] ] = []

    def insert(self, table: str, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        self.inserts.append((table, records))
        return records


def test_context_omits_credentials_and_redacts_auth_values() -> None:
    assert sanitize_context({
        "api_key": "must-not-be-stored",
        "Authorization": "Bearer must-not-be-stored",
        "request_count": 4,
        "provider_message": "Bearer should-not-be-stored",
    }) == {"request_count": 4}


def test_quota_warning_records_only_normalized_quota_values() -> None:
    db = FakeDb()

    assert record_api_diagnostics(
        db,
        component="fetch_fixtures",
        diagnostics={
            "requests": 12,
            "rate_limit": {"x-ratelimit-requests-remaining": "50"},
            "warning": "API-Football remaining quota is low",
        },
    ) is True

    _, rows = db.inserts[0]
    assert rows[0]["severity"] == "warning"
    assert rows[0]["context"] == {"requests": 12, "remaining": 50}


def test_event_persistence_failure_is_non_blocking() -> None:
    class FailingDb:
        def insert(self, *_: Any, **__: Any) -> list[dict[str, Any]]:
            raise RuntimeError("unavailable")

    assert record_event(
        FailingDb(),  # type: ignore[arg-type]
        severity="error",
        component="worker",
        event_type="failed",
        message="safe error",
    ) is False


def test_operational_report_only_exposes_aggregate_diagnostics() -> None:
    report = build_report([
        {"occurred_at": "2026-09-02T10:00:00Z", "severity": "error", "component": "fetch_results", "event_type": "operation_failed"},
        {"occurred_at": "2026-09-02T09:00:00Z", "severity": "info", "component": "fetch_results", "event_type": "api_quota"},
    ])

    assert report["by_severity"] == {"error": 1, "info": 1}
    assert report["latest_warning_or_error"] == {
        "occurred_at": "2026-09-02T10:00:00Z",
        "severity": "error",
        "component": "fetch_results",
        "event_type": "operation_failed",
    }
