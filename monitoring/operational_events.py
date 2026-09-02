"""Persist bounded operational events without allowing telemetry to break jobs."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, Literal

from db.db_client import SupabaseRestClient


Severity = Literal["info", "warning", "error"]
_MAX_MESSAGE_LENGTH = 300
_MAX_CONTEXT_ITEMS = 20
_SENSITIVE_KEY = re.compile(
    r"(?:api[_-]?key|token|secret|password|authorization|cookie|credential)",
    re.IGNORECASE,
)
_SENSITIVE_VALUE = re.compile(
    r"(?:Bearer\s+|x-apisports-key\s*[:=]|https?://[^\s@]+:[^\s@]+@)",
    re.IGNORECASE,
)


def _safe_text(value: object, *, limit: int = _MAX_MESSAGE_LENGTH) -> str:
    """Produce a bounded string and remove values that resemble credentials."""
    text = " ".join(str(value).split())[:limit]
    return "[redacted]" if _SENSITIVE_VALUE.search(text) else text


def sanitize_context(context: Mapping[str, object] | None) -> dict[str, Any]:
    """Keep only small primitive telemetry and omit sensitive key/value pairs."""
    if not context:
        return {}

    safe: dict[str, Any] = {}
    for key, value in list(context.items())[:_MAX_CONTEXT_ITEMS]:
        name = _safe_text(key, limit=80)
        if _SENSITIVE_KEY.search(name):
            continue
        if isinstance(value, bool | int | float) or value is None:
            safe[name] = value
        elif isinstance(value, str):
            normalized = _safe_text(value, limit=160)
            if normalized != "[redacted]":
                safe[name] = normalized
        elif isinstance(value, (list, tuple)):
            safe[name] = [_safe_text(item, limit=80) for item in value[:10]]
    return safe


def record_event(
    db: SupabaseRestClient,
    *,
    severity: Severity,
    component: str,
    event_type: str,
    message: str,
    context: Mapping[str, object] | None = None,
) -> bool:
    """Record an event, returning false instead of interrupting the caller on failure."""
    try:
        db.insert(
            "operational_events",
            [{
                "severity": severity,
                "component": _safe_text(component, limit=80),
                "event_type": _safe_text(event_type, limit=80),
                "message": _safe_text(message),
                "context": sanitize_context(context),
            }],
        )
    except Exception as error:  # Telemetry must never block the production workflow.
        print(f"Operational event persistence skipped: {type(error).__name__}")
        return False
    return True


def record_exception(
    db: SupabaseRestClient,
    *,
    component: str,
    operation: str,
    error: BaseException,
    context: Mapping[str, object] | None = None,
) -> bool:
    """Store the exception class only; provider response bodies may contain secrets."""
    return record_event(
        db,
        severity="error",
        component=component,
        event_type="operation_failed",
        message=f"{operation} failed: {type(error).__name__}",
        context=context,
    )


def record_api_diagnostics(
    db: SupabaseRestClient, *, component: str, diagnostics: Mapping[str, Any]
) -> bool:
    """Store quota telemetry without persisting raw response headers."""
    raw_headers = diagnostics.get("rate_limit")
    remaining: int | None = None
    if isinstance(raw_headers, Mapping):
        for key, value in raw_headers.items():
            if "remaining" not in str(key).lower():
                continue
            try:
                remaining = int(str(value))
            except ValueError:
                pass
            break
    warning = diagnostics.get("warning")
    return record_event(
        db,
        severity="warning" if warning else "info",
        component=component,
        event_type="api_quota",
        message="API-Football quota warning" if warning else "API-Football quota snapshot",
        context={"requests": diagnostics.get("requests", 0), "remaining": remaining},
    )
