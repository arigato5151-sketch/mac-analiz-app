"""Pure calculations for the authenticated personal decision report."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime, timedelta, timezone
from typing import Any


def summarize_decisions(
    rows: Iterable[Mapping[str, Any]], *, now: datetime | None = None
) -> dict[str, Any]:
    current = now or datetime.now(timezone.utc)
    start = current - timedelta(days=7)
    recent = []
    for row in rows:
        created = row.get("created_at")
        try:
            timestamp = datetime.fromisoformat(str(created).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            continue
        if timestamp >= start:
            recent.append(row)
    played = [row for row in recent if row.get("decision") == "played"]
    settled = [row for row in played if row.get("was_correct") is not None]
    correct = [row for row in settled if row.get("was_correct")]
    return {
        "window_start": start,
        "decision_count": len(recent),
        "played_count": len(played),
        "passed_count": sum(row.get("decision") == "passed" for row in recent),
        "settled_count": len(settled),
        "correct_count": len(correct),
        "accuracy": len(correct) / len(settled) if settled else None,
        "recent": recent,
    }
