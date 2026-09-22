"""Interpret freshness and severity of the live squad-availability context."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable

from app.components.freshness import (
    FRESHNESS_CURRENT,
    FRESHNESS_STALE,
    FRESHNESS_UNKNOWN,
    freshness_status,
)

# A squad cannot plausibly carry more unavailable players than this; larger
# totals indicate a pipeline defect and are surfaced as an explicit warning.
MAX_PLAUSIBLE_ABSENCES = 15


@dataclass(frozen=True)
class AvailabilitySummary:
    team_id: int
    status: str
    refreshed_at: datetime | None
    injured: int
    suspended: int
    doubtful: int
    quality_warning: str | None = None


def summarize_availability(
    rows: Iterable[dict[str, Any]],
    snapshot: dict[str, Any] | None,
    *,
    team_id: int,
    now: datetime,
) -> AvailabilitySummary:
    """Return a conservative availability state without inferring missing data."""
    if snapshot is None or not snapshot.get("refreshed_at"):
        return AvailabilitySummary(team_id, FRESHNESS_UNKNOWN, None, 0, 0, 0)

    refreshed_at = datetime.fromisoformat(
        str(snapshot["refreshed_at"]).replace("Z", "+00:00")
    )
    if freshness_status(refreshed_at, source="availability", now=now) == FRESHNESS_STALE:
        return AvailabilitySummary(team_id, FRESHNESS_STALE, refreshed_at, 0, 0, 0)

    counts = {"injured": 0, "suspended": 0, "doubtful": 0}
    for row in rows:
        if int(row["team_id"]) != team_id:
            continue
        status = str(row.get("status"))
        if status in counts:
            counts[status] += 1
    total = sum(counts.values())
    quality_warning = None
    if total > MAX_PLAUSIBLE_ABSENCES:
        quality_warning = (
            f"{total} eksik kaydı şüpheli derecede yüksek; veri hattı hatası "
            "olabilir, sayılar doğrulanmadan güvenilir kabul edilmez."
        )
    return AvailabilitySummary(
        team_id,
        FRESHNESS_CURRENT,
        refreshed_at,
        counts["injured"],
        counts["suspended"],
        counts["doubtful"],
        quality_warning,
    )
