"""Interpret freshness and severity of the live squad-availability context."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Iterable


FRESHNESS_LIMIT = timedelta(hours=30)


@dataclass(frozen=True)
class AvailabilitySummary:
    team_id: int
    status: str
    refreshed_at: datetime | None
    injured: int
    suspended: int
    doubtful: int


def summarize_availability(
    rows: Iterable[dict[str, Any]],
    snapshot: dict[str, Any] | None,
    *,
    team_id: int,
    now: datetime,
) -> AvailabilitySummary:
    """Return a conservative availability state without inferring missing data."""
    if snapshot is None or not snapshot.get("refreshed_at"):
        return AvailabilitySummary(team_id, "Bilinmiyor", None, 0, 0, 0)

    refreshed_at = datetime.fromisoformat(
        str(snapshot["refreshed_at"]).replace("Z", "+00:00")
    )
    if now - refreshed_at > FRESHNESS_LIMIT:
        return AvailabilitySummary(team_id, "Güncel değil", refreshed_at, 0, 0, 0)

    counts = {"injured": 0, "suspended": 0, "doubtful": 0}
    for row in rows:
        if int(row["team_id"]) != team_id:
            continue
        status = str(row.get("status"))
        if status in counts:
            counts[status] += 1
    return AvailabilitySummary(
        team_id,
        "Güncel",
        refreshed_at,
        counts["injured"],
        counts["suspended"],
        counts["doubtful"],
    )
