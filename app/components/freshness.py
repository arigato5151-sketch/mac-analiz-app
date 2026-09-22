"""Shared freshness vocabulary for data-age labels across all screens.

Every "Güncel" label in the UI must come from this module so the freshness
limits and the label strings stay consistent and are tuned in one place.
"""

from __future__ import annotations

from datetime import datetime, timedelta

FRESHNESS_CURRENT = "Güncel"
FRESHNESS_STALE = "Güncel değil"
FRESHNESS_UNKNOWN = "Bilinmiyor"

# Per-source freshness limits used by the UI. A source without a limit always
# reports FRESHNESS_UNKNOWN instead of pretending the data is fresh.
SOURCE_LIMITS: dict[str, timedelta] = {
    "availability": timedelta(hours=30),
    "lineups": timedelta(hours=24),
    "odds": timedelta(hours=12),
}


def freshness_status(
    reference: datetime | None, *, source: str, now: datetime
) -> str:
    """Map a data reference time to the shared freshness vocabulary."""
    limit = SOURCE_LIMITS.get(source)
    if reference is None or limit is None:
        return FRESHNESS_UNKNOWN
    if now - reference > limit:
        return FRESHNESS_STALE
    return FRESHNESS_CURRENT
