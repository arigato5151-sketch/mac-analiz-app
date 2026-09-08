"""Robust ISO-8601 timestamp parsing shared across the pipeline."""

from __future__ import annotations

from datetime import datetime


def parse_iso_datetime(value: object) -> datetime:
    """Parse an ISO-8601 timestamp, tolerating a ``Z`` suffix.

    Supabase timestamptz and API payloads may render UTC as either ``Z`` or
    ``+00:00``; ``datetime.fromisoformat`` rejects bare ``Z`` on some platforms.
    Comparing raw strings instead (e.g. ``str(a) > str(b)``) silently mis-orders
    values where fractional seconds or the offset spellings differ, so callers
    should compare parsed datetimes rather than the serialized forms.
    """
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))