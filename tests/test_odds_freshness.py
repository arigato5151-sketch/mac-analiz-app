"""Live value claims must use the same freshness limit as their UI label."""

from datetime import datetime, timedelta, timezone

import pandas as pd

from app.components.freshness import odds_are_current


def test_current_odds_accepts_a_recent_timestamp() -> None:
    now = datetime(2026, 10, 1, 12, tzinfo=timezone.utc)
    assert odds_are_current(now - timedelta(hours=11), now=now)
    assert odds_are_current(pd.Timestamp(now - timedelta(hours=12)), now=now)


def test_current_odds_rejects_stale_missing_and_future_quotes() -> None:
    now = datetime(2026, 10, 1, 12, tzinfo=timezone.utc)
    assert not odds_are_current(now - timedelta(hours=12, seconds=1), now=now)
    assert not odds_are_current(None, now=now)
    assert not odds_are_current("not a date", now=now)
    assert not odds_are_current(now + timedelta(minutes=1), now=now)
    assert not odds_are_current(datetime(2026, 10, 1, 10), now=now)
