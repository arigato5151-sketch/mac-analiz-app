from __future__ import annotations

from datetime import datetime, timezone

from data_pipeline.odds import attach_pre_match_odds, vig_free_market_probabilities


def test_pre_match_quote_excludes_post_kickoff_quote() -> None:
    matches = [{"id": 7, "match_date": "2026-01-02T12:00:00+00:00"}]
    rows = attach_pre_match_odds(matches, [
        {"match_id": 7, "captured_at": "2026-01-02T11:00:00+00:00", "odds": {"home_win": "2", "draw": "3", "away_win": "4"}},
        {"match_id": 7, "captured_at": "2026-01-02T13:00:00+00:00", "odds": {"home_win": "9", "draw": "9", "away_win": "9"}},
    ], observed_at=datetime(2026, 1, 2, 14, tzinfo=timezone.utc))

    assert rows[0]["market_odds"]["home_win"] == "2"


def test_vig_free_market_probabilities_normalize_1x2() -> None:
    probabilities = vig_free_market_probabilities({"home_win": "2", "draw": "4", "away_win": "4"})

    assert probabilities is not None
    assert sum(probabilities[key] for key in ("market_implied_home_win", "market_implied_draw", "market_implied_away_win")) == 1
