from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from app.components.decision_board import (
    build_match_decisions,
    countdown_to,
    requested_match_id,
)


def _match_frame() -> pd.DataFrame:
    base = datetime(2026, 9, 22, 12, tzinfo=timezone.utc)
    return pd.DataFrame(
        [
            {
                "id": 10,
                "league_id": 135,
                "match_date": pd.Timestamp(base),
                "home_team": "Como",
                "away_team": "Parma",
                "prob_home_win": 0.61,
                "prob_draw": 0.22,
                "prob_away_win": 0.17,
                "prob_over_2_5": 0.56,
                "prob_btts": 0.48,
                "model_version": "v1",
                "predicted_at": pd.Timestamp(base - timedelta(hours=2)),
            },
            {
                "id": 11,
                "league_id": 61,
                "match_date": pd.Timestamp(base + timedelta(hours=3)),
                "home_team": "Lyon",
                "away_team": "Rennes",
                "prob_home_win": 0.44,
                "prob_draw": 0.26,
                "prob_away_win": 0.30,
                "prob_over_2_5": 0.50,
                "prob_btts": 0.52,
                "model_version": "v1",
                "predicted_at": pd.Timestamp(base - timedelta(hours=2)),
            },
            {
                "id": 12,
                "league_id": 207,
                "match_date": pd.Timestamp(base + timedelta(hours=5)),
                "home_team": "Sion",
                "away_team": "Zurich",
                "prob_home_win": None,
                "prob_draw": None,
                "prob_away_win": None,
                "prob_over_2_5": None,
                "prob_btts": None,
                "model_version": None,
                "predicted_at": None,
            },
        ]
    )


def test_requested_match_id_resolves_only_known_ids() -> None:
    matches = _match_frame()

    assert requested_match_id({"match_id": "11"}, matches) == 11
    assert requested_match_id({"match_id": "999"}, matches) is None
    assert requested_match_id({"match_id": "abc"}, matches) is None
    assert requested_match_id({}, matches) is None


def test_countdown_formats_hours_minutes_and_days() -> None:
    now = datetime(2026, 9, 22, 12, tzinfo=timezone.utc)

    assert countdown_to(now - timedelta(minutes=5), now) == "başladı"
    assert countdown_to(now + timedelta(minutes=45), now) == "45 dk sonra"
    assert countdown_to(now + timedelta(hours=3, minutes=10), now) == "3 sa 10 dk sonra"
    assert countdown_to(now + timedelta(hours=50), now) == "2 gün sonra"


def test_decisions_feature_only_threshold_clearing_signals() -> None:
    now = datetime(2026, 9, 22, 12, tzinfo=timezone.utc)
    decisions = build_match_decisions(_match_frame(), now=now)
    by_id = {decision.match_id: decision for decision in decisions}

    assert by_id[10].featured is True
    assert by_id[10].confidence == "Güçlü"
    # %50-54 signals sit in the Düşük band and never appear as featured.
    assert by_id[11].featured is False
    assert by_id[11].confidence == "Düşük"
    assert "yayın eşiğinin altında" in (by_id[11].pas_reason or "")
    assert by_id[12].featured is False
    assert by_id[12].pas_reason == "Model tahmini henüz hazırlanmadı"


def test_decisions_show_the_market_gap_only_with_odds() -> None:
    now = datetime(2026, 9, 22, 12, tzinfo=timezone.utc)
    odds = {
        10: {
            "odds": {"home_win": "2.00", "draw": "3.50", "away_win": "4.00"},
            "captured_at": now - timedelta(hours=1),
        }
    }

    with_odds = {
        decision.match_id: decision
        for decision in build_match_decisions(_match_frame(), odds, now=now)
    }
    without_odds = {
        decision.match_id: decision
        for decision in build_match_decisions(_match_frame(), now=now)
    }

    assert with_odds[10].market_gap is not None
    assert "piyasa" in with_odds[10].market_gap
    assert without_odds[10].market_gap is None


def test_stale_odds_never_create_a_market_gap() -> None:
    now = datetime(2026, 9, 22, 12, tzinfo=timezone.utc)
    stale_quote = {
        10: {
            "odds": {"home_win": "2.00", "draw": "3.50", "away_win": "4.00"},
            "captured_at": now - timedelta(hours=13),
        }
    }

    decision = build_match_decisions(_match_frame(), stale_quote, now=now)[0]

    assert decision.market_gap is None
    assert decision.data_status == "Tahmin hazır · güncel oran yok"
