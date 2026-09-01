from __future__ import annotations

import pytest

from data_pipeline.odds import closing_line_value, parse_match_odds
from data_pipeline.value_bets import analyze_value_bets
from notifications.pre_match import pre_match_message


def test_parse_match_odds_extracts_model_markets() -> None:
    odds = parse_match_odds(
        [
            {
                "bookmakers": [
                    {
                        "name": "Bet365",
                        "bets": [
                            {"name": "Match Winner", "values": [{"value": "Home", "odd": "1.80"}, {"value": "Draw", "odd": "3.40"}, {"value": "Away", "odd": "4.20"}]},
                            {"name": "Goals Over/Under", "values": [{"value": "Over 2.5", "odd": "1.95"}, {"value": "Under 2.5", "odd": "1.85"}]},
                            {"name": "Both Teams Score", "values": [{"value": "Yes", "odd": "1.70"}, {"value": "No", "odd": "2.10"}]},
                        ],
                    }
                ]
            }
        ]
    )

    assert odds is not None
    assert odds.home_win == "1.80"
    assert odds.over_2_5 == "1.95"
    assert odds.btts_no == "2.10"
    assert odds.as_snapshot()["home_win"] == "1.80"


def test_pre_match_message_stays_compact_when_bookmaker_odds_exist() -> None:
    odds = parse_match_odds(
        [{"bookmakers": [{"name": "Bet365", "bets": [{"name": "Match Winner", "values": [{"value": "Home", "odd": "1.80"}, {"value": "Draw", "odd": "3.40"}, {"value": "Away", "odd": "4.20"}]}]}]}]
    )
    message = pre_match_message(
        {"match_date": "2026-08-29T16:00:00+00:00"},
        {"prob_home_win": 0.6, "prob_draw": 0.2, "prob_away_win": 0.2, "prob_over_2_5": 0.55, "prob_btts": 0.45},
        home_team="Ev",
        away_team="Deplasman",
        league_name="Lig",
        odds=odds,
    )

    assert "Bet365" not in message
    assert "Tahmin: Ev kazanır %60" in message


def test_closing_line_value_requires_valid_decimal_odds() -> None:
    assert closing_line_value("2.00", "1.80") == pytest.approx(1 / 9)
    assert closing_line_value("1.00", "1.80") is None


def test_value_bet_analysis_uses_vig_free_opening_and_closing_markets() -> None:
    signals = analyze_value_bets(
        {
            "prob_home_win": 0.60,
            "prob_draw": 0.22,
            "prob_away_win": 0.18,
            "prob_over_2_5": 0.55,
            "prob_btts": 0.48,
        },
        [
            {
                "captured_at": "2026-09-02T09:00:00+00:00",
                "odds": {"home_win": "2.10", "draw": "3.40", "away_win": "3.80"},
            },
            {
                "captured_at": "2026-09-02T17:00:00+00:00",
                "odds": {"home_win": "2.00", "draw": "3.50", "away_win": "4.00"},
            },
        ],
    )

    home = next(signal for signal in signals if signal.selection == "home_win")
    assert home.opening_odds == 2.10
    assert home.closing_odds == 2.00
    assert home.implied_probability_change > 0
    assert home.expected_value == pytest.approx(0.20)
    assert home.is_value_bet is True


def test_value_bet_analysis_skips_incomplete_markets() -> None:
    signals = analyze_value_bets(
        {
            "prob_home_win": 0.5,
            "prob_draw": 0.25,
            "prob_away_win": 0.25,
            "prob_over_2_5": 0.5,
            "prob_btts": 0.5,
        },
        [{"captured_at": "2026-09-02T09:00:00+00:00", "odds": {"home_win": "2.0"}}],
    )

    assert signals == []
