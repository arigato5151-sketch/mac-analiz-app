from __future__ import annotations

from data_pipeline.odds import parse_match_odds
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


def test_pre_match_message_includes_available_bookmaker_odds() -> None:
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

    assert "Bet365 1-X-2: 1 @ 1.80 · X @ 3.40 · 2 @ 4.20" in message
