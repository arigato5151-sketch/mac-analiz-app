from __future__ import annotations

import pandas as pd

from app.components.ui import evaluated_result_display, outcome_prediction_signal


def test_result_display_shows_score_prediction_and_correctness() -> None:
    frame = pd.DataFrame(
        [
            {
                "match_date": pd.Timestamp("2026-08-29T16:30:00", tz="Europe/Istanbul"),
                "league_name": "Süper Lig",
                "home_team": "Ev", "away_team": "Deplasman",
                "home_score": 2, "away_score": 1,
                "actual_result": "home_win", "was_correct": True, "brier_score": 0.21,
                "prob_home_win": 0.61, "prob_draw": 0.22, "prob_away_win": 0.17,
                "prob_over_2_5": 0.57, "prob_btts": 0.48,
            }
        ]
    )

    displayed = evaluated_result_display(frame)

    assert displayed.loc[0, "Skor"] == "2 – 1"
    assert displayed.loc[0, "Model tahmini"] == "Ev kazanır (%61.0)"
    assert displayed.loc[0, "Güven"] == "Güçlü"
    assert displayed.loc[0, "Durum"] == "✓ Doğru"


def test_result_prediction_uses_only_the_evaluated_1x2_market() -> None:
    row = pd.Series(
        {
            "prob_home_win": 0.48,
            "prob_draw": 0.21,
            "prob_away_win": 0.31,
            "prob_over_2_5": 0.70,
        }
    )

    assert outcome_prediction_signal(row) == ("Ev kazanır", 0.48, "Düşük")
