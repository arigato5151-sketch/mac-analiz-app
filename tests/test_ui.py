from __future__ import annotations

import pandas as pd

from app.components.ui import prediction_signal, prediction_signal_text


def test_prediction_signal_prefers_the_highest_available_market() -> None:
    row = pd.Series(
        {
            "prob_home_win": 0.48,
            "prob_draw": 0.21,
            "prob_away_win": 0.31,
            "prob_over_2_5": 0.63,
            "prob_btts": 0.55,
        }
    )

    assert prediction_signal(row) == ("Üst 2.5", 0.63, "Güçlü")
    assert prediction_signal_text(row) == "Üst 2.5 · %63.0 · Güçlü"


def test_prediction_signal_handles_missing_predictions() -> None:
    market, probability, confidence = prediction_signal(pd.Series(dtype=object))

    assert (market, probability, confidence) == ("Tahmin bekleniyor", None, "—")
