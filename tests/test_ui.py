from __future__ import annotations

import pandas as pd

from app.components.ui import (
    dashboard_display,
    diversified_dashboard_display,
    diversified_prediction_text,
    prediction_signal,
    prediction_signal_text,
)


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


def test_dashboard_displays_persisted_diversified_markets() -> None:
    row = pd.Series(
        {
            "market_probabilities": {
                "double_chance": {"1X": 0.82, "X2": 0.43, "12": 0.75},
                "total_goals": {"over_1_5": 0.71, "under_3_5": 0.68},
                "team_goals": {"home_over_0_5": 0.74},
                "correct_scores": [
                    {"score": "1-0", "probability": 0.18},
                    {"score": "2-0", "probability": 0.14},
                    {"score": "1-1", "probability": 0.12},
                ],
            }
        }
    )

    text = diversified_prediction_text(row)

    assert "Çifte şans: 1X %82" in text
    assert "Gol çizgisi: Üst 1.5 %71 · Alt 3.5 %68" in text
    assert "Takım golü: Ev 0.5 Üst %74" in text
    assert "Olası skorlar: 1-0 %18 · 2-0 %14 · 1-1 %12" in text


def test_dashboard_handles_missing_or_malformed_diversified_markets() -> None:
    assert diversified_prediction_text(pd.Series(dtype=object)) == "—"
    assert diversified_prediction_text(pd.Series({"market_probabilities": "not-json"})) == "—"


def test_diversified_dashboard_display_includes_secondary_markets() -> None:
    frame = pd.DataFrame(
        [
            {
                "match_date": pd.Timestamp("2026-09-14T19:30:00+03:00"),
                "league_name": "Serie A",
                "home_team": "Como",
                "away_team": "Parma",
                "prob_home_win": 0.75,
                "prob_draw": 0.16,
                "prob_away_win": 0.09,
                "prob_over_2_5": 0.60,
                "prob_btts": 0.52,
                "market_probabilities": '{"double_chance":{"1X":0.91}}',
            }
        ]
    )

    display = diversified_dashboard_display(frame)

    assert display.loc[0, "Ek tahminler"] == "Çifte şans: 1X %91"
    assert list(display.columns) == ["Tarih", "Maç", "Ek tahminler"]


def test_primary_dashboard_stays_compact() -> None:
    frame = pd.DataFrame(
        [
            {
                "match_date": pd.Timestamp("2026-09-14T19:30:00+03:00"),
                "league_name": "Serie A",
                "home_team": "Como",
                "away_team": "Parma",
                "prob_home_win": 0.75,
                "prob_draw": 0.16,
                "prob_away_win": 0.09,
                "prob_over_2_5": 0.60,
                "prob_btts": 0.52,
            }
        ]
    )

    assert "Ek tahminler" not in dashboard_display(frame).columns
