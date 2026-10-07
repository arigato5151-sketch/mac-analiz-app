from __future__ import annotations

import pandas as pd

from app.components.ui import (
    SECONDARY_MARKET_COLUMNS,
    binary_market_evaluation,
    compact_dashboard_display,
    dashboard_display,
    diversified_dashboard_display,
    diversified_prediction_cells,
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

    assert prediction_signal(row) == ("2,5 Gol Alt/Üst Üst", 0.63, "Güçlü")
    assert prediction_signal_text(row) == "2,5 Gol Alt/Üst Üst · %63.0 · Güçlü"


def test_prediction_signal_handles_missing_predictions() -> None:
    market, probability, confidence = prediction_signal(pd.Series(dtype=object))

    assert (market, probability, confidence) == ("Tahmin bekleniyor", None, "—")


def test_prediction_signal_uses_publish_threshold_for_medium_confidence() -> None:
    assert prediction_signal(pd.Series({"prob_home_win": 0.55})) == (
        "Maç Sonucu 1",
        0.55,
        "Orta",
    )
    assert prediction_signal(pd.Series({"prob_home_win": 0.54}))[2] == "Düşük"


def test_binary_market_evaluation_remains_independent_from_section_helpers() -> None:
    assert binary_market_evaluation(
        0.65,
        actual_positive=True,
        positive_label="Üst",
        negative_label="Alt",
    ) == ("Üst (%65.0)", "Üst", True)


def test_dashboard_formats_each_diversified_market_as_a_separate_cell() -> None:
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

    cells = diversified_prediction_cells(row)

    assert cells["Çifte Şans"] == "ÇŞ 1-X · %82.0"
    assert cells["1,5 Gol Alt/Üst Üst"] == "%71.0"
    assert cells["3,5 Gol Alt/Üst Alt"] == "%68.0"
    assert cells["Ev Sahibi 0,5 Gol Alt/Üst Üst"] == "%74.0"
    assert cells["Deplasman 0,5 Gol Alt/Üst Üst"] == "—"
    assert not any(column.startswith("Skor ") for column in cells)


def test_dashboard_handles_missing_or_malformed_diversified_markets() -> None:
    empty = {column: "—" for column in SECONDARY_MARKET_COLUMNS}
    assert diversified_prediction_cells(pd.Series(dtype=object)) == empty
    assert diversified_prediction_cells(
        pd.Series({"market_probabilities": "not-json"})
    ) == empty


def test_dashboard_display_includes_each_secondary_market_next_to_match() -> None:
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
                "market_probabilities": (
                    '{"double_chance":{"1X":0.91},'
                    '"correct_scores":[{"score":"1-0","probability":0.18}]}'
                ),
            }
        ]
    )

    display = dashboard_display(frame)

    assert display.loc[0, "Çifte Şans"] == "ÇŞ 1-X · %91.0"
    assert not any(column.startswith("Skor ") for column in display.columns)
    assert list(display.columns[3:10]) == list(SECONDARY_MARKET_COLUMNS)


def test_compact_dashboard_hides_advanced_markets() -> None:
    frame = pd.DataFrame(
        [{
            "match_date": pd.Timestamp("2026-09-14T19:30:00+03:00"),
            "league_name": "Serie A",
            "home_team": "Como",
            "away_team": "Parma",
            "prob_home_win": 0.61,
            "prob_draw": 0.22,
            "prob_away_win": 0.17,
            "prob_over_2_5": 0.56,
            "prob_btts": 0.48,
        }]
    )

    display = compact_dashboard_display(frame)

    assert list(display.columns) == [
        "Tarih", "Lig", "Maç", "1", "X", "2", "2,5 Gol Alt/Üst", "Karşılıklı Gol", "Öne çıkan"
    ]
    assert "Çifte Şans" not in display.columns


def test_dashboard_marks_each_missing_secondary_market() -> None:
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

    display = dashboard_display(frame)
    assert all(display.loc[0, column] == "—" for column in SECONDARY_MARKET_COLUMNS)


def test_legacy_diversified_dashboard_import_remains_compatible() -> None:
    frame = pd.DataFrame(
        [
            {
                "match_date": pd.Timestamp("2026-09-14T19:30:00+03:00"),
                "home_team": "Como",
                "away_team": "Parma",
                "market_probabilities": {"double_chance": {"1X": 0.91}},
            }
        ]
    )

    display = diversified_dashboard_display(frame)

    assert display.loc[0, "Çifte Şans"] == "ÇŞ 1-X · %91.0"
