import numpy as np

from app.components.data import _nesine_market_key
from data_pipeline.nesine_odds import _is_requested_nesine_market
from models.market_forecast import derive_requested_market_probabilities
from models.poisson_model import PoissonPrediction


def test_nesine_market_labels_map_requested_markets():
    cases = [
        ({"market_name": "Maç Sonucu", "selection_name": "1"}, "home_win"),
        ({"market_name": "Maç Sonucu", "selection_name": "MS 1"}, "home_win"),
        ({"market_name": "Maç Sonucu", "selection_name": "MS X"}, "draw"),
        ({"market_name": "Çifte Şans", "selection_name": "1X"}, "double_chance_1x"),
        ({"market_name": "Çifte Şans", "selection_name": "ÇŞ 1-X"}, "double_chance_1x"),
        ({"market_name": "Karşılıklı Gol", "selection_name": "KG Var"}, "btts_yes"),
        ({"market_name": "1. Yarı Maç Sonucu", "selection_name": "X"}, "first_half_draw"),
        ({"market_name": "2. Yarı Maç Sonucu", "selection_name": "2"}, "second_half_away_win"),
        ({"market_name": "Maç Sonucu ve 3.5 Alt/Üst", "selection_name": "1 ve Alt"}, "home_win_under_3_5"),
        ({"market_name": "Maç Sonucu ve 1.5 Alt/Üst", "selection_name": "MS1 & Alt"}, "home_win_under_1_5"),
        ({"market_name": "Maç Sonucu ve Karşılıklı Gol", "selection_name": "MS2 & Var"}, "away_win_btts_yes"),
        ({"market_name": "İlk Yarı/Maç Sonucu", "selection_name": "1/1"}, None),
        ({"market_name": "Karşılıklı Gol", "selection_name": "Var"}, "btts_yes"),
        ({"market_name": "Ev Sahibi 1.Y 0,5 Gol Alt/Üst", "selection_name": "Üst"}, "home_first_half_over_0_5"),
        ({"market_name": "Toplam Korner Alt/Üst", "selection_name": "Üst 9,5"}, "corners_over_9_5"),
    ]
    assert [_nesine_market_key(row) for row, _ in cases] == [expected for _, expected in cases]


def test_nesine_collector_excludes_half_time_full_time_combo_from_half_result():
    assert not _is_requested_nesine_market("İlk Yarı/Maç Sonucu", "1/1")
    assert _is_requested_nesine_market("1. Yarı Maç Sonucu", "1")


def test_requested_market_probabilities_cover_combinations_and_halves():
    matrix = np.zeros((7, 7))
    matrix[0, 0] = 1 / 3
    matrix[1, 1] = 1 / 3
    matrix[1, 2] = 1 / 3
    prediction = PoissonPrediction(
        home_expected_goals=1.4,
        away_expected_goals=1.1,
        score_matrix=matrix,
        prob_home_win=1 / 3,
        prob_draw=1 / 3,
        prob_away_win=1 / 3,
        prob_over_2_5=0.5,
        prob_btts=0.5,
        dixon_coles_rho=0.0,
    )
    markets = derive_requested_market_probabilities(
        prediction,
        {"first_half_home_win": 0.4, "second_half_away_win": 0.3},
    )
    assert markets["double_chance_1x"] == 2 / 3
    assert 0 <= markets["home_win_over_3_5"] <= 1
    assert markets["first_half_home_win"] == 0.4
