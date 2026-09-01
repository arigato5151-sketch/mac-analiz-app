import numpy as np
import pytest

from models.poisson_model import (
    dixon_coles_tau,
    estimate_expected_goals,
    predict_score_probabilities,
)


def test_dixon_coles_adjusts_only_low_score_cells_and_rho_zero_is_compatible() -> None:
    independent = predict_score_probabilities(1.2, 0.9)
    corrected = predict_score_probabilities(1.2, 0.9, dixon_coles_rho=-0.10)

    assert dixon_coles_tau(2, 2, 1.2, 0.9, -0.10) == 1.0
    assert corrected.score_probability(0, 0) > independent.score_probability(0, 0)
    assert corrected.score_probability(1, 1) > independent.score_probability(1, 1)
    assert corrected.score_matrix.sum() == pytest.approx(1.0, abs=1e-12)


def test_dixon_coles_rejects_rho_that_makes_a_low_score_probability_invalid() -> None:
    with pytest.raises(ValueError, match="non-positive"):
        predict_score_probabilities(1.0, 1.0, dixon_coles_rho=-1.0)


def test_probability_matrix_and_1x2_are_normalized() -> None:
    prediction = predict_score_probabilities(1.7, 1.1)

    assert prediction.score_matrix.sum() == pytest.approx(1.0, abs=1e-12)
    assert (
        prediction.prob_home_win
        + prediction.prob_draw
        + prediction.prob_away_win
    ) == pytest.approx(1.0, abs=1e-12)
    assert np.all(prediction.score_matrix >= 0)


def test_symmetric_teams_have_symmetric_outcomes() -> None:
    prediction = predict_score_probabilities(1.3, 1.3)

    assert prediction.prob_home_win == pytest.approx(prediction.prob_away_win)
    assert prediction.most_likely_score == (1, 1)


def test_stronger_home_lambda_increases_home_win_probability() -> None:
    balanced = predict_score_probabilities(1.2, 1.2)
    stronger_home = predict_score_probabilities(2.2, 0.8)

    assert stronger_home.prob_home_win > balanced.prob_home_win
    assert stronger_home.prob_home_win > 0.65


def test_estimate_expected_goals_clamps_extreme_values() -> None:
    home_lambda, away_lambda = estimate_expected_goals(
        home_avg_scored=20,
        home_avg_conceded=0,
        away_avg_scored=0,
        away_avg_conceded=20,
        league_avg_home_goals=1.5,
        league_avg_away_goals=1.2,
    )

    assert home_lambda == 6.0
    assert away_lambda == 0.05


@pytest.mark.parametrize(("home", "away"), [(0, 1), (-1, 1), (float("nan"), 1)])
def test_invalid_expected_goals_are_rejected(home: float, away: float) -> None:
    with pytest.raises(ValueError):
        predict_score_probabilities(home, away)
