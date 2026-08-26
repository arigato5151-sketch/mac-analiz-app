"""Independent Poisson baseline for football score probabilities."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np
from numpy.typing import NDArray
from scipy.stats import poisson


@dataclass(frozen=True, slots=True)
class PoissonPrediction:
    home_expected_goals: float
    away_expected_goals: float
    score_matrix: NDArray[np.float64]
    prob_home_win: float
    prob_draw: float
    prob_away_win: float
    prob_over_2_5: float
    prob_btts: float

    @property
    def most_likely_score(self) -> tuple[int, int]:
        flat_index = int(np.argmax(self.score_matrix))
        return tuple(int(value) for value in np.unravel_index(flat_index, self.score_matrix.shape))

    def score_probability(self, home_goals: int, away_goals: int) -> float:
        if home_goals < 0 or away_goals < 0:
            raise ValueError("Goal counts cannot be negative")
        if home_goals >= self.score_matrix.shape[0] or away_goals >= self.score_matrix.shape[1]:
            return 0.0
        return float(self.score_matrix[home_goals, away_goals])

    def as_probabilities(self) -> dict[str, float]:
        return {
            "home_win": self.prob_home_win,
            "draw": self.prob_draw,
            "away_win": self.prob_away_win,
            "over_2_5": self.prob_over_2_5,
            "btts": self.prob_btts,
        }


def _validate_expected_goals(home: float, away: float, max_goals: int) -> None:
    if not all(isfinite(value) and value > 0 for value in (home, away)):
        raise ValueError("Expected goals must be finite positive values")
    if max_goals < 5 or max_goals > 20:
        raise ValueError("max_goals must be between 5 and 20")


def predict_score_probabilities(
    home_expected_goals: float,
    away_expected_goals: float,
    *,
    max_goals: int = 10,
) -> PoissonPrediction:
    """Build a normalized score matrix and derived market probabilities."""
    _validate_expected_goals(home_expected_goals, away_expected_goals, max_goals)

    goals = np.arange(max_goals + 1)
    home_pmf = poisson.pmf(goals, home_expected_goals)
    away_pmf = poisson.pmf(goals, away_expected_goals)
    matrix = np.outer(home_pmf, away_pmf).astype(np.float64)

    # Normalize the tiny truncated tail so all downstream probabilities are
    # internally consistent and sum exactly to one.
    matrix_sum = float(matrix.sum())
    if matrix_sum <= 0:
        raise ArithmeticError("Poisson score matrix has zero probability mass")
    matrix /= matrix_sum

    home_win = float(np.tril(matrix, k=-1).sum())
    draw = float(np.trace(matrix))
    away_win = float(np.triu(matrix, k=1).sum())
    indices = np.indices(matrix.shape)
    over_2_5 = float(matrix[(indices[0] + indices[1]) >= 3].sum())
    btts = float(matrix[1:, 1:].sum())

    return PoissonPrediction(
        home_expected_goals=float(home_expected_goals),
        away_expected_goals=float(away_expected_goals),
        score_matrix=matrix,
        prob_home_win=home_win,
        prob_draw=draw,
        prob_away_win=away_win,
        prob_over_2_5=over_2_5,
        prob_btts=btts,
    )


def estimate_expected_goals(
    *,
    home_avg_scored: float,
    home_avg_conceded: float,
    away_avg_scored: float,
    away_avg_conceded: float,
    league_avg_home_goals: float,
    league_avg_away_goals: float,
) -> tuple[float, float]:
    """Estimate lambdas from home/away attack and defensive strengths."""
    inputs = (
        home_avg_scored,
        home_avg_conceded,
        away_avg_scored,
        away_avg_conceded,
        league_avg_home_goals,
        league_avg_away_goals,
    )
    if not all(isfinite(value) and value >= 0 for value in inputs):
        raise ValueError("Goal averages must be finite non-negative values")
    if league_avg_home_goals <= 0 or league_avg_away_goals <= 0:
        raise ValueError("League goal averages must be greater than zero")

    home_attack = home_avg_scored / league_avg_home_goals
    away_defence = away_avg_conceded / league_avg_home_goals
    away_attack = away_avg_scored / league_avg_away_goals
    home_defence = home_avg_conceded / league_avg_away_goals

    home_lambda = home_attack * away_defence * league_avg_home_goals
    away_lambda = away_attack * home_defence * league_avg_away_goals
    return (
        float(np.clip(home_lambda, 0.05, 6.0)),
        float(np.clip(away_lambda, 0.05, 6.0)),
    )


def predict_from_averages(**averages: float) -> PoissonPrediction:
    home_lambda, away_lambda = estimate_expected_goals(**averages)
    return predict_score_probabilities(home_lambda, away_lambda)
