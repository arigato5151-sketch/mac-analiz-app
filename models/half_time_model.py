"""Half-time and second-half probabilities from expected-goal lambdas.

The model uses a configurable goal-share prior and the same independent Poisson
assumption as the existing full-match baseline. It is intentionally separate so
it can be backtested before being allowed into coupons.
"""

from __future__ import annotations

from math import isfinite
from typing import Any

import numpy as np
from scipy.stats import poisson


def _score_matrix(home_lambda: float, away_lambda: float, max_goals: int = 8) -> np.ndarray:
    if not all(isfinite(value) and value >= 0 for value in (home_lambda, away_lambda)):
        raise ValueError("Half-time expected goals must be finite and non-negative")
    goals = np.arange(max_goals + 1)
    matrix = np.outer(poisson.pmf(goals, home_lambda), poisson.pmf(goals, away_lambda))
    return matrix / matrix.sum()


def _markets(home_lambda: float, away_lambda: float, prefix: str) -> dict[str, float]:
    matrix = _score_matrix(home_lambda, away_lambda)
    home, away = np.indices(matrix.shape)
    return {
        f"{prefix}_home_win": float(np.tril(matrix, -1).sum()),
        f"{prefix}_draw": float(np.trace(matrix)),
        f"{prefix}_away_win": float(np.triu(matrix, 1).sum()),
        f"{prefix}_over_0_5": float(matrix[(home + away) >= 1].sum()),
        f"{prefix}_over_1_5": float(matrix[(home + away) >= 2].sum()),
        f"{prefix}_under_1_5": float(matrix[(home + away) <= 1].sum()),
        f"{prefix}_btts_yes": float(matrix[1:, 1:].sum()),
    }


def predict_half_time_markets(
    home_expected_goals: float,
    away_expected_goals: float,
    *,
    first_half_share: float = 0.46,
) -> dict[str, float]:
    """Return first/second-half probabilities using a documented goal-share prior."""
    if not 0 < first_half_share < 1:
        raise ValueError("first_half_share must be between 0 and 1")
    first_home, first_away = home_expected_goals * first_half_share, away_expected_goals * first_half_share
    second_home, second_away = home_expected_goals - first_home, away_expected_goals - first_away
    result = _markets(first_home, first_away, "first_half")
    result.update(_markets(second_home, second_away, "second_half"))
    return result
