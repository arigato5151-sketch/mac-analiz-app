"""Independent Poisson baseline for football score probabilities."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, log
from math import factorial as math_factorial
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar
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
    dixon_coles_rho: float

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


def dixon_coles_tau(
    home_goals: int,
    away_goals: int,
    home_expected_goals: float,
    away_expected_goals: float,
    rho: float,
) -> float:
    """Return the Dixon-Coles low-score dependence correction factor.

    Only the 0-0, 0-1, 1-0 and 1-1 scorelines are adjusted. This keeps the
    independent Poisson tail intact while correcting its known low-score bias.
    """
    if not isfinite(rho):
        raise ValueError("Dixon-Coles rho must be finite")
    if home_goals == 0 and away_goals == 0:
        tau = 1 - (home_expected_goals * away_expected_goals * rho)
    elif home_goals == 0 and away_goals == 1:
        tau = 1 + (home_expected_goals * rho)
    elif home_goals == 1 and away_goals == 0:
        tau = 1 + (away_expected_goals * rho)
    elif home_goals == 1 and away_goals == 1:
        tau = 1 - rho
    else:
        return 1.0
    if tau <= 0:
        raise ValueError("Dixon-Coles rho produces a non-positive score probability")
    return float(tau)


def _dixon_coles_log_likelihood(
    rho: float,
    home_goals: NDArray[np.int_],
    away_goals: NDArray[np.int_],
    home_lambdas: NDArray[np.float64],
    away_lambdas: NDArray[np.float64],
) -> float:
    """Negative log-likelihood for Dixon-Coles rho (for minimization)."""
    total = 0.0
    for hg, ag, hl, al in zip(home_goals, away_goals, home_lambdas, away_lambdas):
        tau = 1.0
        if hg == 0 and ag == 0:
            tau = 1 - (hl * al * rho)
        elif hg == 0 and ag == 1:
            tau = 1 + (hl * rho)
        elif hg == 1 and ag == 0:
            tau = 1 + (al * rho)
        elif hg == 1 and ag == 1:
            tau = 1 - rho
        if tau <= 0:
            return 1e10
        log_tau = log(tau)
        log_poisson_h = hg * log(hl) - hl - log(math_factorial(hg))
        log_poisson_a = ag * log(al) - al - log(math_factorial(ag))
        total += log_tau + log_poisson_h + log_poisson_a
    return -total


def estimate_dixon_coles_rho(
    home_goals: list[int] | NDArray[np.int_],
    away_goals: list[int] | NDArray[np.int_],
    home_lambdas: list[float] | NDArray[np.float64],
    away_lambdas: list[float] | NDArray[np.float64],
) -> float:
    """Estimate Dixon-Coles rho via MLE on historical match data.

    Args:
        home_goals: Actual home goals scored in each match.
        away_goals: Actual away goals scored in each match.
        home_lambdas: Expected home goals (lambda) for each match.
        away_lambdas: Expected away goals (lambda) for each match.

    Returns:
        Estimated rho in [-0.5, 0.5]. Returns 0.0 if optimization fails
        or data is insufficient.
    """
    hg_arr = np.asarray(home_goals, dtype=int)
    ag_arr = np.asarray(away_goals, dtype=int)
    hl_arr = np.asarray(home_lambdas, dtype=float)
    al_arr = np.asarray(away_lambdas, dtype=float)

    if len(hg_arr) < 30:
        return 0.0

    try:
        result = minimize_scalar(
            _dixon_coles_log_likelihood,
            bounds=(-0.5, 0.5),
            args=(hg_arr, ag_arr, hl_arr, al_arr),
            method="bounded",
        )
        if result.success and isfinite(result.x):
            return float(np.clip(result.x, -0.5, 0.5))
    except Exception:
        pass
    return 0.0


def estimate_league_dixon_coles_rhos(
    matches: list[dict[str, Any]],
) -> dict[int, float]:
    """Estimate Dixon-Coles rho for each league from historical matches.

    Args:
        matches: List of completed match dicts with keys:
            - league_id: int
            - home_score: int
            - away_score: int
            - home_expected_goals: float (or will be estimated from 5-match form)
            - away_expected_goals: float

    Returns:
        Dict mapping league_id -> estimated rho (clipped to [-0.5, 0.5]).
        Leagues with insufficient data return 0.0.
    """
    if not matches:
        return {}

    df = pd.DataFrame(matches)
    required_cols = {"league_id", "home_score", "away_score", "home_expected_goals", "away_expected_goals"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        # If expected goals are missing, try to estimate from xG or scores
        if {"home_xg", "away_xg"}.issubset(df.columns):
            df["home_expected_goals"] = df["home_xg"]
            df["away_expected_goals"] = df["away_xg"]
        else:
            # Fallback: use simple goal averages as proxy
            missing = required_cols - set(df.columns)
            if missing:
                return {int(lid): 0.0 for lid in df["league_id"].unique()}

    rhos: dict[int, float] = {}
    for league_id, group in df.groupby("league_id"):
        if len(group) < 30:
            rhos[int(league_id)] = 0.0
            continue
        rho = estimate_dixon_coles_rho(
            group["home_score"].tolist(),
            group["away_score"].tolist(),
            group["home_expected_goals"].tolist(),
            group["away_expected_goals"].tolist(),
        )
        rhos[int(league_id)] = rho
    return rhos


def predict_score_probabilities(
    home_expected_goals: float,
    away_expected_goals: float,
    *,
    max_goals: int = 10,
    dixon_coles_rho: float = 0.0,
) -> PoissonPrediction:
    """Build a normalized Poisson/Dixon-Coles score matrix and market probabilities."""
    _validate_expected_goals(home_expected_goals, away_expected_goals, max_goals)

    goals = np.arange(max_goals + 1)
    home_pmf = poisson.pmf(goals, home_expected_goals)
    away_pmf = poisson.pmf(goals, away_expected_goals)
    matrix = np.outer(home_pmf, away_pmf).astype(np.float64)
    for home_goals, away_goals in ((0, 0), (0, 1), (1, 0), (1, 1)):
        matrix[home_goals, away_goals] *= dixon_coles_tau(
            home_goals,
            away_goals,
            home_expected_goals,
            away_expected_goals,
            dixon_coles_rho,
        )

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
        dixon_coles_rho=float(dixon_coles_rho),
    )


def kelly_criterion(
    probability: float,
    decimal_odds: float,
    *,
    kelly_fraction: float = 1.0,
) -> float:
    """Calculate optimal Kelly stake fraction.

    Args:
        probability: Estimated probability of the outcome (0.0 to 1.0).
        decimal_odds: Decimal odds offered by the bookmaker.
        kelly_fraction: Fraction of full Kelly to apply (0.0 to 1.0).
                        1.0 = Full Kelly, 0.5 = Half Kelly, etc.

    Returns:
        Optimal fraction of bankroll to stake (0.0 to 1.0).
        Returns 0.0 if the bet has no edge (negative expected value).

    Note:
        Full Kelly maximizes long-term growth but has high variance.
        Half Kelly (0.5) is commonly used for risk management.
    """
    if not 0.0 <= probability <= 1.0:
        raise ValueError("Probability must be between 0.0 and 1.0")
    if decimal_odds <= 1.0:
        raise ValueError("Decimal odds must be greater than 1.0")
    if not 0.0 <= kelly_fraction <= 1.0:
        raise ValueError("Kelly fraction must be between 0.0 and 1.0")

    # Kelly formula: f* = (bp - q) / b
    # where b = decimal_odds - 1, p = probability, q = 1 - p
    b = decimal_odds - 1.0
    p = probability
    q = 1.0 - probability

    edge = b * p - q
    if edge <= 0:
        return 0.0

    full_kelly = edge / b
    return float(np.clip(full_kelly * kelly_fraction, 0.0, 1.0))


def kelly_stake(
    bankroll: float,
    probability: float,
    decimal_odds: float,
    *,
    kelly_fraction: float = 1.0,
) -> float:
    """Calculate absolute stake amount using Kelly criterion.

    Args:
        bankroll: Current bankroll amount.
        probability: Estimated probability of the outcome.
        decimal_odds: Decimal odds offered by the bookmaker.
        kelly_fraction: Fraction of full Kelly to apply.

    Returns:
        Recommended stake amount in currency units.
    """
    if bankroll <= 0:
        raise ValueError("Bankroll must be positive")
    fraction = kelly_criterion(probability, decimal_odds, kelly_fraction=kelly_fraction)
    return bankroll * fraction


def expected_value(
    probability: float,
    decimal_odds: float,
) -> float:
    """Calculate expected value of a bet.

    Returns the expected profit per unit stake.
    EV > 0 indicates a value bet.
    """
    return probability * (decimal_odds - 1.0) - (1.0 - probability)


def implied_probability(decimal_odds: float) -> float:
    """Convert decimal odds to implied probability (removing overround)."""
    if decimal_odds <= 1.0:
        raise ValueError("Decimal odds must be greater than 1.0")
    return 1.0 / decimal_odds


def remove_overround(
    home_odds: float,
    draw_odds: float,
    away_odds: float,
) -> tuple[float, float, float]:
    """Remove bookmaker overround to get fair probabilities.

    Uses the proportional method to normalize implied probabilities.
    """
    probs = [1.0 / o for o in (home_odds, draw_odds, away_odds)]
    total = sum(probs)
    return tuple(p / total for p in probs)


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


def predict_from_averages(
    *, dixon_coles_rho: float = 0.0, **averages: float
) -> PoissonPrediction:
    home_lambda, away_lambda = estimate_expected_goals(**averages)
    return predict_score_probabilities(
        home_lambda, away_lambda, dixon_coles_rho=dixon_coles_rho
    )
