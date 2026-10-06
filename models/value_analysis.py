"""Market-implied probability and value calculations for betting analysis."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite

MIN_VALUE_EV = 0.03
MIN_VALUE_SAMPLE = 30
# Larger apparent edges are more likely to indicate a bad quote or an
# uncalibrated probability than a publishable opportunity.
MAX_PUBLISHABLE_EXPECTED_VALUE = 0.50

COMBO_MARKETS = {
    "home_win_over_2_5": ("home_win", "over_2_5"),
    "home_win_under_2_5": ("home_win", "under_2_5"),
    "draw_over_2_5": ("draw", "over_2_5"),
    "draw_under_2_5": ("draw", "under_2_5"),
    "away_win_over_2_5": ("away_win", "over_2_5"),
    "away_win_under_2_5": ("away_win", "under_2_5"),
    "home_win_btts_yes": ("home_win", "btts_yes"),
    "draw_btts_yes": ("draw", "btts_yes"),
    "away_win_btts_yes": ("away_win", "btts_yes"),
    "over_2_5_btts_yes": ("over_2_5", "btts_yes"),
    "under_2_5_btts_no": ("under_2_5", "btts_no"),
}


def derive_combo_probabilities(
    probabilities: Mapping[str, object],
    *,
    score_matrix: object | None = None,
) -> dict[str, float]:
    """Derive combos from the joint Poisson score matrix, never by independence."""
    if score_matrix is None:
        return {}
    try:
        import numpy as np

        matrix = np.asarray(score_matrix, dtype=float)
        if matrix.ndim != 2 or matrix.size == 0 or not np.isfinite(matrix).all():
            return {}
        total = float(matrix.sum())
        if total <= 0 or not np.isclose(total, 1.0, atol=1e-6):
            return {}
        matrix = matrix / total
        home, away = np.indices(matrix.shape)
        result = home > away
        draw = home == away
        away_win = home < away
        over = home + away >= 3
        under = ~over
        btts = (home >= 1) & (away >= 1)
        masks = {
            "home_win_over_2_5": result & over,
            "home_win_under_2_5": result & under,
            "draw_over_2_5": draw & over,
            "draw_under_2_5": draw & under,
            "away_win_over_2_5": away_win & over,
            "away_win_under_2_5": away_win & under,
            "home_win_btts_yes": result & btts,
            "draw_btts_yes": draw & btts,
            "away_win_btts_yes": away_win & btts,
            "over_2_5_btts_yes": over & btts,
            "under_2_5_btts_no": under & ~btts,
        }
        return {key: float(matrix[mask].sum()) for key, mask in masks.items()}
    except (TypeError, ValueError):
        return {}


@dataclass(frozen=True)
class ValueAssessment:
    """Auditable value assessment for one mutually exclusive market outcome."""

    key: str
    odds: float
    model_probability: float
    implied_probability: float
    fair_market_probability: float
    probability_edge: float
    expected_value: float

    @property
    def has_value(self) -> bool:
        return self.expected_value >= MIN_VALUE_EV


@dataclass(frozen=True)
class ValuePerformance:
    """Flat-stake retrospective result for a selected value threshold."""

    bets: int
    wins: int
    stake: float
    profit: float

    @property
    def strike_rate(self) -> float:
        return self.wins / self.bets if self.bets else 0.0

    @property
    def roi(self) -> float:
        return self.profit / self.stake if self.stake else 0.0


def _probability(value: object, label: str) -> float:
    parsed = float(value)
    if not isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{label} must be between 0 and 1")
    return parsed


def _odds(value: object, label: str) -> float:
    parsed = float(value)
    if not isfinite(parsed) or parsed <= 1.0:
        raise ValueError(f"{label} must be greater than 1")
    return parsed


def assess_market_value(
    model_probabilities: Mapping[str, object],
    odds: Mapping[str, object],
) -> list[ValueAssessment]:
    """Compare model probabilities with a bookmaker's mutually-exclusive odds.

    The raw implied probabilities are normalized to remove bookmaker margin.
    Expected value is calculated as ``model_probability * decimal_odds - 1``.
    Outcomes without both a valid model probability and odds are skipped.
    """
    assessments = []
    market_sets = (
        ("home_win", "draw", "away_win"),
        ("over_2_5", "under_2_5"),
        ("btts_yes", "btts_no"),
    )
    for market_set in market_sets:
        if not all(key in odds and key in model_probabilities for key in market_set):
            continue
        candidates: list[tuple[str, float, float]] = []
        for key in market_set:
            try:
                candidates.append((key, _odds(odds[key], f"odds[{key}]"), _probability(model_probabilities[key], f"probability[{key}]")))
            except (TypeError, ValueError):
                candidates = []
                break
        if not candidates:
            continue
        raw_implied = [1.0 / price for _, price, _ in candidates]
        implied_total = sum(raw_implied)
        if implied_total <= 0:
            continue
        for (key, price, probability), implied in zip(candidates, raw_implied):
            fair_market = implied / implied_total
            assessments.append(
                ValueAssessment(
                    key=key,
                    odds=price,
                    model_probability=probability,
                    implied_probability=implied,
                    fair_market_probability=fair_market,
                    probability_edge=probability - fair_market,
                    expected_value=probability * price - 1.0,
                )
            )
    return assessments


def best_value_assessment(
    model_probabilities: Mapping[str, object],
    odds: Mapping[str, object],
    *,
    minimum_ev: float = MIN_VALUE_EV,
) -> ValueAssessment | None:
    """Return the strongest qualifying value signal, if one exists."""
    if minimum_ev < 0:
        raise ValueError("minimum_ev must not be negative")
    candidates = [
        item for item in assess_market_value(model_probabilities, odds)
        if minimum_ev <= item.expected_value <= MAX_PUBLISHABLE_EXPECTED_VALUE
    ]
    return max(candidates, key=lambda item: item.expected_value, default=None)


def format_percentage(value: float) -> str:
    return f"%{value * 100:.1f}"


def evaluate_flat_stakes(
    bets: list[tuple[ValueAssessment, bool]],
) -> ValuePerformance:
    """Evaluate one-unit stakes using only assessments selected before kickoff."""
    wins = sum(1 for _, won in bets if won)
    profit = sum((assessment.odds - 1.0) if won else -1.0 for assessment, won in bets)
    return ValuePerformance(
        bets=len(bets),
        wins=wins,
        stake=float(len(bets)),
        profit=float(profit),
    )


def value_confidence_status(
    performance: ValuePerformance,
    *,
    median_line_move: float | None = None,
    minimum_sample: int = MIN_VALUE_SAMPLE,
) -> str:
    """Classify value evidence without turning a small sample into a promise."""
    if performance.bets < minimum_sample:
        return "Yetersiz örneklem"
    if performance.roi <= 0:
        return "Negatif ROI"
    if median_line_move is not None and median_line_move < 0:
        return "Piyasa karşısında zayıf"
    return "İzlenebilir value"


def fractional_kelly_stake(
    assessment: ValueAssessment,
    *,
    fraction: float = 0.25,
    cap: float = 0.05,
) -> float:
    """Return a capped fractional-Kelly bankroll share for one assessment."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be between 0 and 1")
    if not 0 < cap <= 1:
        raise ValueError("cap must be between 0 and 1")
    net_odds = assessment.odds - 1.0
    raw = (assessment.model_probability * assessment.odds - 1.0) / net_odds
    return max(0.0, min(cap, raw * fraction))
