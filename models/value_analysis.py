"""Market-implied probability and value calculations for betting analysis."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping


MIN_VALUE_EV = 0.03
MIN_VALUE_SAMPLE = 30


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
    candidates: list[tuple[str, float, float]] = []
    for key, raw_odds in odds.items():
        if key not in model_probabilities:
            continue
        try:
            price = _odds(raw_odds, f"odds[{key}]")
            probability = _probability(model_probabilities[key], f"probability[{key}]")
        except (TypeError, ValueError):
            continue
        candidates.append((str(key), price, probability))

    if not candidates:
        return []

    raw_implied = [1.0 / price for _, price, _ in candidates]
    implied_total = sum(raw_implied)
    if implied_total <= 0:
        return []

    assessments = []
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
        if item.expected_value >= minimum_ev
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
