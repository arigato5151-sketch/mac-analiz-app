"""Derive auditable market-movement and model-edge signals from quote history."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import isfinite
from typing import Any, Mapping, Sequence

from data_pipeline.odds import closing_line_value


@dataclass(frozen=True, slots=True)
class ValueBetSignal:
    """One selection's model edge against a vig-free bookmaker market."""

    market: str
    selection: str
    opening_odds: float
    closing_odds: float
    opening_implied_probability: float
    closing_implied_probability: float
    implied_probability_change: float
    closing_line_value: float
    model_probability: float
    expected_value: float
    is_value_bet: bool


MARKETS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("1-X-2", ("home_win", "draw", "away_win")),
    ("Üst/Alt 2.5", ("over_2_5", "under_2_5")),
    ("KG Var/Yok", ("btts_yes", "btts_no")),
)


def _decimal_odd(value: object) -> float | None:
    try:
        odd = float(value)
    except (TypeError, ValueError):
        return None
    return odd if isfinite(odd) and odd > 1.0 else None


def vig_free_probabilities(
    odds: Mapping[str, object], selections: Sequence[str]
) -> dict[str, float] | None:
    """Normalize inverse decimal odds for one complete bookmaker market."""
    parsed = {selection: _decimal_odd(odds.get(selection)) for selection in selections}
    if any(odd is None for odd in parsed.values()):
        return None
    inverse = {selection: 1.0 / float(odd) for selection, odd in parsed.items()}
    total = sum(inverse.values())
    return {selection: value / total for selection, value in inverse.items()}


def model_selection_probabilities(prediction: Mapping[str, object]) -> dict[str, float]:
    """Map stored model outputs to both sides of each supported market."""
    required = ("prob_home_win", "prob_draw", "prob_away_win", "prob_over_2_5", "prob_btts")
    try:
        values = {key: float(prediction[key]) for key in required}
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Prediction is missing valid market probabilities") from error
    if any(not isfinite(value) or not 0 <= value <= 1 for value in values.values()):
        raise ValueError("Prediction probabilities must be finite values between zero and one")
    if abs(values["prob_home_win"] + values["prob_draw"] + values["prob_away_win"] - 1) > 0.001:
        raise ValueError("1-X-2 prediction probabilities must sum to one")
    return {
        "home_win": values["prob_home_win"],
        "draw": values["prob_draw"],
        "away_win": values["prob_away_win"],
        "over_2_5": values["prob_over_2_5"],
        "under_2_5": 1 - values["prob_over_2_5"],
        "btts_yes": values["prob_btts"],
        "btts_no": 1 - values["prob_btts"],
    }


def _ordered_quotes(quotes: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Ignore malformed snapshots and sort the remaining quotes chronologically."""
    valid = [quote for quote in quotes if isinstance(quote.get("odds"), Mapping) and quote.get("captured_at")]
    return sorted(valid, key=lambda quote: datetime.fromisoformat(str(quote["captured_at"]).replace("Z", "+00:00")))


def analyze_value_bets(
    prediction: Mapping[str, object],
    quotes: Sequence[Mapping[str, Any]],
    *,
    min_expected_value: float = 0.02,
) -> list[ValueBetSignal]:
    """Compare a model prediction with each complete opening/closing market.

    ``expected_value`` is the unit-stake return at the latest (closing) decimal
    price: ``model_probability * closing_odds - 1``. It excludes margin by using
    vig-free probabilities for line movement, while preserving actual offered odds
    for the return calculation. A positive signal is analytical, not a wager order.
    """
    if not isfinite(min_expected_value):
        raise ValueError("min_expected_value must be finite")
    model = model_selection_probabilities(prediction)
    ordered = _ordered_quotes(quotes)
    signals: list[ValueBetSignal] = []

    for market, selections in MARKETS:
        normalized = [
            (quote, vig_free_probabilities(quote["odds"], selections))
            for quote in ordered
        ]
        complete = [(quote, probabilities) for quote, probabilities in normalized if probabilities is not None]
        if not complete:
            continue
        opening_quote, opening = complete[0]
        closing_quote, closing = complete[-1]
        for selection in selections:
            opening_odd = _decimal_odd(opening_quote["odds"].get(selection))
            closing_odd = _decimal_odd(closing_quote["odds"].get(selection))
            if opening_odd is None or closing_odd is None:
                continue
            expected_value = model[selection] * closing_odd - 1
            signals.append(
                ValueBetSignal(
                    market=market,
                    selection=selection,
                    opening_odds=opening_odd,
                    closing_odds=closing_odd,
                    opening_implied_probability=opening[selection],
                    closing_implied_probability=closing[selection],
                    implied_probability_change=closing[selection] - opening[selection],
                    closing_line_value=closing_line_value(opening_odd, closing_odd) or 0.0,
                    model_probability=model[selection],
                    expected_value=expected_value,
                    is_value_bet=expected_value >= min_expected_value,
                )
            )
    return signals
