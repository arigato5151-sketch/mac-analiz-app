"""Transparent decision rules applied after probability estimation."""

from __future__ import annotations

from collections.abc import Sequence


MINIMUM_ACTIONABLE_1X2_CONFIDENCE = 0.50


def select_1x2(
    probabilities: Sequence[float],
    *,
    threshold: float = MINIMUM_ACTIONABLE_1X2_CONFIDENCE,
) -> tuple[int, float, bool]:
    """Return top class, its probability, and whether it clears the publish gate."""
    values = tuple(float(value) for value in probabilities)
    if len(values) != 3:
        raise ValueError("1X2 selection requires exactly three probabilities")
    if any(value < 0 or value > 1 for value in values):
        raise ValueError("1X2 probabilities must be between zero and one")
    if abs(sum(values) - 1.0) > 0.001:
        raise ValueError("1X2 probabilities must sum to one")
    if not 0 <= threshold <= 1:
        raise ValueError("1X2 confidence threshold must be between zero and one")

    selected_index = max(range(3), key=values.__getitem__)
    confidence = values[selected_index]
    return selected_index, confidence, confidence >= threshold
