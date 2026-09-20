"""Transparent decision rules applied after probability estimation."""

from __future__ import annotations

from collections.abc import Sequence


# Chronological validation shows a material accuracy lift at 0.55 while still
# retaining useful coverage. Lower-confidence outcomes remain visible as Pass.
MINIMUM_ACTIONABLE_1X2_CONFIDENCE = 0.55

# Training segment metrics are materially weaker in these leagues (Ligue 2 39%,
# Premier League 44% accuracy; MLS carries the worst log-loss at 1.12), so the
# publish gate tightens there to keep low-quality picks out.
LEAGUE_CONFIDENCE_OVERRIDES: dict[int, float] = {
    39: 0.60,  # Premier League
    62: 0.60,  # Ligue 2
    253: 0.60,  # Major League Soccer
}


def minimum_confidence_for_league(league_id: int | None) -> float:
    """Return the publish gate, tightened for historically weak leagues."""
    if league_id is None:
        return MINIMUM_ACTIONABLE_1X2_CONFIDENCE
    return LEAGUE_CONFIDENCE_OVERRIDES.get(
        int(league_id), MINIMUM_ACTIONABLE_1X2_CONFIDENCE
    )


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
