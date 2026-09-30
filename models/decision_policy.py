"""Transparent decision rules applied after probability estimation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

MINIMUM_ACTIONABLE_1X2_CONFIDENCE = 0.55
LEAGUE_CONFIDENCE_OVERRIDES: dict[int, float] = {}


def policy_from_metadata(metadata: Mapping[str, object] | None) -> dict[str, object]:
    policy = (metadata or {}).get("decision_policy") if isinstance(metadata, Mapping) else None
    return dict(policy) if isinstance(policy, Mapping) else {}


def select_threshold_from_calibration(
    labels: Sequence[int], probabilities: np.ndarray, *, minimum_coverage: float = 0.20
) -> tuple[float, dict[str, float]]:
    """Select the actionable threshold using calibration rows only."""
    values = np.asarray(probabilities, dtype=float)
    actual = np.asarray(labels, dtype=int)
    if values.ndim != 2 or values.shape[1] != 3 or len(actual) != len(values):
        raise ValueError("Calibration labels and probabilities must be aligned (n, 3)")
    confidence = values.max(axis=1)
    predicted = values.argmax(axis=1)
    candidates: list[tuple[float, float, float]] = []
    for threshold in np.arange(0.50, 0.701, 0.01):
        mask = confidence >= threshold
        coverage = float(mask.mean())
        if coverage >= minimum_coverage:
            candidates.append((float((predicted[mask] == actual[mask]).mean()), coverage, float(threshold)))
    if not candidates:
        return MINIMUM_ACTIONABLE_1X2_CONFIDENCE, {"coverage": 0.0, "accuracy": 0.0}
    accuracy, coverage, threshold = max(candidates, key=lambda row: (row[0], row[1], -row[2]))
    return threshold, {"coverage": coverage, "accuracy": accuracy}


def minimum_confidence_for_league(
    league_id: int | None, *, policy: Mapping[str, object] | None = None
) -> float:
    """Return a threshold learned from validation metadata; never test metrics."""
    configured = policy or {}
    try:
        overrides = {int(key): float(value) for key, value in (configured.get("league_overrides") or {}).items()}
        default = float(configured.get("global_threshold", MINIMUM_ACTIONABLE_1X2_CONFIDENCE))
        if 0 <= default <= 1:
            return overrides.get(int(league_id), default) if league_id is not None else default
    except (AttributeError, TypeError, ValueError):
        pass
    if league_id is None:
        return MINIMUM_ACTIONABLE_1X2_CONFIDENCE
    return MINIMUM_ACTIONABLE_1X2_CONFIDENCE


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
