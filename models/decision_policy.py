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
    """Select a confidence gate using a conservative accuracy estimate."""
    values = np.asarray(probabilities, dtype=float)
    actual = np.asarray(labels, dtype=int)
    if values.ndim != 2 or values.shape[1] != 3 or len(actual) != len(values):
        raise ValueError("Calibration labels and probabilities must be aligned (n, 3)")
    if len(actual) == 0 or np.any((actual < 0) | (actual > 2)):
        raise ValueError("Calibration labels must be non-empty class indices in [0, 2]")
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("Calibration probabilities must be finite values in [0, 1]")
    if not np.allclose(values.sum(axis=1), 1.0, atol=1e-3):
        raise ValueError("Each calibration probability row must sum to one")
    if not 0 < minimum_coverage <= 1:
        raise ValueError("minimum_coverage must be in (0, 1]")
    confidence = values.max(axis=1)
    predicted = values.argmax(axis=1)
    candidates: list[tuple[float, float, float, float]] = []
    for threshold in np.arange(0.50, 0.701, 0.01):
        mask = confidence >= threshold
        coverage = float(mask.mean())
        if coverage >= minimum_coverage:
            selected_count = int(mask.sum())
            accuracy = float((predicted[mask] == actual[mask]).mean())
            # Wilson lower bound penalizes thresholds supported by few examples.
            z = 1.96
            denominator = 1.0 + z * z / selected_count
            center = accuracy + z * z / (2.0 * selected_count)
            margin = z * np.sqrt(
                accuracy * (1.0 - accuracy) / selected_count
                + z * z / (4.0 * selected_count**2)
            )
            lower_bound = float((center - margin) / denominator)
            candidates.append((lower_bound, coverage, float(threshold), accuracy))
    if not candidates:
        return MINIMUM_ACTIONABLE_1X2_CONFIDENCE, {
            "coverage": 0.0, "accuracy": 0.0, "accuracy_lower_bound": 0.0
        }
    lower_bound, coverage, threshold, accuracy = max(
        candidates, key=lambda row: (row[0], row[1], -row[2])
    )
    return threshold, {
        "coverage": coverage,
        "accuracy": accuracy,
        "accuracy_lower_bound": lower_bound,
    }


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
