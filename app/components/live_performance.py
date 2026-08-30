"""Helpers for interpreting a limited set of live prediction outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Iterable


WILSON_95_Z = 1.959963984540054


@dataclass(frozen=True)
class LivePerformanceSummary:
    """A compact, presentation-ready live performance snapshot."""

    sample_size: int
    accuracy: float
    accuracy_lower: float
    accuracy_upper: float
    brier_score: float
    reference_brier_score: float | None
    reference_accuracy: float | None
    status: str


def wilson_interval(successes: int, total: int, z_score: float = WILSON_95_Z) -> tuple[float, float]:
    """Return the two-sided Wilson confidence interval for a binomial rate."""
    if total <= 0:
        raise ValueError("total must be positive")
    if not 0 <= successes <= total:
        raise ValueError("successes must be between zero and total")

    proportion = successes / total
    z_squared = z_score**2
    denominator = 1 + z_squared / total
    centre = (proportion + z_squared / (2 * total)) / denominator
    margin = (
        z_score
        * sqrt((proportion * (1 - proportion) + z_squared / (4 * total)) / total)
        / denominator
    )
    return max(0.0, centre - margin), min(1.0, centre + margin)


def summarize_live_performance(
    correctness: Iterable[bool],
    brier_scores: Iterable[float],
    reference_brier_score: float | None = None,
    reference_accuracy: float | None = None,
    *,
    minimum_sample_size: int = 100,
    degradation_threshold: float = 0.05,
) -> LivePerformanceSummary:
    """Summarize outcomes and apply a transparent, non-statistical drift heuristic."""
    correct_values = [bool(value) for value in correctness]
    score_values = [float(value) for value in brier_scores]
    if not correct_values or not score_values:
        raise ValueError("at least one evaluated prediction is required")
    if len(correct_values) != len(score_values):
        raise ValueError("correctness and brier_scores must have the same length")
    if minimum_sample_size <= 0:
        raise ValueError("minimum_sample_size must be positive")
    if degradation_threshold < 0:
        raise ValueError("degradation_threshold must not be negative")

    sample_size = len(correct_values)
    accuracy = sum(correct_values) / sample_size
    lower, upper = wilson_interval(sum(correct_values), sample_size)
    brier_score = sum(score_values) / sample_size

    if sample_size < minimum_sample_size:
        status = "Yetersiz örneklem"
    elif (
        reference_brier_score is not None
        and reference_brier_score > 0
        and brier_score > reference_brier_score * (1 + degradation_threshold)
    ):
        status = "İzlenmeli"
    elif reference_accuracy is not None and upper < reference_accuracy:
        status = "İzlenmeli"
    elif (
        reference_brier_score is not None
        and brier_score <= reference_brier_score
        and reference_accuracy is not None
        and lower > reference_accuracy
    ):
        status = "İyileşti"
    else:
        status = "Belirsiz"

    return LivePerformanceSummary(
        sample_size=sample_size,
        accuracy=accuracy,
        accuracy_lower=lower,
        accuracy_upper=upper,
        brier_score=brier_score,
        reference_brier_score=reference_brier_score,
        reference_accuracy=reference_accuracy,
        status=status,
    )
