"""Helpers for interpreting a limited set of live prediction outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Iterable

import pandas as pd


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


def build_performance_breakdowns(
    performance: pd.DataFrame,
    *,
    confidence_edges: tuple[float, ...] = (0.50, 0.60, 0.70, 0.80, 0.90, 1.01),
) -> dict[str, list[dict[str, float | int | str]]]:
    """Group evaluated 1X2 outcomes by league and top-pick confidence.

    Rows missing a required value are excluded rather than treated as failures.
    The returned counts make coverage visible, preventing small high-confidence
    groups from being mistaken for the overall model rate.
    """
    required = {
        "league_name",
        "was_correct",
        "brier_score",
        "prob_home_win",
        "prob_draw",
        "prob_away_win",
    }
    missing = required.difference(performance.columns)
    if missing:
        raise ValueError(f"performance is missing columns: {', '.join(sorted(missing))}")
    if len(confidence_edges) < 2 or any(
        left >= right for left, right in zip(confidence_edges, confidence_edges[1:])
    ):
        raise ValueError("confidence_edges must be strictly increasing")

    frame = performance.copy()
    probability_columns = ["prob_home_win", "prob_draw", "prob_away_win"]
    frame[probability_columns] = frame[probability_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    frame["brier_score"] = pd.to_numeric(frame["brier_score"], errors="coerce")
    frame = frame.dropna(
        subset=["league_name", "was_correct", "brier_score", *probability_columns]
    )
    if frame.empty:
        return {"league": [], "confidence": []}

    frame["confidence"] = frame[probability_columns].max(axis=1)

    def summarize(group: pd.DataFrame, label: str) -> dict[str, float | int | str]:
        sample_size = len(group)
        return {
            "Grup": label,
            "Örneklem": sample_size,
            "Kapsama": sample_size / len(frame),
            "İsabet": group["was_correct"].astype(bool).mean(),
            "Brier": group["brier_score"].mean(),
        }

    league_rows = [
        summarize(group, str(league))
        for league, group in frame.groupby("league_name", dropna=False)
    ]
    confidence_rows: list[dict[str, float | int | str]] = []
    for lower, upper in zip(confidence_edges, confidence_edges[1:]):
        group = frame[(frame["confidence"] >= lower) & (frame["confidence"] < upper)]
        if group.empty:
            continue
        label = f"%{lower * 100:.0f}–%{min(upper, 1.0) * 100:.0f}"
        confidence_rows.append(summarize(group, label))

    return {
        "league": sorted(league_rows, key=lambda row: (-int(row["Örneklem"]), str(row["Grup"]))),
        "confidence": confidence_rows,
    }
