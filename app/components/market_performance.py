"""Aggregate only the diversified market signals that users actually saw."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Iterable

from models.market_forecast import (
    MINIMUM_DOUBLE_CHANCE_CONFIDENCE,
    MINIMUM_GOAL_MARKET_CONFIDENCE,
)


MARKET_LABELS = {
    ("total_goals", "over_1_5"): "Üst 1.5",
    ("total_goals", "under_3_5"): "Alt 3.5",
    ("team_goals", "home_over_0_5"): "Ev 0.5 Üst",
    ("team_goals", "away_over_0_5"): "Dep. 0.5 Üst",
    ("team_goals", "home_over_1_5"): "Ev 1.5 Üst",
    ("team_goals", "away_over_1_5"): "Dep. 1.5 Üst",
}


def _json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    return {}


def summarize_diversified_market_performance(
    rows: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Summarize threshold-qualified signals using their immutable evaluations."""
    buckets: defaultdict[str, list[tuple[bool, float | None]]] = defaultdict(list)
    for row in rows:
        probabilities = _json_object(row.get("market_probabilities"))
        performance = _json_object(row.get("market_performance"))

        double_chance = probabilities.get("double_chance") or {}
        if double_chance:
            key, probability = max(
                ((str(key), float(value)) for key, value in double_chance.items()),
                key=lambda item: item[1],
            )
            evaluation = (performance.get("double_chance") or {}).get(key) or {}
            if probability >= MINIMUM_DOUBLE_CHANCE_CONFIDENCE and "actual" in evaluation:
                buckets["Çifte şans"].append(
                    (bool(evaluation["actual"]), float(evaluation["brier_score"]))
                )

        for (group, key), label in MARKET_LABELS.items():
            probability = float((probabilities.get(group) or {}).get(key, 0.0))
            evaluation = (performance.get(group) or {}).get(key) or {}
            if probability >= MINIMUM_GOAL_MARKET_CONFIDENCE and "actual" in evaluation:
                buckets[label].append(
                    (bool(evaluation["actual"]), float(evaluation["brier_score"]))
                )

        correct_score = performance.get("correct_score") or {}
        if "top_3_hit" in correct_score:
            buckets["İlk 3 olası skor"].append(
                (bool(correct_score["top_3_hit"]), None)
            )

    summaries: list[dict[str, Any]] = []
    for label, values in buckets.items():
        brier_values = [brier for _, brier in values if brier is not None]
        summaries.append(
            {
                "Pazar": label,
                "Örneklem": len(values),
                "İsabet": sum(actual for actual, _ in values) / len(values),
                "Brier": (
                    sum(brier_values) / len(brier_values) if brier_values else None
                ),
            }
        )
    return sorted(summaries, key=lambda item: (-int(item["Örneklem"]), str(item["Pazar"])))
