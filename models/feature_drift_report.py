"""Compare training/live availability feature coverage and distributions.

Usage:
    python -m models.feature_drift_report --training-json train.json --live-json live.json
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

FEATURES = (
    "home_lineup_confirmed",
    "away_lineup_confirmed",
    "home_available_count",
    "away_available_count",
    "home_impact_score",
    "away_impact_score",
    "impact_score_diff",
)
DEFAULT_MIN_COVERAGE = 0.80


def _numeric(values: Iterable[Any]) -> list[float]:
    return [float(value) for value in values if value is not None and value != ""]


def _summary(rows: list[dict[str, Any]], feature: str, minimum_coverage: float) -> dict[str, Any]:
    values = _numeric(row.get(feature) for row in rows)
    coverage = len(values) / len(rows) if rows else 0.0
    result: dict[str, Any] = {
        "rows": len(rows),
        "observed": len(values),
        "coverage": round(coverage, 6),
        "mean": round(sum(values) / len(values), 6) if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "low_coverage": coverage < minimum_coverage,
    }
    if feature.endswith("lineup_confirmed"):
        result["true_rate"] = round(sum(bool(value) for value in values) / len(values), 6) if values else None
    return result


def build_report(
    training_rows: list[dict[str, Any]],
    live_rows: list[dict[str, Any]],
    *,
    minimum_coverage: float = DEFAULT_MIN_COVERAGE,
) -> dict[str, Any]:
    if not 0 <= minimum_coverage <= 1:
        raise ValueError("minimum_coverage must be between 0 and 1")
    features: dict[str, Any] = {}
    recommendations: list[str] = []
    for feature in FEATURES:
        training = _summary(training_rows, feature, minimum_coverage)
        live = _summary(live_rows, feature, minimum_coverage)
        features[feature] = {"training": training, "live": live}
        if training["low_coverage"] or live["low_coverage"]:
            recommendations.append(
                f"{feature}: doluluk düşük; modelden çıkarma adayı olarak incelenmeli "
                f"(eğitim={training['coverage']:.1%}, canlı={live['coverage']:.1%})."
            )
    return {
        "minimum_coverage": minimum_coverage,
        "training_rows": len(training_rows),
        "live_rows": len(live_rows),
        "features": features,
        "recommendations": recommendations,
    }


def _load_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("rows", payload.get("data"))
    if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
        raise ValueError(f"{path} must contain a JSON array of objects (or rows/data wrapper)")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-json", type=Path, required=True)
    parser.add_argument("--live-json", type=Path, required=True)
    parser.add_argument("--minimum-coverage", type=float, default=DEFAULT_MIN_COVERAGE)
    args = parser.parse_args()
    report = build_report(
        _load_rows(args.training_json),
        _load_rows(args.live_json),
        minimum_coverage=args.minimum_coverage,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
