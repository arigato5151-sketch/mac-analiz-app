from __future__ import annotations

import pandas as pd

from models.feature_engineering import FEATURE_COLUMNS
from models.train_model import (
    confidence_coverage_report,
    normalize_multiclass_probabilities,
    select_blend_weight,
    walk_forward_report,
)


def test_walk_forward_skips_insufficient_history() -> None:
    features = pd.DataFrame([{column: 0.0 for column in FEATURE_COLUMNS}] * 10)
    labels = pd.DataFrame(
        {"result": [0] * 10, "over_2_5": [0] * 10, "btts": [0] * 10,
         "match_date": pd.date_range("2026-01-01", periods=10, tz="UTC")}
    )

    assert walk_forward_report(features, labels) == []


def test_confidence_coverage_report_exposes_accuracy_tradeoff() -> None:
    labels = pd.Series([0, 1, 2, 0]).to_numpy()
    probabilities = pd.DataFrame(
        [[0.70, 0.20, 0.10], [0.40, 0.35, 0.25], [0.10, 0.20, 0.70], [0.45, 0.30, 0.25]]
    ).to_numpy()

    rows = confidence_coverage_report(labels, probabilities, thresholds=(0.0, 0.60))

    assert rows[0]["matches"] == 4
    assert rows[0]["accuracy"] == 0.75
    assert rows[1]["matches"] == 2
    assert rows[1]["coverage"] == 0.5
    assert rows[1]["accuracy"] == 1.0


def test_blend_selection_can_prefer_stronger_anchor() -> None:
    labels = pd.Series([0, 1, 2, 0]).to_numpy()
    model = pd.DataFrame(
        [[0.1, 0.8, 0.1], [0.8, 0.1, 0.1], [0.8, 0.1, 0.1], [0.1, 0.8, 0.1]]
    ).to_numpy()
    anchor = pd.DataFrame(
        [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8], [0.8, 0.1, 0.1]]
    ).to_numpy()

    weight, loss = select_blend_weight(labels, model, anchor)

    assert weight == 0.0
    assert loss < 0.3


def test_multiclass_normalization_removes_float_drift() -> None:
    normalized = normalize_multiclass_probabilities(
        pd.DataFrame([[0.2, 0.3, 0.5000002]]).to_numpy()
    )

    assert abs(float(normalized.sum(axis=1)[0]) - 1.0) < 1e-12
