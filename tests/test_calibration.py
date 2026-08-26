from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import log_loss

from models.calibration import (
    apply_binary_temperature,
    apply_multiclass_temperature,
    expected_calibration_error,
    fit_multiclass_temperature,
)
from models.train_model import recency_sample_weights


def test_multiclass_temperature_preserves_normalization_and_ranking() -> None:
    raw = np.array([[0.8, 0.15, 0.05], [0.2, 0.3, 0.5]])

    calibrated = apply_multiclass_temperature(raw, 2.0)

    assert calibrated.sum(axis=1) == pytest.approx(np.ones(2))
    assert calibrated.argmax(axis=1).tolist() == raw.argmax(axis=1).tolist()
    assert calibrated[0, 0] < raw[0, 0]


def test_binary_temperature_preserves_probability_bounds() -> None:
    calibrated = apply_binary_temperature(np.array([0.01, 0.5, 0.99]), 1.5)

    assert np.all((calibrated > 0) & (calibrated < 1))
    assert calibrated[1] == pytest.approx(0.5)


def test_temperature_fit_improves_overconfident_calibration_sample() -> None:
    labels = np.array(([0, 1, 2] * 40), dtype=int)
    raw = np.full((len(labels), 3), 0.02)
    for index, label in enumerate(labels):
        predicted = label if index % 2 == 0 else (label + 1) % 3
        raw[index, predicted] = 0.96

    temperature = fit_multiclass_temperature(labels, raw)
    calibrated = apply_multiclass_temperature(raw, temperature)

    assert temperature > 1.0
    assert log_loss(labels, calibrated, labels=[0, 1, 2]) < log_loss(
        labels, raw, labels=[0, 1, 2]
    )


def test_expected_calibration_error_is_zero_for_perfect_confidence() -> None:
    labels = np.array([0, 1, 2])
    probabilities = np.eye(3)

    assert expected_calibration_error(labels, probabilities) == pytest.approx(0.0)


def test_recency_weights_prioritize_newer_matches_and_normalize() -> None:
    dates = pd.Series(pd.to_datetime(["2024-01-01", "2025-01-01", "2026-01-01"], utc=True))

    weights = recency_sample_weights(dates, half_life_days=365)

    assert weights[0] < weights[1] < weights[2]
    assert weights.mean() == pytest.approx(1.0)
