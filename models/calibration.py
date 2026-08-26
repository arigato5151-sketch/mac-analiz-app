"""Leakage-safe temperature calibration for model probabilities."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import log_loss


EPSILON = 1e-7


def apply_multiclass_temperature(
    probabilities: np.ndarray, temperature: float
) -> np.ndarray:
    """Scale multiclass log-probabilities and normalize each row."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("multiclass probabilities must have shape (n, classes)")
    logits = np.log(np.clip(values, EPSILON, 1.0)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    exponentiated = np.exp(logits)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


def apply_binary_temperature(
    probabilities: np.ndarray, temperature: float
) -> np.ndarray:
    """Scale binary logits without changing their ranking."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    values = np.clip(np.asarray(probabilities, dtype=float), EPSILON, 1 - EPSILON)
    logits = np.log(values / (1.0 - values)) / temperature
    return 1.0 / (1.0 + np.exp(-logits))


def fit_multiclass_temperature(
    y_true: np.ndarray, probabilities: np.ndarray
) -> float:
    """Fit one temperature on a dedicated chronological calibration set."""
    labels = np.asarray(y_true, dtype=int)
    values = np.asarray(probabilities, dtype=float)
    if len(labels) != len(values) or len(labels) < 50:
        raise ValueError("At least 50 aligned calibration rows are required")
    result = minimize_scalar(
        lambda temperature: log_loss(
            labels,
            apply_multiclass_temperature(values, float(temperature)),
            labels=list(range(values.shape[1])),
        ),
        bounds=(0.35, 4.0),
        method="bounded",
        options={"xatol": 1e-4},
    )
    return float(result.x) if result.success else 1.0


def fit_binary_temperature(
    y_true: np.ndarray, probabilities: np.ndarray
) -> float:
    labels = np.asarray(y_true, dtype=int)
    values = np.asarray(probabilities, dtype=float)
    if len(labels) != len(values) or len(labels) < 50:
        raise ValueError("At least 50 aligned calibration rows are required")
    result = minimize_scalar(
        lambda temperature: log_loss(
            labels, apply_binary_temperature(values, float(temperature)), labels=[0, 1]
        ),
        bounds=(0.35, 4.0),
        method="bounded",
        options={"xatol": 1e-4},
    )
    return float(result.x) if result.success else 1.0


def expected_calibration_error(
    y_true: np.ndarray, probabilities: np.ndarray, *, bins: int = 10
) -> float:
    """Return confidence-weighted multiclass expected calibration error."""
    if bins < 2:
        raise ValueError("bins must be at least 2")
    labels = np.asarray(y_true, dtype=int)
    values = np.asarray(probabilities, dtype=float)
    confidences = values.max(axis=1)
    predictions = values.argmax(axis=1)
    correct = predictions == labels
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (confidences > lower) & (confidences <= upper)
        if not np.any(mask):
            continue
        error += float(mask.mean()) * abs(
            float(correct[mask].mean()) - float(confidences[mask].mean())
        )
    return error
