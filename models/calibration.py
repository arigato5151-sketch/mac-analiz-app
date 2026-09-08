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


def guarded_multiclass_temperature(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    minimum_improvement: float = 1e-4,
) -> float:
    """Accept calibration only when it improves a later chronological guard set."""
    labels = np.asarray(y_true, dtype=int)
    values = np.asarray(probabilities, dtype=float)
    if len(labels) != len(values) or len(labels) < 100:
        raise ValueError("At least 100 aligned calibration rows are required")
    split = max(50, int(len(labels) * 0.6))
    if len(labels) - split < 50:
        split = len(labels) - 50
    candidate = fit_multiclass_temperature(labels[:split], values[:split])
    raw_loss = log_loss(
        labels[split:], values[split:], labels=list(range(values.shape[1]))
    )
    candidate_loss = log_loss(
        labels[split:],
        apply_multiclass_temperature(values[split:], candidate),
        labels=list(range(values.shape[1])),
    )
    if candidate_loss > raw_loss - minimum_improvement:
        return 1.0
    return fit_multiclass_temperature(labels, values)


def guarded_binary_temperature(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    minimum_improvement: float = 1e-4,
) -> float:
    """Guard binary temperature scaling against chronological degradation."""
    labels = np.asarray(y_true, dtype=int)
    values = np.asarray(probabilities, dtype=float)
    if len(labels) != len(values) or len(labels) < 100:
        raise ValueError("At least 100 aligned calibration rows are required")
    split = max(50, int(len(labels) * 0.6))
    if len(labels) - split < 50:
        split = len(labels) - 50
    candidate = fit_binary_temperature(labels[:split], values[:split])
    raw_loss = log_loss(labels[split:], values[split:], labels=[0, 1])
    candidate_loss = log_loss(
        labels[split:],
        apply_binary_temperature(values[split:], candidate),
        labels=[0, 1],
    )
    if candidate_loss > raw_loss - minimum_improvement:
        return 1.0
    return fit_binary_temperature(labels, values)


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
