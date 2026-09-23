"""Reference baselines the model must beat before its accuracy means anything.

Raw accuracy alone cannot justify a model: these helpers compute the majority
class, always-home, Poisson and (when odds exist) vig-free market references
on the same rows, audit the draw distribution and detect a collapsing class
in the upcoming prediction distribution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import log_loss

from models.calibration import expected_calibration_error

CLASS_LABELS = ("home_win", "draw", "away_win")

# A healthy 1X-2 model never concentrates every pick on one class and never
# squeezes any class out of the distribution entirely.
MIN_CLASS_SHARE = 0.10
MIN_TOP_PICK_CLASS_SHARE = 0.05
MIN_TOP_PICK_AUDIT_SAMPLE = 20
MAX_TOP_PICK_CONCENTRATION = 0.90


@dataclass(frozen=True)
class BaselineResult:
    name: str
    log_loss: float
    brier_score: float
    accuracy: float


def _one_hot(labels: np.ndarray) -> np.ndarray:
    return np.eye(len(CLASS_LABELS))[labels.astype(int)]


def _metrics(labels: np.ndarray, probabilities: np.ndarray, name: str) -> BaselineResult:
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape != (len(labels), len(CLASS_LABELS)):
        raise ValueError("Probabilities must be an aligned (n, 3) matrix")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError("Probabilities must be finite values between zero and one")
    row_totals = probabilities.sum(axis=1)
    if not np.allclose(row_totals, 1.0, atol=1e-6):
        raise ValueError("Probability rows must sum to one")
    # Stored decimal probabilities can differ from one by a few machine
    # epsilon. Normalize after validation so sklearn scores them consistently.
    probabilities = probabilities / row_totals[:, np.newaxis]
    one_hot = _one_hot(labels)
    correct = (probabilities.argmax(axis=1) == labels).astype(float)
    return BaselineResult(
        name=name,
        log_loss=float(log_loss(labels, probabilities, labels=list(range(len(CLASS_LABELS))))),
        brier_score=float(np.mean((probabilities - one_hot) ** 2)),
        accuracy=float(correct.mean()),
    )


def majority_class_baseline(labels: np.ndarray) -> BaselineResult:
    """Always predict the historically most frequent class."""
    counts = np.bincount(labels.astype(int), minlength=len(CLASS_LABELS))
    probabilities = np.tile(
        (counts / counts.sum()).astype(float), (len(labels), 1)
    )
    return _metrics(labels, probabilities, "Çoğunluk sınıfı")


def home_pick_baseline(labels: np.ndarray) -> BaselineResult:
    """Always predict the home win."""
    probabilities = np.tile(np.array([1.0, 0.0, 0.0]), (len(labels), 1))
    return _metrics(labels, probabilities, "Basit ev sahibi seçimi")


def given_probabilities_baseline(
    labels: np.ndarray, probabilities: np.ndarray, name: str
) -> BaselineResult:
    """Score an externally supplied probability matrix (Poisson or market)."""
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(CLASS_LABELS) or len(values) != len(labels):
        raise ValueError("Baseline probabilities must be aligned (n, 3) matrices")
    return _metrics(labels, values, name)


def compare_to_baselines(
    labels: np.ndarray,
    model_probabilities: np.ndarray,
    *,
    poisson_probabilities: np.ndarray | None = None,
    market_probabilities: np.ndarray | None = None,
) -> dict[str, object]:
    """Compare the model with every available baseline on the same rows."""
    model = _metrics(labels, np.asarray(model_probabilities, dtype=float), "Model")
    baselines: list[BaselineResult] = [
        majority_class_baseline(labels),
        home_pick_baseline(labels),
    ]
    if poisson_probabilities is not None:
        baselines.append(
            given_probabilities_baseline(labels, poisson_probabilities, "Poisson")
        )
    if market_probabilities is not None:
        baselines.append(
            given_probabilities_baseline(
                labels, market_probabilities, "Piyasa (marjsız)"
            )
        )
    return {
        "model": model,
        "baselines": baselines,
        "model_ece": float(
            expected_calibration_error(labels, np.asarray(model_probabilities, dtype=float))
        ),
        "beats_all_available": all(
            model.log_loss < baseline.log_loss for baseline in baselines
        ),
    }


@dataclass(frozen=True)
class DistributionAudit:
    predicted_share: dict[str, float]
    realized_share: dict[str, float]
    top_pick_share: dict[str, float]
    top_pick_concentration: float
    collapsed_class: str | None
    warning: str | None


def detect_upcoming_collapse(probabilities: np.ndarray) -> str | None:
    """Collapse check for the upcoming (label-less) prediction distribution."""
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(CLASS_LABELS) or not len(values):
        return None
    predicted_share = {
        CLASS_LABELS[index]: float(values[:, index].mean())
        for index in range(len(CLASS_LABELS))
    }
    for label in CLASS_LABELS:
        if predicted_share[label] < MIN_CLASS_SHARE:
            return (
                f"{label} sınıfı tahmin dağılımında çöktü: ortalama olasılık "
                f"%{predicted_share[label] * 100:.1f}; tahmin hattı sağlıklı değil."
            )
    picks = np.bincount(values.argmax(axis=1), minlength=len(CLASS_LABELS))
    top_pick_share = picks / picks.sum()
    concentration = float(top_pick_share.max())
    if concentration > MAX_TOP_PICK_CONCENTRATION:
        return (
            "Seçimlerin "
            f"%{concentration * 100:.1f}'i tek sınıfta toplandı; dağılım bozuk."
        )
    if len(values) >= MIN_TOP_PICK_AUDIT_SAMPLE:
        collapsed_pick_index = next(
            (
                index
                for index, share in enumerate(top_pick_share)
                if share < MIN_TOP_PICK_CLASS_SHARE
            ),
            None,
        )
        if collapsed_pick_index is not None:
            label = CLASS_LABELS[collapsed_pick_index]
            return (
                f"{label} sınıfı seçimlerden silindi: en güçlü tahmin olma payı "
                f"%{top_pick_share[collapsed_pick_index] * 100:.1f}; modelin sınıf "
                "ayrımı izlenmeli."
            )
    return None


def audit_class_distribution(
    labels: np.ndarray, probabilities: np.ndarray
) -> DistributionAudit:
    """Audit draw share and detect a class collapsing out of the distribution."""
    values = np.asarray(probabilities, dtype=float)
    realized_counts = np.bincount(labels.astype(int), minlength=len(CLASS_LABELS))
    realized_share = {
        CLASS_LABELS[index]: float(realized_counts[index] / realized_counts.sum())
        for index in range(len(CLASS_LABELS))
    }
    predicted_share = {
        CLASS_LABELS[index]: float(values[:, index].mean())
        for index in range(len(CLASS_LABELS))
    }
    picks = np.bincount(values.argmax(axis=1), minlength=len(CLASS_LABELS))
    top_pick_share = {
        CLASS_LABELS[index]: float(picks[index] / picks.sum())
        for index in range(len(CLASS_LABELS))
    }
    concentration = float(picks.max() / picks.sum())
    collapsed_class = next(
        (
            label
            for label in CLASS_LABELS
            if predicted_share[label] < MIN_CLASS_SHARE
        ),
        None,
    )
    warning = detect_upcoming_collapse(values)
    return DistributionAudit(
        predicted_share=predicted_share,
        realized_share=realized_share,
        top_pick_share=top_pick_share,
        top_pick_concentration=concentration,
        collapsed_class=collapsed_class,
        warning=warning,
    )
