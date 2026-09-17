"""Unit tests for Optuna hyperparameter optimization and chronological guards."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from xgboost import XGBClassifier

from models.optimizer import (
    OPTUNA_AVAILABLE,
    OptimizationResult,
    binary_brier_score,
    build_candidate_xgb,
    compute_composite_score,
    multiclass_brier_score,
    optimize_xgb_hyperparameters,
)


def _synthetic_dataset(
    n_samples: int = 400, n_features: int = 10, binary: bool = False
) -> tuple[pd.DataFrame, np.ndarray, pd.Series]:
    rng = np.random.default_rng(42)
    x_data = rng.standard_normal((n_samples, n_features))
    feature_cols = [f"f_{i}" for i in range(n_features)]
    df = pd.DataFrame(x_data, columns=feature_cols)

    # Chronological timestamps
    base_date = pd.Timestamp("2024-01-01", tz="UTC")
    dates = pd.Series([base_date + pd.Timedelta(days=i) for i in range(n_samples)])

    if binary:
        # Logistic signal on first feature
        logits = x_data[:, 0] * 1.5
        probs = 1.0 / (1.0 + np.exp(-logits))
        y = (rng.uniform(size=n_samples) < probs).astype(int)
    else:
        # Multiclass 3-class signal
        logits = np.column_stack([
            x_data[:, 0] * 1.2,
            np.zeros(n_samples),
            x_data[:, 1] * 1.2,
        ])
        exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        y = np.array([rng.choice([0, 1, 2], p=p) for p in probs])

    return df, y, dates


def test_multiclass_and_binary_brier_score():
    y_true_multi = np.array([0, 1, 2])
    probs_multi = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    assert multiclass_brier_score(y_true_multi, probs_multi) == 0.0

    # Binary brier score
    y_true_bin = np.array([1, 0])
    probs_bin = np.array([1.0, 0.0])
    assert binary_brier_score(y_true_bin, probs_bin) == 0.0


def test_compute_composite_score_includes_all_metrics():
    y_true = np.array([0, 1, 2, 0])
    probs = np.array([
        [0.8, 0.1, 0.1],
        [0.1, 0.7, 0.2],
        [0.2, 0.2, 0.6],
        [0.5, 0.3, 0.2],
    ])
    score, metrics = compute_composite_score(y_true, probs, binary=False)
    assert "log_loss" in metrics
    assert "brier_score" in metrics
    assert "ece" in metrics
    assert "accuracy" in metrics
    assert score > 0
    # Expected weighted sum
    assert score == pytest.approx(
        metrics["log_loss"] + metrics["brier_score"] + 0.5 * metrics["ece"]
    )


def test_chronological_split_and_no_data_leakage():
    df, y, dates = _synthetic_dataset(n_samples=500, binary=False)
    # Split chronologically
    fit_idx = slice(0, 350)
    val_idx = slice(350, 450)

    fit_dates = dates.iloc[fit_idx]
    val_dates = dates.iloc[val_idx]

    # Verification: Validation strictly after fit in time
    assert fit_dates.max() < val_dates.min()

    # Fit a quick baseline
    baseline = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=30,
        max_depth=3,
        eval_metric="mlogloss",
        random_state=42,
    )
    baseline.fit(
        df.iloc[fit_idx],
        y[fit_idx],
        eval_set=[(df.iloc[val_idx], y[val_idx])],
        verbose=False,
    )

    result = optimize_xgb_hyperparameters(
        binary=False,
        x_fit=df.iloc[fit_idx],
        y_fit=y[fit_idx],
        fit_weights=None,
        x_validation=df.iloc[val_idx],
        y_validation=y[val_idx],
        baseline_model=baseline,
        n_trials=3,
        timeout_seconds=30,
        seed=42,
    )
    assert isinstance(result, OptimizationResult)
    assert result.trials_count >= 1
    # Check that trial report records duration, metrics and params
    for trial in result.trials_report:
        assert trial["duration_seconds"] > 0
        assert "log_loss" in trial
        assert "brier_score" in trial
        assert "params" in trial


def test_promotion_guard_retains_baseline_when_candidate_is_worse():
    df, y, _ = _synthetic_dataset(n_samples=300, binary=True)
    fit_idx = slice(0, 200)
    val_idx = slice(200, 300)

    # Create a baseline model
    baseline = XGBClassifier(
        objective="binary:logistic",
        n_estimators=30,
        max_depth=3,
        eval_metric="logloss",
        random_state=42,
    )
    baseline.fit(
        df.iloc[fit_idx],
        y[fit_idx],
        eval_set=[(df.iloc[val_idx], y[val_idx])],
        verbose=False,
    )

    # When n_trials is 0, baseline must be retained without modification
    result = optimize_xgb_hyperparameters(
        binary=True,
        x_fit=df.iloc[fit_idx],
        y_fit=y[fit_idx],
        fit_weights=None,
        x_validation=df.iloc[val_idx],
        y_validation=y[val_idx],
        baseline_model=baseline,
        n_trials=0,
        seed=42,
    )
    assert result.improved is False
    assert result.selected_strategy == "baseline_preset"
    assert result.best_model == baseline


def test_reproducibility_with_seed():
    if not OPTUNA_AVAILABLE:
        pytest.skip("Optuna is not installed")

    df, y, _ = _synthetic_dataset(n_samples=250, binary=False)
    fit_idx = slice(0, 180)
    val_idx = slice(180, 250)

    baseline = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=20,
        max_depth=3,
        eval_metric="mlogloss",
        random_state=42,
    )
    baseline.fit(
        df.iloc[fit_idx],
        y[fit_idx],
        eval_set=[(df.iloc[val_idx], y[val_idx])],
        verbose=False,
    )

    run1 = optimize_xgb_hyperparameters(
        binary=False,
        x_fit=df.iloc[fit_idx],
        y_fit=y[fit_idx],
        fit_weights=None,
        x_validation=df.iloc[val_idx],
        y_validation=y[val_idx],
        baseline_model=baseline,
        n_trials=2,
        seed=123,
    )
    run2 = optimize_xgb_hyperparameters(
        binary=False,
        x_fit=df.iloc[fit_idx],
        y_fit=y[fit_idx],
        fit_weights=None,
        x_validation=df.iloc[val_idx],
        y_validation=y[val_idx],
        baseline_model=baseline,
        n_trials=2,
        seed=123,
    )
    assert run1.best_params == run2.best_params
    assert run1.best_composite_score == pytest.approx(run2.best_composite_score)

