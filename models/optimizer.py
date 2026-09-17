"""Optuna hyperparameter optimization for XGBoost with strict chronological validation."""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from xgboost import XGBClassifier

from models.calibration import expected_calibration_error

LOGGER = logging.getLogger(__name__)

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    optuna = None
    OPTUNA_AVAILABLE = False


@dataclass(frozen=True, slots=True)
class TrialMetric:
    trial_number: int
    duration_seconds: float
    log_loss: float
    brier_score: float
    ece: float
    accuracy: float
    composite_score: float
    params: dict[str, Any]
    state: str


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    best_model: XGBClassifier
    best_params: dict[str, Any]
    best_composite_score: float
    baseline_composite_score: float
    improved: bool
    selected_strategy: str
    trials_count: int
    best_metrics: dict[str, float]
    baseline_metrics: dict[str, float]
    trials_report: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "best_params": self.best_params,
            "best_composite_score": self.best_composite_score,
            "baseline_composite_score": self.baseline_composite_score,
            "improved": self.improved,
            "selected_strategy": self.selected_strategy,
            "trials_count": self.trials_count,
            "best_metrics": self.best_metrics,
            "baseline_metrics": self.baseline_metrics,
            "trials_report": self.trials_report,
        }


def multiclass_brier_score(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """Compute multiclass Brier score sum((p - y)^2)."""
    one_hot = np.eye(probabilities.shape[1], dtype=float)[y_true]
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def binary_brier_score(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """Compute binary Brier score mean((p - y)^2)."""
    return float(np.mean((probabilities - y_true) ** 2))


def compute_composite_score(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    binary: bool,
    weight_log_loss: float = 1.0,
    weight_brier: float = 1.0,
    weight_ece: float = 0.5,
) -> tuple[float, dict[str, float]]:
    """Compute multi-metric objective: Log Loss + Brier Score + 0.5 * ECE."""
    y_arr = np.asarray(y_true, dtype=int)
    prob_arr = np.asarray(probabilities, dtype=float)

    if binary:
        prob_1 = prob_arr[:, 1] if prob_arr.ndim == 2 else prob_arr
        loss = float(log_loss(y_arr, prob_1, labels=[0, 1]))
        brier = binary_brier_score(y_arr, prob_1)
        # Expected calibration error for binary
        pred_label = (prob_1 >= 0.5).astype(int)
        acc = float(accuracy_score(y_arr, pred_label))
        # 2D probabilities for ECE computation
        prob_2d = np.column_stack([1.0 - prob_1, prob_1]) if prob_arr.ndim == 1 else prob_arr
        ece = float(expected_calibration_error(y_arr, prob_2d))
    else:
        loss = float(log_loss(y_arr, prob_arr, labels=[0, 1, 2]))
        brier = multiclass_brier_score(y_arr, prob_arr)
        acc = float(accuracy_score(y_arr, prob_arr.argmax(axis=1)))
        ece = float(expected_calibration_error(y_arr, prob_arr))

    composite = (
        weight_log_loss * loss
        + weight_brier * brier
        + weight_ece * ece
    )
    metrics = {
        "log_loss": loss,
        "brier_score": brier,
        "ece": ece,
        "accuracy": acc,
        "composite_score": composite,
    }
    return composite, metrics


def build_candidate_xgb(
    params: dict[str, Any],
    *,
    binary: bool,
    random_state: int = 42,
) -> XGBClassifier:
    """Build an XGBClassifier with specified hyperparameter dict."""
    common_args = {
        "n_estimators": int(params.get("n_estimators", 500 if binary else 600)),
        "learning_rate": float(params.get("learning_rate", 0.04 if binary else 0.035)),
        "max_depth": int(params.get("max_depth", 4)),
        "min_child_weight": int(params.get("min_child_weight", 5)),
        "subsample": float(params.get("subsample", 0.85)),
        "colsample_bytree": float(params.get("colsample_bytree", 0.85)),
        "reg_alpha": float(params.get("reg_alpha", 0.1)),
        "reg_lambda": float(params.get("reg_lambda", 2.0)),
        "gamma": float(params.get("gamma", 0.0)),
        "random_state": random_state,
        "n_jobs": -1,
        "tree_method": "hist",
    }
    if binary:
        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            early_stopping_rounds=int(params.get("early_stopping_rounds", 35)),
            **common_args,
        )
    return XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        early_stopping_rounds=int(params.get("early_stopping_rounds", 40)),
        **common_args,
    )


def optimize_xgb_hyperparameters(
    *,
    binary: bool,
    x_fit: pd.DataFrame,
    y_fit: np.ndarray,
    fit_weights: np.ndarray | None,
    x_validation: pd.DataFrame,
    y_validation: np.ndarray,
    baseline_model: XGBClassifier,
    n_trials: int = 25,
    timeout_seconds: float | None = 120.0,
    seed: int = 42,
) -> OptimizationResult:
    """Optimize XGBoost hyperparameters with Optuna using strict chronological validation.
    
    Guarantees:
    - Never uses future data in fit slice.
    - Evaluates strictly on chronological future validation slice.
    - Uses multi-metric objective: Log Loss + Brier score + 0.5 * ECE.
    - Preserves baseline model if candidate trials do not improve performance.
    """
    # 1. Evaluate baseline model on validation set
    baseline_probabilities = baseline_model.predict_proba(x_validation)
    baseline_composite, baseline_metrics = compute_composite_score(
        y_validation, baseline_probabilities, binary=binary
    )

    if not OPTUNA_AVAILABLE or n_trials <= 0:
        LOGGER.info("Optuna is not available or n_trials <= 0; returning baseline preset")
        return OptimizationResult(
            best_model=baseline_model,
            best_params=baseline_model.get_params(),
            best_composite_score=baseline_composite,
            baseline_composite_score=baseline_composite,
            improved=False,
            selected_strategy="baseline_preset",
            trials_count=0,
            best_metrics=baseline_metrics,
            baseline_metrics=baseline_metrics,
            trials_report=[],
        )

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    trials_report: list[dict[str, Any]] = []
    models_by_trial: dict[int, XGBClassifier] = {}

    def objective(trial: optuna.Trial) -> float:
        start_time = time.perf_counter()
        params = {
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.08, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 2, 15),
            "subsample": trial.suggest_float("subsample", 0.65, 0.95),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 0.95),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 2.0),
            "n_estimators": 500 if binary else 600,
        }

        model = build_candidate_xgb(params, binary=binary, random_state=seed)
        model.fit(
            x_fit,
            y_fit,
            sample_weight=fit_weights,
            eval_set=[(x_validation, y_validation)],
            verbose=False,
        )
        probabilities = model.predict_proba(x_validation)
        composite_score, metrics = compute_composite_score(
            y_validation, probabilities, binary=binary
        )
        duration = time.perf_counter() - start_time

        trial_record = {
            "trial_number": trial.number,
            "duration_seconds": round(duration, 3),
            "log_loss": round(metrics["log_loss"], 5),
            "brier_score": round(metrics["brier_score"], 5),
            "ece": round(metrics["ece"], 5),
            "accuracy": round(metrics["accuracy"], 5),
            "composite_score": round(composite_score, 5),
            "params": params,
            "state": "COMPLETE",
        }
        trials_report.append(trial_record)
        models_by_trial[trial.number] = model
        return composite_score

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout_seconds,
        catch=(Exception,),
    )

    if not study.trials or len(models_by_trial) == 0:
        LOGGER.warning("No Optuna trials completed successfully; falling back to baseline")
        return OptimizationResult(
            best_model=baseline_model,
            best_params=baseline_model.get_params(),
            best_composite_score=baseline_composite,
            baseline_composite_score=baseline_composite,
            improved=False,
            selected_strategy="baseline_preset",
            trials_count=0,
            best_metrics=baseline_metrics,
            baseline_metrics=baseline_metrics,
            trials_report=trials_report,
        )

    best_trial = study.best_trial
    best_score = float(best_trial.value)
    best_trial_params = best_trial.params

    # Find metrics of best trial
    best_trial_record = next(
        (t for t in trials_report if t["trial_number"] == best_trial.number), None
    )
    best_metrics = {
        "log_loss": best_trial_record["log_loss"] if best_trial_record else best_score,
        "brier_score": best_trial_record["brier_score"] if best_trial_record else best_score,
        "ece": best_trial_record["ece"] if best_trial_record else 0.0,
        "accuracy": best_trial_record["accuracy"] if best_trial_record else 0.0,
        "composite_score": best_score,
    }

    # Model promotion guard:
    # Only select candidate if it strictly improves over baseline on composite loss.
    # If candidate is equal or worse, keep the baseline model!
    if best_score < baseline_composite - 1e-5:
        LOGGER.info(
            "Optuna found superior model: composite %.4f vs baseline %.4f (improvement: %.4f)",
            best_score,
            baseline_composite,
            baseline_composite - best_score,
        )
        return OptimizationResult(
            best_model=models_by_trial[best_trial.number],
            best_params=best_trial_params,
            best_composite_score=best_score,
            baseline_composite_score=baseline_composite,
            improved=True,
            selected_strategy="optuna",
            trials_count=len(study.trials),
            best_metrics=best_metrics,
            baseline_metrics=baseline_metrics,
            trials_report=trials_report,
        )

    LOGGER.info(
        "Candidate Optuna model (%.4f) did not beat baseline (%.4f); retaining baseline",
        best_score,
        baseline_composite,
    )
    return OptimizationResult(
        best_model=baseline_model,
        best_params=baseline_model.get_params(),
        best_composite_score=best_score,
        baseline_composite_score=baseline_composite,
        improved=False,
        selected_strategy="baseline_preset",
        trials_count=len(study.trials),
        best_metrics=best_metrics,
        baseline_metrics=baseline_metrics,
        trials_report=trials_report,
    )

