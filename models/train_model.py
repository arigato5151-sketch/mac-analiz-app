"""Train and chronologically validate XGBoost prediction models."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from xgboost import XGBClassifier

from config.settings import PROJECT_ROOT, get_settings
from db.db_client import SupabaseRestClient
from models.calibration import (
    apply_binary_temperature,
    apply_multiclass_temperature,
    expected_calibration_error,
    fit_binary_temperature,
    fit_multiclass_temperature,
)
from models.feature_engineering import FEATURE_COLUMNS, build_training_dataset


@dataclass(frozen=True, slots=True)
class EvaluationMetrics:
    log_loss: float
    raw_log_loss: float
    brier_score: float
    raw_brier_score: float
    expected_calibration_error: float
    raw_expected_calibration_error: float
    accuracy: float
    baseline_log_loss: float
    over_2_5_log_loss: float
    btts_log_loss: float
    test_size: int
    calibration_size: int
    test_start: str
    test_end: str


def multiclass_brier_score(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    one_hot = np.eye(probabilities.shape[1], dtype=float)[y_true]
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def _result_model() -> XGBClassifier:
    return XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=600,
        learning_rate=0.035,
        max_depth=4,
        min_child_weight=5,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=2.0,
        early_stopping_rounds=40,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )


def _binary_model() -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        n_estimators=500,
        learning_rate=0.04,
        max_depth=4,
        min_child_weight=5,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=2.0,
        early_stopping_rounds=35,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )


def _chronological_slices(size: int) -> tuple[slice, slice, slice, slice]:
    if size < 1_000:
        raise ValueError("At least 1000 chronological matches are required")
    validation_start = int(size * 0.65)
    calibration_start = int(size * 0.75)
    test_start = int(size * 0.80)
    return (
        slice(0, validation_start),
        slice(validation_start, calibration_start),
        slice(calibration_start, test_start),
        slice(test_start, size),
    )


def recency_sample_weights(
    dates: pd.Series, *, half_life_days: float = 365.0
) -> np.ndarray:
    """Weight recent training matches more heavily without using future data."""
    if half_life_days <= 0:
        raise ValueError("half_life_days must be positive")
    timestamps = pd.to_datetime(dates, utc=True)
    ages = (timestamps.max() - timestamps).dt.total_seconds().to_numpy() / 86400.0
    weights = np.power(0.5, ages / half_life_days)
    return weights / weights.mean()


def _segment_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    league_ids: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for league_id in sorted(np.unique(league_ids)):
        mask = league_ids == league_id
        if int(mask.sum()) < 20:
            continue
        segment_y = y_true[mask]
        segment_p = probabilities[mask]
        rows.append(
            {
                "league_id": int(league_id),
                "matches": int(mask.sum()),
                "log_loss": float(log_loss(segment_y, segment_p, labels=[0, 1, 2])),
                "brier_score": multiclass_brier_score(segment_y, segment_p),
                "accuracy": float(accuracy_score(segment_y, segment_p.argmax(axis=1))),
                "ece": expected_calibration_error(segment_y, segment_p),
            }
        )
    return rows


def train_models(
    features: pd.DataFrame, labels: pd.DataFrame
) -> tuple[dict[str, Any], EvaluationMetrics]:
    if tuple(features.columns) != FEATURE_COLUMNS:
        raise ValueError("Feature columns do not match the model contract")
    if len(features) != len(labels):
        raise ValueError("Feature and label row counts must match")

    fit_slice, validation_slice, calibration_slice, test_slice = _chronological_slices(
        len(features)
    )
    x_fit = features.iloc[fit_slice]
    x_validation = features.iloc[validation_slice]
    x_calibration = features.iloc[calibration_slice]
    x_test = features.iloc[test_slice]

    y_result = labels["result"].to_numpy(dtype=int)
    y_over = labels["over_2_5"].to_numpy(dtype=int)
    y_btts = labels["btts"].to_numpy(dtype=int)
    fit_weights = recency_sample_weights(labels.iloc[fit_slice]["match_date"])

    result_model = _result_model()
    result_model.fit(
        x_fit,
        y_result[fit_slice],
        sample_weight=fit_weights,
        eval_set=[(x_validation, y_result[validation_slice])],
        verbose=False,
    )
    over_model = _binary_model()
    over_model.fit(
        x_fit,
        y_over[fit_slice],
        sample_weight=fit_weights,
        eval_set=[(x_validation, y_over[validation_slice])],
        verbose=False,
    )
    btts_model = _binary_model()
    btts_model.fit(
        x_fit,
        y_btts[fit_slice],
        sample_weight=fit_weights,
        eval_set=[(x_validation, y_btts[validation_slice])],
        verbose=False,
    )

    calibration_result_probabilities = result_model.predict_proba(x_calibration)
    result_temperature = fit_multiclass_temperature(
        y_result[calibration_slice], calibration_result_probabilities
    )
    over_temperature = fit_binary_temperature(
        y_over[calibration_slice], over_model.predict_proba(x_calibration)[:, 1]
    )
    btts_temperature = fit_binary_temperature(
        y_btts[calibration_slice], btts_model.predict_proba(x_calibration)[:, 1]
    )

    raw_result_probabilities = result_model.predict_proba(x_test)
    result_probabilities = apply_multiclass_temperature(
        raw_result_probabilities, result_temperature
    )
    result_predictions = np.argmax(result_probabilities, axis=1)
    class_priors = np.bincount(y_result[: test_slice.start], minlength=3).astype(float)
    class_priors /= class_priors.sum()
    baseline_probabilities = np.tile(class_priors, (len(x_test), 1))

    evaluated_labels = y_result[test_slice]
    model_log_loss = float(log_loss(evaluated_labels, result_probabilities, labels=[0, 1, 2]))
    raw_model_log_loss = float(
        log_loss(evaluated_labels, raw_result_probabilities, labels=[0, 1, 2])
    )
    baseline_loss = float(
        log_loss(evaluated_labels, baseline_probabilities, labels=[0, 1, 2])
    )
    if model_log_loss > baseline_loss + 0.02:
        raise RuntimeError(
            f"Model log loss {model_log_loss:.4f} is materially worse than "
            f"prior baseline {baseline_loss:.4f}"
        )

    over_probabilities = apply_binary_temperature(
        over_model.predict_proba(x_test)[:, 1], over_temperature
    )
    btts_probabilities = apply_binary_temperature(
        btts_model.predict_proba(x_test)[:, 1], btts_temperature
    )
    test_dates = labels.iloc[test_slice]["match_date"]
    metrics = EvaluationMetrics(
        log_loss=model_log_loss,
        raw_log_loss=raw_model_log_loss,
        brier_score=multiclass_brier_score(evaluated_labels, result_probabilities),
        raw_brier_score=multiclass_brier_score(
            evaluated_labels, raw_result_probabilities
        ),
        expected_calibration_error=expected_calibration_error(
            evaluated_labels, result_probabilities
        ),
        raw_expected_calibration_error=expected_calibration_error(
            evaluated_labels, raw_result_probabilities
        ),
        accuracy=float(accuracy_score(evaluated_labels, result_predictions)),
        baseline_log_loss=baseline_loss,
        over_2_5_log_loss=float(log_loss(y_over[test_slice], over_probabilities)),
        btts_log_loss=float(log_loss(y_btts[test_slice], btts_probabilities)),
        test_size=len(x_test),
        calibration_size=len(x_calibration),
        test_start=test_dates.iloc[0].isoformat(),
        test_end=test_dates.iloc[-1].isoformat(),
    )
    bundle = {
        "result_model": result_model,
        "over_2_5_model": over_model,
        "btts_model": btts_model,
        "feature_columns": list(FEATURE_COLUMNS),
        "class_labels": ["home_win", "draw", "away_win"],
        "trained_rows": len(features),
        "training_end": labels["match_date"].iloc[-1].isoformat(),
        "metrics": asdict(metrics),
        "calibration": {
            "method": "chronological_temperature_scaling",
            "result_temperature": result_temperature,
            "over_2_5_temperature": over_temperature,
            "btts_temperature": btts_temperature,
            "training_half_life_days": 365.0,
        },
        "league_metrics": _segment_metrics(
            evaluated_labels,
            result_probabilities,
            x_test["league_id"].to_numpy(dtype=int),
        ),
    }
    return bundle, metrics


def save_model_bundle(
    bundle: dict[str, Any], output_dir: Path, *, publish_latest: bool = False
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    version = datetime.now(timezone.utc).strftime("v%Y%m%dT%H%M%SZ")
    model_path = output_dir / f"model_{version}.joblib"
    metadata_path = output_dir / f"model_{version}.json"
    versioned_bundle = {**bundle, "model_version": model_path.stem}
    temporary_path = model_path.with_suffix(".tmp")
    joblib.dump(versioned_bundle, temporary_path, compress=3)
    os.replace(temporary_path, model_path)
    metadata_path.write_text(
        json.dumps(
            {
                key: value
                for key, value in versioned_bundle.items()
                if not key.endswith("_model")
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    if publish_latest:
        latest_path = output_dir / "latest.joblib"
        temporary_latest = output_dir / "latest.tmp"
        joblib.dump(versioned_bundle, temporary_latest, compress=3)
        os.replace(temporary_latest, latest_path)
    return model_path, metadata_path


def load_completed_matches(db: SupabaseRestClient) -> list[dict[str, Any]]:
    return db.select_all(
        "matches",
        columns=(
            "id,league_id,home_team_id,away_team_id,match_date,status,"
            "home_score,away_score"
        ),
        filters={
            "status": "eq.finished",
            "home_score": "not.is.null",
            "away_score": "not.is.null",
        },
        order="match_date.asc,id.asc",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "models" / "saved_models",
    )
    parser.add_argument("--publish-latest", action="store_true")
    args = parser.parse_args()
    settings = get_settings()
    db = SupabaseRestClient(settings.supabase_url, settings.supabase_service_role_key)
    matches = load_completed_matches(db)
    features, labels = build_training_dataset(matches)
    bundle, metrics = train_models(features, labels)
    model_path, metadata_path = save_model_bundle(
        bundle, args.output_dir, publish_latest=args.publish_latest
    )
    print(
        json.dumps(
            {
                "rows": len(features),
                "metrics": asdict(metrics),
                "model_path": str(model_path),
                "metadata_path": str(metadata_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
