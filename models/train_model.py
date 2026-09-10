"""Train and chronologically validate XGBoost prediction models."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from bisect import bisect_right
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from xgboost import XGBClassifier

from config.settings import PROJECT_ROOT, get_settings
from db.db_client import SupabaseRestClient
from data_pipeline.odds import attach_pre_match_odds
from models.calibration import (
    apply_binary_temperature,
    apply_multiclass_temperature,
    expected_calibration_error,
    guarded_binary_temperature,
    guarded_multiclass_temperature,
)
from models.feature_engineering import (
    BINARY_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    build_training_dataset,
)


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


def normalize_multiclass_probabilities(probabilities: np.ndarray) -> np.ndarray:
    """Clip numerical noise and make every multiclass row sum exactly to one."""
    values = np.clip(np.asarray(probabilities, dtype=float), 1e-12, None)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("Multiclass probabilities must have shape (n, classes)")
    totals = values.sum(axis=1, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError("Multiclass probability rows must have positive mass")
    return values / totals


RESULT_MODEL_PRESETS: dict[str, dict[str, float | int]] = {
    "balanced": {
        "max_depth": 4,
        "min_child_weight": 5,
        "reg_alpha": 0.1,
        "reg_lambda": 2.0,
    },
    "regularized": {
        "max_depth": 3,
        "min_child_weight": 10,
        "reg_alpha": 0.3,
        "reg_lambda": 4.0,
    },
}
BINARY_MODEL_PRESETS = RESULT_MODEL_PRESETS
DEFAULT_MARKET_RESULT_MODEL_WEIGHT = 0.25
MINIMUM_MARKET_BLEND_SAMPLE = 100


def _result_model(preset: str = "balanced") -> XGBClassifier:
    if preset not in RESULT_MODEL_PRESETS:
        raise ValueError(f"Unknown result model preset: {preset}")
    return XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=600,
        learning_rate=0.035,
        **RESULT_MODEL_PRESETS[preset],
        subsample=0.85,
        colsample_bytree=0.85,
        early_stopping_rounds=40,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )


def _binary_model(preset: str = "balanced") -> XGBClassifier:
    if preset not in BINARY_MODEL_PRESETS:
        raise ValueError(f"Unknown binary model preset: {preset}")
    return XGBClassifier(
        objective="binary:logistic",
        n_estimators=500,
        learning_rate=0.04,
        **BINARY_MODEL_PRESETS[preset],
        subsample=0.85,
        colsample_bytree=0.85,
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


def confidence_coverage_report(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    thresholds: tuple[float, ...] = (0.0, 0.45, 0.50, 0.55, 0.60, 0.65),
) -> list[dict[str, Any]]:
    """Measure the accuracy/coverage trade-off without changing probabilities."""
    labels = np.asarray(y_true, dtype=int)
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 2 or len(labels) != len(values):
        raise ValueError("Labels and multiclass probabilities must be aligned")
    confidence = values.max(axis=1)
    predicted = values.argmax(axis=1)
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        if threshold < 0 or threshold > 1:
            raise ValueError("Confidence thresholds must be between 0 and 1")
        mask = confidence >= threshold
        selected = int(mask.sum())
        rows.append(
            {
                "threshold": float(threshold),
                "matches": selected,
                "coverage": float(mask.mean()),
                "accuracy": (
                    float(accuracy_score(labels[mask], predicted[mask]))
                    if selected
                    else None
                ),
            }
        )
    return rows


def select_blend_weight(
    y_true: np.ndarray,
    model_probabilities: np.ndarray,
    anchor_probabilities: np.ndarray,
) -> tuple[float, float]:
    """Choose a conservative model/market blend on chronological validation data."""
    labels = np.asarray(y_true, dtype=int)
    model_values = np.asarray(model_probabilities, dtype=float)
    anchor_values = np.asarray(anchor_probabilities, dtype=float)
    if model_values.shape != anchor_values.shape or len(labels) != len(model_values):
        raise ValueError("Blend inputs must have aligned shapes")
    label_set = list(range(model_values.shape[1])) if model_values.ndim == 2 else [0, 1]
    best_weight = 1.0
    best_loss = float("inf")
    for weight in np.linspace(0.0, 1.0, 21):
        blended = weight * model_values + (1.0 - weight) * anchor_values
        if blended.ndim == 2:
            blended = normalize_multiclass_probabilities(blended)
        loss = float(log_loss(labels, blended, labels=label_set))
        if loss < best_loss:
            best_weight, best_loss = float(weight), loss
    return best_weight, best_loss


def select_source_aware_result_weights(
    y_true: np.ndarray,
    model_probabilities: np.ndarray,
    anchor_probabilities: np.ndarray,
    market_available: np.ndarray,
    *,
    minimum_market_sample: int = MINIMUM_MARKET_BLEND_SAMPLE,
    default_market_model_weight: float = DEFAULT_MARKET_RESULT_MODEL_WEIGHT,
) -> tuple[float, float]:
    """Select separate 1X2 weights for real odds and the Poisson fallback.

    Historical odds coverage starts much later than match history. A single
    validation weight otherwise learns only the Poisson fallback and silently
    discards stronger closing-market information in live predictions.
    """
    labels = np.asarray(y_true, dtype=int)
    model_values = np.asarray(model_probabilities, dtype=float)
    anchor_values = np.asarray(anchor_probabilities, dtype=float)
    available = np.asarray(market_available, dtype=bool)
    if (
        model_values.shape != anchor_values.shape
        or len(labels) != len(model_values)
        or len(available) != len(labels)
    ):
        raise ValueError("Source-aware blend inputs must have aligned shapes")
    if minimum_market_sample < 1:
        raise ValueError("minimum_market_sample must be positive")
    if not 0 <= default_market_model_weight <= 1:
        raise ValueError("default_market_model_weight must be between zero and one")

    fallback_mask = ~available
    fallback_weight = (
        select_blend_weight(
            labels[fallback_mask],
            model_values[fallback_mask],
            anchor_values[fallback_mask],
        )[0]
        if fallback_mask.any()
        else 1.0
    )
    market_weight = (
        select_blend_weight(
            labels[available],
            model_values[available],
            anchor_values[available],
        )[0]
        if int(available.sum()) >= minimum_market_sample
        else float(default_market_model_weight)
    )
    return fallback_weight, market_weight


def _fit_best_model(
    *,
    binary: bool,
    x_fit: pd.DataFrame,
    y_fit: np.ndarray,
    fit_weights: np.ndarray,
    x_validation: pd.DataFrame,
    y_validation: np.ndarray,
) -> tuple[XGBClassifier, str, float]:
    """Select a small, bounded preset search on a future validation slice."""
    best: tuple[XGBClassifier, str, float] | None = None
    presets = BINARY_MODEL_PRESETS if binary else RESULT_MODEL_PRESETS
    for preset in presets:
        model = _binary_model(preset) if binary else _result_model(preset)
        model.fit(
            x_fit,
            y_fit,
            sample_weight=fit_weights,
            eval_set=[(x_validation, y_validation)],
            verbose=False,
        )
        probabilities = model.predict_proba(x_validation)
        loss = float(
            log_loss(
                y_validation,
                probabilities[:, 1] if binary else probabilities,
                labels=[0, 1] if binary else [0, 1, 2],
            )
        )
        if best is None or loss < best[2]:
            best = model, preset, loss
    if best is None:  # Defensive: preset dictionaries are module constants.
        raise RuntimeError("No model candidate was evaluated")
    return best


def walk_forward_report(
    features: pd.DataFrame, labels: pd.DataFrame, *, folds: int = 3,
) -> list[dict[str, Any]]:
    """Evaluate expanding chronological training windows without changing promotion guards."""
    if folds < 2:
        raise ValueError("folds must be at least 2")
    if len(features) <= 3_000:
        return []
    if tuple(features.columns) != FEATURE_COLUMNS or len(features) != len(labels):
        raise ValueError("Features and labels must match the model contract")

    # Keep the first half as a stable initial fit, then test equal future blocks.
    first_test_start = len(features) // 2
    block_size = (len(features) - first_test_start) // folds
    y_result = labels["result"].to_numpy(dtype=int)
    report: list[dict[str, Any]] = []
    for fold in range(folds):
        test_start = first_test_start + fold * block_size
        test_end = len(features) if fold == folds - 1 else test_start + block_size
        validation_start = max(1_000, int(test_start * 0.85))
        model, preset, _ = _fit_best_model(
            binary=False,
            x_fit=features.iloc[:validation_start],
            y_fit=y_result[:validation_start],
            fit_weights=recency_sample_weights(labels.iloc[:validation_start]["match_date"]),
            x_validation=features.iloc[validation_start:test_start],
            y_validation=y_result[validation_start:test_start],
        )
        probabilities = model.predict_proba(features.iloc[test_start:test_end])
        actual = y_result[test_start:test_end]
        report.append(
            {
                "fold": fold + 1,
                "preset": preset,
                "train_rows": test_start,
                "test_rows": test_end - test_start,
                "test_start": labels.iloc[test_start]["match_date"].isoformat(),
                "test_end": labels.iloc[test_end - 1]["match_date"].isoformat(),
                "log_loss": float(log_loss(actual, probabilities, labels=[0, 1, 2])),
                "brier_score": multiclass_brier_score(actual, probabilities),
                "expected_calibration_error": expected_calibration_error(actual, probabilities),
            }
        )
    return report


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

    result_model, result_preset, result_validation_loss = _fit_best_model(
        binary=False,
        x_fit=x_fit,
        y_fit=y_result[fit_slice],
        fit_weights=fit_weights,
        x_validation=x_validation,
        y_validation=y_result[validation_slice],
    )
    over_model, over_preset, over_validation_loss = _fit_best_model(
        binary=True,
        x_fit=x_fit.loc[:, BINARY_FEATURE_COLUMNS],
        y_fit=y_over[fit_slice],
        fit_weights=fit_weights,
        x_validation=x_validation.loc[:, BINARY_FEATURE_COLUMNS],
        y_validation=y_over[validation_slice],
    )
    btts_model, btts_preset, btts_validation_loss = _fit_best_model(
        binary=True,
        x_fit=x_fit.loc[:, BINARY_FEATURE_COLUMNS],
        y_fit=y_btts[fit_slice],
        fit_weights=fit_weights,
        x_validation=x_validation.loc[:, BINARY_FEATURE_COLUMNS],
        y_validation=y_btts[validation_slice],
    )

    validation_result_anchor = x_validation[
        ["market_implied_home_win", "market_implied_draw", "market_implied_away_win"]
    ].to_numpy(dtype=float)
    result_blend_weight, result_market_model_weight = select_source_aware_result_weights(
        y_result[validation_slice],
        result_model.predict_proba(x_validation),
        validation_result_anchor,
        x_validation["market_odds_available"].to_numpy(dtype=bool),
    )
    over_blend_weight, _ = select_blend_weight(
        y_over[validation_slice],
        over_model.predict_proba(x_validation.loc[:, BINARY_FEATURE_COLUMNS])[:, 1],
        x_validation["market_implied_over_2_5"].to_numpy(dtype=float),
    )
    btts_blend_weight, _ = select_blend_weight(
        y_btts[validation_slice],
        btts_model.predict_proba(x_validation.loc[:, BINARY_FEATURE_COLUMNS])[:, 1],
        x_validation["market_implied_btts"].to_numpy(dtype=float),
    )

    def result_blend(frame: pd.DataFrame) -> np.ndarray:
        model_values = result_model.predict_proba(frame)
        anchor = frame[
            ["market_implied_home_win", "market_implied_draw", "market_implied_away_win"]
        ].to_numpy(dtype=float)
        weights = np.where(
            frame["market_odds_available"].to_numpy(dtype=bool),
            result_market_model_weight,
            result_blend_weight,
        )[:, None]
        return normalize_multiclass_probabilities(weights * model_values + (1.0 - weights) * anchor)

    def binary_blend(
        model: XGBClassifier, frame: pd.DataFrame, column: str, weight: float
    ) -> np.ndarray:
        return weight * model.predict_proba(frame.loc[:, BINARY_FEATURE_COLUMNS])[
            :, 1
        ] + (1.0 - weight) * frame[column].to_numpy(dtype=float)

    calibration_result_probabilities = result_blend(x_calibration)
    result_temperature = guarded_multiclass_temperature(
        y_result[calibration_slice], calibration_result_probabilities
    )
    over_temperature = guarded_binary_temperature(
        y_over[calibration_slice],
        binary_blend(
            over_model,
            x_calibration,
            "market_implied_over_2_5",
            over_blend_weight,
        ),
    )
    btts_temperature = guarded_binary_temperature(
        y_btts[calibration_slice],
        binary_blend(
            btts_model,
            x_calibration,
            "market_implied_btts",
            btts_blend_weight,
        ),
    )

    raw_result_probabilities = result_blend(x_test)
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
        binary_blend(
            over_model, x_test, "market_implied_over_2_5", over_blend_weight
        ),
        over_temperature,
    )
    btts_probabilities = apply_binary_temperature(
        binary_blend(
            btts_model, x_test, "market_implied_btts", btts_blend_weight
        ),
        btts_temperature,
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
        "binary_feature_columns": list(BINARY_FEATURE_COLUMNS),
        "class_labels": ["home_win", "draw", "away_win"],
        "trained_rows": len(features),
        "training_end": labels["match_date"].iloc[-1].isoformat(),
        "metrics": asdict(metrics),
        "calibration": {
            "method": "guarded_chronological_temperature_scaling",
            "result_temperature": result_temperature,
            "over_2_5_temperature": over_temperature,
            "btts_temperature": btts_temperature,
            "training_half_life_days": 365.0,
        },
        "model_selection": {
            "method": "bounded_chronological_preset_search",
            "result": {
                "preset": result_preset,
                "validation_log_loss": result_validation_loss,
            },
            "over_2_5": {
                "preset": over_preset,
                "validation_log_loss": over_validation_loss,
            },
            "btts": {
                "preset": btts_preset,
                "validation_log_loss": btts_validation_loss,
            },
        },
        "blend": {
            "anchor": "source_aware_vig_free_market_or_poisson",
            "result_model_weight": result_blend_weight,
            "result_market_model_weight": result_market_model_weight,
            "minimum_market_blend_sample": MINIMUM_MARKET_BLEND_SAMPLE,
            "over_2_5_model_weight": over_blend_weight,
            "btts_model_weight": btts_blend_weight,
        },
        "league_metrics": _segment_metrics(
            evaluated_labels,
            result_probabilities,
            x_test["league_id"].to_numpy(dtype=int),
        ),
        "confidence_coverage": confidence_coverage_report(
            evaluated_labels, result_probabilities
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


def load_historical_matches(db: SupabaseRestClient) -> list[dict[str, Any]]:
    """Load finished matches with scores/xg only — the causal feature builder's input.

    Inference paths (`predict`, `pre_match`, `shadow`) feed history exclusively
    into ``CausalFeatureState``, which reads team ids, scores, xg, league and
    date. Market/availability/lineup context columns are never consumed there, so
    avoid transferring the (fast-growing) historical odds quote table once per
    scheduled run. Training and the UI still call ``load_completed_matches``.
    """
    return db.select_all(
        "matches",
        columns=(
            "id,league_id,home_team_id,away_team_id,match_date,status,"
            "home_score,away_score,home_xg,away_xg,home_xa,away_xa"
        ),
        filters={
            "status": "eq.finished",
            "home_score": "not.is.null",
            "away_score": "not.is.null",
        },
        order="match_date.asc,id.asc",
    )


def load_completed_matches(db: SupabaseRestClient) -> list[dict[str, Any]]:
    matches = load_historical_matches(db)
    quotes = db.select_all(
        "odds_quote_history",
        columns="match_id,odds,captured_at",
        order="captured_at.asc",
    )
    availability = db.select_all(
        "team_availability_history",
        columns="team_id,refreshed_at,available_count",
        order="refreshed_at.asc",
    )
    lineups = db.select_all(
        "fixture_lineups",
        columns="match_id,team_id,confirmed_at",
        order="confirmed_at.asc",
    )
    return attach_historical_context(
        attach_pre_match_odds(matches, quotes),
        availability_history=availability,
        lineups=lineups,
    )


def attach_historical_context(
    matches: list[dict[str, Any]],
    *,
    availability_history: list[dict[str, Any]],
    lineups: list[dict[str, Any]],
    decision_lead_minutes: int = 20,
) -> list[dict[str, Any]]:
    """Attach only context that existed before each historical kickoff."""
    if decision_lead_minutes < 0:
        raise ValueError("decision_lead_minutes must not be negative")
    availability_by_team: dict[int, list[tuple[datetime, int]]] = {}
    for item in availability_history:
        refreshed_at = datetime.fromisoformat(
            str(item["refreshed_at"]).replace("Z", "+00:00")
        )
        availability_by_team.setdefault(int(item["team_id"]), []).append(
            (refreshed_at, int(item["available_count"]))
        )
    for observations in availability_by_team.values():
        observations.sort(key=lambda item: item[0])

    confirmed_at_by_fixture = {
        (int(item["match_id"]), int(item["team_id"])): datetime.fromisoformat(
            str(item["confirmed_at"]).replace("Z", "+00:00")
        )
        for item in lineups
    }
    enriched: list[dict[str, Any]] = []
    for match in matches:
        row = dict(match)
        kickoff = datetime.fromisoformat(str(row["match_date"]).replace("Z", "+00:00"))
        decision_at = kickoff - timedelta(minutes=decision_lead_minutes)
        for side in ("home", "away"):
            team_id = int(row[f"{side}_team_id"])
            observations = availability_by_team.get(team_id, [])
            timestamps = [item[0] for item in observations]
            index = bisect_right(timestamps, decision_at) - 1
            if index >= 0:
                available_count = observations[index][1]
                row[f"{side}_available_count"] = available_count
                row[f"{side}_unavailable_count"] = max(0, 22 - available_count)
            confirmed_at = confirmed_at_by_fixture.get((int(row["id"]), team_id))
            row[f"{side}_lineup_confirmed"] = bool(
                confirmed_at is not None and confirmed_at <= decision_at
            )
        enriched.append(row)
    return enriched


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "models" / "saved_models",
    )
    parser.add_argument("--publish-latest", action="store_true")
    parser.add_argument("--report-walk-forward", action="store_true")
    args = parser.parse_args()
    settings = get_settings()
    db = SupabaseRestClient(settings.supabase_url, settings.supabase_service_role_key)
    matches = load_completed_matches(db)
    features, labels = build_training_dataset(matches)
    bundle, metrics = train_models(features, labels)
    walk_forward = walk_forward_report(features, labels) if args.report_walk_forward else []
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
                "walk_forward": walk_forward,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
