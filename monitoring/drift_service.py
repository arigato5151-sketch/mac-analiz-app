"""Data and model drift monitoring service with statistical tests."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from models.calibration import expected_calibration_error

LOGGER = logging.getLogger(__name__)

# Sensitive pattern regex for sanitization
SENSITIVE_KEY_SUB = re.compile(
    r"(?i)(api[_-]?key|secret|password|token|bearer|auth|database_url)=([^\s&]+)"
)
URL_CREDENTIAL_SUB = re.compile(
    r"(?i)([a-z0-9+.-]+://)([^:@\s]+):([^@\s]+)@([^\s]+)"
)

_RESULT_LABEL_MAP = {"home_win": 0, "draw": 1, "away_win": 2}
_RESULT_INT_MAP = {"H": 0, "D": 1, "A": 2, "home_win": 0, "draw": 1, "away_win": 2}


def sanitize_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Recursively scrub any sensitive keys or values from report dictionaries."""
    sanitized: dict[str, Any] = {}
    for k, v in data.items():
        k_lower = str(k).lower()
        if any(
            term in k_lower
            for term in ("key", "secret", "password", "token", "credential")
        ):
            sanitized[k] = "[REDACTED]"
            continue
        if isinstance(v, dict):
            sanitized[k] = sanitize_dict(v)
        elif isinstance(v, list):
            sanitized[k] = [
                sanitize_dict(item) if isinstance(item, dict) else item for item in v
            ]
        elif isinstance(v, str):
            val = SENSITIVE_KEY_SUB.sub(r"\1=[REDACTED]", v)
            val = URL_CREDENTIAL_SUB.sub(r"\1\2:[REDACTED]@\4", val)
            sanitized[k] = val
        else:
            sanitized[k] = v
    return sanitized


def calculate_psi(
    reference: np.ndarray,
    current: np.ndarray,
    num_buckets: int = 10,
    epsilon: float = 1e-4,
) -> float:
    """Calculate Population Stability Index (PSI) between reference and current."""
    ref = reference[~np.isnan(reference)]
    cur = current[~np.isnan(current)]
    if len(ref) == 0 or len(cur) == 0:
        return 0.0

    percentiles = np.linspace(0, 100, num_buckets + 1)
    try:
        bin_edges = np.percentile(ref, percentiles)
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) < 2:
            return 0.0
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf
    except Exception:  # noqa: BLE001
        return 0.0

    ref_counts, _ = np.histogram(ref, bins=bin_edges)
    cur_counts, _ = np.histogram(cur, bins=bin_edges)

    ref_pct = np.clip(ref_counts / len(ref), epsilon, None)
    cur_pct = np.clip(cur_counts / len(cur), epsilon, None)
    ref_pct /= ref_pct.sum()
    cur_pct /= cur_pct.sum()

    psi_val = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi_val)


def calculate_feature_drift(
    reference: pd.Series,
    current: pd.Series,
    psi_threshold: float = 0.20,
) -> dict[str, Any]:
    """Evaluate drift for a single numeric or categorical feature."""
    ref_clean = reference.dropna()
    cur_clean = current.dropna()

    missing_ref = float(reference.isna().mean())
    missing_cur = float(current.isna().mean())

    if len(ref_clean) == 0 or len(cur_clean) == 0:
        return {
            "drift_score": 0.0,
            "drift_detected": False,
            "missing_ref": missing_ref,
            "missing_cur": missing_cur,
            "test": "insufficient_data",
        }

    if pd.api.types.is_numeric_dtype(reference):
        ks_res = stats.ks_2samp(ref_clean, cur_clean)
        psi = calculate_psi(ref_clean.to_numpy(), cur_clean.to_numpy())
        drift_detected = (ks_res.pvalue < 0.01) and (psi > psi_threshold)
        return {
            "drift_score": round(psi, 4),
            "p_value": round(float(ks_res.pvalue), 5),
            "drift_detected": bool(drift_detected),
            "missing_ref": round(missing_ref, 4),
            "missing_cur": round(missing_cur, 4),
            "test": "ks_and_psi",
        }

    ref_dist = ref_clean.value_counts(normalize=True)
    cur_dist = cur_clean.value_counts(normalize=True)
    all_categories = sorted(set(ref_dist.index).union(set(cur_dist.index)))
    p = np.array([ref_dist.get(cat, 1e-4) for cat in all_categories])
    q = np.array([cur_dist.get(cat, 1e-4) for cat in all_categories])
    p /= p.sum()
    q /= q.sum()
    psi = float(np.sum((q - p) * np.log(q / p)))
    return {
        "drift_score": round(psi, 4),
        "drift_detected": bool(psi > psi_threshold),
        "missing_ref": round(missing_ref, 4),
        "missing_cur": round(missing_cur, 4),
        "test": "categorical_psi",
    }


def _normalize_proba_rows(proba: np.ndarray, epsilon: float = 1e-7) -> np.ndarray:
    """Row-wise normalize a probability matrix and clip to [epsilon, 1-epsilon].

    Handles NaN, inf, zero-sum rows gracefully.
    """
    proba = np.array(proba, dtype=float)
    # Replace inf/nan with 0
    proba = np.where(np.isfinite(proba), proba, 0.0)
    row_sums = proba.sum(axis=1, keepdims=True)
    # Rows that sum to 0 get uniform distribution
    zero_rows = (row_sums == 0).flatten()
    proba[zero_rows] = 1.0 / proba.shape[1]
    row_sums = proba.sum(axis=1, keepdims=True)
    proba = proba / row_sums
    return np.clip(proba, epsilon, 1.0 - epsilon)


def compute_performance_log_loss(
    performance_df: pd.DataFrame,
) -> dict[str, Any]:
    """Compute multiclass log loss from actual_result + prob columns.

    The ``log_loss`` column is not stored in the DB view; we derive it from
    the probability columns that ARE present. Invalid rows (missing probs,
    unrecognised actual_result) are silently dropped.

    Returns
    -------
    dict with keys: ``log_loss`` (float | None), ``sample_size`` (int),
    ``status`` (str: 'ok' | 'insufficient' | 'unavailable')
    """
    required_cols = {"prob_home_win", "prob_draw", "prob_away_win", "actual_result"}
    if performance_df is None or performance_df.empty:
        return {"log_loss": None, "sample_size": 0, "status": "unavailable"}
    if not required_cols.issubset(performance_df.columns):
        missing = required_cols - set(performance_df.columns)
        LOGGER.debug("compute_performance_log_loss: missing columns %s", missing)
        return {"log_loss": None, "sample_size": 0, "status": "unavailable"}

    df = performance_df[list(required_cols)].copy()

    # Map actual_result to int
    df["_y"] = df["actual_result"].map(_RESULT_INT_MAP)
    df = df.dropna(subset=["_y", "prob_home_win", "prob_draw", "prob_away_win"])
    if df.empty:
        return {"log_loss": None, "sample_size": 0, "status": "insufficient"}

    proba = df[["prob_home_win", "prob_draw", "prob_away_win"]].to_numpy(dtype=float)
    proba = _normalize_proba_rows(proba)
    y = df["_y"].to_numpy(dtype=int)

    if len(y) < 5:
        return {"log_loss": None, "sample_size": len(y), "status": "insufficient"}

    try:
        # Manual per-row log loss to avoid sklearn warning about non-normalized probs
        row_indices = np.arange(len(y))
        loss = float(-np.mean(np.log(np.clip(proba[row_indices, y], 1e-15, 1.0))))
        return {"log_loss": round(loss, 4), "sample_size": len(y), "status": "ok"}
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Log loss computation failed: %s", exc)
        return {"log_loss": None, "sample_size": len(y), "status": "unavailable"}


@dataclass(frozen=True, slots=True)
class DriftThresholds:
    max_drifted_features_share: float = 0.30
    max_missing_ratio: float = 0.05
    max_log_loss_degradation: float = 0.15
    max_brier_degradation: float = 0.10


@dataclass(frozen=True, slots=True)
class DriftReportSummary:
    generated_at: str
    status: str  # OK, WARNING, CRITICAL
    drifted_features_count: int
    total_features_count: int
    drifted_features_share: float
    features_drift: dict[str, Any]
    missing_values_summary: dict[str, float]
    class_distribution_shift: dict[str, Any]
    confidence_summary: dict[str, float]
    segment_metrics: list[dict[str, Any]]
    performance_drift: dict[str, Any]
    thresholds: dict[str, float]
    critical_alerts: list[str]
    # Snapshot provenance
    data_source: str = "unknown"
    reference_rows: int = 0
    current_rows: int = 0
    reference_period: str = ""
    current_period: str = ""

    def to_dict(self) -> dict[str, Any]:
        return sanitize_dict(asdict(self))


class DriftMonitoringService:
    """Generate feature drift and performance degradation reports."""

    def __init__(self, thresholds: DriftThresholds | None = None) -> None:
        self.thresholds = thresholds or DriftThresholds()

    def generate_report(
        self,
        *,
        reference_features: pd.DataFrame,
        current_features: pd.DataFrame,
        feature_columns: list[str],
        reference_predictions: pd.DataFrame | None = None,
        current_predictions: pd.DataFrame | None = None,
        performance_evaluations: pd.DataFrame | None = None,
        reference_metrics: dict[str, float] | None = None,
        data_source: str = "unknown",
        reference_period: str = "",
        current_period: str = "",
    ) -> DriftReportSummary:
        """Compute complete drift report comparing reference and current datasets."""
        now_iso = datetime.now(timezone.utc).isoformat()
        critical_alerts: list[str] = []

        ref_rows = len(reference_features) if reference_features is not None else 0
        cur_rows = len(current_features) if current_features is not None else 0

        # ------------------------------------------------------------------ #
        # 1. Feature Drift Analysis                                            #
        # ------------------------------------------------------------------ #
        features_drift: dict[str, Any] = {}
        drifted_count = 0
        missing_summary: dict[str, float] = {}

        if (
            reference_features is not None
            and current_features is not None
            and not reference_features.empty
            and not current_features.empty
        ):
            cols_to_check = [
                col
                for col in feature_columns
                if col in reference_features.columns and col in current_features.columns
            ]
            for col in cols_to_check:
                drift_info = calculate_feature_drift(
                    reference_features[col], current_features[col]
                )
                features_drift[col] = drift_info
                if drift_info["drift_detected"]:
                    drifted_count += 1
                missing_summary[col] = drift_info["missing_cur"]
        else:
            cols_to_check = []

        total_cols = len(cols_to_check) or 1
        drifted_share = float(drifted_count / total_cols)

        if cols_to_check and drifted_share > self.thresholds.max_drifted_features_share:
            critical_alerts.append(
                f"Kritik seviyede özellik sapması: %{drifted_share * 100:.1f} özellik saptı "
                f"(Eşik: %{self.thresholds.max_drifted_features_share * 100:.0f})"
            )

        # ------------------------------------------------------------------ #
        # 2. Missing Value Checks                                              #
        # ------------------------------------------------------------------ #
        overall_missing = 0.0
        if (
            current_features is not None
            and not current_features.empty
            and cols_to_check
        ):
            overall_missing = float(
                current_features[cols_to_check].isna().mean().mean()
            )
            if overall_missing > self.thresholds.max_missing_ratio:
                critical_alerts.append(
                    f"Yüksek eksik değer oranı: %{overall_missing * 100:.1f} "
                    f"(Eşik: %{self.thresholds.max_missing_ratio * 100:.0f})"
                )

        # ------------------------------------------------------------------ #
        # 3. Class Distribution & Confidence Level                            #
        # ------------------------------------------------------------------ #
        class_distribution_shift: dict[str, Any] = {}
        confidence_summary: dict[str, float] = {}

        if current_predictions is not None and not current_predictions.empty:
            prob_cols = ["prob_home_win", "prob_draw", "prob_away_win"]
            available_probs = [c for c in prob_cols if c in current_predictions.columns]
            if len(available_probs) == 3:
                proba = _normalize_proba_rows(
                    current_predictions[available_probs].to_numpy(dtype=float)
                )
                predicted_classes = proba.argmax(axis=1)
                dist = {
                    "home_win": float(np.mean(predicted_classes == 0)),
                    "draw": float(np.mean(predicted_classes == 1)),
                    "away_win": float(np.mean(predicted_classes == 2)),
                }
                confidences = proba.max(axis=1)
                class_distribution_shift["current"] = dist
                confidence_summary = {
                    "mean_confidence": float(np.mean(confidences)),
                    "median_confidence": float(np.median(confidences)),
                    "std_confidence": float(np.std(confidences)),
                    "low_confidence_share": float(np.mean(confidences < 0.40)),
                }

                if reference_predictions is not None and not reference_predictions.empty:
                    ref_available = [
                        c for c in prob_cols if c in reference_predictions.columns
                    ]
                    if len(ref_available) == 3:
                        ref_proba = _normalize_proba_rows(
                            reference_predictions[ref_available].to_numpy(dtype=float)
                        )
                        ref_classes = ref_proba.argmax(axis=1)
                        class_distribution_shift["reference"] = {
                            "home_win": float(np.mean(ref_classes == 0)),
                            "draw": float(np.mean(ref_classes == 1)),
                            "away_win": float(np.mean(ref_classes == 2)),
                        }

        # ------------------------------------------------------------------ #
        # 4. Performance Degradation — log_loss derived from probs            #
        # ------------------------------------------------------------------ #
        performance_drift: dict[str, Any] = {}
        segment_metrics: list[dict[str, Any]] = []

        if performance_evaluations is not None and not performance_evaluations.empty:
            ll_result = compute_performance_log_loss(performance_evaluations)
            current_log_loss = ll_result["log_loss"]  # may be None

            current_brier: float | None = None
            if "brier_score" in performance_evaluations.columns:
                try:
                    current_brier = float(
                        performance_evaluations["brier_score"].astype(float).mean()
                    )
                except Exception:  # noqa: BLE001
                    pass

            current_acc: float | None = None
            if "was_correct" in performance_evaluations.columns:
                try:
                    current_acc = float(
                        performance_evaluations["was_correct"].astype(float).mean()
                    )
                except Exception:  # noqa: BLE001
                    pass

            performance_drift = {
                "sample_size": ll_result["sample_size"],
                "log_loss": current_log_loss,
                "log_loss_status": ll_result["status"],
                "brier_score": round(current_brier, 4) if current_brier is not None else None,
                "accuracy": round(current_acc, 4) if current_acc is not None else None,
            }

            if reference_metrics:
                ref_log_loss = reference_metrics.get("log_loss")
                ref_brier = reference_metrics.get("brier_score")
                if (
                    ref_log_loss
                    and current_log_loss is not None
                    and current_log_loss
                    > ref_log_loss * (1.0 + self.thresholds.max_log_loss_degradation)
                ):
                    critical_alerts.append(
                        f"Log Loss performansı %{(current_log_loss / ref_log_loss - 1.0) * 100:.1f} kötüleşti"
                    )
                if (
                    ref_brier
                    and current_brier is not None
                    and current_brier
                    > ref_brier * (1.0 + self.thresholds.max_brier_degradation)
                ):
                    critical_alerts.append(
                        f"Brier Skoru performansı %{(current_brier / ref_brier - 1.0) * 100:.1f} kötüleşti"
                    )

            # League-level segmentation
            if "league_id" in performance_evaluations.columns:
                for league_id, group in performance_evaluations.groupby("league_id"):
                    if len(group) >= 5:
                        seg_ll = compute_performance_log_loss(group)
                        seg_brier: float | None = None
                        if "brier_score" in group.columns:
                            try:
                                seg_brier = round(
                                    float(group["brier_score"].astype(float).mean()), 4
                                )
                            except Exception:  # noqa: BLE001
                                pass
                        seg_acc: float | None = None
                        if "was_correct" in group.columns:
                            try:
                                seg_acc = round(
                                    float(group["was_correct"].astype(float).mean()), 4
                                )
                            except Exception:  # noqa: BLE001
                                pass
                        segment_metrics.append(
                            {
                                "segment_type": "league",
                                "segment_id": int(league_id),
                                "matches": len(group),
                                "log_loss": seg_ll["log_loss"],
                                "brier_score": seg_brier,
                                "accuracy": seg_acc,
                            }
                        )

        # ------------------------------------------------------------------ #
        # 5. Overall Status                                                    #
        # ------------------------------------------------------------------ #
        if critical_alerts:
            status = "CRITICAL"
        elif drifted_share > 0.15 or overall_missing > 0.02:
            status = "WARNING"
        else:
            status = "OK"

        return DriftReportSummary(
            generated_at=now_iso,
            status=status,
            drifted_features_count=drifted_count,
            total_features_count=total_cols,
            drifted_features_share=round(drifted_share, 4),
            features_drift=features_drift,
            missing_values_summary=missing_summary,
            class_distribution_shift=class_distribution_shift,
            confidence_summary=confidence_summary,
            segment_metrics=segment_metrics,
            performance_drift=performance_drift,
            thresholds=asdict(self.thresholds),
            critical_alerts=critical_alerts,
            data_source=data_source,
            reference_rows=ref_rows,
            current_rows=cur_rows,
            reference_period=reference_period,
            current_period=current_period,
        )
