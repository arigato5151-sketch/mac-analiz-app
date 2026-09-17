"""Data and model drift monitoring service with statistical tests and Evidently integration."""

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
from sklearn.metrics import accuracy_score, log_loss

from models.calibration import expected_calibration_error

LOGGER = logging.getLogger(__name__)

# Sensitive pattern regex for sanitization
SENSITIVE_KEY_SUB = re.compile(r"(?i)(api[_-]?key|secret|password|token|bearer|auth|database_url)=([^\s&]+)")
URL_CREDENTIAL_SUB = re.compile(r"(?i)([a-z0-9+.-]+://)([^:@\s]+):([^@\s]+)@([^\s]+)")


def sanitize_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Recursively scrub any sensitive keys or values from report dictionaries."""
    sanitized: dict[str, Any] = {}
    for k, v in data.items():
        k_lower = str(k).lower()
        if any(secret_term in k_lower for secret_term in ("key", "secret", "password", "token", "credential")):
            sanitized[k] = "[REDACTED]"
            continue
        if isinstance(v, dict):
            sanitized[k] = sanitize_dict(v)
        elif isinstance(v, list):
            sanitized[k] = [sanitize_dict(item) if isinstance(item, dict) else item for item in v]
        elif isinstance(v, str):
            val = SENSITIVE_KEY_SUB.sub(r"\1=[REDACTED]", v)
            val = URL_CREDENTIAL_SUB.sub(r"\1\2:[REDACTED]@\4", val)
            sanitized[k] = val
        else:
            sanitized[k] = v
    return sanitized


def calculate_psi(
    reference: np.ndarray, current: np.ndarray, num_buckets: int = 10, epsilon: float = 1e-4
) -> float:
    """Calculate Population Stability Index (PSI) between reference and current samples."""
    ref = reference[~np.isnan(reference)]
    cur = current[~np.isnan(current)]
    if len(ref) == 0 or len(cur) == 0:
        return 0.0

    percentiles = np.linspace(0, 100, num_buckets + 1)
    try:
        bin_edges = np.percentile(ref, percentiles)
        bin_edges[0] -= 1e-5
        bin_edges[-1] += 1e-5
        # Ensure strictly increasing bins
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) < 2:
            return 0.0
    except Exception:
        return 0.0

    ref_counts, _ = np.histogram(ref, bins=bin_edges)
    cur_counts, _ = np.histogram(cur, bins=bin_edges)

    ref_pct = np.clip(ref_counts / len(ref), epsilon, None)
    cur_pct = np.clip(cur_counts / len(cur), epsilon, None)

    # Re-normalize
    ref_pct /= ref_pct.sum()
    cur_pct /= cur_pct.sum()

    psi_val = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi_val)


def calculate_feature_drift(
    reference: pd.Series, current: pd.Series, psi_threshold: float = 0.20
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

    # Numeric feature test
    if pd.api.types.is_numeric_dtype(reference):
        # Kolmogorov-Smirnov test and PSI
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

    # Categorical feature test
    ref_dist = ref_clean.value_counts(normalize=True)
    cur_dist = cur_clean.value_counts(normalize=True)
    all_categories = sorted(set(ref_dist.index).union(set(cur_dist.index)))
    p = np.array([ref_dist.get(cat, 1e-4) for cat in all_categories])
    q = np.array([cur_dist.get(cat, 1e-4) for cat in all_categories])
    p /= p.sum()
    q /= q.sum()
    psi = float(np.sum((q - p) * np.log(q / p)))
    drift_detected = psi > psi_threshold
    return {
        "drift_score": round(psi, 4),
        "drift_detected": bool(drift_detected),
        "missing_ref": round(missing_ref, 4),
        "missing_cur": round(missing_cur, 4),
        "test": "categorical_psi",
    }


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

    def to_dict(self) -> dict[str, Any]:
        return sanitize_dict(asdict(self))


class DriftMonitoringService:
    """Service to generate comprehensive feature drift and performance degradation reports."""

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
    ) -> DriftReportSummary:
        """Compute complete drift report comparing reference and current datasets."""
        now_iso = datetime.now(timezone.utc).isoformat()
        critical_alerts: list[str] = []

        # 1. Feature Drift Analysis
        features_drift: dict[str, Any] = {}
        drifted_count = 0
        missing_summary: dict[str, float] = {}

        cols_to_check = [col for col in feature_columns if col in reference_features.columns and col in current_features.columns]
        for col in cols_to_check:
            drift_info = calculate_feature_drift(reference_features[col], current_features[col])
            features_drift[col] = drift_info
            if drift_info["drift_detected"]:
                drifted_count += 1
            missing_summary[col] = drift_info["missing_cur"]

        total_cols = len(cols_to_check) or 1
        drifted_share = float(drifted_count / total_cols)

        if drifted_share > self.thresholds.max_drifted_features_share:
            critical_alerts.append(
                f"Kritik seviyede özellik sapması: %{drifted_share * 100:.1f} özellik saptı (Eşik: %{self.thresholds.max_drifted_features_share * 100:.0f})"
            )

        # 2. Missing Value Checks
        overall_missing = float(current_features[cols_to_check].isna().mean().mean()) if cols_to_check else 0.0
        if overall_missing > self.thresholds.max_missing_ratio:
            critical_alerts.append(
                f"Yüksek eksik değer oranı: %{overall_missing * 100:.1f} (Eşik: %{self.thresholds.max_missing_ratio * 100:.0f})"
            )

        # 3. Class Distribution & Confidence Level
        class_distribution_shift: dict[str, Any] = {}
        confidence_summary: dict[str, float] = {}

        if current_predictions is not None and not current_predictions.empty:
            prob_cols = ["prob_home_win", "prob_draw", "prob_away_win"]
            available_probs = [c for c in prob_cols if c in current_predictions.columns]
            if len(available_probs) == 3:
                probs = current_predictions[available_probs].to_numpy(dtype=float)
                predicted_classes = probs.argmax(axis=1)
                dist = {
                    "home_win": float(np.mean(predicted_classes == 0)),
                    "draw": float(np.mean(predicted_classes == 1)),
                    "away_win": float(np.mean(predicted_classes == 2)),
                }
                confidences = probs.max(axis=1)
                class_distribution_shift["current"] = dist
                confidence_summary = {
                    "mean_confidence": float(np.mean(confidences)),
                    "median_confidence": float(np.median(confidences)),
                    "std_confidence": float(np.std(confidences)),
                    "low_confidence_share": float(np.mean(confidences < 0.40)),
                }

                if reference_predictions is not None and not reference_predictions.empty:
                    ref_probs = reference_predictions[available_probs].to_numpy(dtype=float)
                    ref_classes = ref_probs.argmax(axis=1)
                    class_distribution_shift["reference"] = {
                        "home_win": float(np.mean(ref_classes == 0)),
                        "draw": float(np.mean(ref_classes == 1)),
                        "away_win": float(np.mean(ref_classes == 2)),
                    }

        # 4. Performance Degradation & Segment Metrics
        performance_drift: dict[str, Any] = {}
        segment_metrics: list[dict[str, Any]] = []

        if performance_evaluations is not None and not performance_evaluations.empty:
            current_log_loss = float(performance_evaluations["log_loss"].mean())
            current_brier = float(performance_evaluations["brier_score"].mean())
            current_acc = float(performance_evaluations["was_correct"].astype(float).mean())

            performance_drift = {
                "sample_size": len(performance_evaluations),
                "log_loss": round(current_log_loss, 4),
                "brier_score": round(current_brier, 4),
                "accuracy": round(current_acc, 4),
            }

            if reference_metrics:
                ref_log_loss = reference_metrics.get("log_loss")
                ref_brier = reference_metrics.get("brier_score")
                if ref_log_loss and current_log_loss > ref_log_loss * (1.0 + self.thresholds.max_log_loss_degradation):
                    critical_alerts.append(
                        f"Log Loss performansı %{(current_log_loss / ref_log_loss - 1.0) * 100:.1f} kötüleşti"
                    )
                if ref_brier and current_brier > ref_brier * (1.0 + self.thresholds.max_brier_degradation):
                    critical_alerts.append(
                        f"Brier Skoru performansı %{(current_brier / ref_brier - 1.0) * 100:.1f} kötüleşti"
                    )

            # League-level segmentation
            if "league_id" in performance_evaluations.columns:
                for league_id, group in performance_evaluations.groupby("league_id"):
                    if len(group) >= 5:
                        segment_metrics.append({
                            "segment_type": "league",
                            "segment_id": int(league_id),
                            "matches": len(group),
                            "log_loss": round(float(group["log_loss"].mean()), 4),
                            "brier_score": round(float(group["brier_score"].mean()), 4),
                            "accuracy": round(float(group["was_correct"].astype(float).mean()), 4),
                        })

        # Overall Status
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
        )
