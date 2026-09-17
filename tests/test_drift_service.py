"""Unit tests for drift monitoring service, quality gates, and sanitization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from evaluation.drift_check import run_drift_check
from monitoring.drift_service import (
    DriftMonitoringService,
    DriftThresholds,
    calculate_feature_drift,
    calculate_psi,
    compute_performance_log_loss,
    sanitize_dict,
)
from monitoring.feature_snapshot import save_feature_snapshot


def test_sanitize_dict_redacts_secrets_and_urls():
    raw = {
        "model_version": "v1.0",
        "api_key": "secret123456",
        "db_password": "super_secret_pw",
        "nested": {
            "token": "bearer_abc",
            "connection": "postgres://user:pass123@db.supabase.co:5432/postgres",
        },
        "clean_metric": 0.85,
    }
    sanitized = sanitize_dict(raw)
    assert sanitized["api_key"] == "[REDACTED]"
    assert sanitized["db_password"] == "[REDACTED]"
    assert sanitized["nested"]["token"] == "[REDACTED]"
    assert "pass123" not in str(sanitized)
    assert sanitized["clean_metric"] == 0.85


def test_calculate_psi_detects_distribution_shifts():
    rng = np.random.default_rng(42)
    ref = rng.normal(0, 1, 1000)
    cur = rng.normal(0, 1, 1000)
    psi_stable = calculate_psi(ref, cur)
    assert psi_stable < 0.10

    shifted = rng.normal(3, 1, 1000)
    psi_shifted = calculate_psi(ref, shifted)
    assert psi_shifted > 0.50


def test_drift_service_detects_feature_drift_and_alerts():
    rng = np.random.default_rng(42)
    feature_cols = ["f1", "f2", "f3"]
    ref_df = pd.DataFrame(rng.normal(0, 1, (300, 3)), columns=feature_cols)
    cur_df = pd.DataFrame(rng.normal(5, 1, (300, 3)), columns=feature_cols)

    service = DriftMonitoringService(
        thresholds=DriftThresholds(max_drifted_features_share=0.30)
    )
    report = service.generate_report(
        reference_features=ref_df,
        current_features=cur_df,
        feature_columns=feature_cols,
    )
    assert report.status == "CRITICAL"
    assert report.drifted_features_share >= 0.60
    assert len(report.critical_alerts) >= 1
    assert "Kritik seviyede özellik sapması" in report.critical_alerts[0]


def test_drift_service_detects_high_missing_values():
    feature_cols = ["f1", "f2"]
    ref_df = pd.DataFrame({"f1": [1.0] * 100, "f2": [2.0] * 100})
    cur_df = pd.DataFrame({
        "f1": [1.0] * 80 + [np.nan] * 20,
        "f2": [2.0] * 100,
    })

    service = DriftMonitoringService(
        thresholds=DriftThresholds(max_missing_ratio=0.05)
    )
    report = service.generate_report(
        reference_features=ref_df,
        current_features=cur_df,
        feature_columns=feature_cols,
    )
    assert report.status == "CRITICAL"
    assert any("Yüksek eksik değer oranı" in alert for alert in report.critical_alerts)


def test_drift_service_computes_class_and_confidence_shift():
    feature_cols = ["f1"]
    ref_df = pd.DataFrame({"f1": [1.0] * 100})
    cur_df = pd.DataFrame({"f1": [1.0] * 100})

    cur_preds = pd.DataFrame({
        "prob_home_win": [0.7, 0.2, 0.1],
        "prob_draw": [0.2, 0.6, 0.2],
        "prob_away_win": [0.1, 0.2, 0.7],
    })

    service = DriftMonitoringService()
    report = service.generate_report(
        reference_features=ref_df,
        current_features=cur_df,
        feature_columns=feature_cols,
        current_predictions=cur_preds,
    )
    assert "current" in report.class_distribution_shift
    assert report.confidence_summary["mean_confidence"] == pytest.approx(0.7, abs=0.05)


# ---------------------------------------------------------------------------
# New Regression Tests
# ---------------------------------------------------------------------------

def test_performance_drift_computed_without_log_loss_column():
    """Verify performance report works when log_loss column is absent from DB view."""
    eval_df = pd.DataFrame({
        "actual_result": ["home_win", "draw", "away_win", "home_win", "draw", "away_win"],
        "prob_home_win": [0.60, 0.25, 0.15, 0.70, 0.20, 0.10],
        "prob_draw": [0.25, 0.50, 0.25, 0.20, 0.60, 0.20],
        "prob_away_win": [0.15, 0.25, 0.60, 0.10, 0.20, 0.70],
        "brier_score": [0.22, 0.35, 0.18, 0.15, 0.28, 0.19],
        "was_correct": [True, True, True, True, True, True],
    })
    assert "log_loss" not in eval_df.columns

    service = DriftMonitoringService()
    report = service.generate_report(
        reference_features=pd.DataFrame({"f1": [1.0, 2.0]}),
        current_features=pd.DataFrame({"f1": [1.0, 2.0]}),
        feature_columns=["f1"],
        performance_evaluations=eval_df,
    )

    p_drift = report.performance_drift
    assert p_drift["log_loss"] is not None
    assert p_drift["sample_size"] == 6
    assert p_drift["log_loss_status"] == "ok"
    assert isinstance(p_drift["log_loss"], float)
    assert p_drift["log_loss"] > 0.0


def test_compute_performance_log_loss_accuracy():
    """Verify exact multiclass log loss computation from probabilities and outcomes."""
    # 2 home_wins with prob 0.8, 1 draw with prob 0.5
    # log_loss = - (log(0.8) + log(0.8) + log(0.5) + log(0.8) + log(0.8)) / 5
    # Each row is normalized
    eval_df = pd.DataFrame({
        "actual_result": ["home_win", "home_win", "draw", "home_win", "home_win"],
        "prob_home_win": [0.8, 0.8, 0.2, 0.8, 0.8],
        "prob_draw": [0.1, 0.1, 0.5, 0.1, 0.1],
        "prob_away_win": [0.1, 0.1, 0.3, 0.1, 0.1],
    })
    res = compute_performance_log_loss(eval_df)
    assert res["status"] == "ok"
    assert res["sample_size"] == 5
    expected = - (4 * np.log(0.8) + np.log(0.5)) / 5
    assert res["log_loss"] == pytest.approx(expected, abs=0.01)


def test_corrupt_or_empty_probabilities_handled_gracefully():
    """NaN, inf, and empty inputs should not raise exceptions."""
    # Empty DataFrame
    res_empty = compute_performance_log_loss(pd.DataFrame())
    assert res_empty["log_loss"] is None
    assert res_empty["status"] == "unavailable"

    # All NaNs
    corrupt_df = pd.DataFrame({
        "actual_result": ["home_win", "draw"],
        "prob_home_win": [np.nan, np.nan],
        "prob_draw": [np.nan, np.nan],
        "prob_away_win": [np.nan, np.nan],
    })
    res_corrupt = compute_performance_log_loss(corrupt_df)
    assert res_corrupt["log_loss"] is None
    assert res_corrupt["sample_size"] == 0


def test_ci_drift_check_skips_when_snapshots_missing(tmp_path: Path):
    """CI quality gate outputs SKIPPED and exit 0 when genuine feature snapshots are absent."""
    missing_ref = tmp_path / "no_ref.parquet"
    missing_cur = tmp_path / "no_cur.parquet"

    code = run_drift_check(
        fail_on_critical=True,
        reference_snapshot_path=missing_ref,
        current_snapshot_path=missing_cur,
        allow_skip=True,
    )
    assert code == 0


def test_ci_drift_check_fails_on_critical_with_real_snapshots(tmp_path: Path):
    """CI gate with --fail-on-critical returns exit code 1 when genuine feature drift is critical."""
    rng = np.random.default_rng(42)
    feature_cols = [f"feat_{i}" for i in range(10)]

    ref_df = pd.DataFrame(rng.normal(0, 1, (200, 10)), columns=feature_cols)
    cur_df = pd.DataFrame(rng.normal(10, 1, (200, 10)), columns=feature_cols)  # massive shift

    ref_path = tmp_path / "ref.parquet"
    cur_path = tmp_path / "cur.parquet"
    save_feature_snapshot(ref_df, ref_path)
    save_feature_snapshot(cur_df, cur_path)

    code = run_drift_check(
        fail_on_critical=True,
        reference_snapshot_path=ref_path,
        current_snapshot_path=cur_path,
        allow_skip=False,
    )
    assert code == 1
