"""Unit tests for drift monitoring service, quality gates, and sanitization."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from monitoring.drift_service import (
    DriftMonitoringService,
    DriftThresholds,
    calculate_feature_drift,
    calculate_psi,
    sanitize_dict,
)


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
    # Identical distributions should have near-zero PSI
    ref = rng.normal(0, 1, 1000)
    cur = rng.normal(0, 1, 1000)
    psi_stable = calculate_psi(ref, cur)
    assert psi_stable < 0.10

    # Heavily shifted distribution should have high PSI
    shifted = rng.normal(3, 1, 1000)
    psi_shifted = calculate_psi(ref, shifted)
    assert psi_shifted > 0.50


def test_drift_service_detects_feature_drift_and_alerts():
    rng = np.random.default_rng(42)
    feature_cols = ["f1", "f2", "f3"]
    ref_df = pd.DataFrame(rng.normal(0, 1, (300, 3)), columns=feature_cols)

    # Shift all 3 features significantly in current
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
    # 20% missing in f1
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

