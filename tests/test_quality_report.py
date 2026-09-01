from __future__ import annotations

import pytest

from evaluation.quality_report import summarize_quality


def test_quality_summary_weights_daily_rows_by_sample_size() -> None:
    report = summarize_quality([
        {"model_version": "v1", "sample_size": 2, "accuracy": 0.5, "avg_brier_score": 0.6, "avg_log_loss": 1.1},
        {"model_version": "v1", "sample_size": 8, "accuracy": 0.75, "avg_brier_score": 0.4, "avg_log_loss": 0.8},
    ])

    assert report == [{
        "model_version": "v1",
        "sample_size": 10,
        "accuracy": pytest.approx(0.70),
        "avg_brier_score": pytest.approx(0.44),
        "avg_log_loss": pytest.approx(0.86),
    }]
