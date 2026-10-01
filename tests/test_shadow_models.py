from __future__ import annotations

import json

import joblib
import pytest

import models.shadow as shadow
from models.artifact_store import ArtifactStoreError, build_artifact_manifest
from models.shadow import (
    MAXIMUM_PROMOTION_BRIER,
    MINIMUM_PROMOTION_ACCURACY,
    MINIMUM_PROMOTION_SAMPLE,
    evaluate_shadow_predictions,
    paired_market_comparison,
    promotion_decision,
)


def test_recovery_candidate_uses_its_own_verified_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(shadow, "PROJECT_ROOT", tmp_path)
    version = "model_v20261001T082640Z"
    requests: list[str] = []

    def fake_download(name, *, dest_dir):
        requests.append(name)
        if not name.startswith("recovery/"):
            raise ArtifactStoreError("legacy artifact missing")
        target = dest_dir / name.rsplit("/", 1)[-1]
        dest_dir.mkdir(parents=True, exist_ok=True)
        if target.suffix == ".joblib":
            joblib.dump({"model_version": version}, target)
        else:
            target.write_text(json.dumps(build_artifact_manifest(dest_dir)), encoding="utf-8")
        return target

    monkeypatch.setattr(shadow, "download_model", fake_download)
    path = shadow.candidate_path(version)

    assert path.parent.name == version
    assert requests[-1] == f"recovery/{version}/artifact_manifest.json"


def test_explicit_registration_rejects_mismatched_version(tmp_path):
    path = tmp_path / "model_v20261001T082640Z.joblib"
    joblib.dump({"model_version": "other"}, path)
    (tmp_path / "artifact_manifest.json").write_text(
        json.dumps(build_artifact_manifest(tmp_path)), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="does not match"):
        shadow.register_candidate_artifact(object(), path)


class _NoFinishedMatchesDb:
    def __init__(self) -> None:
        self.queried: set[str] = set()
        self.orders: dict[str, str | None] = {}

    def select_all(self, table: str, **kwargs: object) -> list[dict]:
        self.queried.add(table)
        self.orders[table] = kwargs.get("order")
        if table == "shadow_prediction_performance":
            assert kwargs["order"] == "shadow_prediction_id.asc"
            return []
        if table == "matches":
            return []
        if table == "shadow_predictions":
            raise AssertionError("shadow_predictions must not be scanned with no finished fixtures")
        return []

    def upsert(self, table: str, records: list[dict], **kwargs: object) -> list[dict]:
        return records


def test_shadow_evaluation_skips_scan_when_no_finished_matches() -> None:
    db = _NoFinishedMatchesDb()

    assert evaluate_shadow_predictions(db) == []
    assert db.queried <= {"shadow_prediction_performance", "matches"}
    assert db.orders["shadow_prediction_performance"] == "shadow_prediction_id.asc"


def test_shadow_candidate_needs_a_sufficient_same_match_sample() -> None:
    accepted, reason = promotion_decision(
        candidate_brier=0.40,
        candidate_accuracy=0.60,
        production_brier=0.50,
        production_accuracy=0.55,
        sample_size=MINIMUM_PROMOTION_SAMPLE - 1,
    )

    assert accepted is False
    assert "yetersiz" in reason


def test_shadow_candidate_requires_better_brier_and_no_accuracy_regression() -> None:
    assert promotion_decision(
        candidate_brier=0.42,
        candidate_accuracy=0.60,
        production_brier=0.50,
        production_accuracy=0.60,
        sample_size=MINIMUM_PROMOTION_SAMPLE,
    )[0] is True
    assert promotion_decision(
        candidate_brier=0.50,
        candidate_accuracy=0.65,
        production_brier=0.50,
        production_accuracy=0.60,
        sample_size=MINIMUM_PROMOTION_SAMPLE,
    )[0] is False
    assert promotion_decision(
        candidate_brier=0.42,
        candidate_accuracy=0.59,
        production_brier=0.50,
        production_accuracy=0.60,
        sample_size=MINIMUM_PROMOTION_SAMPLE,
    )[0] is False


def test_shadow_candidate_must_clear_absolute_quality_floors() -> None:
    assert promotion_decision(
        candidate_brier=MAXIMUM_PROMOTION_BRIER - 0.01,
        candidate_accuracy=MINIMUM_PROMOTION_ACCURACY - 0.01,
        production_brier=0.70,
        production_accuracy=0.40,
        sample_size=MINIMUM_PROMOTION_SAMPLE,
    )[0] is False
    assert promotion_decision(
        candidate_brier=MAXIMUM_PROMOTION_BRIER + 0.01,
        candidate_accuracy=MINIMUM_PROMOTION_ACCURACY + 0.05,
        production_brier=0.70,
        production_accuracy=0.40,
        sample_size=MINIMUM_PROMOTION_SAMPLE,
    )[0] is False


def test_shadow_candidate_with_worse_calibration_than_raw_is_rejected() -> None:
    accepted, reason = promotion_decision(
        candidate_brier=0.42,
        candidate_accuracy=0.60,
        production_brier=0.50,
        production_accuracy=0.55,
        sample_size=MINIMUM_PROMOTION_SAMPLE,
        candidate_calibrated_log_loss=1.02,
        candidate_raw_log_loss=1.01,
    )

    assert accepted is False
    assert "ham modelden kötü" in reason


def test_shadow_candidate_with_unacceptable_ece_regression_is_rejected() -> None:
    accepted, reason = promotion_decision(
        candidate_brier=0.42,
        candidate_accuracy=0.60,
        production_brier=0.50,
        production_accuracy=0.55,
        sample_size=MINIMUM_PROMOTION_SAMPLE,
        candidate_ece=0.10,
        production_ece=0.01,
    )

    assert accepted is False
    assert "ECE" in reason


def test_shadow_candidate_must_beat_the_frequency_baseline() -> None:
    accepted, reason = promotion_decision(
        candidate_brier=0.42,
        candidate_accuracy=0.60,
        production_brier=0.50,
        production_accuracy=0.55,
        sample_size=MINIMUM_PROMOTION_SAMPLE,
        candidate_log_loss=1.07,
        candidate_baseline_log_loss=1.073,
    )

    assert accepted is False
    assert "baseline" in reason
    assert promotion_decision(
        candidate_brier=0.42,
        candidate_accuracy=0.60,
        production_brier=0.50,
        production_accuracy=0.55,
        sample_size=MINIMUM_PROMOTION_SAMPLE,
        candidate_log_loss=1.01,
        candidate_baseline_log_loss=1.073,
    )[0] is True


def test_new_gate_checks_stay_silent_when_metrics_are_unavailable() -> None:
    assert promotion_decision(
        candidate_brier=0.42,
        candidate_accuracy=0.60,
        production_brier=0.50,
        production_accuracy=0.55,
        sample_size=MINIMUM_PROMOTION_SAMPLE,
    )[0] is True


def test_paired_market_comparison_requires_evidence_and_bootstrap_advantage() -> None:
    assert paired_market_comparison([(0.5, 0.5)] * 199)["status"] == "insufficient_evidence"
    result = paired_market_comparison([(0.40, 0.42)] * 200, bootstrap_samples=100)
    assert result["status"] == "passed"
    assert result["bootstrap_ci_upper"] < 0
