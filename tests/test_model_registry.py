from __future__ import annotations

import json

import joblib
import pytest

import app.components.model_registry as registry


@pytest.fixture(autouse=True)
def _registry_cache():
    registry.load_model_registry.clear()
    yield
    registry.load_model_registry.clear()


@pytest.fixture
def artifact_dir(tmp_path, monkeypatch):
    model_dir = tmp_path / "models" / "saved_models"
    model_dir.mkdir(parents=True)
    monkeypatch.setattr(registry, "PROJECT_ROOT", tmp_path)
    joblib.dump(
        {
            "model_version": "model_v20260102T000000Z",
            "training_end": "2026-01-02T00:00:00+00:00",
        },
        model_dir / "latest.joblib",
    )
    artifacts = {
        "model_v20260101T000000Z.json": {"training_end": "2026-01-01T00:00:00+00:00"},
        "model_v20260102T000000Z.json": {"training_end": "2026-01-02T00:00:00+00:00"},
        "model_v20260105T000000Z.json": {"training_end": "2026-01-05T00:00:00+00:00"},
    }
    for name, meta in artifacts.items():
        (model_dir / name).write_text(json.dumps(meta), encoding="utf-8")
    return model_dir


def test_registry_assigns_roles_from_artifacts(artifact_dir) -> None:
    roles = {info.version: info.status for info in registry.load_model_registry()}

    assert roles["model_v20260102T000000Z"] == registry.STATUS_PRODUCTION
    assert roles["model_v20260105T000000Z"] == registry.STATUS_CANDIDATE
    assert roles["model_v20260101T000000Z"] == registry.STATUS_RETIRED


def test_production_version_for_prefers_the_registry_role(artifact_dir) -> None:
    assert (
        registry.production_version_for(
            ["model_v20260101T000000Z", "model_v20260102T000000Z"]
        )
        == "model_v20260102T000000Z"
    )
    assert registry.production_version_for(["model_v20260101T000000Z"]) is None
    assert registry.production_version_for([]) is None


def test_status_label_tr_never_fakes_unknown_versions(artifact_dir) -> None:
    assert registry.status_label_tr("model_v20260102T000000Z") == "Üretim"
    assert registry.status_label_tr("model_v20260105T000000Z") == "Aday"
    assert registry.status_label_tr("model_v20260101T000000Z") == "Eski sürüm"
    assert registry.status_label_tr("model_v9999") == "Durum bilinmiyor"


def test_active_production_version_works_without_data_rows(artifact_dir) -> None:
    assert registry.active_production_version() == "model_v20260102T000000Z"


def test_version_label_carries_role_and_sample_size(artifact_dir) -> None:
    assert (
        registry.version_label("model_v20260102T000000Z", sample_size=630)
        == "model_v20260102T000000Z · Üretim · n=630"
    )
    assert registry.version_label("model_v20260102T000000Z").endswith("· Üretim")
    assert registry.version_label("model_v9999").endswith("Durum bilinmiyor")


def test_corrupt_metadata_falls_back_to_filename_timestamp(artifact_dir) -> None:
    (artifact_dir / "model_v20260107T000000Z.json").write_text(
        "{invalid", encoding="utf-8"
    )

    roles = {info.version: info.status for info in registry.load_model_registry()}

    assert roles["model_v20260107T000000Z"] == registry.STATUS_CANDIDATE
