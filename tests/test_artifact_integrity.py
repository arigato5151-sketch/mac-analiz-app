import json

import joblib
import pytest

from models.artifact_store import (
    ArtifactIntegrityError,
    build_artifact_manifest,
    load_verified_joblib,
)


def test_verified_loader_accepts_matching_manifest(tmp_path):
    artifact = tmp_path / "model.joblib"
    joblib.dump({"model_version": "test"}, artifact)
    manifest = tmp_path / "artifact_manifest.json"
    manifest.write_text(json.dumps(build_artifact_manifest(tmp_path)), encoding="utf-8")

    assert load_verified_joblib(artifact)["model_version"] == "test"


def test_verified_loader_rejects_missing_manifest(tmp_path):
    artifact = tmp_path / "model.joblib"
    joblib.dump({"model_version": "test"}, artifact)

    with pytest.raises(ArtifactIntegrityError, match="manifest is missing"):
        load_verified_joblib(artifact)


def test_verified_loader_rejects_tampered_artifact(tmp_path):
    artifact = tmp_path / "model.joblib"
    joblib.dump({"model_version": "test"}, artifact)
    manifest = tmp_path / "artifact_manifest.json"
    manifest.write_text(json.dumps(build_artifact_manifest(tmp_path)), encoding="utf-8")
    artifact.write_bytes(artifact.read_bytes() + b"tampered")

    with pytest.raises(ArtifactIntegrityError, match="checksum mismatch"):
        load_verified_joblib(artifact)
