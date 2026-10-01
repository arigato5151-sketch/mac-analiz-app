"""A fresh training run creates its own manifest without blessing old binaries."""

import joblib
import pytest

from models.artifact_store import ArtifactIntegrityError, load_verified_joblib
from models.train_model import save_model_bundle


def test_fresh_training_bundle_has_matching_manifest(tmp_path):
    model_path, _ = save_model_bundle({"training_end": "2026-10-01T00:00:00Z"}, tmp_path, publish_latest=True)

    assert load_verified_joblib(model_path)["model_version"] == model_path.stem
    assert load_verified_joblib(tmp_path / "latest.joblib")["model_version"] == model_path.stem


def test_training_refuses_to_bless_preexisting_unverified_binary(tmp_path):
    joblib.dump({"model_version": "unverified"}, tmp_path / "latest.joblib")

    with pytest.raises(ArtifactIntegrityError, match="trusted manifest"):
        save_model_bundle({"training_end": "2026-10-01T00:00:00Z"}, tmp_path)

    assert not (tmp_path / "artifact_manifest.json").exists()
