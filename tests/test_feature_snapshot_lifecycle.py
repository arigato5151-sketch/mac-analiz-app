"""Unit tests for feature snapshot artifact lifecycle, storage upload/pull, and inference telemetry."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from models.artifact_store import (
    ArtifactStoreError,
    download_model_artifacts,
    push_local_models,
)
from models.predict import record_inference_feature_snapshot
from monitoring.feature_snapshot import (
    CURRENT_SNAPSHOT_NAME,
    REFERENCE_SNAPSHOT_NAME,
    extract_snapshot_metadata,
    load_feature_snapshot,
    resolve_snapshot_path,
    save_feature_snapshot,
    snapshot_filename,
)


def test_parquet_snapshot_save_load_and_versioning(tmp_path: Path):
    """Verify versioned snapshot saving, companion metadata, and path resolution."""
    rng = np.random.default_rng(42)
    features = pd.DataFrame(rng.normal(0, 1, (50, 5)), columns=[f"f_{i}" for i in range(5)])
    model_version = "model_v20260918T120000Z"

    # 1. Save reference snapshot with version
    ref_path = tmp_path / snapshot_filename("ref", model_version)
    meta = {
        "model_version": model_version,
        "period_start": "2024-01-01T00:00:00Z",
        "period_end": "2024-06-01T00:00:00Z",
        "source": "training_reference",
    }
    saved_meta = save_feature_snapshot(features, ref_path, metadata=meta)
    assert saved_meta["available"] is True
    assert saved_meta["row_count"] == 50
    assert saved_meta["source"] == "training_reference"
    assert ref_path.is_file()
    assert ref_path.with_suffix(".json").is_file()

    # 2. Path resolution: versioned vs unversioned fallback
    resolved = resolve_snapshot_path(tmp_path, kind="ref", model_version=model_version)
    assert resolved == ref_path

    # Fallback to default alias if specific version doesn't exist
    alias_path = tmp_path / REFERENCE_SNAPSHOT_NAME
    save_feature_snapshot(features, alias_path, metadata=meta)
    resolved_alias = resolve_snapshot_path(tmp_path, kind="ref", model_version="non_existent_version")
    assert resolved_alias == alias_path

    # 3. Load snapshot & metadata
    loaded_df = load_feature_snapshot(ref_path)
    assert loaded_df is not None
    assert len(loaded_df) == 50
    assert list(loaded_df.columns[:5]) == [f"f_{i}" for i in range(5)]

    extracted_meta = extract_snapshot_metadata(loaded_df, path=ref_path)
    assert extracted_meta["model_version"] == model_version
    assert extracted_meta["period_start"] == "2024-01-01T00:00:00Z"
    assert extracted_meta["period_end"] == "2024-06-01T00:00:00Z"
    assert extracted_meta["source"] == "training_reference"


def test_record_inference_feature_snapshot(tmp_path: Path):
    """Verify live inference features are persisted with production telemetry metadata."""
    rng = np.random.default_rng(42)
    features = pd.DataFrame(rng.normal(0, 1, (20, 4)), columns=["feat_a", "feat_b", "feat_c", "feat_d"])
    upcoming_matches = [
        {"id": 101, "match_date": datetime(2026, 9, 20, 18, 0, tzinfo=timezone.utc)},
        {"id": 102, "match_date": datetime(2026, 9, 22, 20, 0, tzinfo=timezone.utc)},
    ]
    model_version = "model_v20260918T120000Z"

    alias_path = record_inference_feature_snapshot(
        features,
        upcoming_matches,
        model_version=model_version,
        output_dir=tmp_path,
    )
    assert alias_path is not None
    assert alias_path.name == CURRENT_SNAPSHOT_NAME
    assert alias_path.is_file()

    versioned_path = tmp_path / snapshot_filename("current", model_version)
    assert versioned_path.is_file()

    meta = extract_snapshot_metadata(load_feature_snapshot(versioned_path), path=versioned_path)
    assert meta["source"] == "production_inference"
    assert meta["model_version"] == model_version
    assert meta["row_count"] == 20
    assert meta["rows"] == 20
    assert "2026-09-20" in meta["period_start"]
    assert "2026-09-22" in meta["period_end"]


def test_push_local_models_includes_parquet_and_metadata(tmp_path: Path):
    """Verify push_local_models uploads .joblib, .json, and .parquet snapshots."""
    # Create mock artifacts
    (tmp_path / "model_v1.joblib").write_bytes(b"mock_binary")
    (tmp_path / "model_v1.json").write_text("{}", encoding="utf-8")
    (tmp_path / "feature_snapshot_ref_model_v1.parquet").write_bytes(b"mock_parquet")
    (tmp_path / "feature_snapshot_ref_model_v1.json").write_text("{}", encoding="utf-8")
    (tmp_path / "feature_snapshot_current.parquet").write_bytes(b"mock_current_parquet")

    uploaded_names: list[str] = []

    def mock_upload(path, name=None, upsert=True):
        uploaded_names.append(name or Path(path).name)
        return name or Path(path).name

    with patch("models.artifact_store.upload_model", side_effect=mock_upload):
        result = push_local_models(dest_dir=tmp_path)

    assert "model_v1.joblib" in uploaded_names
    assert "model_v1.json" in uploaded_names
    assert "feature_snapshot_ref_model_v1.parquet" in uploaded_names
    assert "feature_snapshot_ref_model_v1.json" in uploaded_names
    assert "feature_snapshot_current.parquet" in uploaded_names
    assert len(result) == 5


def test_download_model_artifacts_lifecycle(tmp_path: Path):
    """Verify download_model_artifacts pulls model, json and snapshots, handling storage errors."""
    downloaded_files: dict[str, Path] = {}

    def mock_download(name, dest_dir=None, local_path=None):
        if "missing" in name:
            raise ArtifactStoreError(f"Object not found: {name}")
        out_path = Path(dest_dir) / name
        out_path.write_bytes(b"dummy_content")
        return out_path

    with patch("models.artifact_store.download_model", side_effect=mock_download):
        artifacts = download_model_artifacts("latest", dest_dir=tmp_path)

    assert "model" in artifacts
    assert "metadata" in artifacts
    assert artifacts["model"].name == "latest.joblib"
    assert (tmp_path / "latest.joblib").is_file()


def test_download_model_artifacts_storage_unreachable_fallback(tmp_path: Path):
    """When storage is unreachable, download_model_artifacts raises ArtifactStoreError without crashing."""
    with patch("models.artifact_store.download_model", side_effect=ArtifactStoreError("Network down")):
        with pytest.raises(ArtifactStoreError):
            download_model_artifacts("latest", dest_dir=tmp_path)

