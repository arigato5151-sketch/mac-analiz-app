"""Rebuild a missing training reference snapshot from the model's fit period."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from config.settings import PROJECT_ROOT, get_settings
from db.db_client import SupabaseRestClient
from models.artifact_store import MANIFEST_NAME, load_verified_joblib
from models.feature_engineering import build_training_dataset
from models.train_model import _chronological_slices, load_completed_matches
from monitoring.feature_snapshot import (
    REFERENCE_SNAPSHOT_NAME,
    save_feature_snapshot,
    snapshot_filename,
)


def ensure_reference_snapshot(
    *, directory: Path | None = None, db: SupabaseRestClient | None = None
) -> dict[str, Any]:
    """Keep an existing reference, or reproduce the exact fit window from DB history."""
    models_dir = directory or PROJECT_ROOT / "models" / "saved_models"
    bundle_path = models_dir / "latest.joblib"
    bundle = load_verified_joblib(
        bundle_path, manifest_path=models_dir / MANIFEST_NAME
    )
    model_version = str(bundle.get("model_version", "")).strip()
    if not model_version:
        raise RuntimeError("The production model bundle has no model_version")

    versioned_path = models_dir / snapshot_filename("ref", model_version)
    alias_path = models_dir / REFERENCE_SNAPSHOT_NAME
    if versioned_path.is_file() or alias_path.is_file():
        return {"created": False, "model_version": model_version}

    settings = get_settings()
    database = db or SupabaseRestClient(
        settings.supabase_url, settings.supabase_service_role_key
    )
    training_end = pd.to_datetime(bundle.get("training_end"), utc=True, errors="coerce")
    expected_rows = int(bundle.get("trained_rows", 0))
    if pd.isna(training_end) or expected_rows < 1:
        raise RuntimeError("The production model bundle lacks its training window metadata")

    matches = load_completed_matches(database)
    cutoff = training_end.to_pydatetime()
    historical_matches = [
        match
        for match in matches
        if datetime.fromisoformat(str(match["match_date"]).replace("Z", "+00:00"))
        <= cutoff
    ]
    features, labels = build_training_dataset(historical_matches)
    expected_columns = list(bundle.get("feature_columns") or [])
    missing_columns = sorted(set(expected_columns) - set(features.columns))
    if not expected_columns or missing_columns:
        raise RuntimeError(
            "Rebuilt features are missing production model columns: "
            + ", ".join(missing_columns)
        )
    if len(features) != expected_rows:
        raise RuntimeError(
            f"Rebuilt training rows differ from model metadata: "
            f"expected {expected_rows}, found {len(features)}"
        )

    fit_slice = _chronological_slices(len(features))[0]
    reference = features.iloc[fit_slice].loc[:, expected_columns]
    period_start = pd.to_datetime(labels.iloc[fit_slice.start or 0]["match_date"], utc=True)
    period_end = pd.to_datetime(labels.iloc[fit_slice.stop - 1]["match_date"], utc=True)
    metadata = {
        "model_version": model_version,
        "period_start": period_start.isoformat(),
        "period_end": period_end.isoformat(),
        "source": "reconstructed_training_fit",
    }
    save_feature_snapshot(reference, versioned_path, metadata=metadata)
    save_feature_snapshot(reference, alias_path, metadata=metadata)
    return {
        "created": versioned_path.is_file() and alias_path.is_file(),
        "model_version": model_version,
        "rows": len(reference),
        "training_rows": len(features),
    }


if __name__ == "__main__":
    print(json.dumps(ensure_reference_snapshot(), ensure_ascii=False))
