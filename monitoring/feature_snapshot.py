"""
Feature Snapshot — save and load feature distribution snapshots.

Used for drift monitoring:
- Reference snapshot is saved at training time from the training fit slice
  (source: "training_reference").
- Current snapshot is saved during live prediction/inference from the actual
  features calculated for upcoming fixtures (source: "production_inference").

Rules:
- Only numeric feature values are stored — no labels, no predictions,
  no player names, no API keys, no PII.
- Files are stored as Parquet next to the model artifacts with companion JSON metadata.
- Callers receive None when a snapshot is absent so drift can report
  "unavailable" rather than producing a misleading "OK".
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

LOGGER = logging.getLogger(__name__)

FEATURE_SCHEMA_VERSION = "1.0"

# Standard alias names (relative to saved_models dir)
REFERENCE_SNAPSHOT_NAME = "feature_snapshot_ref.parquet"
CURRENT_SNAPSHOT_NAME = "feature_snapshot_current.parquet"


def snapshot_filename(kind: str, model_version: str | None = None) -> str:
    """Return the canonical filename for a snapshot kind ('ref' or 'current')."""
    if model_version:
        clean_version = model_version.strip()
        return f"feature_snapshot_{kind}_{clean_version}.parquet"
    return f"feature_snapshot_{kind}.parquet"


def save_feature_snapshot(
    features: pd.DataFrame,
    path: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist a feature DataFrame as a Parquet snapshot with companion JSON metadata.

    Only numeric columns are stored. Secrets and PII are strictly excluded.
    A companion JSON sidecar file (<path>.json) is saved alongside the Parquet file.

    Parameters
    ----------
    features:
        Feature DataFrame (e.g. x_fit from training or upcoming features from inference).
    path:
        Destination Parquet file path.
    metadata:
        Optional dictionary containing:
        - model_version
        - period_start (ISO string)
        - period_end (ISO string)
        - source ("training_reference" or "production_inference")
        - feature_schema_version
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if features is None or features.empty:
        LOGGER.warning(
            "save_feature_snapshot: empty DataFrame — snapshot not written to %s", path
        )
        return {"available": False, "rows": 0, "columns": []}

    # Keep only numeric columns to avoid PII / secret leakage
    numeric_cols = features.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        LOGGER.warning("save_feature_snapshot: no numeric columns found — snapshot not written")
        return {"available": False, "rows": 0, "columns": []}

    now_iso = datetime.now(timezone.utc).isoformat()
    snapshot = features[numeric_cols].copy()
    snapshot["_snapshot_saved_at"] = now_iso

    meta: dict[str, Any] = {
        "model_version": (metadata or {}).get("model_version", "unknown"),
        "created_at": now_iso,
        "period_start": (metadata or {}).get("period_start", ""),
        "period_end": (metadata or {}).get("period_end", ""),
        "row_count": len(snapshot),
        "feature_schema_version": (metadata or {}).get(
            "feature_schema_version", FEATURE_SCHEMA_VERSION
        ),
        "source": (metadata or {}).get("source", "unknown"),
        "columns": numeric_cols,
        "available": True,
    }

    try:
        snapshot.to_parquet(path, index=False)
        meta_path = path.with_suffix(".json")
        meta_path.write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        LOGGER.info(
            "Feature snapshot saved: %s (%d rows, %d features, source=%s)",
            path.name,
            len(snapshot),
            len(numeric_cols),
            meta.get("source"),
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Feature snapshot could not be saved to %s: %s", path, exc)

    return meta


def load_feature_snapshot(path: str | Path) -> pd.DataFrame | None:
    """Load a feature snapshot Parquet file.

    Returns None if missing or corrupted.
    """
    path = Path(path)
    if not path.exists():
        LOGGER.debug("Feature snapshot not found: %s", path)
        return None

    try:
        df = pd.read_parquet(path)
        LOGGER.debug(
            "Feature snapshot loaded: %s (%d rows, %d cols)",
            path.name,
            len(df),
            len(df.columns),
        )
        return df
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Feature snapshot could not be loaded from %s: %s", path, exc)
        return None


def load_snapshot_metadata(path: str | Path) -> dict[str, Any]:
    """Load the companion JSON metadata file for a snapshot if it exists."""
    meta_path = Path(path).with_suffix(".json")
    if meta_path.is_file():
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("Failed reading snapshot metadata at %s: %s", meta_path, exc)
    return {}


def extract_snapshot_metadata(
    df: pd.DataFrame | None,
    path: str | Path | None = None,
) -> dict[str, Any]:
    """Extract summary metadata combining companion JSON file and DataFrame structure."""
    file_meta = load_snapshot_metadata(path) if path else {}

    if df is None or df.empty:
        return {
            "rows": file_meta.get("row_count", 0),
            "columns": file_meta.get("columns", []),
            "saved_at": file_meta.get("created_at"),
            "model_version": file_meta.get("model_version", "unknown"),
            "period_start": file_meta.get("period_start", ""),
            "period_end": file_meta.get("period_end", ""),
            "source": file_meta.get("source", "unknown"),
            "feature_schema_version": file_meta.get(
                "feature_schema_version", FEATURE_SCHEMA_VERSION
            ),
            "available": False,
        }

    feature_cols = [c for c in df.columns if not c.startswith("_")]
    saved_at = file_meta.get("created_at")
    if not saved_at and "_snapshot_saved_at" in df.columns:
        saved_at = str(df["_snapshot_saved_at"].iloc[0])

    return {
        "rows": len(df),
        "columns": feature_cols,
        "saved_at": saved_at,
        "model_version": file_meta.get("model_version", "unknown"),
        "period_start": file_meta.get("period_start", ""),
        "period_end": file_meta.get("period_end", ""),
        "source": file_meta.get("source", "unknown"),
        "feature_schema_version": file_meta.get(
            "feature_schema_version", FEATURE_SCHEMA_VERSION
        ),
        "available": True,
    }


def resolve_snapshot_path(
    saved_models_dir: Path | str,
    kind: str = "ref",
    model_version: str | None = None,
) -> Path:
    """Resolve the appropriate snapshot path for a model version or default alias.

    Prefers versioned snapshot (e.g. feature_snapshot_ref_{version}.parquet).
    Falls back to unversioned alias (feature_snapshot_ref.parquet).
    """
    directory = Path(saved_models_dir)
    if model_version:
        versioned_name = snapshot_filename(kind, model_version)
        versioned_path = directory / versioned_name
        if versioned_path.is_file():
            return versioned_path

    # Fallback to default alias
    return directory / snapshot_filename(kind)


def snapshot_feature_columns(df: pd.DataFrame | None) -> list[str]:
    """Return feature column names from a snapshot (excludes _snapshot_* columns)."""
    if df is None or df.empty:
        return []
    return [c for c in df.columns if not c.startswith("_")]
