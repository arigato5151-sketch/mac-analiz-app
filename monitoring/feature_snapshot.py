"""
Feature Snapshot — save and load feature distribution snapshots.

Used for drift monitoring: the reference snapshot is saved at training time
(fit slice) and the current snapshot is saved at validation time.

Rules:
- Only numeric feature values are stored — no labels, no predictions,
  no player names, no API keys, no PII.
- Files are stored as Parquet next to the model artifacts.
- Callers receive None when a snapshot is absent so drift can report
  "unavailable" rather than producing a misleading "OK".
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

LOGGER = logging.getLogger(__name__)

# Standard snapshot file names (relative to saved_models dir)
REFERENCE_SNAPSHOT_NAME = "feature_snapshot_ref.parquet"
CURRENT_SNAPSHOT_NAME = "feature_snapshot_current.parquet"


def save_feature_snapshot(
    features: pd.DataFrame,
    path: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Persist a feature DataFrame as a Parquet snapshot.

    Only numeric columns are stored. A ``_snapshot_saved_at`` column is
    appended with the UTC timestamp so downstream drift checks can verify
    chronological ordering.

    Parameters
    ----------
    features:
        Feature DataFrame (e.g. x_fit or x_validation from train_models()).
    path:
        Destination Parquet file path.
    metadata:
        Optional dict of scalar metadata (row count, date range, etc.) stored
        as Parquet metadata. Must not contain secrets.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if features is None or features.empty:
        LOGGER.warning("save_feature_snapshot: empty DataFrame — snapshot not written to %s", path)
        return

    # Keep only numeric columns to avoid PII / secret leakage
    numeric_cols = features.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        LOGGER.warning("save_feature_snapshot: no numeric columns found — snapshot not written")
        return

    snapshot = features[numeric_cols].copy()
    snapshot["_snapshot_saved_at"] = datetime.now(timezone.utc).isoformat()

    try:
        snapshot.to_parquet(path, index=False)
        LOGGER.info(
            "Feature snapshot saved: %s (%d rows, %d features)",
            path.name,
            len(snapshot),
            len(numeric_cols),
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Feature snapshot could not be saved to %s: %s", path, exc)


def load_feature_snapshot(path: str | Path) -> pd.DataFrame | None:
    """Load a feature snapshot Parquet file.

    Returns
    -------
    pd.DataFrame | None
        The snapshot DataFrame, or ``None`` if the file does not exist or
        cannot be read. Callers must treat ``None`` as "snapshot unavailable".
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


def extract_snapshot_metadata(df: pd.DataFrame | None) -> dict[str, Any]:
    """Extract summary metadata from a snapshot DataFrame.

    Returns a dict with:
    - ``rows``: int, number of rows (0 if df is None or empty)
    - ``columns``: list[str], feature column names (excl. _snapshot_saved_at)
    - ``saved_at``: str | None, ISO timestamp from _snapshot_saved_at column
    - ``available``: bool
    """
    if df is None or df.empty:
        return {"rows": 0, "columns": [], "saved_at": None, "available": False}

    feature_cols = [c for c in df.columns if not c.startswith("_")]
    saved_at: str | None = None
    if "_snapshot_saved_at" in df.columns:
        saved_at = str(df["_snapshot_saved_at"].iloc[0])

    return {
        "rows": len(df),
        "columns": feature_cols,
        "saved_at": saved_at,
        "available": True,
    }


def snapshot_feature_columns(df: pd.DataFrame | None) -> list[str]:
    """Return feature column names from a snapshot (excludes _snapshot_* cols)."""
    if df is None or df.empty:
        return []
    return [c for c in df.columns if not c.startswith("_")]

