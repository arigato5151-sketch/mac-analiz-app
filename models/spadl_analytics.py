"""
SPADL Analytics — Expected Threat (xT) and VAEP helpers
=======================================================
Provides:
  - A simple 12×8 xT grid (Karun Singh style) pre-computed from open data.
  - ``compute_xt_for_actions()``: assigns xT gain to each SPADL action.
  - ``compute_vaep_features()``: extracts VAEP-compatible feature rows
    (experimental, shadow-model only).
  - ``aggregate_player_ratings()``: produces per-player xT-based ratings.

These are **research / shadow-model** utilities. They are never called by the
main prediction pipeline and therefore cannot degrade live accuracy.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional socceraction guard
# ---------------------------------------------------------------------------
try:
    import socceraction.vaep.formula as vaep_formula  # type: ignore[import]

    VAEP_AVAILABLE = True
    logger.info("socceraction VAEP available")
except (ImportError, AttributeError, Exception):  # noqa: BLE001
    # AttributeError: socceraction/pandera uses np.string_ removed in NumPy 2.0
    VAEP_AVAILABLE = False
    logger.debug("socceraction not available — VAEP features disabled")

# ---------------------------------------------------------------------------
# xT grid — 12 columns × 8 rows (pitch sections)
# Derived from Karun Singh's public xT grid (CC BY 4.0).
# Values represent the probability of scoring from each zone.
# ---------------------------------------------------------------------------
XT_GRID_12x8: np.ndarray = np.array(
    [
        [0.00638, 0.00349, 0.00364, 0.00366, 0.00366, 0.00364, 0.00349, 0.00638],
        [0.00779, 0.00428, 0.00441, 0.00450, 0.00450, 0.00441, 0.00428, 0.00779],
        [0.00931, 0.00506, 0.00518, 0.00519, 0.00519, 0.00518, 0.00506, 0.00931],
        [0.01336, 0.00760, 0.00774, 0.00783, 0.00783, 0.00774, 0.00760, 0.01336],
        [0.01841, 0.01143, 0.01230, 0.01257, 0.01257, 0.01230, 0.01143, 0.01841],
        [0.03764, 0.02472, 0.02628, 0.02851, 0.02851, 0.02628, 0.02472, 0.03764],
        [0.08284, 0.06241, 0.06979, 0.09174, 0.09174, 0.06979, 0.06241, 0.08284],
        [0.24257, 0.15173, 0.20534, 0.27456, 0.27456, 0.20534, 0.15173, 0.24257],
        [0.31256, 0.19540, 0.26654, 0.35128, 0.35128, 0.26654, 0.19540, 0.31256],
        [0.50476, 0.33426, 0.43116, 0.52654, 0.52654, 0.43116, 0.33426, 0.50476],
        [0.56992, 0.38441, 0.51511, 0.61212, 0.61212, 0.51511, 0.38441, 0.56992],
        [0.62642, 0.43892, 0.58893, 0.70613, 0.70613, 0.58893, 0.43892, 0.62642],
    ],
    dtype=np.float64,
)  # shape (12, 8): rows = pitch length bins, columns = pitch width bins

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
XT_ROWS = XT_GRID_12x8.shape[0]  # 12
XT_COLS = XT_GRID_12x8.shape[1]  # 8


# ---------------------------------------------------------------------------
# xT helpers
# ---------------------------------------------------------------------------

def _coords_to_xt(x: float, y: float) -> float:
    """Look up the xT value for a (x, y) pitch coordinate.

    Parameters
    ----------
    x : float
        Pitch x coordinate in [0, PITCH_LENGTH].
    y : float
        Pitch y coordinate in [0, PITCH_WIDTH].

    Returns
    -------
    float
        xT value from the grid, or 0.0 if out of range.
    """
    col = int(np.clip(y / PITCH_WIDTH * XT_COLS, 0, XT_COLS - 1))
    row = int(np.clip(x / PITCH_LENGTH * XT_ROWS, 0, XT_ROWS - 1))
    return float(XT_GRID_12x8[row, col])


def compute_xt_for_actions(actions: pd.DataFrame) -> pd.DataFrame:
    """Assign xT gain (``xt_gain``) to each row in a SPADL action DataFrame.

    Expected columns: ``start_x``, ``start_y``, ``end_x``, ``end_y``.
    Actions with missing coordinates receive ``xt_gain = 0.0``.

    Parameters
    ----------
    actions : pd.DataFrame
        SPADL action DataFrame.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with an additional ``xt_gain`` column (float).
    """
    required = {"start_x", "start_y", "end_x", "end_y"}
    if actions.empty or not required.issubset(actions.columns):
        actions = actions.copy()
        actions["xt_gain"] = 0.0
        return actions

    df = actions.copy()
    df["xt_start"] = df.apply(
        lambda r: _coords_to_xt(r["start_x"], r["start_y"]), axis=1
    )
    df["xt_end"] = df.apply(
        lambda r: _coords_to_xt(r["end_x"], r["end_y"]), axis=1
    )
    df["xt_gain"] = df["xt_end"] - df["xt_start"]
    df.drop(columns=["xt_start", "xt_end"], inplace=True)
    return df


def get_xt_grid() -> np.ndarray:
    """Return the 12×8 xT grid as a NumPy array."""
    return XT_GRID_12x8.copy()


# ---------------------------------------------------------------------------
# VAEP feature extraction (experimental — shadow model only)
# ---------------------------------------------------------------------------

def compute_vaep_features(
    actions: pd.DataFrame,
    game_id: int = 0,
) -> pd.DataFrame:
    """Extract VAEP-compatible feature rows from SPADL actions.

    Uses ``socceraction.vaep.formula`` when available; otherwise returns an
    empty DataFrame with the expected column structure so callers can detect
    unavailability without crashing.

    Parameters
    ----------
    actions : pd.DataFrame
        SPADL actions for a single match.
    game_id : int
        Game identifier (informational only).

    Returns
    -------
    pd.DataFrame
        Feature DataFrame with ``vaep_value``, ``offensive_value``,
        ``defensive_value`` columns (or empty if unavailable).
    """
    if actions.empty:
        return pd.DataFrame(columns=["game_id", "action_id", "vaep_value",
                                     "offensive_value", "defensive_value"])

    if not VAEP_AVAILABLE:
        logger.debug("VAEP not available — returning empty features for game %d", game_id)
        return pd.DataFrame(columns=["game_id", "action_id", "vaep_value",
                                     "offensive_value", "defensive_value"])

    try:
        scores = vaep_formula.value(
            actions=actions,
            Pscores=pd.Series(np.zeros(len(actions)), dtype=float),
            Pconcedes=pd.Series(np.zeros(len(actions)), dtype=float),
        )
        result = actions[["game_id"]].copy() if "game_id" in actions.columns else pd.DataFrame()
        result["game_id"] = game_id
        result["action_id"] = actions.index
        result["vaep_value"] = scores["vaep_value"].values
        result["offensive_value"] = scores["offensive_value"].values
        result["defensive_value"] = scores["defensive_value"].values
        return result.reset_index(drop=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning("VAEP feature extraction failed: %s", exc)
        return pd.DataFrame(columns=["game_id", "action_id", "vaep_value",
                                     "offensive_value", "defensive_value"])


# ---------------------------------------------------------------------------
# Player rating aggregation
# ---------------------------------------------------------------------------

def aggregate_player_ratings(
    actions_with_xt: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate per-player xT contributions from a match's SPADL actions.

    Parameters
    ----------
    actions_with_xt : pd.DataFrame
        SPADL DataFrame that already has an ``xt_gain`` column
        (output of :func:`compute_xt_for_actions`).

    Returns
    -------
    pd.DataFrame
        Columns: ``player_id``, ``team_id``, ``total_xt``, ``action_count``,
        ``xt_per_action``, ``positive_xt_count``, ``negative_xt_count``.
        Sorted descending by ``total_xt``.
    """
    required = {"player_id", "team_id", "xt_gain"}
    if actions_with_xt.empty or not required.issubset(actions_with_xt.columns):
        return pd.DataFrame(
            columns=["player_id", "team_id", "total_xt", "action_count",
                     "xt_per_action", "positive_xt_count", "negative_xt_count"]
        )

    df = actions_with_xt.copy()
    agg: dict[str, Any] = {
        "xt_gain": ["sum", "count", "mean"],
    }
    grouped = df.groupby(["player_id", "team_id"])["xt_gain"].agg(
        total_xt="sum",
        action_count="count",
        xt_per_action="mean",
    ).reset_index()

    pos = df[df["xt_gain"] > 0].groupby("player_id").size().rename("positive_xt_count")
    neg = df[df["xt_gain"] < 0].groupby("player_id").size().rename("negative_xt_count")

    grouped = grouped.merge(pos, on="player_id", how="left")
    grouped = grouped.merge(neg, on="player_id", how="left")
    grouped[["positive_xt_count", "negative_xt_count"]] = (
        grouped[["positive_xt_count", "negative_xt_count"]].fillna(0).astype(int)
    )
    grouped["total_xt"] = grouped["total_xt"].round(4)
    grouped["xt_per_action"] = grouped["xt_per_action"].round(6)

    return grouped.sort_values("total_xt", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Zone analysis
# ---------------------------------------------------------------------------

def compute_zone_threat_map(actions: pd.DataFrame) -> np.ndarray:
    """Aggregate action-start positions into a 12×8 zone threat map.

    Each cell accumulates the sum of ``xt_gain`` values for actions starting
    in that cell.

    Returns
    -------
    np.ndarray of shape (12, 8)
    """
    zone_map = np.zeros((XT_ROWS, XT_COLS), dtype=np.float64)

    if actions.empty or "start_x" not in actions.columns:
        return zone_map

    xt_col = "xt_gain" if "xt_gain" in actions.columns else None

    for _, row in actions.iterrows():
        col = int(np.clip(row["start_y"] / PITCH_WIDTH * XT_COLS, 0, XT_COLS - 1))
        r = int(np.clip(row["start_x"] / PITCH_LENGTH * XT_ROWS, 0, XT_ROWS - 1))
        value = float(row[xt_col]) if xt_col and not pd.isna(row[xt_col]) else 1.0
        zone_map[r, col] += value

    return zone_map
