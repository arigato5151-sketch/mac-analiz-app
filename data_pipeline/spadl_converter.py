"""
SPADL Converter — Phase 4: socceraction integration
====================================================
Converts StatsBomb (or generic) event data to SPADL format using socceraction.
Protected by an import guard — if socceraction is unavailable the module
degrades gracefully and all public functions return empty DataFrames / None.

SPADL (Soccer Action Description Language) standardises diverse event-log
formats into a single tabular structure with columns:
    game_id, period_id, time_seconds, player_id, team_id,
    start_x, start_y, end_x, end_y, result_id, bodypart_id, type_id
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    pass  # only used for type hints, no runtime import needed

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency guard
# ---------------------------------------------------------------------------
try:
    import socceraction.spadl as spadl
    import socceraction.spadl.statsbomb as sb_spadl

    SOCCERACTION_AVAILABLE = True
    logger.info("socceraction available — SPADL conversion enabled")
except (ImportError, AttributeError, Exception):  # noqa: BLE001
    # AttributeError: socceraction/pandera uses np.string_ removed in NumPy 2.0
    SOCCERACTION_AVAILABLE = False
    logger.warning(
        "socceraction not available (not installed or incompatible numpy version). "
        "SPADL conversion is disabled. "
        "Install via: pip install socceraction  (use requirements-research.txt)"
    )

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------
SPADL_COLUMNS = [
    "game_id",
    "period_id",
    "time_seconds",
    "player_id",
    "team_id",
    "start_x",
    "start_y",
    "end_x",
    "end_y",
    "result_id",
    "bodypart_id",
    "type_id",
]

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0


# ---------------------------------------------------------------------------
# Core conversion helpers
# ---------------------------------------------------------------------------

def _empty_spadl() -> pd.DataFrame:
    """Return an empty SPADL-shaped DataFrame (used as fallback)."""
    return pd.DataFrame(columns=SPADL_COLUMNS)


def convert_statsbomb_events_to_spadl(
    events: pd.DataFrame,
    game_id: int = 0,
    freeze_frames: pd.DataFrame | None = None,  # noqa: ARG001 — reserved for future use
) -> pd.DataFrame:
    """Convert StatsBomb event rows to SPADL format.

    Parameters
    ----------
    events:
        Raw StatsBomb event DataFrame as returned by ``statsbombpy`` or the
        project's :mod:`data_pipeline.statsbomb_adapter`.
    game_id:
        Integer identifier to assign to every action row.
    freeze_frames:
        Optional freeze-frame DataFrame (currently unused, reserved).

    Returns
    -------
    pd.DataFrame
        SPADL-formatted actions, or an empty DataFrame if ``socceraction`` is
        unavailable or conversion fails.
    """
    if not SOCCERACTION_AVAILABLE:
        logger.debug("convert_statsbomb_events_to_spadl: socceraction unavailable, returning empty")
        return _empty_spadl()

    if events is None or events.empty:
        logger.debug("convert_statsbomb_events_to_spadl: empty events input")
        return _empty_spadl()

    try:
        actions = sb_spadl.convert_to_actions(events=events, home_team_id=0)
        if "game_id" not in actions.columns:
            actions.insert(0, "game_id", game_id)
        else:
            actions["game_id"] = game_id
        logger.info("SPADL conversion complete: %d actions from %d events", len(actions), len(events))
        return actions
    except Exception as exc:  # noqa: BLE001
        logger.warning("SPADL conversion failed: %s", exc)
        return _empty_spadl()


def get_spadl_action_types() -> dict[int, str]:
    """Return the SPADL action-type mapping (id → name).

    Falls back to a hard-coded subset if ``socceraction`` is unavailable.
    """
    if SOCCERACTION_AVAILABLE:
        try:
            at = spadl.config.actiontypes
            return dict(enumerate(at))
        except Exception:  # noqa: BLE001
            pass

    # Minimal fallback — covers the most common types
    return {
        0: "pass",
        1: "cross",
        2: "throw_in",
        3: "freekick_crossed",
        4: "freekick_short",
        5: "corner_crossed",
        6: "corner_short",
        7: "take_on",
        8: "foul",
        9: "tackle",
        10: "interception",
        11: "shot",
        12: "shot_penalty",
        13: "shot_freekick",
        14: "keeper_save",
        15: "keeper_claim",
        16: "keeper_punch",
        17: "keeper_pick_up",
        18: "clearance",
        19: "bad_touch",
        20: "non_action",
        21: "dribble",
        22: "goalkick",
    }


def get_spadl_result_types() -> dict[int, str]:
    """Return the SPADL result mapping (id → name)."""
    if SOCCERACTION_AVAILABLE:
        try:
            rt = spadl.config.results
            return dict(enumerate(rt))
        except Exception:  # noqa: BLE001
            pass

    return {
        0: "fail",
        1: "success",
        2: "offside",
        3: "owngoal",
        4: "yellow_card",
        5: "red_card",
    }


# ---------------------------------------------------------------------------
# Aggregate statistics helpers
# ---------------------------------------------------------------------------

def compute_shot_statistics(actions: pd.DataFrame) -> dict:
    """Summarise shot-related SPADL actions.

    Parameters
    ----------
    actions:
        SPADL action DataFrame (output of :func:`convert_statsbomb_events_to_spadl`).

    Returns
    -------
    dict with keys: total_shots, shots_on_target, goals, shot_conversion_rate
    """
    if actions.empty or "type_id" not in actions.columns:
        return {"total_shots": 0, "shots_on_target": 0, "goals": 0, "shot_conversion_rate": 0.0}

    action_types = get_spadl_action_types()
    shot_type_ids = {k for k, v in action_types.items() if "shot" in v}
    result_types = get_spadl_result_types()
    success_id = next((k for k, v in result_types.items() if v == "success"), 1)

    shots = actions[actions["type_id"].isin(shot_type_ids)]
    total = len(shots)
    goals = int((shots["result_id"] == success_id).sum()) if "result_id" in shots.columns else 0
    on_target = goals  # SPADL: success = goal for shots; approximate

    return {
        "total_shots": total,
        "shots_on_target": on_target,
        "goals": goals,
        "shot_conversion_rate": round(goals / total, 4) if total > 0 else 0.0,
    }


def compute_pass_statistics(actions: pd.DataFrame) -> dict:
    """Summarise pass-related SPADL actions.

    Returns
    -------
    dict with keys: total_passes, successful_passes, pass_accuracy
    """
    if actions.empty or "type_id" not in actions.columns:
        return {"total_passes": 0, "successful_passes": 0, "pass_accuracy": 0.0}

    action_types = get_spadl_action_types()
    pass_type_ids = {k for k, v in action_types.items() if "pass" in v or "cross" in v}
    result_types = get_spadl_result_types()
    success_id = next((k for k, v in result_types.items() if v == "success"), 1)

    passes = actions[actions["type_id"].isin(pass_type_ids)]
    total = len(passes)
    successful = int((passes["result_id"] == success_id).sum()) if "result_id" in passes.columns else 0

    return {
        "total_passes": total,
        "successful_passes": successful,
        "pass_accuracy": round(successful / total, 4) if total > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Disk cache helpers
# ---------------------------------------------------------------------------

def save_spadl_to_parquet(actions: pd.DataFrame, path: str | Path) -> None:
    """Save SPADL actions to Parquet for re-use."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not actions.empty:
        actions.to_parquet(path, index=False)
        logger.debug("SPADL saved to %s", path)


def load_spadl_from_parquet(path: str | Path) -> pd.DataFrame:
    """Load cached SPADL actions from Parquet.

    Returns an empty DataFrame if the file does not exist or cannot be read.
    """
    path = Path(path)
    if not path.exists():
        return _empty_spadl()
    try:
        df = pd.read_parquet(path)
        logger.debug("SPADL loaded from %s (%d rows)", path, len(df))
        return df
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not load SPADL parquet %s: %s", path, exc)
        return _empty_spadl()
