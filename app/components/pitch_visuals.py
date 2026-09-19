"""Pitch and tactical visualizations using mplsoccer for event data."""

from __future__ import annotations

import logging
from typing import Any

import matplotlib

# Streamlit and CI render figures without a desktop display server. Select a
# non-interactive backend before importing pyplot to avoid Tk initialization.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

try:
    from mplsoccer import Pitch, PyPizza, VerticalPitch
    MPLSOCCER_AVAILABLE = True
except ImportError:
    Pitch = None
    VerticalPitch = None
    PyPizza = None
    MPLSOCCER_AVAILABLE = False


def render_shot_map(
    events: pd.DataFrame,
    *,
    team_name: str | None = None,
    pitch_color: str = "#101720",
    line_color: str = "#555555",
) -> plt.Figure | None:
    """Render a shot map with xG bubble sizes and goal markers."""
    if not MPLSOCCER_AVAILABLE:
        LOGGER.warning("mplsoccer is not available")
        return None

    if events.empty or "type" not in events.columns:
        return None

    shots = events[events["type"] == "Shot"].copy()
    if team_name and "team" in shots.columns:
        shots = shots[shots["team"] == team_name]

    if shots.empty:
        return None

    pitch = Pitch(
        pitch_type="statsbomb",
        pitch_color=pitch_color,
        line_color=line_color,
        half=True,
    )
    fig, ax = pitch.draw(figsize=(8, 6))

    # Extract coordinates
    if "location" in shots.columns:
        locs = shots["location"].dropna()
        x = locs.apply(lambda l: l[0] if isinstance(l, (list, tuple)) and len(l) >= 2 else np.nan)
        y = locs.apply(lambda l: l[1] if isinstance(l, (list, tuple)) and len(l) >= 2 else np.nan)
    else:
        x = shots.get("x", pd.Series())
        y = shots.get("y", pd.Series())

    valid_mask = x.notna() & y.notna()
    if not valid_mask.any():
        plt.close(fig)
        return None

    valid_shots = shots[valid_mask]
    vx = x[valid_mask]
    vy = y[valid_mask]

    xg = valid_shots.get("shot_statsbomb_xg", pd.Series(0.1, index=valid_shots.index)).fillna(0.1)
    sizes = np.clip(xg * 600 + 40, 40, 700)

    # Goal vs Non-goal
    outcomes = valid_shots.get("shot_outcome", pd.Series("", index=valid_shots.index))
    is_goal = outcomes.str.lower() == "goal"

    # Non-goals
    pitch.scatter(
        vx[~is_goal],
        vy[~is_goal],
        s=sizes[~is_goal],
        c="#3b82f6",
        alpha=0.6,
        edgecolors="#ffffff",
        ax=ax,
        label="Şut",
    )
    # Goals
    if is_goal.any():
        pitch.scatter(
            vx[is_goal],
            vy[is_goal],
            s=sizes[is_goal],
            c="#ef4444",
            marker="*",
            edgecolors="#ffffff",
            ax=ax,
            label="Gol",
        )

    title = f"{team_name} — Şut Haritası" if team_name else "Şut Haritası"
    ax.set_title(title, color="white", fontsize=14, pad=12)
    ax.legend(facecolor=pitch_color, edgecolor=line_color, labelcolor="white", loc="upper left")
    plt.tight_layout()
    return fig


def render_pass_network(
    events: pd.DataFrame,
    team_name: str,
    *,
    pitch_color: str = "#101720",
    line_color: str = "#555555",
    min_passes: int = 3,
) -> plt.Figure | None:
    """Render a tactical pass network showing average player positions and passing links."""
    if not MPLSOCCER_AVAILABLE or events.empty:
        return None

    passes = events[(events["type"] == "Pass") & (events["team"] == team_name)].copy()
    if passes.empty or "player" not in passes.columns or "pass_recipient" not in passes.columns:
        return None

    # Location parsing
    if "location" in passes.columns:
        locs = passes["location"].dropna()
        passes["x"] = locs.apply(lambda l: l[0] if isinstance(l, (list, tuple)) and len(l) >= 2 else np.nan)
        passes["y"] = locs.apply(lambda l: l[1] if isinstance(l, (list, tuple)) and len(l) >= 2 else np.nan)

    passes = passes.dropna(subset=["x", "y", "player", "pass_recipient"])
    if len(passes) < 15:
        return None

    # Average player positions
    avg_loc = passes.groupby("player")[["x", "y"]].mean()
    pass_counts = passes.groupby("player").size()

    pitch = Pitch(pitch_type="statsbomb", pitch_color=pitch_color, line_color=line_color)
    fig, ax = pitch.draw(figsize=(9, 6))

    # Pairwise pass combinations
    pairs = passes.groupby(["player", "pass_recipient"]).size().reset_index(name="count")
    significant_pairs = pairs[pairs["count"] >= min_passes]

    for _, row in significant_pairs.iterrows():
        p1, p2, count = row["player"], row["pass_recipient"], row["count"]
        if p1 in avg_loc.index and p2 in avg_loc.index:
            pitch.lines(
                avg_loc.loc[p1, "x"],
                avg_loc.loc[p1, "y"],
                avg_loc.loc[p2, "x"],
                avg_loc.loc[p2, "y"],
                lw=min(count * 0.8, 6.0),
                color="#f59e0b",
                alpha=0.6,
                ax=ax,
            )

    # Plot player nodes
    for player, row in avg_loc.iterrows():
        n_passes = pass_counts.get(player, 10)
        pitch.scatter(
            row["x"],
            row["y"],
            s=min(n_passes * 15 + 100, 600),
            c="#10b981",
            edgecolors="#ffffff",
            ax=ax,
            zorder=3,
        )
        # Short name label
        short_name = str(player).split()[-1]
        ax.text(
            row["x"],
            row["y"] + 2.5,
            short_name,
            color="white",
            fontsize=8,
            ha="center",
            va="center",
            fontweight="bold",
        )

    ax.set_title(f"{team_name} — Pas Ağı", color="white", fontsize=14, pad=12)
    plt.tight_layout()
    return fig


def render_action_heatmap(
    events: pd.DataFrame,
    team_name: str,
    *,
    pitch_color: str = "#101720",
    line_color: str = "#555555",
) -> plt.Figure | None:
    """Render a bivariate action density heatmap for offensive and defensive actions."""
    if not MPLSOCCER_AVAILABLE or events.empty:
        return None

    actions = events[events["team"] == team_name].copy()
    if actions.empty:
        return None

    if "location" in actions.columns:
        locs = actions["location"].dropna()
        actions["x"] = locs.apply(lambda l: l[0] if isinstance(l, (list, tuple)) and len(l) >= 2 else np.nan)
        actions["y"] = locs.apply(lambda l: l[1] if isinstance(l, (list, tuple)) and len(l) >= 2 else np.nan)

    actions = actions.dropna(subset=["x", "y"])
    if len(actions) < 20:
        return None

    pitch = Pitch(pitch_type="statsbomb", pitch_color=pitch_color, line_color=line_color)
    fig, ax = pitch.draw(figsize=(9, 6))

    try:
        pitch.kdeplot(
            actions["x"],
            actions["y"],
            ax=ax,
            cmap="magma",
            fill=True,
            n_levels=25,
            alpha=0.6,
        )
    except Exception as exc:
        LOGGER.warning("KDE plot generation failed: %s", exc)
        plt.close(fig)
        return None

    ax.set_title(f"{team_name} — Aksiyon Yoğunluk Haritası", color="white", fontsize=14, pad=12)
    plt.tight_layout()
    return fig


def render_player_radar(
    params: list[str],
    values: list[float],
    *,
    player_name: str = "Oyuncu",
    slice_colors: list[str] | None = None,
) -> plt.Figure | None:
    """Render a pizza radar chart using mplsoccer's PyPizza."""
    if not MPLSOCCER_AVAILABLE or not params or not values or len(params) != len(values):
        return None

    if slice_colors is None:
        slice_colors = ["#1a78cf"] * len(params)

    baker = PyPizza(
        params=params,
        background_color="#101720",
        straight_line_color="#555555",
        last_circle_color="#555555",
        other_circle_color="#333333",
        inner_circle_size=5,
    )

    fig, ax = baker.make_pizza(
        values,
        figsize=(7, 7),
        slice_colors=slice_colors,
        kwargs_slices=dict(edgecolor="#222222", zorder=2, linewidth=1),
        kwargs_params=dict(color="#e2e8f0", fontsize=10, va="center"),
        kwargs_values=dict(
            color="#ffffff",
            fontsize=9,
            zorder=3,
            bbox=dict(
                edgecolor="#222222",
                facecolor="#1e293b",
                boxstyle="round,pad=0.2",
                lw=1,
            ),
        ),
    )

    fig.text(
        0.515,
        0.97,
        player_name,
        size=15,
        ha="center",
        color="#ffffff",
        weight="bold",
    )
    plt.tight_layout()
    return fig
