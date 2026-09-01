"""Pure data builders for the match-detail comparison charts."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from models.feature_engineering import TeamState


RADAR_METRICS = ("Form", "Hücum", "Savunma", "Elo", "Dinlenme")


def _clamp_percent(value: float) -> float:
    return float(np.clip(value * 100, 0, 100))


def _average(values: list[float], default: float) -> float:
    return float(np.mean(values)) if values else default


def _team_radar_values(team: TeamState, *, match_at: datetime) -> dict[str, float]:
    """Normalize comparable team state measures to a transparent 0–100 scale."""
    recent = list(team.results)[-5:]
    goals_for = _average([float(result.goals_for) for result in recent], 1.25)
    goals_against = _average([float(result.goals_against) for result in recent], 1.25)
    form = _average([float(result.points) / 3 for result in recent], 1 / 3)
    if team.last_match_at is None:
        rest_days = 7.0
    else:
        rest_days = max(0.0, (match_at - team.last_match_at).total_seconds() / 86_400)
    return {
        "Form": _clamp_percent(form),
        "Hücum": _clamp_percent(goals_for / 3),
        "Savunma": _clamp_percent(1 - goals_against / 3),
        "Elo": _clamp_percent((team.elo - 1_200) / 600),
        "Dinlenme": _clamp_percent(rest_days / 7),
    }


def build_radar_comparison(
    home: TeamState, away: TeamState, *, match_at: datetime
) -> pd.DataFrame:
    """Return one normalized radar row per team without presentation dependencies."""
    home_values = _team_radar_values(home, match_at=match_at)
    away_values = _team_radar_values(away, match_at=match_at)
    return pd.DataFrame(
        {
            "Metrik": RADAR_METRICS,
            "Ev": [home_values[metric] for metric in RADAR_METRICS],
            "Deplasman": [away_values[metric] for metric in RADAR_METRICS],
        }
    )


def build_form_comparison(home: TeamState, away: TeamState) -> pd.DataFrame:
    """Align each team's last five actual match points from oldest to newest."""
    rows: list[dict[str, Any]] = []
    for label, team in (("Ev", home), ("Deplasman", away)):
        points = [result.points for result in list(team.results)[-5:]]
        padded = [None] * (5 - len(points)) + points
        for index, value in enumerate(padded, start=1):
            rows.append({"Takım": label, "Maç": f"M-{6 - index}", "Puan": value})
    return pd.DataFrame(rows)
