from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.components.match_visuals import build_form_comparison, build_radar_comparison
from models.feature_engineering import MatchResult, TeamState


def _team(*, elo: float, results: list[MatchResult]) -> TeamState:
    state = TeamState(elo=elo)
    state.results.extend(results)
    return state


def test_radar_comparison_returns_bounded_normalized_metrics() -> None:
    home = _team(elo=1650, results=[MatchResult(2, 0, 3, True)])
    away = _team(elo=1450, results=[MatchResult(0, 2, 0, False)])

    frame = build_radar_comparison(
        home, away, match_at=datetime(2026, 9, 3, 18, tzinfo=timezone.utc)
    )

    assert frame["Metrik"].tolist() == ["Form", "Hücum", "Savunma", "Elo", "Dinlenme"]
    assert frame[["Ev", "Deplasman"]].to_numpy().min() >= 0
    assert frame[["Ev", "Deplasman"]].to_numpy().max() <= 100
    assert frame.loc[0, "Ev"] > frame.loc[0, "Deplasman"]


def test_form_comparison_aligns_recent_points_with_missing_history() -> None:
    home = _team(elo=1500, results=[MatchResult(1, 0, 3, True)])
    away = _team(elo=1500, results=[MatchResult(1, 1, 1, False), MatchResult(0, 2, 0, True)])

    frame = build_form_comparison(home, away)

    assert len(frame) == 10
    assert frame.loc[(frame["Takım"] == "Ev") & (frame["Maç"] == "M-1"), "Puan"].iloc[0] == 3
    assert frame.loc[(frame["Takım"] == "Deplasman") & (frame["Maç"] == "M-1"), "Puan"].iloc[0] == 0
