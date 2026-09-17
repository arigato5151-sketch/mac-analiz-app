"""Unit tests for StatsBomb adapter and mplsoccer pitch visuals."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from app.components.pitch_visuals import (
    render_action_heatmap,
    render_pass_network,
    render_player_radar,
    render_shot_map,
)
from data_pipeline.statsbomb_adapter import StatsBombAdapter


def test_statsbomb_adapter_graceful_fallback_on_network_error(tmp_path):
    adapter = StatsBombAdapter(cache_dir=tmp_path)
    with patch("data_pipeline.statsbomb_adapter.sb.competitions", side_effect=Exception("Network error")):
        df = adapter.get_competitions()
        assert isinstance(df, pd.DataFrame)
        assert df.empty


def test_statsbomb_adapter_caching(tmp_path):
    adapter = StatsBombAdapter(cache_dir=tmp_path)
    mock_comps = pd.DataFrame([
        {"competition_id": 43, "season_id": 106, "competition_name": "FIFA World Cup"}
    ])
    with patch("data_pipeline.statsbomb_adapter.sb.competitions", return_value=mock_comps):
        df1 = adapter.get_competitions()
        assert len(df1) == 1
        assert (tmp_path / "competitions.parquet").exists()

    # Second read from cache without hitting sb.competitions
    with patch("data_pipeline.statsbomb_adapter.sb.competitions", side_effect=RuntimeError("Must not call")):
        df2 = adapter.get_competitions()
        assert len(df2) == 1
        assert df2.iloc[0]["competition_name"] == "FIFA World Cup"


def test_statsbomb_missing_match_returns_none(tmp_path):
    adapter = StatsBombAdapter(cache_dir=tmp_path)
    with patch("data_pipeline.statsbomb_adapter.sb.competitions", return_value=pd.DataFrame()):
        result = adapter.find_match_by_teams("NonExistent FC", "Fake United")
        assert result is None


def test_render_shot_map_with_synthetic_events():
    events = pd.DataFrame([
        {
            "type": "Shot",
            "team": "Team A",
            "location": [105.0, 40.0],
            "shot_statsbomb_xg": 0.45,
            "shot_outcome": "Goal",
        },
        {
            "type": "Shot",
            "team": "Team A",
            "location": [95.0, 35.0],
            "shot_statsbomb_xg": 0.15,
            "shot_outcome": "Saved",
        },
    ])
    fig = render_shot_map(events, team_name="Team A")
    assert fig is not None
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

    # Empty events returns None gracefully
    assert render_shot_map(pd.DataFrame()) is None


def test_render_pass_network_with_synthetic_events():
    events_list = []
    players = ["Player 1", "Player 2", "Player 3", "Player 4"]
    for i in range(25):
        p1, p2 = players[i % 4], players[(i + 1) % 4]
        events_list.append({
            "type": "Pass",
            "team": "Team A",
            "player": p1,
            "pass_recipient": p2,
            "location": [50.0 + (i % 5) * 5, 30.0 + (i % 4) * 10],
        })
    events = pd.DataFrame(events_list)
    fig = render_pass_network(events, team_name="Team A", min_passes=2)
    assert fig is not None
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

    # Insufficient events returns None gracefully
    assert render_pass_network(pd.DataFrame(), team_name="Team A") is None


def test_render_action_heatmap_with_synthetic_events():
    events_list = []
    rng = np.random.default_rng(42)
    for _ in range(40):
        events_list.append({
            "type": "Action",
            "team": "Team A",
            "location": [float(rng.uniform(10, 110)), float(rng.uniform(10, 70))],
        })
    events = pd.DataFrame(events_list)
    fig = render_action_heatmap(events, team_name="Team A")
    assert fig is not None
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_render_player_radar():
    params = ["Pas", "Şut", "Dribbling", "Defans", "Fizik"]
    values = [85.0, 70.0, 90.0, 45.0, 80.0]
    fig = render_player_radar(params, values, player_name="Test Oyuncu")
    assert fig is not None
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

    # Mismatched params and values returns None gracefully
    assert render_player_radar(params, values[:3]) is None

