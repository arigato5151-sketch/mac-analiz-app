"""Unit tests for StatsBomb adapter with dependency injection and mplsoccer visuals."""

from __future__ import annotations

from pathlib import Path
from typing import Any

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


class FakeStatsBombProvider:
    """Mock StatsBomb provider to avoid real network calls and test DI."""

    def __init__(
        self,
        comps: pd.DataFrame | None = None,
        matches: pd.DataFrame | None = None,
        events: pd.DataFrame | None = None,
        raise_on_comps: bool = False,
        raise_on_events: bool = False,
    ) -> None:
        self._comps = (
            comps
            if comps is not None
            else pd.DataFrame([
                {
                    "competition_id": 43,
                    "season_id": 106,
                    "competition_name": "FIFA World Cup",
                    "season_name": "2022",
                }
            ])
        )
        self._matches = (
            matches
            if matches is not None
            else pd.DataFrame([
                {
                    "match_id": 3869685,
                    "match_date": "2022-12-18",
                    "home_team": "Argentina",
                    "away_team": "France",
                }
            ])
        )
        self._events = (
            events
            if events is not None
            else pd.DataFrame([
                {
                    "type": "Shot",
                    "team": "Argentina",
                    "location": [105.0, 40.0],
                    "shot_statsbomb_xg": 0.45,
                    "shot_outcome": "Goal",
                }
            ])
        )
        self.raise_on_comps = raise_on_comps
        self.raise_on_events = raise_on_events

    def competitions(self) -> pd.DataFrame:
        if self.raise_on_comps:
            raise ConnectionError("Simulated network error on competitions")
        return self._comps

    def matches(self, competition_id: int, season_id: int) -> pd.DataFrame:
        return self._matches

    def events(self, match_id: int) -> pd.DataFrame:
        if self.raise_on_events:
            raise ConnectionError("Simulated network error on events")
        return self._events


def test_statsbomb_adapter_package_not_installed(tmp_path: Path):
    """When no provider is given and statsbombpy is not available, returns empty results gracefully."""
    adapter = StatsBombAdapter(cache_dir=tmp_path)
    adapter._provider = None
    adapter._available = False

    assert adapter.is_available is False
    assert adapter.get_competitions().empty
    assert adapter.get_matches(43, 106).empty
    assert adapter.get_events(12345).empty
    assert adapter.find_match_by_teams("Argentina", "France") is None


def test_statsbomb_adapter_network_error_fallback(tmp_path: Path):
    """When the provider raises a network error, empty DataFrame is returned without crashing."""
    failing_provider = FakeStatsBombProvider(raise_on_comps=True)
    adapter = StatsBombAdapter(cache_dir=tmp_path, provider=failing_provider)

    df = adapter.get_competitions()
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_statsbomb_adapter_cache_miss_then_hit(tmp_path: Path):
    """Cache miss fetches from provider; subsequent call reads from disk cache even if provider fails."""
    provider = FakeStatsBombProvider()
    adapter = StatsBombAdapter(cache_dir=tmp_path, provider=provider)

    # Cache miss -> reads from provider, writes parquet
    df1 = adapter.get_competitions()
    assert len(df1) == 1
    assert (tmp_path / "competitions.parquet").exists()

    # Cache hit -> now provider fails, but adapter reads from parquet
    failing_provider = FakeStatsBombProvider(raise_on_comps=True)
    adapter_cached = StatsBombAdapter(cache_dir=tmp_path, provider=failing_provider)
    df2 = adapter_cached.get_competitions()
    assert len(df2) == 1
    assert df2.iloc[0]["competition_name"] == "FIFA World Cup"


def test_statsbomb_adapter_match_found_with_normalization(tmp_path: Path):
    """Fuzzy / token-based matching correctly matches names with different casing or accents."""
    provider = FakeStatsBombProvider()
    adapter = StatsBombAdapter(cache_dir=tmp_path, provider=provider)

    match = adapter.find_match_by_teams("argentina", "FRANCE", year=2022)
    assert match is not None
    assert int(match["match_id"]) == 3869685


def test_statsbomb_adapter_match_not_found(tmp_path: Path):
    """Non-existent team returns None without error."""
    provider = FakeStatsBombProvider()
    adapter = StatsBombAdapter(cache_dir=tmp_path, provider=provider)

    result = adapter.find_match_by_teams("Arsenal FC", "Chelsea FC")
    assert result is None


def test_statsbomb_adapter_empty_events(tmp_path: Path):
    """Empty events returned gracefully."""
    empty_provider = FakeStatsBombProvider(events=pd.DataFrame())
    adapter = StatsBombAdapter(cache_dir=tmp_path, provider=empty_provider)

    events = adapter.get_events(3869685)
    assert isinstance(events, pd.DataFrame)
    assert events.empty


# ---------------------------------------------------------------------------
# Visuals tests (mplsoccer)
# ---------------------------------------------------------------------------

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

    assert render_player_radar(params, values[:3]) is None


# ---------------------------------------------------------------------------
# StatsBomb timeout non-blocking regression test
# ---------------------------------------------------------------------------

def test_call_with_timeout_does_not_block_ui_thread(tmp_path: Path):
    """Timeout must not block the calling thread beyond timeout_seconds + small margin.

    The test simulates a hung network call (sleep 60 s) and verifies that the
    adapter returns within 2 seconds — proving the UI / Streamlit thread is
    never held waiting for the background executor to terminate.
    """
    import time

    class SlowProvider:
        def competitions(self) -> pd.DataFrame:
            time.sleep(5.0)  # simulates a slow network call
            return pd.DataFrame()

    adapter = StatsBombAdapter(
        cache_dir=tmp_path,
        provider=SlowProvider(),
        timeout_seconds=0.5,
    )
    start = time.perf_counter()
    result = adapter.get_competitions()
    elapsed = time.perf_counter() - start

    assert isinstance(result, pd.DataFrame)
    assert result.empty, "Should return empty DataFrame on timeout"
    assert elapsed < 2.5, (
        f"UI thread was blocked for {elapsed:.2f}s, expected < 2.5s. "
        "Check that executor.shutdown(wait=False) is used on timeout/error paths."
    )
