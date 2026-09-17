"""
Tests for data_pipeline.soccerdata_adapter
==========================================
All network calls are mocked — no real HTTP requests are made.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from data_pipeline.soccerdata_adapter import (
    DEFAULT_CACHE_DIR,
    DEFAULT_MAX_RETRIES,
    SoccerdataAdapter,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_schedule_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-08-13", "2023-08-20"]),
            "home_team": ["Arsenal", "Chelsea"],
            "away_team": ["Nottm Forest", "Liverpool"],
            "home_score": [2, 1],
            "away_score": [1, 4],
        }
    )


def _make_standings_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "rank": [1, 2, 3],
            "team": ["Arsenal", "Man City", "Liverpool"],
            "points": [40, 38, 35],
        }
    )


@pytest.fixture()
def adapter(tmp_path: Path) -> SoccerdataAdapter:
    """Return an adapter configured with a temp cache dir and no_cache=True."""
    return SoccerdataAdapter(
        league="ENG-Premier League",
        season="2324",
        cache_dir=tmp_path / "soccerdata_cache",
        no_cache=True,  # skip disk cache in most tests
    )


# ---------------------------------------------------------------------------
# Construction / defaults
# ---------------------------------------------------------------------------


class TestAdapterConstruction:
    def test_default_league(self):
        a = SoccerdataAdapter()
        assert a.league == "ENG-Premier League"

    def test_default_season(self):
        a = SoccerdataAdapter()
        assert a.season == "2324"

    def test_default_cache_dir(self):
        a = SoccerdataAdapter()
        assert a.cache_dir == DEFAULT_CACHE_DIR

    def test_default_max_retries(self):
        a = SoccerdataAdapter()
        assert a.max_retries == DEFAULT_MAX_RETRIES

    def test_custom_league(self):
        a = SoccerdataAdapter(league="ESP-La Liga")
        assert a.league == "ESP-La Liga"

    def test_custom_no_cache(self, tmp_path):
        a = SoccerdataAdapter(cache_dir=tmp_path, no_cache=True)
        assert a.no_cache is True


# ---------------------------------------------------------------------------
# Graceful degradation — soccerdata unavailable
# ---------------------------------------------------------------------------


class TestGracefulDegradation:
    """When soccerdata is not installed all public methods return empty DataFrames."""

    def test_get_schedule_returns_empty_when_unavailable(self, adapter):
        with patch("data_pipeline.soccerdata_adapter.SOCCERDATA_AVAILABLE", False):
            df = adapter.get_schedule()
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_get_standings_returns_empty_when_unavailable(self, adapter):
        with patch("data_pipeline.soccerdata_adapter.SOCCERDATA_AVAILABLE", False):
            df = adapter.get_standings()
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_get_player_season_stats_empty_when_unavailable(self, adapter):
        with patch("data_pipeline.soccerdata_adapter.SOCCERDATA_AVAILABLE", False):
            df = adapter.get_player_season_stats()
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_get_team_season_stats_empty_when_unavailable(self, adapter):
        with patch("data_pipeline.soccerdata_adapter.SOCCERDATA_AVAILABLE", False):
            df = adapter.get_team_season_stats()
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_list_available_leagues_empty_when_unavailable(self, adapter):
        with patch("data_pipeline.soccerdata_adapter.SOCCERDATA_AVAILABLE", False):
            leagues = adapter.list_available_leagues()
        assert leagues == []


# ---------------------------------------------------------------------------
# Mocked network calls
# ---------------------------------------------------------------------------


class TestGetScheduleMocked:
    """Mock sd.FBref to avoid real network calls."""

    def _patch_fbref(self, adapter: SoccerdataAdapter, schedule_df: pd.DataFrame):
        mock_fbref = MagicMock()
        mock_fbref.read_schedule.return_value = schedule_df
        adapter._fbref = mock_fbref  # inject mock directly
        return mock_fbref

    def test_returns_dataframe(self, adapter):
        schedule = _make_schedule_df()
        with patch("data_pipeline.soccerdata_adapter.SOCCERDATA_AVAILABLE", True):
            self._patch_fbref(adapter, schedule)
            df = adapter.get_schedule()
        assert isinstance(df, pd.DataFrame)

    def test_has_expected_columns(self, adapter):
        schedule = _make_schedule_df()
        with patch("data_pipeline.soccerdata_adapter.SOCCERDATA_AVAILABLE", True):
            self._patch_fbref(adapter, schedule)
            df = adapter.get_schedule()
        for col in ["home_team", "away_team"]:
            assert col in df.columns

    def test_returns_correct_row_count(self, adapter):
        schedule = _make_schedule_df()
        with patch("data_pipeline.soccerdata_adapter.SOCCERDATA_AVAILABLE", True):
            self._patch_fbref(adapter, schedule)
            df = adapter.get_schedule()
        assert len(df) == 2


class TestGetStandingsMocked:
    def test_returns_dataframe(self, adapter):
        standings = _make_standings_df()
        mock_fbref = MagicMock()
        mock_fbref.read_league_table.return_value = standings
        adapter._fbref = mock_fbref
        with patch("data_pipeline.soccerdata_adapter.SOCCERDATA_AVAILABLE", True):
            df = adapter.get_standings()
        assert isinstance(df, pd.DataFrame)
        assert "team" in df.columns


# ---------------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------------


class TestRetryLogic:
    def test_retries_on_exception(self, adapter, tmp_path):
        """Fetch function fails twice then succeeds on 3rd attempt."""
        call_count = 0

        def flaky_fetch():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Network error")
            return _make_schedule_df()

        with patch("data_pipeline.soccerdata_adapter.SOCCERDATA_AVAILABLE", True):
            adapter.retry_delay = 0.0  # speed up test
            result = adapter._fetch_with_retry(flaky_fetch, "schedule_retry_test")

        assert call_count == 3
        assert not result.empty

    def test_returns_empty_after_max_retries(self, adapter):
        """All retries exhausted → empty DataFrame returned, no exception raised."""
        def always_fail():
            raise TimeoutError("Always times out")

        with patch("data_pipeline.soccerdata_adapter.SOCCERDATA_AVAILABLE", True):
            adapter.retry_delay = 0.0
            result = adapter._fetch_with_retry(always_fail, "always_fail")

        assert isinstance(result, pd.DataFrame)
        assert result.empty


# ---------------------------------------------------------------------------
# Disk cache
# ---------------------------------------------------------------------------


class TestDiskCache:
    def test_cache_write_and_read(self, tmp_path):
        adapter = SoccerdataAdapter(
            league="ENG-Premier League",
            season="2324",
            cache_dir=tmp_path / "cache",
            no_cache=False,
        )
        schedule = _make_schedule_df()
        adapter._save_cache(schedule, "schedule")
        loaded = adapter._load_cache("schedule")
        assert loaded is not None
        assert len(loaded) == len(schedule)

    def test_no_cache_skips_disk(self, tmp_path):
        adapter = SoccerdataAdapter(
            cache_dir=tmp_path / "cache",
            no_cache=True,
        )
        schedule = _make_schedule_df()
        adapter._save_cache(schedule, "schedule")
        loaded = adapter._load_cache("schedule")
        assert loaded is None

    def test_clear_cache(self, tmp_path):
        adapter = SoccerdataAdapter(
            league="ENG-Premier League",
            season="2324",
            cache_dir=tmp_path / "cache",
            no_cache=False,
        )
        adapter._save_cache(_make_schedule_df(), "schedule")
        adapter._save_cache(_make_standings_df(), "standings")
        deleted = adapter.clear_cache()
        assert deleted == 2
        assert adapter._load_cache("schedule") is None


# ---------------------------------------------------------------------------
# find_match
# ---------------------------------------------------------------------------


class TestFindMatch:
    def test_finds_existing_match(self, adapter):
        schedule = _make_schedule_df()
        with patch.object(adapter, "get_schedule", return_value=schedule):
            result = adapter.find_match("Arsenal", "Nottm")
        assert result is not None
        assert "Arsenal" in result["home_team"]

    def test_returns_none_when_not_found(self, adapter):
        schedule = _make_schedule_df()
        with patch.object(adapter, "get_schedule", return_value=schedule):
            result = adapter.find_match("Barcelona", "Real Madrid")
        assert result is None

    def test_returns_none_on_empty_schedule(self, adapter):
        with patch.object(adapter, "get_schedule", return_value=pd.DataFrame()):
            result = adapter.find_match("Arsenal", "Chelsea")
        assert result is None

