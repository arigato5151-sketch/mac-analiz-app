"""
Tests for data_pipeline.spadl_converter and models.spadl_analytics
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data_pipeline.spadl_converter import (
    SPADL_COLUMNS,
    _empty_spadl,
    compute_pass_statistics,
    compute_shot_statistics,
    convert_statsbomb_events_to_spadl,
    get_spadl_action_types,
    get_spadl_result_types,
    load_spadl_from_parquet,
    save_spadl_to_parquet,
)
from models.spadl_analytics import (
    XT_GRID_12x8,
    _coords_to_xt,
    aggregate_player_ratings,
    compute_xt_for_actions,
    compute_zone_threat_map,
    get_xt_grid,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_synthetic_actions(n: int = 20) -> pd.DataFrame:
    """Create a minimal synthetic SPADL DataFrame."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "game_id": 1,
            "period_id": rng.integers(1, 3, size=n),
            "time_seconds": rng.integers(0, 5400, size=n),
            "player_id": rng.integers(1, 12, size=n),
            "team_id": rng.choice([1, 2], size=n),
            "start_x": rng.uniform(0, 105, size=n),
            "start_y": rng.uniform(0, 68, size=n),
            "end_x": rng.uniform(0, 105, size=n),
            "end_y": rng.uniform(0, 68, size=n),
            "result_id": rng.integers(0, 2, size=n),
            "bodypart_id": rng.integers(0, 3, size=n),
            "type_id": rng.integers(0, 22, size=n),
        }
    )
    return df


# ---------------------------------------------------------------------------
# spadl_converter tests
# ---------------------------------------------------------------------------


class TestEmptySpadl:
    def test_returns_dataframe(self):
        df = _empty_spadl()
        assert isinstance(df, pd.DataFrame)

    def test_has_expected_columns(self):
        df = _empty_spadl()
        for col in SPADL_COLUMNS:
            assert col in df.columns, f"Missing column: {col}"

    def test_is_empty(self):
        df = _empty_spadl()
        assert len(df) == 0


class TestConvertStatsbombEvents:
    """convert_statsbomb_events_to_spadl — graceful fallback behaviour."""

    def test_empty_events_returns_empty(self):
        result = convert_statsbomb_events_to_spadl(pd.DataFrame())
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_none_events_returns_empty(self):
        result = convert_statsbomb_events_to_spadl(None)  # type: ignore[arg-type]
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_returns_dataframe_regardless_of_availability(self):
        """Function must never raise — always returns a DataFrame."""
        events = pd.DataFrame({"id": [1], "type": [{"id": 30, "name": "Pass"}]})
        result = convert_statsbomb_events_to_spadl(events, game_id=99)
        assert isinstance(result, pd.DataFrame)


class TestActionTypes:
    def test_get_action_types_returns_dict(self):
        at = get_spadl_action_types()
        assert isinstance(at, dict)
        assert len(at) >= 10

    def test_action_types_values_are_strings(self):
        for k, v in get_spadl_action_types().items():
            assert isinstance(k, int), f"Key not int: {k}"
            assert isinstance(v, str), f"Value not str: {v}"

    def test_shot_in_action_types(self):
        types = get_spadl_action_types()
        assert any("shot" in v for v in types.values())

    def test_get_result_types_returns_dict(self):
        rt = get_spadl_result_types()
        assert isinstance(rt, dict)
        assert len(rt) >= 4

    def test_success_in_result_types(self):
        rt = get_spadl_result_types()
        assert any("success" in v for v in rt.values())


class TestShotAndPassStatistics:
    def test_shot_stats_empty_df(self):
        stats = compute_shot_statistics(pd.DataFrame())
        assert stats["total_shots"] == 0
        assert stats["shot_conversion_rate"] == 0.0

    def test_pass_stats_empty_df(self):
        stats = compute_pass_statistics(pd.DataFrame())
        assert stats["total_passes"] == 0
        assert stats["pass_accuracy"] == 0.0

    def test_shot_stats_with_synthetic_data(self):
        actions = _make_synthetic_actions(50)
        stats = compute_shot_statistics(actions)
        assert "total_shots" in stats
        assert 0 <= stats["shot_conversion_rate"] <= 1.0

    def test_pass_stats_with_synthetic_data(self):
        actions = _make_synthetic_actions(50)
        stats = compute_pass_statistics(actions)
        assert "total_passes" in stats
        assert 0 <= stats["pass_accuracy"] <= 1.0

    def test_shot_stats_conversion_rate_zero_when_no_shots(self):
        actions = _make_synthetic_actions(10)
        # Force type_id to non-shot values (0 = pass in fallback types)
        actions["type_id"] = 0
        stats = compute_shot_statistics(actions)
        assert stats["total_shots"] == 0
        assert stats["shot_conversion_rate"] == 0.0


class TestParquetCacheRoundtrip:
    def test_save_and_load(self, tmp_path):
        actions = _make_synthetic_actions(15)
        path = tmp_path / "test_spadl.parquet"
        save_spadl_to_parquet(actions, path)
        loaded = load_spadl_from_parquet(path)
        assert len(loaded) == len(actions)
        assert set(SPADL_COLUMNS).issubset(loaded.columns)

    def test_load_nonexistent_returns_empty(self, tmp_path):
        path = tmp_path / "nonexistent.parquet"
        result = load_spadl_from_parquet(path)
        assert result.empty

    def test_save_empty_does_not_create_file(self, tmp_path):
        path = tmp_path / "empty.parquet"
        save_spadl_to_parquet(pd.DataFrame(), path)
        assert not path.exists()


# ---------------------------------------------------------------------------
# spadl_analytics tests
# ---------------------------------------------------------------------------


class TestXTGrid:
    def test_grid_shape(self):
        grid = get_xt_grid()
        assert grid.shape == (12, 8)

    def test_grid_values_in_range(self):
        grid = get_xt_grid()
        assert (grid >= 0).all()
        assert (grid <= 1).all()

    def test_higher_xt_near_goal(self):
        """Zone close to opponent goal (high x) should have higher xT."""
        near_goal = _coords_to_xt(100.0, 34.0)
        near_own_goal = _coords_to_xt(5.0, 34.0)
        assert near_goal > near_own_goal

    def test_xt_grid_is_copy(self):
        """Mutating the returned grid should not affect the module-level constant."""
        grid1 = get_xt_grid()
        grid2 = get_xt_grid()
        grid1[0, 0] = 999.0
        assert grid2[0, 0] != 999.0

    def test_xt_grid_constant_unchanged(self):
        assert XT_GRID_12x8[0, 0] == pytest.approx(0.00638, rel=1e-3)


class TestCoordsToXT:
    def test_midfield_centre(self):
        val = _coords_to_xt(52.5, 34.0)
        assert 0 < val < 0.2

    def test_penalty_box_centre(self):
        val = _coords_to_xt(100.0, 34.0)
        assert val > 0.3

    def test_own_half(self):
        val = _coords_to_xt(5.0, 34.0)
        assert val < 0.05

    def test_boundary_x_max(self):
        val = _coords_to_xt(105.0, 34.0)
        assert isinstance(val, float)

    def test_boundary_y_max(self):
        val = _coords_to_xt(52.5, 68.0)
        assert isinstance(val, float)


class TestComputeXTForActions:
    def test_adds_xt_gain_column(self):
        actions = _make_synthetic_actions(10)
        result = compute_xt_for_actions(actions)
        assert "xt_gain" in result.columns

    def test_no_xt_start_or_end_columns_in_output(self):
        actions = _make_synthetic_actions(10)
        result = compute_xt_for_actions(actions)
        assert "xt_start" not in result.columns
        assert "xt_end" not in result.columns

    def test_xt_gain_dtype_float(self):
        actions = _make_synthetic_actions(10)
        result = compute_xt_for_actions(actions)
        assert result["xt_gain"].dtype == np.float64

    def test_empty_df_returns_xt_gain_zero(self):
        result = compute_xt_for_actions(pd.DataFrame())
        assert "xt_gain" in result.columns
        assert len(result) == 0

    def test_missing_coords_returns_xt_gain_zero(self):
        df = pd.DataFrame({"type_id": [0, 1]})
        result = compute_xt_for_actions(df)
        assert (result["xt_gain"] == 0.0).all()

    def test_xt_gain_for_goal_scoring_position(self):
        """Action ending in front of goal should have positive xT gain."""
        df = pd.DataFrame(
            {
                "start_x": [10.0],
                "start_y": [34.0],
                "end_x": [100.0],
                "end_y": [34.0],
            }
        )
        result = compute_xt_for_actions(df)
        assert result["xt_gain"].iloc[0] > 0


class TestAggregatePlayerRatings:
    def test_returns_dataframe(self):
        actions = _make_synthetic_actions(30)
        actions_with_xt = compute_xt_for_actions(actions)
        ratings = aggregate_player_ratings(actions_with_xt)
        assert isinstance(ratings, pd.DataFrame)

    def test_expected_columns(self):
        actions = _make_synthetic_actions(30)
        actions_with_xt = compute_xt_for_actions(actions)
        ratings = aggregate_player_ratings(actions_with_xt)
        for col in ["player_id", "team_id", "total_xt", "action_count"]:
            assert col in ratings.columns, f"Missing column: {col}"

    def test_sorted_by_total_xt_desc(self):
        actions = _make_synthetic_actions(50)
        actions_with_xt = compute_xt_for_actions(actions)
        ratings = aggregate_player_ratings(actions_with_xt)
        if len(ratings) > 1:
            assert ratings["total_xt"].iloc[0] >= ratings["total_xt"].iloc[-1]

    def test_empty_returns_empty(self):
        ratings = aggregate_player_ratings(pd.DataFrame())
        assert ratings.empty

    def test_missing_xt_gain_column_returns_empty(self):
        df = _make_synthetic_actions(10)
        # Remove xt_gain column
        ratings = aggregate_player_ratings(df)
        assert ratings.empty


class TestZoneThreatMap:
    def test_shape(self):
        actions = _make_synthetic_actions(20)
        actions_with_xt = compute_xt_for_actions(actions)
        zone_map = compute_zone_threat_map(actions_with_xt)
        assert zone_map.shape == (12, 8)

    def test_empty_returns_zeros(self):
        zone_map = compute_zone_threat_map(pd.DataFrame())
        assert zone_map.shape == (12, 8)
        assert (zone_map == 0).all()

    def test_values_are_floats(self):
        actions = _make_synthetic_actions(10)
        actions_with_xt = compute_xt_for_actions(actions)
        zone_map = compute_zone_threat_map(actions_with_xt)
        assert zone_map.dtype == np.float64

