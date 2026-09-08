from __future__ import annotations

import numpy as np
import pytest

from models.feature_engineering import FEATURE_COLUMNS
from models.predict import generate_prediction_rows


class FixedMulticlassModel:
    def __init__(self, expected_columns: list[str]) -> None:
        self.expected_columns = expected_columns

    def predict_proba(self, features):
        assert list(features.columns) == self.expected_columns
        return np.tile([0.6, 0.25, 0.15], (len(features), 1))


class FixedBinaryModel:
    def __init__(self, expected_columns: list[str], probability: float) -> None:
        self.expected_columns = expected_columns
        self.probability = probability

    def predict_proba(self, features):
        assert list(features.columns) == self.expected_columns
        positive = np.full(len(features), self.probability)
        return np.column_stack((1.0 - positive, positive))


def test_legacy_model_contract_remains_usable_after_feature_expansion() -> None:
    new_columns = {
        "home_xg_diff_5",
        "away_xg_diff_5",
        "home_venue_xg_for_5",
        "away_venue_xg_for_5",
        "market_home_move",
        "market_draw_move",
        "market_away_move",
        "market_over_2_5_move",
        "market_btts_move",
    }
    legacy_columns = [column for column in FEATURE_COLUMNS if column not in new_columns]
    bundle = {
        "feature_columns": legacy_columns,
        "result_model": FixedMulticlassModel(legacy_columns),
        "over_2_5_model": FixedBinaryModel(legacy_columns, 0.55),
        "btts_model": FixedBinaryModel(legacy_columns, 0.45),
        "calibration": {},
    }
    history = [{
        "id": 1,
        "league_id": 39,
        "home_team_id": 1,
        "away_team_id": 2,
        "match_date": "2026-01-01T12:00:00+00:00",
        "home_score": 2,
        "away_score": 0,
    }]
    upcoming = [{
        "id": 2,
        "league_id": 39,
        "home_team_id": 1,
        "away_team_id": 2,
        "match_date": "2026-01-08T12:00:00+00:00",
        "home_score": None,
        "away_score": None,
    }]

    row = generate_prediction_rows(
        bundle, history, upcoming, model_version="legacy-v1"
    )[0]

    assert row["prob_home_win"] == pytest.approx(0.6)
    assert row["prob_over_2_5"] == pytest.approx(0.55)
    assert row["prob_btts"] == pytest.approx(0.45)
