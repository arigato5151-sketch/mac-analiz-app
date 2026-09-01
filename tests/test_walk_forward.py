from __future__ import annotations

import pandas as pd

from models.feature_engineering import FEATURE_COLUMNS
from models.train_model import walk_forward_report


def test_walk_forward_skips_insufficient_history() -> None:
    features = pd.DataFrame([{column: 0.0 for column in FEATURE_COLUMNS}] * 10)
    labels = pd.DataFrame(
        {"result": [0] * 10, "over_2_5": [0] * 10, "btts": [0] * 10,
         "match_date": pd.date_range("2026-01-01", periods=10, tz="UTC")}
    )

    assert walk_forward_report(features, labels) == []
