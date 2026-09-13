from __future__ import annotations

import pandas as pd
import pytest

from app.components.metrics import (
    BinaryAccuracySummary,
    format_accuracy,
    summarize_binary_accuracy,
)


def test_binary_accuracy_ignores_missing_historical_evaluations() -> None:
    result = summarize_binary_accuracy(pd.Series([True, None, False, pd.NA]))

    assert result == BinaryAccuracySummary(
        correct=1,
        sample_size=2,
        accuracy=0.5,
    )
    assert format_accuracy(result) == "%50.0"


def test_binary_accuracy_reports_empty_sample_without_fabricating_rate() -> None:
    result = summarize_binary_accuracy(pd.Series([None, pd.NA, float("nan")]))

    assert result == BinaryAccuracySummary(
        correct=0,
        sample_size=0,
        accuracy=None,
    )
    assert format_accuracy(result) == "—"


def test_binary_accuracy_rejects_non_boolean_database_values() -> None:
    with pytest.raises(ValueError, match="must be booleans or null"):
        summarize_binary_accuracy(pd.Series([True, "false"]))
