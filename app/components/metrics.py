"""Pure helpers for user-facing prediction performance summaries."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class BinaryAccuracySummary:
    """Accuracy and effective sample size for an evaluated binary market."""

    correct: int
    sample_size: int
    accuracy: float | None


def summarize_binary_accuracy(values: pd.Series) -> BinaryAccuracySummary:
    """Ignore missing evaluations and summarize genuine boolean outcomes."""
    evaluated = values.dropna()
    if evaluated.empty:
        return BinaryAccuracySummary(correct=0, sample_size=0, accuracy=None)

    invalid = evaluated.map(lambda value: not isinstance(value, (bool, np.bool_)))
    if invalid.any():
        raise ValueError("Binary accuracy values must be booleans or null")

    correct = int(evaluated.astype(bool).sum())
    sample_size = int(len(evaluated))
    return BinaryAccuracySummary(
        correct=correct,
        sample_size=sample_size,
        accuracy=correct / sample_size,
    )


def format_accuracy(summary: BinaryAccuracySummary) -> str:
    """Format an accuracy ratio for Streamlit without inventing missing data."""
    return f"%{summary.accuracy * 100:.1f}" if summary.accuracy is not None else "—"
