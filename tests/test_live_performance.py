from __future__ import annotations

import pytest

from app.components.live_performance import summarize_live_performance, wilson_interval


def test_wilson_interval_contains_observed_accuracy() -> None:
    lower, upper = wilson_interval(successes=18, total=30)

    assert lower < 0.6 < upper
    assert 0 < lower < upper < 1


def test_live_summary_marks_small_samples_as_insufficient() -> None:
    summary = summarize_live_performance(
        [True, False, True], [0.2, 0.9, 0.3], reference_brier_score=0.4
    )

    assert summary.sample_size == 3
    assert summary.accuracy == pytest.approx(2 / 3)
    assert summary.status == "Yetersiz örneklem"


def test_live_summary_flags_brier_degradation_after_sufficient_sample() -> None:
    summary = summarize_live_performance(
        [True] * 40 + [False] * 10,
        [0.70] * 50,
        reference_brier_score=0.50,
    )

    assert summary.status == "İzlenmeli"


def test_live_summary_rejects_mismatched_series() -> None:
    with pytest.raises(ValueError, match="same length"):
        summarize_live_performance([True], [0.1, 0.2])
