from __future__ import annotations

import pytest

from evaluation.track_performance import actual_result, build_performance_row


@pytest.mark.parametrize(
    ("home", "away", "expected"),
    [(2, 0, "home_win"), (1, 1, "draw"), (0, 3, "away_win")],
)
def test_actual_result(home: int, away: int, expected: str) -> None:
    assert actual_result(home, away) == expected


def test_build_performance_row_calculates_multiclass_brier_score() -> None:
    prediction = {
        "id": 9,
        "prob_home_win": 0.60,
        "prob_draw": 0.25,
        "prob_away_win": 0.15,
    }
    match = {"id": 100, "home_score": 2, "away_score": 1}

    row = build_performance_row(
        prediction, match, evaluated_at="2026-08-26T12:00:00+00:00"
    )

    assert row["actual_result"] == "home_win"
    assert row["was_correct"] is True
    assert row["brier_score"] == pytest.approx(0.245)


def test_build_performance_row_rejects_invalid_probability_sum() -> None:
    prediction = {
        "id": 9,
        "prob_home_win": 0.8,
        "prob_draw": 0.3,
        "prob_away_win": 0.1,
    }

    with pytest.raises(ValueError, match="sum to one"):
        build_performance_row(
            prediction,
            {"id": 100, "home_score": 0, "away_score": 0},
            evaluated_at="2026-08-26T12:00:00+00:00",
        )
