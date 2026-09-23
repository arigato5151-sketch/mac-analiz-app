from __future__ import annotations

import pandas as pd
import pytest

from app.components.live_performance import build_performance_breakdowns


def test_breakdowns_report_league_and_confidence_coverage() -> None:
    frame = pd.DataFrame(
        [
            {
                "league_name": "Lig A",
                "was_correct": True,
                "brier_score": 0.10,
                "prob_home_win": 0.70,
                "prob_draw": 0.20,
                "prob_away_win": 0.10,
            },
            {
                "league_name": "Lig A",
                "was_correct": False,
                "brier_score": 0.60,
                "prob_home_win": 0.55,
                "prob_draw": 0.30,
                "prob_away_win": 0.15,
            },
            {
                "league_name": "Lig B",
                "was_correct": True,
                "brier_score": 0.20,
                "prob_home_win": 0.91,
                "prob_draw": 0.05,
                "prob_away_win": 0.04,
            },
        ]
    )

    result = build_performance_breakdowns(frame)

    assert result["league"][0]["Grup"] == "Lig A"
    assert result["league"][0]["Örneklem"] == 2
    assert result["league"][0]["Kapsama"] == pytest.approx(2 / 3)
    assert [row["Grup"] for row in result["confidence"]] == [
        "%50–%60",
        "%70–%80",
        "%90–%100",
    ]


def test_breakdowns_reject_missing_columns() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        build_performance_breakdowns(pd.DataFrame())


def test_breakdowns_keep_confidence_when_league_is_unavailable() -> None:
    frame = pd.DataFrame(
        [
            {
                "was_correct": True,
                "brier_score": 0.10,
                "prob_home_win": 0.70,
                "prob_draw": 0.20,
                "prob_away_win": 0.10,
            }
        ]
    )

    result = build_performance_breakdowns(frame)

    assert result["league"] == []
    assert result["confidence"][0]["Örneklem"] == 1
