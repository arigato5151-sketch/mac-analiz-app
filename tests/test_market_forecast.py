from __future__ import annotations

import pytest

from models.market_forecast import (
    derive_market_probabilities,
    evaluate_market_probabilities,
    format_market_summary,
)
from models.poisson_model import predict_score_probabilities


def test_derived_markets_are_normalized_and_return_three_scores() -> None:
    markets = derive_market_probabilities(
        (0.55, 0.25, 0.20), predict_score_probabilities(1.7, 0.8)
    )

    assert markets["double_chance"]["1X"] == pytest.approx(0.80)
    assert markets["total_goals"]["over_1_5"] + markets["total_goals"]["under_1_5"] == pytest.approx(1.0)
    assert markets["total_goals"]["over_3_5"] + markets["total_goals"]["under_3_5"] == pytest.approx(1.0)
    assert len(markets["correct_scores"]) == 3
    assert markets["correct_scores"][0]["probability"] >= markets["correct_scores"][1]["probability"]


def test_summary_suppresses_weak_markets_but_keeps_score_scenarios() -> None:
    markets = derive_market_probabilities(
        (0.38, 0.31, 0.31), predict_score_probabilities(1.1, 1.0)
    )

    lines = format_market_summary(markets)

    assert not any(line.startswith("Çifte şans:") for line in lines)
    assert any(line.startswith("Olası skorlar:") for line in lines)


def test_market_evaluation_scores_binary_markets_and_top_three() -> None:
    markets = derive_market_probabilities(
        (0.55, 0.25, 0.20), predict_score_probabilities(1.7, 0.8)
    )

    evaluation = evaluate_market_probabilities(markets, home_score=1, away_score=0)

    assert evaluation["double_chance"]["1X"]["actual"] is True
    assert evaluation["total_goals"]["under_1_5"]["actual"] is True
    assert evaluation["team_goals"]["away_over_0_5"]["actual"] is False
    assert 0.0 <= evaluation["double_chance"]["1X"]["brier_score"] <= 1.0
    assert evaluation["correct_score"]["actual"] == "1-0"


def test_derived_markets_reject_invalid_result_probabilities() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        derive_market_probabilities(
            (0.7, 0.3, 0.2), predict_score_probabilities(1.0, 1.0)
        )
