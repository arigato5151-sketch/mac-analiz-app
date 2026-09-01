from __future__ import annotations

import pandas as pd
import pytest

from app.components.monte_carlo import simulate_top_pick_accuracy


def test_monte_carlo_returns_repeatable_bounded_accuracy_distribution() -> None:
    predictions = pd.DataFrame(
        [{"prob_home_win": 0.60, "prob_draw": 0.25, "prob_away_win": 0.15}]
    )

    first = simulate_top_pick_accuracy(predictions, simulations=2_000, seed=7)
    second = simulate_top_pick_accuracy(predictions, simulations=2_000, seed=7)

    assert first.equals(second)
    assert len(first) == 2_000
    assert first["Tahmini isabet"].between(0, 1).all()
    assert first["Tahmini isabet"].mean() == pytest.approx(0.60, abs=0.04)


def test_monte_carlo_rejects_invalid_probability_rows() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        simulate_top_pick_accuracy(
            pd.DataFrame([{"prob_home_win": 0.6, "prob_draw": 0.2, "prob_away_win": 0.1}])
        )
