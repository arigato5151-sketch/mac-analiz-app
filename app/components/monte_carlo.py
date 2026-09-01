"""Deterministic Monte Carlo helpers for displaying forecast uncertainty."""

from __future__ import annotations

import numpy as np
import pandas as pd


OUTCOME_COLUMNS = ("prob_home_win", "prob_draw", "prob_away_win")


def simulate_top_pick_accuracy(
    predictions: pd.DataFrame,
    *,
    simulations: int = 10_000,
    seed: int = 20260902,
) -> pd.DataFrame:
    """Simulate the accuracy distribution of each fixture's strongest 1-X-2 pick.

    This samples only from probabilities already produced by the model. It is an
    uncertainty view of the upcoming fixture set, not a claim about realised
    model performance.
    """
    if simulations < 1 or simulations > 100_000:
        raise ValueError("simulations must be between 1 and 100000")
    if predictions.empty:
        raise ValueError("at least one prediction is required")
    try:
        probabilities = predictions.loc[:, OUTCOME_COLUMNS].to_numpy(dtype=float)
    except KeyError as error:
        raise ValueError("Predictions must include all 1-X-2 probability columns") from error
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("Prediction probabilities must be finite and non-negative")
    totals = probabilities.sum(axis=1)
    if (totals <= 0).any() or not np.allclose(totals, 1.0, atol=1e-3):
        raise ValueError("Each 1-X-2 probability row must sum to one")

    probabilities = probabilities / totals[:, None]
    generator = np.random.default_rng(seed)
    draws = generator.random((simulations, len(probabilities)))
    sampled_outcomes = (draws[..., None] > np.cumsum(probabilities, axis=1)).sum(axis=2)
    top_picks = probabilities.argmax(axis=1)
    accuracies = (sampled_outcomes == top_picks).mean(axis=1)
    return pd.DataFrame({"Tahmini isabet": accuracies})
