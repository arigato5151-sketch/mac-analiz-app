"""Derive auditable football markets from calibrated results and a score matrix."""

from __future__ import annotations

from collections.abc import Sequence
from math import exp
from math import isfinite
from typing import Any

import numpy as np

from models.poisson_model import PoissonPrediction

MINIMUM_DOUBLE_CHANCE_CONFIDENCE = 0.70
MINIMUM_GOAL_MARKET_CONFIDENCE = 0.65


def _probability(value: float, *, label: str) -> float:
    parsed = float(value)
    if not isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{label} must be a finite probability")
    return parsed


def _matrix_probability(matrix: np.ndarray, mask: np.ndarray) -> float:
    return float(np.asarray(matrix, dtype=float)[mask].sum())


def derive_market_probabilities(
    result_probabilities: Sequence[float],
    poisson: PoissonPrediction,
) -> dict[str, Any]:
    """Build JSON-safe secondary markets from one immutable score forecast."""
    if len(result_probabilities) != 3:
        raise ValueError("Result probabilities must contain home, draw and away")
    home, draw, away = (
        _probability(value, label="result probability")
        for value in result_probabilities
    )
    if not np.isclose(home + draw + away, 1.0, atol=1e-6):
        raise ValueError("Result probabilities must sum to one")

    matrix = np.asarray(poisson.score_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.size == 0 or not np.isclose(matrix.sum(), 1.0):
        raise ValueError("Poisson score matrix must be normalized")
    home_goals, away_goals = np.indices(matrix.shape)
    totals = home_goals + away_goals

    over_1_5 = _matrix_probability(matrix, totals >= 2)
    over_3_5 = _matrix_probability(matrix, totals >= 4)
    home_over_0_5 = _matrix_probability(matrix, home_goals >= 1)
    away_over_0_5 = _matrix_probability(matrix, away_goals >= 1)
    home_over_1_5 = _matrix_probability(matrix, home_goals >= 2)
    away_over_1_5 = _matrix_probability(matrix, away_goals >= 2)

    ranked_scores = sorted(
        (
            (float(matrix[h, a]), int(h), int(a))
            for h in range(matrix.shape[0])
            for a in range(matrix.shape[1])
        ),
        key=lambda item: (-item[0], item[1], item[2]),
    )[:3]

    return {
        "score_matrix": matrix.tolist(),
        "double_chance": {
            "1X": home + draw,
            "X2": draw + away,
            "12": home + away,
        },
        "total_goals": {
            "over_1_5": over_1_5,
            "under_1_5": 1.0 - over_1_5,
            "over_3_5": over_3_5,
            "under_3_5": 1.0 - over_3_5,
        },
        "team_goals": {
            "home_over_0_5": home_over_0_5,
            "away_over_0_5": away_over_0_5,
            "home_over_1_5": home_over_1_5,
            "away_over_1_5": away_over_1_5,
        },
        "expected_goals": {
            "home": float(poisson.home_expected_goals),
            "away": float(poisson.away_expected_goals),
        },
        "correct_scores": [
            {"score": f"{home_score}-{away_score}", "probability": probability}
            for probability, home_score, away_score in ranked_scores
        ],
    }


def derive_requested_market_probabilities(
    poisson: PoissonPrediction,
    half_time_markets: dict[str, float],
) -> dict[str, float]:
    """Derive probabilities for requested result, total, team, combo and half markets."""
    matrix = np.asarray(poisson.score_matrix, dtype=float)
    home_goals, away_goals = np.indices(matrix.shape)
    total = home_goals + away_goals
    outcomes = {
        "1": home_goals > away_goals,
        "x": home_goals == away_goals,
        "2": home_goals < away_goals,
    }
    result_names = {"1": "home_win", "x": "draw", "2": "away_win"}
    probabilities: dict[str, float] = {}
    for result, mask in outcomes.items():
        probabilities[result_names[result]] = float(matrix[mask].sum())
    probabilities.update({
        "double_chance_1x": probabilities["home_win"] + probabilities["draw"],
        "double_chance_x2": probabilities["draw"] + probabilities["away_win"],
        "double_chance_12": probabilities["home_win"] + probabilities["away_win"],
        "btts_yes": float(matrix[(home_goals > 0) & (away_goals > 0)].sum()),
    })
    probabilities["btts_no"] = 1.0 - probabilities["btts_yes"]
    for line in (1.5, 2.5, 3.5, 4.5):
        suffix = f"{int(line)}_5"
        over = float(matrix[total > line].sum())
        probabilities[f"over_{suffix}"] = over
        probabilities[f"under_{suffix}"] = 1.0 - over
        for result, result_mask in outcomes.items():
            key = result_names[result]
            for direction, total_mask in (("over", total > line), ("under", total <= line)):
                probabilities[f"{key}_{direction}_{suffix}"] = float(matrix[result_mask & total_mask].sum())
    for team, goals in (("home", home_goals), ("away", away_goals)):
        for line in (0.5, 1.5, 2.5, 3.5, 4.5):
            suffix = f"{int(line)}_5"
            over = float(matrix[goals > line].sum())
            probabilities[f"{team}_over_{suffix}"] = over
            probabilities[f"{team}_under_{suffix}"] = 1.0 - over
            for period in ("first_half", "second_half"):
                half_goals = half_time_markets.get(
                    f"{period}_{team}_over_1_5", 0.0
                ) if line == 1.5 else half_time_markets.get(
                    f"{period}_{team}_over_0_5", 0.0
                )
                if line in (0.5, 1.5):
                    probabilities[f"{team}_{period}_over_{suffix}"] = float(half_goals)
                    probabilities[f"{team}_{period}_under_{suffix}"] = 1.0 - float(half_goals)
    for result, result_mask in outcomes.items():
        key = result_names[result]
        btts = (home_goals > 0) & (away_goals > 0)
        probabilities[f"{key}_btts_yes"] = float(matrix[result_mask & btts].sum())
        probabilities[f"{key}_btts_no"] = float(matrix[result_mask & ~btts].sum())
    probabilities.update(half_time_markets)
    return probabilities


def derive_corner_probabilities(
    home_expected_corners: float,
    away_expected_corners: float,
    *,
    lines: tuple[float, ...] = (7.5, 8.5, 9.5, 10.5, 11.5),
    max_corners: int = 35,
) -> dict[str, float]:
    """Build total-corner over/under probabilities from independent Poisson counts."""
    if not all(isfinite(float(value)) and float(value) > 0 for value in (home_expected_corners, away_expected_corners)):
        raise ValueError("Expected corners must be finite and positive")
    if max_corners < 1 or any(line <= 0 or line % 1 == 0 for line in lines):
        raise ValueError("Corner lines must be positive half-step lines")

    def pmf(mean: float) -> list[float]:
        values = [exp(-mean)]
        for goals in range(1, max_corners + 1):
            values.append(values[-1] * mean / goals)
        return values

    home, away = pmf(float(home_expected_corners)), pmf(float(away_expected_corners))
    total_pmf = np.convolve(home, away)
    total_pmf = total_pmf / total_pmf.sum()
    result: dict[str, float] = {
        "expected_home_corners": float(home_expected_corners),
        "expected_away_corners": float(away_expected_corners),
        "expected_total_corners": float(home_expected_corners + away_expected_corners),
    }
    for line in lines:
        suffix = str(line).replace(".", "_")
        over = float(total_pmf[int(line) + 1:].sum())
        result[f"over_{suffix}"] = over
        result[f"under_{suffix}"] = 1.0 - over
    return result


def format_market_summary(markets: dict[str, Any]) -> list[str]:
    """Return concise Telegram/UI lines, suppressing weak secondary signals."""
    lines: list[str] = []
    double_chance = markets.get("double_chance") or {}
    if double_chance:
        label, probability = max(
            ((str(key), float(value)) for key, value in double_chance.items()),
            key=lambda item: item[1],
        )
        if probability >= MINIMUM_DOUBLE_CHANCE_CONFIDENCE:
            lines.append(f"Çifte şans: {label} %{probability * 100:.0f}")

    totals = markets.get("total_goals") or {}
    total_signals = []
    for key, label in (("over_1_5", "Üst 1.5"), ("under_3_5", "Alt 3.5")):
        probability = float(totals.get(key, 0.0))
        if probability >= MINIMUM_GOAL_MARKET_CONFIDENCE:
            total_signals.append(f"{label} %{probability * 100:.0f}")
    if total_signals:
        lines.append("Gol çizgisi: " + " · ".join(total_signals))

    team_goals = markets.get("team_goals") or {}
    team_signals = []
    for key, label in (
        ("home_over_0_5", "Ev 0.5 Üst"),
        ("away_over_0_5", "Dep. 0.5 Üst"),
        ("home_over_1_5", "Ev 1.5 Üst"),
        ("away_over_1_5", "Dep. 1.5 Üst"),
    ):
        probability = float(team_goals.get(key, 0.0))
        if probability >= MINIMUM_GOAL_MARKET_CONFIDENCE:
            team_signals.append(f"{label} %{probability * 100:.0f}")
    if team_signals:
        lines.append("Takım golü: " + " · ".join(team_signals))

    scores = markets.get("correct_scores") or []
    if scores:
        formatted = [
            f"{item['score']!s} %{float(item['probability']) * 100:.0f}"
            for item in scores[:3]
        ]
        lines.append("Olası skorlar: " + " · ".join(formatted))
    return lines


def format_telegram_market_lines(markets: dict[str, Any]) -> list[str]:
    """Return one Telegram line per confident secondary prediction type."""
    lines: list[str] = []
    double_chance = markets.get("double_chance") or {}
    candidates: list[tuple[str, float]] = []
    for label in ("1X", "X2", "12"):
        try:
            probability = _probability(
                double_chance[label], label=f"double chance {label}"
            )
        except (KeyError, TypeError, ValueError):
            continue
        candidates.append((label, probability))
    if candidates:
        label, probability = max(candidates, key=lambda item: item[1])
        if probability >= MINIMUM_DOUBLE_CHANCE_CONFIDENCE:
            lines.append(f"Çifte şans: {label} %{probability * 100:.0f}")

    for group, key, label in (
        ("total_goals", "over_1_5", "Üst 1.5"),
        ("total_goals", "under_3_5", "Alt 3.5"),
        ("team_goals", "home_over_0_5", "Ev 0.5 Üst"),
        ("team_goals", "away_over_0_5", "Dep. 0.5 Üst"),
        ("team_goals", "home_over_1_5", "Ev 1.5 Üst"),
        ("team_goals", "away_over_1_5", "Dep. 1.5 Üst"),
    ):
        try:
            probability = _probability(
                (markets.get(group) or {})[key], label=label
            )
        except (KeyError, TypeError, ValueError):
            continue
        if probability >= MINIMUM_GOAL_MARKET_CONFIDENCE:
            lines.append(f"{label}: %{probability * 100:.0f}")

    score_rank = 0
    for item in (markets.get("correct_scores") or [])[:3]:
        try:
            score = str(item["score"]).strip()
            probability = _probability(item["probability"], label="correct score")
        except (KeyError, TypeError, ValueError):
            continue
        if not score:
            continue
        score_rank += 1
        lines.append(f"Skor {score_rank}: {score} %{probability * 100:.0f}")
    return lines


def evaluate_market_probabilities(
    markets: dict[str, Any], *, home_score: int, away_score: int
) -> dict[str, Any]:
    """Score every persisted derived market without treating it as a published pick."""
    if home_score < 0 or away_score < 0:
        raise ValueError("Final scores cannot be negative")
    result = "1" if home_score > away_score else "X" if home_score == away_score else "2"
    total = home_score + away_score
    actuals = {
        "double_chance": {
            "1X": result in {"1", "X"},
            "X2": result in {"X", "2"},
            "12": result in {"1", "2"},
        },
        "total_goals": {
            "over_1_5": total >= 2,
            "under_1_5": total < 2,
            "over_3_5": total >= 4,
            "under_3_5": total < 4,
        },
        "team_goals": {
            "home_over_0_5": home_score >= 1,
            "away_over_0_5": away_score >= 1,
            "home_over_1_5": home_score >= 2,
            "away_over_1_5": away_score >= 2,
        },
    }
    evaluated: dict[str, Any] = {}
    for group, group_actuals in actuals.items():
        probabilities = markets.get(group) or {}
        evaluated[group] = {
            key: {
                "actual": actual,
                "probability": (probability := float(probabilities[key])),
                "brier_score": (probability - float(actual)) ** 2,
            }
            for key, actual in group_actuals.items()
            if key in probabilities
        }
    predicted_scores = {
        str(item.get("score")) for item in markets.get("correct_scores") or []
    }
    evaluated["correct_score"] = {
        "actual": f"{home_score}-{away_score}",
        "top_3_hit": f"{home_score}-{away_score}" in predicted_scores,
    }
    return evaluated
