"""Causal, chronological feature engineering for football matches."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
import json

import numpy as np
import pandas as pd

from models.poisson_model import predict_score_probabilities
from data_pipeline.odds import vig_free_market_probabilities


FEATURE_COLUMNS: tuple[str, ...] = (
    "league_id",
    "home_win_rate_5",
    "away_win_rate_5",
    "home_win_rate_10",
    "away_win_rate_10",
    "home_goal_diff_5",
    "away_goal_diff_5",
    "home_venue_win_rate_5",
    "away_venue_win_rate_5",
    "home_venue_goals_for_5",
    "away_venue_goals_for_5",
    "home_elo",
    "away_elo",
    "elo_diff",
    "home_rest_days",
    "away_rest_days",
    "h2h_home_win_rate_5",
    "h2h_draw_rate_5",
    "poisson_home_win",
    "poisson_draw",
    "poisson_away_win",
    "poisson_over_2_5",
    "poisson_btts",
    "market_implied_home_win",
    "market_implied_draw",
    "market_implied_away_win",
    "market_implied_over_2_5",
    "market_implied_btts",
    "market_odds_available",
)


@dataclass(frozen=True, slots=True)
class MatchResult:
    goals_for: int
    goals_against: int
    points: int
    was_home: bool


@dataclass(slots=True)
class TeamState:
    elo: float = 1500.0
    results: deque[MatchResult] = field(default_factory=lambda: deque(maxlen=10))
    last_match_at: datetime | None = None


def _safe_mean(values: list[float], default: float) -> float:
    return float(np.mean(values)) if values else default


def _recent(state: TeamState, n: int) -> list[MatchResult]:
    return list(state.results)[-n:]


def _win_rate(state: TeamState, n: int, *, venue_home: bool | None = None) -> float:
    results = _recent(state, n)
    if venue_home is not None:
        results = [result for result in results if result.was_home is venue_home][-n:]
    return _safe_mean([float(result.points == 3) for result in results], 0.33)


def _goal_average(
    state: TeamState,
    n: int,
    *,
    scored: bool,
    venue_home: bool | None = None,
) -> float:
    results = _recent(state, 10)
    if venue_home is not None:
        results = [result for result in results if result.was_home is venue_home]
    results = results[-n:]
    values = [
        float(result.goals_for if scored else result.goals_against)
        for result in results
    ]
    return _safe_mean(values, 1.25)


def _goal_diff(state: TeamState, n: int) -> float:
    results = _recent(state, n)
    return _safe_mean(
        [float(result.goals_for - result.goals_against) for result in results], 0.0
    )


def _rest_days(last_match: datetime | None, current: datetime) -> float:
    if last_match is None:
        return 7.0
    return float(np.clip((current - last_match).total_seconds() / 86400, 0, 30))


def _expected_score(actual_home: float, actual_away: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((actual_away - actual_home) / 400.0))


def _update_elo(home: TeamState, away: TeamState, home_score: float, k: float = 24.0) -> None:
    expected_home = _expected_score(home.elo + 65.0, away.elo)
    delta = k * (home_score - expected_home)
    home.elo += delta
    away.elo -= delta


def _result_label(home_score: int, away_score: int) -> int:
    if home_score > away_score:
        return 0
    if home_score == away_score:
        return 1
    return 2


class CausalFeatureState:
    """Mutable chronological state shared by training and live inference."""

    def __init__(self) -> None:
        self.states: defaultdict[int, TeamState] = defaultdict(TeamState)
        self.h2h: defaultdict[tuple[int, int], deque[tuple[int, int]]] = defaultdict(
            lambda: deque(maxlen=5)
        )

    def poisson_baseline(self, row: dict[str, Any]):
        """Return the causal Poisson baseline available before a target match."""
        home = self.states[int(row["home_team_id"])]
        away = self.states[int(row["away_team_id"])]
        home_for = _goal_average(home, 5, scored=True, venue_home=True)
        away_against = _goal_average(away, 5, scored=False, venue_home=False)
        away_for = _goal_average(away, 5, scored=True, venue_home=False)
        home_against = _goal_average(home, 5, scored=False, venue_home=True)
        home_lambda = float(np.clip((home_for + away_against) / 2, 0.05, 6.0))
        away_lambda = float(np.clip((away_for + home_against) / 2, 0.05, 6.0))
        return predict_score_probabilities(home_lambda, away_lambda)

    def feature_row(self, row: dict[str, Any]) -> dict[str, float]:
        home_id = int(row["home_team_id"])
        away_id = int(row["away_team_id"])
        match_at = datetime.fromisoformat(str(row["match_date"]).replace("Z", "+00:00"))
        home = self.states[home_id]
        away = self.states[away_id]

        pair = tuple(sorted((home_id, away_id)))
        pair_history = self.h2h[pair]
        home_h2h_wins = [winner == home_id for winner, _ in pair_history]
        h2h_draws = [winner == 0 for winner, _ in pair_history]

        home_for = _goal_average(home, 5, scored=True, venue_home=True)
        away_for = _goal_average(away, 5, scored=True, venue_home=False)
        poisson = self.poisson_baseline(row)
        market = vig_free_market_probabilities(row.get("market_odds") or {})
        market_features = market or {
            "market_implied_home_win": poisson.prob_home_win,
            "market_implied_draw": poisson.prob_draw,
            "market_implied_away_win": poisson.prob_away_win,
            "market_implied_over_2_5": poisson.prob_over_2_5,
            "market_implied_btts": poisson.prob_btts,
        }

        return {
            "league_id": float(row["league_id"]),
            "home_win_rate_5": _win_rate(home, 5),
            "away_win_rate_5": _win_rate(away, 5),
            "home_win_rate_10": _win_rate(home, 10),
            "away_win_rate_10": _win_rate(away, 10),
            "home_goal_diff_5": _goal_diff(home, 5),
            "away_goal_diff_5": _goal_diff(away, 5),
            "home_venue_win_rate_5": _win_rate(home, 5, venue_home=True),
            "away_venue_win_rate_5": _win_rate(away, 5, venue_home=False),
            "home_venue_goals_for_5": home_for,
            "away_venue_goals_for_5": away_for,
            "home_elo": home.elo,
            "away_elo": away.elo,
            "elo_diff": home.elo - away.elo,
            "home_rest_days": _rest_days(home.last_match_at, match_at),
            "away_rest_days": _rest_days(away.last_match_at, match_at),
            "h2h_home_win_rate_5": _safe_mean(
                [float(value) for value in home_h2h_wins], 0.33
            ),
            "h2h_draw_rate_5": _safe_mean(
                [float(value) for value in h2h_draws], 0.28
            ),
            "poisson_home_win": poisson.prob_home_win,
            "poisson_draw": poisson.prob_draw,
            "poisson_away_win": poisson.prob_away_win,
            "poisson_over_2_5": poisson.prob_over_2_5,
            "poisson_btts": poisson.prob_btts,
            **market_features,
            "market_odds_available": float(market is not None),
        }

    def update(self, row: dict[str, Any]) -> None:
        home_id = int(row["home_team_id"])
        away_id = int(row["away_team_id"])
        home_score = int(row["home_score"])
        away_score = int(row["away_score"])
        match_at = datetime.fromisoformat(str(row["match_date"]).replace("Z", "+00:00"))
        home = self.states[home_id]
        away = self.states[away_id]

        home_points = 3 if home_score > away_score else 1 if home_score == away_score else 0
        away_points = 3 if away_score > home_score else 1 if home_score == away_score else 0
        home.results.append(MatchResult(home_score, away_score, home_points, True))
        away.results.append(MatchResult(away_score, home_score, away_points, False))
        score = 1.0 if home_score > away_score else 0.5 if home_score == away_score else 0.0
        _update_elo(home, away, score)
        home.last_match_at = match_at
        away.last_match_at = match_at
        winner = home_id if home_score > away_score else away_id if away_score > home_score else 0
        self.h2h[tuple(sorted((home_id, away_id)))].append(
            (winner, int(home_score == away_score))
        )


def build_training_dataset(
    matches: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return features and labels using only information before each match."""
    valid = [
        row
        for row in matches
        if row.get("home_score") is not None
        and row.get("away_score") is not None
        and row.get("match_date")
    ]
    valid.sort(key=lambda row: (row["match_date"], int(row["id"])))
    if not valid:
        raise ValueError("No completed matches available for feature engineering")

    state = CausalFeatureState()
    feature_rows: list[dict[str, float]] = []
    label_rows: list[dict[str, Any]] = []
    for row in valid:
        home_score = int(row["home_score"])
        away_score = int(row["away_score"])
        match_at = datetime.fromisoformat(str(row["match_date"]).replace("Z", "+00:00"))
        feature_rows.append(state.feature_row(row))
        label_rows.append(
            {
                "match_id": int(row["id"]),
                "match_date": match_at,
                "result": _result_label(home_score, away_score),
                "over_2_5": int(home_score + away_score >= 3),
                "btts": int(home_score > 0 and away_score > 0),
            }
        )
        state.update(row)

    features = pd.DataFrame(feature_rows, columns=FEATURE_COLUMNS)
    labels = pd.DataFrame(label_rows)
    return features, labels


def build_upcoming_features(
    historical_matches: list[dict[str, Any]],
    upcoming_matches: list[dict[str, Any]],
    *,
    team_form_by_id: dict[int, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Build target features without updating state from unknown outcomes."""
    completed = [
        row
        for row in historical_matches
        if row.get("home_score") is not None
        and row.get("away_score") is not None
        and row.get("match_date")
    ]
    completed.sort(key=lambda row: (row["match_date"], int(row["id"])))
    targets = sorted(
        upcoming_matches, key=lambda row: (row["match_date"], int(row["id"]))
    )
    if not completed or not targets:
        raise ValueError("Completed history and upcoming matches are required")

    state = CausalFeatureState()
    for row in completed:
        state.update(row)
    rows = [state.feature_row(row) for row in targets]
    if team_form_by_id:
        for features, match in zip(rows, targets, strict=True):
            home_id = int(match["home_team_id"])
            away_id = int(match["away_team_id"])
            for prefix, team_id, venue_key in (
                ("home", home_id, "home_win_rate"),
                ("away", away_id, "away_win_rate"),
            ):
                form = team_form_by_id.get(team_id)
                if not form:
                    continue
                features[f"{prefix}_win_rate_5"] = float(form["win_rate_last5"])
                features[f"{prefix}_goal_diff_5"] = float(
                    form["avg_goals_scored_last5"]
                ) - float(form["avg_goals_conceded_last5"])
                split = form.get("home_away_split") or {}
                if isinstance(split, str):
                    split = json.loads(split)
                venue_value = split.get(venue_key)
                if venue_value is not None:
                    features[f"{prefix}_venue_win_rate_5"] = float(venue_value)
                features[f"{prefix}_elo"] = float(form["elo_rating"])
            features["elo_diff"] = features["home_elo"] - features["away_elo"]
    return pd.DataFrame(rows, index=[int(row["id"]) for row in targets], columns=FEATURE_COLUMNS)
