"""Causal, chronological feature engineering for football matches."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from models.poisson_model import predict_score_probabilities
from data_pipeline.odds import vig_free_market_probabilities
from config.leagues import home_advantage_for_league


FEATURE_COLUMNS: tuple[str, ...] = (
    "league_id",
    "home_win_rate_5",
    "away_win_rate_5",
    "home_win_rate_10",
    "away_win_rate_10",
    "home_draw_rate_5",
    "away_draw_rate_5",
    "home_points_per_game_5",
    "away_points_per_game_5",
    "form_points_diff_5",
    "home_goal_diff_5",
    "away_goal_diff_5",
    "goal_diff_form_difference",
    "home_xg_diff_5",
    "away_xg_diff_5",
    "xg_form_difference",
    "home_venue_xg_for_5",
    "away_venue_xg_for_5",
    "home_venue_win_rate_5",
    "away_venue_win_rate_5",
    "venue_win_rate_diff_5",
    "home_venue_goals_for_5",
    "away_venue_goals_for_5",
    "home_elo",
    "away_elo",
    "elo_diff",
    "elo_abs_diff",
    "home_rest_days",
    "away_rest_days",
    "rest_days_diff",
    "h2h_home_win_rate_5",
    "h2h_draw_rate_5",
    "league_home_win_rate_200",
    "league_draw_rate_200",
    "league_away_win_rate_200",
    "poisson_home_win",
    "poisson_draw",
    "poisson_away_win",
    "poisson_result_margin",
    "poisson_over_2_5",
    "poisson_btts",
    "market_implied_home_win",
    "market_implied_draw",
    "market_implied_away_win",
    "market_result_margin",
    "market_implied_over_2_5",
    "market_implied_btts",
    "market_odds_available",
    "market_home_move",
    "market_draw_move",
    "market_away_move",
    "market_over_2_5_move",
    "market_btts_move",
    "home_available_count",
    "away_available_count",
    "home_impact_score",
    "away_impact_score",
    "impact_score_diff",
    "home_lineup_confirmed",
    "away_lineup_confirmed",
)

# These features target the three-way result boundary. Keeping them out of the
# binary goal models avoids letting a 1X2 experiment regress Üst/KG performance.
RESULT_ONLY_FEATURE_COLUMNS: frozenset[str] = frozenset(
    {
        "home_draw_rate_5",
        "away_draw_rate_5",
        "home_points_per_game_5",
        "away_points_per_game_5",
        "form_points_diff_5",
        "goal_diff_form_difference",
        "xg_form_difference",
        "venue_win_rate_diff_5",
        "elo_abs_diff",
        "rest_days_diff",
        "league_home_win_rate_200",
        "league_draw_rate_200",
        "league_away_win_rate_200",
        "poisson_result_margin",
        "market_result_margin",
    }
)
BINARY_FEATURE_COLUMNS: tuple[str, ...] = tuple(
    column for column in FEATURE_COLUMNS if column not in RESULT_ONLY_FEATURE_COLUMNS
)


AVAILABILITY_IMPACT_WEIGHTS: dict[str, float] = {
    "injured": 1.00,
    "suspended": 1.15,
    "doubtful": 0.35,
}
STARTING_XI_SIZE = 11


def availability_impact_score(
    unavailable_players: list[dict[str, Any]] | None,
    *,
    fallback_unavailable_count: int | None = None,
) -> float:
    """Convert known availability problems into a bounded team Impact Score.

    The source has no reliable player-value or expected-minutes field, so the score
    measures availability pressure rather than pretending to know a player's talent:
    injured=1.00, suspended=1.15, doubtful=0.35 first-XI equivalents. If only a
    historical aggregate count exists, it is conservatively treated as injuries.
    """
    if unavailable_players:
        weighted_absences = sum(
            AVAILABILITY_IMPACT_WEIGHTS.get(
                str(player.get("status", "injured")).lower(),
                AVAILABILITY_IMPACT_WEIGHTS["injured"],
            )
            for player in unavailable_players
        )
    else:
        weighted_absences = max(0, int(fallback_unavailable_count or 0))
    return float(np.clip(weighted_absences / STARTING_XI_SIZE, 0.0, 1.0))


@dataclass(frozen=True, slots=True)
class MatchResult:
    goals_for: int
    goals_against: int
    points: int
    was_home: bool
    xg_for: float | None = None
    xg_against: float | None = None


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


def _draw_rate(state: TeamState, n: int) -> float:
    return _safe_mean(
        [float(result.points == 1) for result in _recent(state, n)],
        0.28,
    )


def _points_per_game(state: TeamState, n: int) -> float:
    return _safe_mean(
        [float(result.points) for result in _recent(state, n)],
        1.33,
    )


def _top_probability_margin(probabilities: tuple[float, float, float]) -> float:
    ordered = sorted(probabilities, reverse=True)
    return float(ordered[0] - ordered[1])


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


def _xg_average(
    state: TeamState,
    n: int,
    *,
    for_team: bool,
    venue_home: bool | None = None,
) -> float:
    results = _recent(state, 10)
    if venue_home is not None:
        results = [result for result in results if result.was_home is venue_home]
    results = results[-n:]
    values = [
        result.xg_for if for_team else result.xg_against
        for result in results
    ]
    observed = [float(value) for value in values if value is not None]
    return _safe_mean(observed, 1.25)


def _xg_diff(state: TeamState, n: int) -> float:
    results = _recent(state, n)
    values = [
        float(result.xg_for - result.xg_against)
        for result in results
        if result.xg_for is not None and result.xg_against is not None
    ]
    return _safe_mean(values, 0.0)


def _rest_days(last_match: datetime | None, current: datetime) -> float:
    if last_match is None:
        return 7.0
    return float(np.clip((current - last_match).total_seconds() / 86400, 0, 30))


def _expected_score(actual_home: float, actual_away: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((actual_away - actual_home) / 400.0))


def margin_of_victory_multiplier(goal_diff: int, elo_diff_winner: float) -> float:
    """FiveThirtyEight-style Elo scaling: a 4-0 win moves more than a 1-0 win."""
    if goal_diff <= 0:
        return 1.0
    return float(np.log(goal_diff + 1) * (2.2 / ((abs(elo_diff_winner) * 0.001) + 2.2)))


def regress_elo_to_league_mean(last_known_elo: float | None, league_average: float = 1500.0) -> float:
    """Use a conservative 30% off-season regression for a returning team."""
    if last_known_elo is None:
        return league_average
    return float(league_average + 0.7 * (last_known_elo - league_average))


def _update_elo(
    home: TeamState, away: TeamState, home_score: float, *, goal_diff: int,
    home_advantage: float, k: float = 24.0,
) -> None:
    expected_home = _expected_score(home.elo + home_advantage, away.elo)
    winner_gap = home.elo - away.elo if home_score >= 0.5 else away.elo - home.elo
    delta = k * margin_of_victory_multiplier(goal_diff, winner_gap) * (home_score - expected_home)
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
        self.league_results: defaultdict[int, deque[int]] = defaultdict(
            lambda: deque(maxlen=200)
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
        opening_market = vig_free_market_probabilities(
            row.get("market_opening_odds") or {}
        )
        market_features = market or {
            "market_implied_home_win": poisson.prob_home_win,
            "market_implied_draw": poisson.prob_draw,
            "market_implied_away_win": poisson.prob_away_win,
            "market_implied_over_2_5": poisson.prob_over_2_5,
            "market_implied_btts": poisson.prob_btts,
        }
        home_impact_score = availability_impact_score(
            row.get("home_unavailable_players"),
            fallback_unavailable_count=row.get("home_unavailable_count"),
        )
        away_impact_score = availability_impact_score(
            row.get("away_unavailable_players"),
            fallback_unavailable_count=row.get("away_unavailable_count"),
        )
        home_points = _points_per_game(home, 5)
        away_points = _points_per_game(away, 5)
        home_goal_diff = _goal_diff(home, 5)
        away_goal_diff = _goal_diff(away, 5)
        home_xg_diff = _xg_diff(home, 5)
        away_xg_diff = _xg_diff(away, 5)
        home_venue_win_rate = _win_rate(home, 5, venue_home=True)
        away_venue_win_rate = _win_rate(away, 5, venue_home=False)
        home_rest_days = _rest_days(home.last_match_at, match_at)
        away_rest_days = _rest_days(away.last_match_at, match_at)
        league_history = self.league_results[int(row["league_id"])]
        league_home_rate = _safe_mean(
            [float(result == 0) for result in league_history], 0.44
        )
        league_draw_rate = _safe_mean(
            [float(result == 1) for result in league_history], 0.26
        )
        league_away_rate = _safe_mean(
            [float(result == 2) for result in league_history], 0.30
        )
        poisson_margin = _top_probability_margin(
            (poisson.prob_home_win, poisson.prob_draw, poisson.prob_away_win)
        )
        market_margin = _top_probability_margin(
            (
                market_features["market_implied_home_win"],
                market_features["market_implied_draw"],
                market_features["market_implied_away_win"],
            )
        )

        return {
            "league_id": float(row["league_id"]),
            "home_win_rate_5": _win_rate(home, 5),
            "away_win_rate_5": _win_rate(away, 5),
            "home_win_rate_10": _win_rate(home, 10),
            "away_win_rate_10": _win_rate(away, 10),
            "home_draw_rate_5": _draw_rate(home, 5),
            "away_draw_rate_5": _draw_rate(away, 5),
            "home_points_per_game_5": home_points,
            "away_points_per_game_5": away_points,
            "form_points_diff_5": home_points - away_points,
            "home_goal_diff_5": home_goal_diff,
            "away_goal_diff_5": away_goal_diff,
            "goal_diff_form_difference": home_goal_diff - away_goal_diff,
            "home_xg_diff_5": home_xg_diff,
            "away_xg_diff_5": away_xg_diff,
            "xg_form_difference": home_xg_diff - away_xg_diff,
            "home_venue_xg_for_5": _xg_average(
                home, 5, for_team=True, venue_home=True
            ),
            "away_venue_xg_for_5": _xg_average(
                away, 5, for_team=True, venue_home=False
            ),
            "home_venue_win_rate_5": home_venue_win_rate,
            "away_venue_win_rate_5": away_venue_win_rate,
            "venue_win_rate_diff_5": home_venue_win_rate - away_venue_win_rate,
            "home_venue_goals_for_5": home_for,
            "away_venue_goals_for_5": away_for,
            "home_elo": home.elo,
            "away_elo": away.elo,
            "elo_diff": home.elo - away.elo,
            "elo_abs_diff": abs(home.elo - away.elo),
            "home_rest_days": home_rest_days,
            "away_rest_days": away_rest_days,
            "rest_days_diff": home_rest_days - away_rest_days,
            "h2h_home_win_rate_5": _safe_mean(
                [float(value) for value in home_h2h_wins], 0.33
            ),
            "h2h_draw_rate_5": _safe_mean(
                [float(value) for value in h2h_draws], 0.28
            ),
            "league_home_win_rate_200": league_home_rate,
            "league_draw_rate_200": league_draw_rate,
            "league_away_win_rate_200": league_away_rate,
            "poisson_home_win": poisson.prob_home_win,
            "poisson_draw": poisson.prob_draw,
            "poisson_away_win": poisson.prob_away_win,
            "poisson_result_margin": poisson_margin,
            "poisson_over_2_5": poisson.prob_over_2_5,
            "poisson_btts": poisson.prob_btts,
            **market_features,
            "market_result_margin": market_margin,
            "market_odds_available": float(market is not None),
            "market_home_move": float(
                market_features["market_implied_home_win"]
                - (opening_market or market_features)["market_implied_home_win"]
            ),
            "market_draw_move": float(
                market_features["market_implied_draw"]
                - (opening_market or market_features)["market_implied_draw"]
            ),
            "market_away_move": float(
                market_features["market_implied_away_win"]
                - (opening_market or market_features)["market_implied_away_win"]
            ),
            "market_over_2_5_move": float(
                market_features["market_implied_over_2_5"]
                - (opening_market or market_features)["market_implied_over_2_5"]
            ),
            "market_btts_move": float(
                market_features["market_implied_btts"]
                - (opening_market or market_features)["market_implied_btts"]
            ),
            "home_available_count": float(row.get("home_available_count", 22)),
            "away_available_count": float(row.get("away_available_count", 22)),
            "home_impact_score": home_impact_score,
            "away_impact_score": away_impact_score,
            "impact_score_diff": home_impact_score - away_impact_score,
            "home_lineup_confirmed": float(bool(row.get("home_lineup_confirmed", False))),
            "away_lineup_confirmed": float(bool(row.get("away_lineup_confirmed", False))),
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
        home_xg = float(row["home_xg"]) if row.get("home_xg") is not None else None
        away_xg = float(row["away_xg"]) if row.get("away_xg") is not None else None
        home.results.append(
            MatchResult(home_score, away_score, home_points, True, home_xg, away_xg)
        )
        away.results.append(
            MatchResult(away_score, home_score, away_points, False, away_xg, home_xg)
        )
        score = 1.0 if home_score > away_score else 0.5 if home_score == away_score else 0.0
        _update_elo(
            home, away, score, goal_diff=abs(home_score - away_score),
            home_advantage=home_advantage_for_league(int(row["league_id"])),
        )
        home.last_match_at = match_at
        away.last_match_at = match_at
        winner = home_id if home_score > away_score else away_id if away_score > home_score else 0
        self.h2h[tuple(sorted((home_id, away_id)))].append(
            (winner, int(home_score == away_score))
        )
        self.league_results[int(row["league_id"])].append(
            _result_label(home_score, away_score)
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
    # Provider team-form rows remain useful for UI/commentary, but the model uses
    # the identical causal state builder in training and inference.
    return pd.DataFrame(rows, index=[int(row["id"]) for row in targets], columns=FEATURE_COLUMNS)
