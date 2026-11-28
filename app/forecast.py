"""Poisson forecasts, team form summaries and market probability tables."""

from __future__ import annotations

import math
import logging
from bisect import bisect_left
from functools import wraps
from time import perf_counter
from typing import Any, Callable, ParamSpec, TypeVar

import pandas as pd
import streamlit as st

from app.metrics import _expected_calibration_error

LOG = logging.getLogger(__name__)
P = ParamSpec("P")
R = TypeVar("R")

DEFAULT_RECENT_FORM_PRIOR = 5.0
MIN_LOG_LOSS_IMPROVEMENT = 0.005
MIN_TUNING_HISTORY = 160
RECENT_FORM_PRIORS = (2.5, 5.0, 8.0, 12.0, 20.0)
TUNING_VALIDATION_MATCHES = 100


def _log_calculation_duration(function: Callable[P, R]) -> Callable[P, R]:
    """Log uncached forecast calculation time without changing its result."""
    @wraps(function)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        started_at = perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            LOG.info(
                "Forecast calculation: function=%s elapsed_seconds=%.3f",
                function.__name__,
                perf_counter() - started_at,
            )

    return wrapped


def team_recent_matches(history: pd.DataFrame, team_id: Any, before: pd.Timestamp, league_id: Any, limit: int = 5) -> pd.DataFrame:
    """Return a team's latest finished matches strictly before kickoff."""
    if history.empty or pd.isna(before):
        return pd.DataFrame()
    team = pd.to_numeric(pd.Series([team_id]), errors="coerce").iloc[0]
    league = pd.to_numeric(pd.Series([league_id]), errors="coerce").iloc[0]
    if pd.isna(team) or pd.isna(league):
        return pd.DataFrame()
    prior = history[
        (history["match_date"] < before)
        & (history["league_id"] == int(league))
        & ((history["home_team_id"] == int(team)) | (history["away_team_id"] == int(team)))
    ].copy()
    return prior.sort_values(["match_date", "id"], ascending=False).head(limit)

def summarize_team_form(recent: pd.DataFrame, team_id: Any) -> dict[str, Any]:
    """Summarize recent results from the team's point of view."""
    summary: dict[str, Any] = {"form": "—", "points": None, "goals_for": None, "goals_against": None, "xg_for": None, "xg_against": None}
    if recent.empty:
        return summary
    team = int(team_id)
    form, points, goals_for, goals_against = [], 0, [], []
    xg_for, xg_against = [], []
    for match in recent.to_dict("records"):
        home = int(match["home_team_id"]) == team
        scored = match["home_score"] if home else match["away_score"]
        conceded = match["away_score"] if home else match["home_score"]
        scored, conceded = int(scored), int(conceded)
        goals_for.append(scored)
        goals_against.append(conceded)
        if scored > conceded:
            form.append("G")
            points += 3
        elif scored == conceded:
            form.append("B")
            points += 1
        else:
            form.append("M")
        for target, key in ((xg_for, "home_xg" if home else "away_xg"), (xg_against, "away_xg" if home else "home_xg")):
            try:
                value = float(match[key])
                if pd.notna(value):
                    target.append(value)
            except (TypeError, ValueError):
                pass
    summary.update({
        "form": " · ".join(form),
        "points": points / len(recent),
        "goals_for": sum(goals_for) / len(goals_for),
        "goals_against": sum(goals_against) / len(goals_against),
        "xg_for": sum(xg_for) / len(xg_for) if xg_for else None,
        "xg_against": sum(xg_against) / len(xg_against) if xg_against else None,
    })
    return summary

def poisson_probabilities(mean: float, max_goals: int = 12) -> list[float]:
    """Return a normalized Poisson goal distribution, including its tail."""
    mean = max(0.05, float(mean))
    probabilities = [math.exp(-mean) * mean**goals / math.factorial(goals) for goals in range(max_goals)]
    probabilities.append(max(0.0, 1.0 - sum(probabilities)))
    total = sum(probabilities)
    return [probability / total for probability in probabilities]

def _expected_goal_means(
    row: dict[str, Any], history: pd.DataFrame, recent_form_prior: float
) -> tuple[float, float] | None:
    """Estimate goal means using only matches completed before this kickoff."""
    kickoff = row["kickoff"]
    if history.empty or pd.isna(kickoff):
        return None
    league_id = pd.to_numeric(pd.Series([row.get("league_id")]), errors="coerce").iloc[0]
    if pd.isna(league_id):
        return None
    league_matches = history[
        (history["league_id"] == int(league_id))
        & (history["match_date"] < kickoff)
    ]
    if len(league_matches) < 10:
        return None
    home_mean = float(league_matches["home_score"].mean())
    away_mean = float(league_matches["away_score"].mean())
    baseline = (home_mean + away_mean) / 2
    if baseline <= 0:
        return None

    def strengths(recent: pd.DataFrame, team_id: Any) -> tuple[float, float]:
        if recent.empty:
            return 1.0, 1.0
        team = int(team_id)
        home_mask = recent["home_team_id"] == team
        goals_for = [
            int(match["home_score"] if home else match["away_score"])
            for match, home in zip(recent.to_dict("records"), home_mask)
        ]
        goals_against = [
            int(match["away_score"] if home else match["home_score"])
            for match, home in zip(recent.to_dict("records"), home_mask)
        ]
        shrink = len(recent) / (len(recent) + recent_form_prior)
        attack = 1 + shrink * (sum(goals_for) / len(goals_for) / baseline - 1)
        defense_weakness = 1 + shrink * (sum(goals_against) / len(goals_against) / baseline - 1)
        return max(0.25, attack), max(0.25, defense_weakness)

    home_recent = team_recent_matches(history, row["home_team_id"], kickoff, league_id)
    away_recent = team_recent_matches(history, row["away_team_id"], kickoff, league_id)
    home_attack, home_defense = strengths(home_recent, row["home_team_id"])
    away_attack, away_defense = strengths(away_recent, row["away_team_id"])
    return (
        home_mean * math.sqrt(home_attack * away_defense),
        away_mean * math.sqrt(away_attack * home_defense),
    )

def _match_outcome_probabilities(expected_home: float, expected_away: float) -> tuple[float, float, float]:
    home_goals, away_goals = poisson_probabilities(expected_home), poisson_probabilities(expected_away)
    probabilities = [
        (home_count, away_count, home_probability * away_probability)
        for home_count, home_probability in enumerate(home_goals)
        for away_count, away_probability in enumerate(away_goals)
    ]
    total = sum(probability for _, _, probability in probabilities)
    outcome_probabilities = []
    outcome_checks = (
        lambda home, away: home > away,
        lambda home, away: home == away,
        lambda home, away: home < away,
    )
    for is_outcome in outcome_checks:
        outcome_probabilities.append(
            sum(
                probability
                for home_count, away_count, probability in probabilities
                if is_outcome(home_count, away_count)
            ) / total
        )
    return tuple(outcome_probabilities)

@st.cache_data(ttl=86_400, max_entries=32, show_spinner=False)
@_log_calculation_duration
def tune_recent_form_prior(league_history: pd.DataFrame) -> dict[str, Any]:
    """Select a form prior on earlier matches and report later holdout performance."""
    ordered = league_history.sort_values(["match_date", "id"], ascending=True).reset_index(drop=True)
    empty_result = {
        "prior": DEFAULT_RECENT_FORM_PRIOR,
        "validation_matches": 0,
        "log_loss": None,
        "baseline_log_loss": None,
        "selection_matches": 0,
        "selection_log_loss": None,
        "market_metrics": {},
    }
    if len(ordered) < MIN_TUNING_HISTORY:
        return empty_result

    first_validation = max(60, len(ordered) - TUNING_VALIDATION_MATCHES)
    predictions: dict[float, list[dict[str, float]]] = {
        prior: [] for prior in RECENT_FORM_PRIORS
    }
    actual_outcomes: list[int] = []
    rows = ordered.to_dict("records")

    league_rows: dict[int, list[dict[str, Any]]] = {}
    team_rows: dict[tuple[int, int], list[tuple[Any, Any, Any, bool]]] = {}
    for history_match in rows:
        kickoff = history_match.get("match_date")
        league_value = pd.to_numeric(
            pd.Series([history_match.get("league_id")]), errors="coerce"
        ).iloc[0]
        if pd.isna(kickoff) or pd.isna(league_value):
            continue
        league_id = int(league_value)
        league_rows.setdefault(league_id, []).append(history_match)
        for team_key, is_home in (
            (history_match.get("home_team_id"), True),
            (history_match.get("away_team_id"), False),
        ):
            team_value = pd.to_numeric(pd.Series([team_key]), errors="coerce").iloc[0]
            if pd.isna(team_value):
                continue
            team_rows.setdefault((league_id, int(team_value)), []).append(
                (
                    kickoff,
                    history_match.get("home_score"),
                    history_match.get("away_score"),
                    is_home,
                )
            )

    league_indexes: dict[int, tuple[list[Any], list[float], list[float], list[int], list[int]]] = {}
    for league_id, matches in league_rows.items():
        dates: list[Any] = []
        home_sums, away_sums = [0.0], [0.0]
        home_counts, away_counts = [0], [0]
        for history_match in matches:
            dates.append(history_match["match_date"])
            previous_home, previous_away = home_sums[-1], away_sums[-1]
            previous_home_count, previous_away_count = home_counts[-1], away_counts[-1]
            try:
                home_score = float(history_match["home_score"])
                if not math.isfinite(home_score):
                    raise ValueError
            except (TypeError, ValueError):
                home_score = None
            try:
                away_score = float(history_match["away_score"])
                if not math.isfinite(away_score):
                    raise ValueError
            except (TypeError, ValueError):
                away_score = None
            home_sums.append(previous_home + (home_score or 0.0))
            away_sums.append(previous_away + (away_score or 0.0))
            home_counts.append(previous_home_count + (home_score is not None))
            away_counts.append(previous_away_count + (away_score is not None))
        league_indexes[league_id] = (
            dates,
            home_sums,
            away_sums,
            home_counts,
            away_counts,
        )

    team_dates = {
        team_key: [entry[0] for entry in matches]
        for team_key, matches in team_rows.items()
    }

    def recent_strength(
        league_id: int, team_id: int, kickoff: Any, baseline: float, prior: float
    ) -> tuple[float, float]:
        key = (league_id, team_id)
        matches = team_rows.get(key, [])
        before = bisect_left(team_dates.get(key, []), kickoff)
        recent = matches[max(0, before - 5):before]
        if not recent:
            return 1.0, 1.0
        goals_for = [
            int(home_score if is_home else away_score)
            for _, home_score, away_score, is_home in recent
        ]
        goals_against = [
            int(away_score if is_home else home_score)
            for _, home_score, away_score, is_home in recent
        ]
        shrink = len(recent) / (len(recent) + prior)
        attack = 1 + shrink * (sum(goals_for) / len(goals_for) / baseline - 1)
        defense_weakness = 1 + shrink * (
            sum(goals_against) / len(goals_against) / baseline - 1
        )
        return max(0.25, attack), max(0.25, defense_weakness)

    for index in range(first_validation, len(rows)):
        match = rows[index]
        if match.get("home_score") is None or match.get("away_score") is None:
            continue
        home_score, away_score = int(match["home_score"]), int(match["away_score"])
        outcome = 0 if home_score > away_score else 1 if home_score == away_score else 2
        actual_outcomes.append(outcome)
        kickoff = match["match_date"]
        league_value = pd.to_numeric(
            pd.Series([match.get("league_id")]), errors="coerce"
        ).iloc[0]
        league_id = int(league_value) if pd.notna(league_value) else None
        league_index = league_indexes.get(league_id) if league_id is not None else None
        league_count = 0
        baseline = None
        home_mean = away_mean = 0.0
        if league_index is not None and pd.notna(kickoff):
            dates, home_sums, away_sums, home_counts, away_counts = league_index
            league_count = bisect_left(dates, kickoff)
            if league_count >= 10 and home_counts[league_count] and away_counts[league_count]:
                home_mean = home_sums[league_count] / home_counts[league_count]
                away_mean = away_sums[league_count] / away_counts[league_count]
                baseline = (home_mean + away_mean) / 2
                if baseline <= 0:
                    baseline = None
        row = {
            "kickoff": kickoff,
            "league_id": league_id,
            "home_team_id": match["home_team_id"],
            "away_team_id": match["away_team_id"],
        }
        home_value = pd.to_numeric(
            pd.Series([row["home_team_id"]]), errors="coerce"
        ).iloc[0]
        away_value = pd.to_numeric(
            pd.Series([row["away_team_id"]]), errors="coerce"
        ).iloc[0]
        for prior in RECENT_FORM_PRIORS:
            means = None
            if baseline is not None:
                home_attack, home_defense = (
                    recent_strength(
                        league_id, int(home_value), kickoff, baseline, prior
                    )
                    if pd.notna(home_value)
                    else (1.0, 1.0)
                )
                away_attack, away_defense = (
                    recent_strength(
                        league_id, int(away_value), kickoff, baseline, prior
                    )
                    if pd.notna(away_value)
                    else (1.0, 1.0)
                )
                means = (
                    home_mean * math.sqrt(home_attack * away_defense),
                    away_mean * math.sqrt(away_attack * home_defense),
                )
            if means is None:
                predictions[prior].append(
                    {
                        "result": float("nan"),
                        "result_0": float("nan"),
                        "result_1": float("nan"),
                        "result_2": float("nan"),
                        "over_2_5": float("nan"),
                        "over_2_5_probability": float("nan"),
                        "over_2_5_actual": float("nan"),
                        "btts": float("nan"),
                        "btts_probability": float("nan"),
                        "btts_actual": float("nan"),
                    }
                )
            else:
                home_probs = poisson_probabilities(means[0])
                away_probs = poisson_probabilities(means[1])
                score_probs = [
                    (home, away, home_probability * away_probability)
                    for home, home_probability in enumerate(home_probs)
                    for away, away_probability in enumerate(away_probs)
                ]
                total_probability = sum(probability for _, _, probability in score_probs)
                over_probability = sum(
                    probability
                    for home, away, probability in score_probs
                    if home + away >= 3
                ) / total_probability
                btts_probability = sum(
                    probability
                    for home, away, probability in score_probs
                    if home > 0 and away > 0
                ) / total_probability
                outcome_probabilities = _match_outcome_probabilities(*means)
                predictions[prior].append(
                    {
                        "result": outcome_probabilities[outcome],
                        "result_0": outcome_probabilities[0],
                        "result_1": outcome_probabilities[1],
                        "result_2": outcome_probabilities[2],
                        "over_2_5": over_probability if home_score + away_score >= 3 else 1 - over_probability,
                        "over_2_5_probability": over_probability,
                        "over_2_5_actual": float(home_score + away_score >= 3),
                        "btts": btts_probability if home_score > 0 and away_score > 0 else 1 - btts_probability,
                        "btts_probability": btts_probability,
                        "btts_actual": float(home_score > 0 and away_score > 0),
                    }
                )

    baseline = predictions[DEFAULT_RECENT_FORM_PRIOR]
    valid = [
        index
        for index, value in enumerate(baseline)
        if not math.isnan(value["result"])
    ]
    if len(valid) < 60:
        empty_result["validation_matches"] = len(valid)
        return empty_result

    selection_end = max(40, int(len(valid) * 0.8))
    selection_indices = valid[:selection_end]
    evaluation_indices = valid[selection_end:]
    scores = {
        prior: sum(
            -math.log(max(1e-12, predictions[prior][index]["result"]))
            for index in selection_indices
        ) / len(selection_indices)
        for prior in RECENT_FORM_PRIORS
        if all(
            not math.isnan(predictions[prior][index]["result"])
            for index in selection_indices
        )
    }
    baseline_loss = scores.get(DEFAULT_RECENT_FORM_PRIOR)
    if baseline_loss is None:
        empty_result["validation_matches"] = len(evaluation_indices)
        empty_result["selection_matches"] = len(selection_indices)
        return empty_result
    best_prior = min(scores, key=scores.get)
    best_loss = scores[best_prior]
    if baseline_loss - best_loss < MIN_LOG_LOSS_IMPROVEMENT:
        best_prior, best_loss = DEFAULT_RECENT_FORM_PRIOR, baseline_loss

    if not evaluation_indices:
        empty_result["selection_matches"] = len(selection_indices)
        empty_result["selection_log_loss"] = best_loss
        return empty_result
    evaluation = {
        prior: sum(
            -math.log(max(1e-12, predictions[prior][index]["result"]))
            for index in evaluation_indices
        ) / len(evaluation_indices)
        for prior in (best_prior, DEFAULT_RECENT_FORM_PRIOR)
        if all(
            not math.isnan(predictions[prior][index]["result"])
            for index in evaluation_indices
        )
    }
    if best_prior not in evaluation:
        best_prior, best_loss = DEFAULT_RECENT_FORM_PRIOR, baseline_loss
    market_metrics: dict[str, dict[str, Any]] = {}
    for market in ("result", "over_2_5", "btts"):
        model_values = [predictions[best_prior][index][market] for index in evaluation_indices]
        baseline_values = [
            predictions[DEFAULT_RECENT_FORM_PRIOR][index][market]
            for index in evaluation_indices
        ]
        valid_market = [
            offset
            for offset, (model_value, baseline_value) in enumerate(
                zip(model_values, baseline_values)
            )
            if not math.isnan(model_value) and not math.isnan(baseline_value)
        ]
        if market == "result":
            model_pairs = [
                (
                    predictions[best_prior][evaluation_indices[offset]][f"result_{outcome}"],
                    float(actual_outcomes[evaluation_indices[offset]] == outcome),
                )
                for offset in valid_market
                for outcome in range(3)
            ]
            baseline_pairs = [
                (
                    predictions[DEFAULT_RECENT_FORM_PRIOR][evaluation_indices[offset]][f"result_{outcome}"],
                    float(actual_outcomes[evaluation_indices[offset]] == outcome),
                )
                for offset in valid_market
                for outcome in range(3)
            ]
            model_brier = sum(
                sum(
                    (model_pairs[offset * 3 + outcome][0] - model_pairs[offset * 3 + outcome][1]) ** 2
                    for outcome in range(3)
                )
                for offset in range(len(valid_market))
            ) / len(valid_market) if valid_market else None
            baseline_brier = sum(
                sum(
                    (baseline_pairs[offset * 3 + outcome][0] - baseline_pairs[offset * 3 + outcome][1]) ** 2
                    for outcome in range(3)
                )
                for offset in range(len(valid_market))
            ) / len(valid_market) if valid_market else None
        else:
            probability_key = f"{market}_probability"
            actual_key = f"{market}_actual"
            model_pairs = [
                (
                    predictions[best_prior][evaluation_indices[offset]][probability_key],
                    predictions[best_prior][evaluation_indices[offset]][actual_key],
                )
                for offset in valid_market
            ]
            baseline_pairs = [
                (
                    predictions[DEFAULT_RECENT_FORM_PRIOR][evaluation_indices[offset]][probability_key],
                    predictions[DEFAULT_RECENT_FORM_PRIOR][evaluation_indices[offset]][actual_key],
                )
                for offset in valid_market
            ]
            model_brier = (
                sum((probability - actual) ** 2 for probability, actual in model_pairs)
                / len(model_pairs) if model_pairs else None
            )
            baseline_brier = (
                sum((probability - actual) ** 2 for probability, actual in baseline_pairs)
                / len(baseline_pairs) if baseline_pairs else None
            )
        market_metrics[market] = {
            "sample_size": len(valid_market),
            "model_log_loss": (
                sum(-math.log(max(1e-12, model_values[offset])) for offset in valid_market)
                / len(valid_market) if valid_market else None
            ),
            "baseline_log_loss": (
                sum(-math.log(max(1e-12, baseline_values[offset])) for offset in valid_market)
                / len(valid_market) if valid_market else None
            ),
            "model_brier_score": model_brier,
            "baseline_brier_score": baseline_brier,
            "model_calibration_error": _expected_calibration_error(model_pairs),
            "baseline_calibration_error": _expected_calibration_error(baseline_pairs),
        }
    return {
        "prior": best_prior,
        "validation_matches": len(evaluation_indices),
        "log_loss": evaluation.get(best_prior),
        "baseline_log_loss": evaluation.get(DEFAULT_RECENT_FORM_PRIOR),
        "selection_matches": len(selection_indices),
        "selection_log_loss": best_loss,
        "market_metrics": market_metrics,
    }

@st.cache_data(ttl=86_400, max_entries=256, show_spinner=False)
@_log_calculation_duration
def model_match_forecast(
    row: dict[str, Any],
    history: pd.DataFrame,
    recent_form_tuning: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Fit a simple independent Poisson goals model from pre-match league history."""
    kickoff = row["kickoff"]
    if history.empty or pd.isna(kickoff):
        return None
    league_id = pd.to_numeric(pd.Series([row.get("league_id")]), errors="coerce").iloc[0]
    if pd.isna(league_id):
        return None
    league_matches = history[
        (history["league_id"] == int(league_id))
        & (history["match_date"] < kickoff)
    ]
    if len(league_matches) < 10:
        return None
    tuning = recent_form_tuning or tune_recent_form_prior(league_matches)
    recent_form_prior = float(tuning["prior"])
    means = _expected_goal_means(row, history, recent_form_prior)
    if means is None:
        return None
    expected_home, expected_away = means
    home_recent = team_recent_matches(history, row["home_team_id"], kickoff, league_id)
    away_recent = team_recent_matches(history, row["away_team_id"], kickoff, league_id)
    home_goals, away_goals = poisson_probabilities(expected_home), poisson_probabilities(expected_away)
    score_probabilities = [
        (home_count, away_count, home_probability * away_probability)
        for home_count, home_probability in enumerate(home_goals)
        for away_count, away_probability in enumerate(away_goals)
    ]
    total = sum(probability for _, _, probability in score_probabilities)
    home_win = sum(probability for home_count, away_count, probability in score_probabilities if home_count > away_count) / total
    draw = sum(probability for home_count, away_count, probability in score_probabilities if home_count == away_count) / total
    away_win = sum(probability for home_count, away_count, probability in score_probabilities if home_count < away_count) / total
    over_2_5 = sum(probability for home_count, away_count, probability in score_probabilities if home_count + away_count >= 3) / total
    btts = sum(probability for home_count, away_count, probability in score_probabilities if home_count > 0 and away_count > 0) / total
    most_likely = max(score_probabilities, key=lambda item: item[2])
    score_matrix = [
        [home_probability * away_probability for away_probability in away_goals]
        for home_probability in home_goals
    ]
    return {
        "home_win": home_win,
        "draw": draw,
        "away_win": away_win,
        "over_2_5": over_2_5,
        "btts": btts,
        "expected_home": expected_home,
        "expected_away": expected_away,
        "most_likely_score": f"{most_likely[0]}–{most_likely[1]}",
        "score_matrix": score_matrix,
        "history_count": len(league_matches),
        "home_sample": len(home_recent),
        "away_sample": len(away_recent),
        "recent_form_prior": recent_form_prior,
        "tuning_matches": tuning["validation_matches"],
        "tuning_log_loss": tuning["log_loss"],
        "tuning_baseline_log_loss": tuning["baseline_log_loss"],
        "tuning_selection_matches": tuning["selection_matches"],
        "tuning_market_metrics": tuning["market_metrics"],
    }

@st.cache_data(ttl=86_400, max_entries=256, show_spinner=False)
@_log_calculation_duration
def model_half_forecast(
    row: dict[str, Any], history: pd.DataFrame, *, half: int
) -> dict[str, Any] | None:
    """Fit a period-specific Poisson model from historical halftime scores."""
    kickoff = row["kickoff"]
    if history.empty or pd.isna(kickoff) or half not in {1, 2}:
        return None
    league_id = pd.to_numeric(pd.Series([row.get("league_id")]), errors="coerce").iloc[0]
    if pd.isna(league_id):
        return None
    sample = history[
        (history["league_id"] == int(league_id)) & (history["match_date"] < kickoff)
    ].copy()
    if half == 1:
        sample["period_home"] = sample["home_halftime_score"]
        sample["period_away"] = sample["away_halftime_score"]
    else:
        sample["period_home"] = sample["home_score"] - sample["home_halftime_score"]
        sample["period_away"] = sample["away_score"] - sample["away_halftime_score"]
    sample = sample.dropna(subset=["period_home", "period_away"])
    sample = sample[(sample["period_home"] >= 0) & (sample["period_away"] >= 0)]
    if len(sample) < 20:
        return None

    league_home = float(sample["period_home"].mean())
    league_away = float(sample["period_away"].mean())
    if league_home + league_away <= 0:
        return None

    def recent_strength(team_id: Any, is_home: bool) -> tuple[float, float, int]:
        team = int(team_id)
        recent = sample[
            (sample["home_team_id"] == team) | (sample["away_team_id"] == team)
        ].sort_values(["match_date", "id"], ascending=False).head(5)
        if recent.empty:
            return 1.0, 1.0, 0
        scored, conceded = [], []
        for match in recent.to_dict("records"):
            team_home = int(match["home_team_id"]) == team
            scored.append(float(match["period_home"] if team_home else match["period_away"]))
            conceded.append(float(match["period_away"] if team_home else match["period_home"]))
        baseline_for = league_home if is_home else league_away
        baseline_against = league_away if is_home else league_home
        shrink = len(recent) / (len(recent) + 5)
        attack = 1 + shrink * (sum(scored) / len(scored) / max(0.05, baseline_for) - 1)
        weakness = 1 + shrink * (sum(conceded) / len(conceded) / max(0.05, baseline_against) - 1)
        return max(0.25, attack), max(0.25, weakness), len(recent)

    home_attack, away_defense, home_n = recent_strength(row["home_team_id"], True)
    away_attack, home_defense, away_n = recent_strength(row["away_team_id"], False)
    expected_home = league_home * math.sqrt(home_attack * home_defense)
    expected_away = league_away * math.sqrt(away_attack * away_defense)
    home_goals = poisson_probabilities(expected_home)
    away_goals = poisson_probabilities(expected_away)
    matrix = [[home_goals[h] * away_goals[a] for a in range(len(away_goals))] for h in range(len(home_goals))]
    total = sum(sum(values) for values in matrix)
    home_win = sum(matrix[h][a] for h in range(len(matrix)) for a in range(len(matrix[h])) if h > a) / total
    draw = sum(matrix[h][a] for h in range(len(matrix)) for a in range(len(matrix[h])) if h == a) / total
    away_win = max(0.0, 1 - home_win - draw)
    btts = sum(matrix[h][a] for h in range(1, len(matrix)) for a in range(1, len(matrix[h]))) / total
    return {
        "score_matrix": matrix,
        "home_win": home_win,
        "draw": draw,
        "away_win": away_win,
        "btts": btts,
        "history_count": len(sample),
        "home_sample": home_n,
        "away_sample": away_n,
        "expected_home": expected_home,
        "expected_away": expected_away,
    }

@st.cache_data(ttl=86_400, max_entries=256, show_spinner=False)
@_log_calculation_duration
def model_corner_forecast(row: dict[str, Any], history: pd.DataFrame) -> dict[str, Any] | None:
    """Estimate full-time corners from historical corner counts in this league."""
    kickoff = row["kickoff"]
    if history.empty or pd.isna(kickoff):
        return None
    league_id = pd.to_numeric(pd.Series([row.get("league_id")]), errors="coerce").iloc[0]
    if pd.isna(league_id):
        return None
    league = history[
        (history["league_id"] == int(league_id)) & (history["match_date"] < kickoff)
    ].dropna(subset=["home_corners", "away_corners"])
    if len(league) < 20:
        return None
    home_mean, away_mean = float(league["home_corners"].mean()), float(league["away_corners"].mean())
    baseline = (home_mean + away_mean) / 2
    if baseline <= 0:
        return None

    def recent_corner_strength(team_id: Any) -> tuple[float, float, int]:
        team = int(team_id)
        recent = history[
            (history["league_id"] == int(league_id))
            & (history["match_date"] < kickoff)
            & ((history["home_team_id"] == team) | (history["away_team_id"] == team))
        ].dropna(subset=["home_corners", "away_corners"])
        recent = recent.sort_values(["match_date", "id"], ascending=False).head(5)
        if recent.empty:
            return 1.0, 1.0, 0
        for_values, against_values = [], []
        for match in recent.to_dict("records"):
            is_home = int(match["home_team_id"]) == team
            for_values.append(float(match["home_corners"] if is_home else match["away_corners"]))
            against_values.append(float(match["away_corners"] if is_home else match["home_corners"]))
        shrink = len(recent) / (len(recent) + 5)
        attack = max(0.25, 1 + shrink * (sum(for_values) / len(for_values) / baseline - 1))
        weakness = max(0.25, 1 + shrink * (sum(against_values) / len(against_values) / baseline - 1))
        return attack, weakness, len(recent)

    home_attack, home_weakness, home_n = recent_corner_strength(row["home_team_id"])
    away_attack, away_weakness, away_n = recent_corner_strength(row["away_team_id"])
    expected_home = home_mean * math.sqrt(home_attack * away_weakness)
    expected_away = away_mean * math.sqrt(away_attack * home_weakness)
    home_dist, away_dist = poisson_probabilities(expected_home), poisson_probabilities(expected_away)
    matrix = [[home_probability * away_probability for away_probability in away_dist] for home_probability in home_dist]
    return {
        "home_expected": expected_home,
        "away_expected": expected_away,
        "score_matrix": matrix,
        "history_count": len(league),
        "home_sample": home_n,
        "away_sample": away_n,
    }

def market_probability_tables(forecast: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Derive standard goal and team-goal markets from the model score matrix."""
    matrix = forecast["score_matrix"]
    total_goals = []
    for line in (0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5):
        over = sum(matrix[h][a] for h in range(len(matrix)) for a in range(len(matrix[h])) if h + a > line)
        total_goals.append({"Gol çizgisi": line, "Üst": f"%{over * 100:.1f}", "Alt": f"%{(1 - over) * 100:.1f}"})

    team_goals = []
    for line in (0.5, 1.5, 2.5, 3.5):
        home_over = sum(matrix[h][a] for h in range(len(matrix)) for a in range(len(matrix[h])) if h > line)
        away_over = sum(matrix[h][a] for h in range(len(matrix)) for a in range(len(matrix[h])) if a > line)
        team_goals.append({
            "Takım gol çizgisi": line,
            "Ev Üst": f"%{home_over * 100:.1f}",
            "Ev Alt": f"%{(1 - home_over) * 100:.1f}",
            "Dep Üst": f"%{away_over * 100:.1f}",
            "Dep Alt": f"%{(1 - away_over) * 100:.1f}",
        })

    home_win, draw, away_win = forecast["home_win"], forecast["draw"], forecast["away_win"]
    btts = forecast["btts"]
    outcomes = pd.DataFrame([
        {"Pazar": "Çifte şans 1X", "Olasılık": f"%{(home_win + draw) * 100:.1f}"},
        {"Pazar": "Çifte şans X2", "Olasılık": f"%{(draw + away_win) * 100:.1f}"},
        {"Pazar": "Çifte şans 12", "Olasılık": f"%{(home_win + away_win) * 100:.1f}"},
        {"Pazar": "Karşılıklı gol Var", "Olasılık": f"%{btts * 100:.1f}"},
        {"Pazar": "Karşılıklı gol Yok", "Olasılık": f"%{(1 - btts) * 100:.1f}"},
    ])
    return pd.DataFrame(total_goals), pd.DataFrame(team_goals), outcomes
