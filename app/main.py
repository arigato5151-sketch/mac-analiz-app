"""Streamlit match center and completed-match statistics archive."""

from __future__ import annotations

import json
import math
import re
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.components.data import (
    load_completed_match_history,
    load_iddaa_market_labels,
    load_match_day,
    load_reference_catalog,
)

ISTANBUL = ZoneInfo("Europe/Istanbul")

COUNTRY_NAMES_TR = {
    "Argentina": "Arjantin", "Austria": "Avusturya", "Belgium": "Belçika",
    "Brazil": "Brezilya", "England": "İngiltere", "France": "Fransa",
    "Germany": "Almanya", "Italy": "İtalya", "Netherlands": "Hollanda",
    "Portugal": "Portekiz", "Russia": "Rusya", "Saudi-Arabia": "Suudi Arabistan",
    "Spain": "İspanya", "Turkey": "Türkiye", "USA": "ABD", "World": "Uluslararası",
}
LEAGUE_NAMES_TR = {
    "Premier League": "Premier Lig", "La Liga": "LaLiga", "Serie A": "Serie A",
    "Bundesliga": "Bundesliga", "Ligue 1": "Ligue 1", "UEFA Champions League": "UEFA Şampiyonlar Ligi",
    "UEFA Europa League": "UEFA Avrupa Ligi", "UEFA Europa Conference League": "UEFA Konferans Ligi",
    "Süper Lig": "Süper Lig", "Primeira Liga": "Portekiz Ligi", "Eredivisie": "Hollanda Ligi",
    "Jupiler Pro League": "Belçika Pro Ligi", "Championship": "İngiltere Championship",
    "Pro League": "Suudi Arabistan Pro Ligi", "Major League Soccer": "MLS",
    "Premier Liga": "Rusya Premier Ligi", "Liga Profesional": "Arjantin Profesyonel Ligi",
    "FIFA World Cup": "FIFA Dünya Kupası", "Euro Championship": "Avrupa Şampiyonası",
    "UEFA Nations League": "UEFA Uluslar Ligi", "Africa Cup of Nations": "Afrika Uluslar Kupası",
    "Asian Cup": "Asya Kupası", "Copa America": "Copa América", "CONCACAF Gold Cup": "CONCACAF Altın Kupa",
}

st.set_page_config(page_title="Maç Merkezi", page_icon="⚽", layout="wide")
st.markdown(
    """
    <style>
    .block-container{max-width:1440px;padding-top:1.6rem;padding-bottom:3rem}
    .hero{padding:1.5rem 1.8rem;border:1px solid rgba(140,160,190,.18);border-radius:20px;
      background:linear-gradient(120deg,#101c31 0%,#152b3c 55%,#153a36 100%);margin-bottom:1.2rem}
    .hero-kicker{color:#84e0be;text-transform:uppercase;letter-spacing:.12em;font-size:.78rem;font-weight:700}
    .hero h1{margin:.4rem 0 .2rem;font-size:2.2rem}.hero p{color:#c1cedb;margin:0}
    .match-title{font-size:1.25rem;font-weight:700;margin:.2rem 0 .7rem}
    .market-head{font-size:1rem;font-weight:700;margin:.4rem 0 .2rem}
    [data-testid="stMetric"]{background:rgba(128,150,180,.08);padding:.8rem 1rem;border-radius:14px}
    </style>
    """,
    unsafe_allow_html=True,
)


def unpack_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def market_rows(payload: dict[str, Any], labels: dict[str, str]) -> list[dict[str, Any]]:
    rows = []
    for market in payload.get("markets", []):
        selections = []
        for selection in market.get("selections", []):
            try:
                # IdDaa provides both the internal odd and the value shown to
                # bettors. Display the latter so the app matches IdDaa's board.
                display_odd = selection.get("display_odds")
                odd = float(display_odd if display_odd is not None else selection.get("odds"))
            except (TypeError, ValueError):
                continue
            if odd > 1:
                selections.append((str(selection.get("name") or selection.get("id") or "Seçim"), odd))
        if len(selections) < 2:
            continue
        implied = [1 / odd for _, odd in selections]
        total = sum(implied)
        line = market.get("line")
        market_code = market.get("market_id")
        sport_type = market.get("sport_type")
        market_key = f"{sport_type if sport_type is not None else ''}:{market_code if market_code is not None else ''}"
        configured_name = (
            labels.get(market_key)
            or labels.get(str(market_code if market_code is not None else ""))
        )
        # The event feed stores a generic market_name for team-goal codes.
        # Preserve the code meaning across the sport-type variants in stored quotes.
        if str(sport_type) in {"2", "4"}:
            team_goal_name = (
                {(2, 603): "Ev Sahibi Gol Alt/Üst", (2, 604): "Deplasman Gol Alt/Üst",
                 (4, 203): "Ev Sahibi Gol Alt/Üst", (4, 207): "Deplasman Gol Alt/Üst"}
                .get((int(sport_type), int(market_code))) if market_code is not None else None
            )
            if team_goal_name and str(configured_name or "").casefold() in {"", "maç sonucu alt/üst", "mac sonucu alt/ust"}:
                configured_name = team_goal_name
        name = str(
            configured_name
            or market.get("market_name")
            or _infer_legacy_market_name(selections)
            or (f"Market {market_code}" if market_code is not None else "Diğer market")
        )
        if line not in (None, "", 0, "0"):
            name = f"{name} · {line}"
        rows.append({
            "name": name,
            "source_name": str(configured_name or market.get("market_name") or name),
            "line": line,
            "market_id": market_code,
            "sport_type": sport_type,
            "selections": [
                {"name": item[0], "odds": item[1], "probability": prob / total}
                for item, prob in zip(selections, implied)
            ],
            "overround": total - 1,
        })
    return rows


def _infer_legacy_market_name(selections: list[tuple[str, float]]) -> str | None:
    """Give old code-less market rows a useful label from their outcomes."""
    names = [name.strip().casefold() for name, _ in selections]
    normalized = {" ".join(name.replace("-", " ").split()) for name in names}
    if normalized.issubset({"1", "x", "2"}) and len(normalized) >= 2:
        return "Maç Sonucu"
    if normalized.issubset({"ev sahibi", "beraberlik", "deplasman"}) and len(normalized) >= 2:
        return "Maç Sonucu"
    if any("üst" in name or "alt" in name or "over" in name or "under" in name for name in names):
        return "Alt/Üst"
    if any("karşılıklı" in name or "both teams" in name or name in {"kg var", "kg yok"} for name in names):
        return "Karşılıklı Gol"
    if normalized.issubset({"evet", "hayır", "yes", "no"}):
        return "Evet / Hayır"
    return None


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


RECENT_FORM_PRIORS = (2.5, 5.0, 8.0, 12.0, 20.0)
DEFAULT_RECENT_FORM_PRIOR = 5.0
MIN_TUNING_HISTORY = 160
TUNING_VALIDATION_MATCHES = 100
MIN_LOG_LOSS_IMPROVEMENT = 0.005


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


def _expected_calibration_error(pairs: list[tuple[float, float]]) -> float | None:
    """Return ten-bin expected calibration error for forecast/actual pairs."""
    if not pairs:
        return None
    error = 0.0
    for bin_index in range(10):
        bucket = [
            (probability, actual)
            for probability, actual in pairs
            if min(9, int(probability * 10)) == bin_index
        ]
        if bucket:
            error += len(bucket) / len(pairs) * abs(
                sum(probability for probability, _ in bucket) / len(bucket)
                - sum(actual for _, actual in bucket) / len(bucket)
            )
    return error


@st.cache_data(ttl=86_400, max_entries=32, show_spinner=False)
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
    for index in range(first_validation, len(rows)):
        match = rows[index]
        if match.get("home_score") is None or match.get("away_score") is None:
            continue
        home_score, away_score = int(match["home_score"]), int(match["away_score"])
        outcome = 0 if home_score > away_score else 1 if home_score == away_score else 2
        actual_outcomes.append(outcome)
        prior_history = ordered.iloc[:index]
        row = {
            "kickoff": match["match_date"],
            "league_id": match["league_id"],
            "home_team_id": match["home_team_id"],
            "away_team_id": match["away_team_id"],
        }
        for prior in RECENT_FORM_PRIORS:
            means = _expected_goal_means(row, prior_history, prior)
            if means is None:
                predictions[prior].append(
                    {
                        "result": float("nan"),
                        "result_home": float("nan"),
                        "result_draw": float("nan"),
                        "result_away": float("nan"),
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
                predictions[prior].append(
                    {
                        "result": _match_outcome_probabilities(*means)[outcome],
                        "result_home": _match_outcome_probabilities(*means)[0],
                        "result_draw": _match_outcome_probabilities(*means)[1],
                        "result_away": _match_outcome_probabilities(*means)[2],
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


def model_match_forecast(row: dict[str, Any], history: pd.DataFrame) -> dict[str, Any] | None:
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
    tuning = tune_recent_form_prior(league_matches)
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


def quote_selection_model_probability(
    market: dict[str, Any], selection_name: str, forecast: dict[str, Any] | None,
    corners: dict[str, Any] | None,
) -> float | None:
    """Match a quoted selection to an outcome computed by the statistical model."""
    if forecast is None:
        return None
    label = selection_name.casefold().strip().replace("ı", "i").replace("ş", "s").replace("ü", "u").replace("ö", "o").replace("ç", "c").replace("ğ", "g")
    if label == "0":
        label = "x"
    if label.replace(" ", "") in {"1ve0", "0ve1"}:
        label = "1x"
    elif label.replace(" ", "") in {"0ve2", "2ve0"}:
        label = "x2"
    elif label.replace(" ", "") in {"1ve2", "2ve1"}:
        label = "12"
    market_name = str(market.get("source_name") or market.get("name") or "").casefold()
    market_name = market_name.replace("ı", "i").replace("ş", "s").replace("ü", "u").replace("ö", "o").replace("ç", "c").replace("ğ", "g")
    market_context = f"{market_name} {label}"
    if any(word in market_name or word in label for word in ("kart", "card", "sari", "kirmizi", "yari", "half time", "ilk yari")):
        return None
    if any(word in market_context for word in ("korner", "corner")):
        matrix = corners["score_matrix"] if corners else None
    else:
        matrix = forecast["score_matrix"]
    if matrix is None:
        return None

    line_value = market.get("line")
    if line_value in (None, "", 0, "0"):
        match = re.search(r"\d+(?:[.,]\d+)?", str(market.get("name") or ""))
        line_value = match.group(0) if match else None
    try:
        line = float(str(line_value).replace(",", ".")) if line_value is not None else None
    except (TypeError, ValueError):
        line = None

    if any(word in market_name for word in ("karsilikli", "both teams", "kg")):
        probability = forecast["btts"]
        return probability if any(word in label for word in ("var", "evet", "yes")) else 1 - probability if any(word in label for word in ("yok", "hayir", "no")) else None
    if any(word in label.replace(" ", "") for word in ("1x", "x2", "12")):
        code = label.replace(" ", "")
        return {"1x": forecast["home_win"] + forecast["draw"], "x2": forecast["draw"] + forecast["away_win"], "12": forecast["home_win"] + forecast["away_win"]}.get(code)
    outcomes = {
        ("x" if str(item.get("name", "")).casefold().strip() == "0" else str(item.get("name", "")).casefold().strip().replace("ı", "i").replace("ş", "s").replace("ü", "u").replace("ö", "o").replace("ç", "c").replace("ğ", "g"))
        for item in market.get("selections", [])
    }
    if outcomes.issubset({"1", "x", "2"}) and len(outcomes) >= 2 and label in {"1", "x", "2"}:
        return {"1": forecast["home_win"], "x": forecast["draw"], "2": forecast["away_win"]}[label]

    selection_line = re.search(r"(\d+)\s*[-–:]\s*(\d+)", label)
    if selection_line:
        home_score, away_score = map(int, selection_line.groups())
        return matrix[home_score][away_score] if home_score < len(matrix) and away_score < len(matrix[home_score]) else 0.0

    is_over = any(word in label for word in ("ust", "over"))
    is_under = any(word in label for word in ("alt", "under"))
    if line is not None and (is_over or is_under):
        is_corner_market = any(word in market_name for word in ("korner", "corner"))
        team_side = "home" if any(word in market_context for word in ("ev sahibi", "home team", "ev gol", "ev korner", "home corner")) else "away" if any(word in market_context for word in ("deplasman", "away team", "dep gol", "dep korner", "away corner")) else None
        has_goal_context = any(word in market_context for word in ("gol", "goal", "alt/ust", "over/under", "toplam", "takim", "team"))
        if not is_corner_market and team_side is None and not has_goal_context:
            return None
        over = sum(
            matrix[h][a]
            for h in range(len(matrix)) for a in range(len(matrix[h]))
            if (h if team_side == "home" else a if team_side == "away" else h + a) > line
        )
        probability = over if is_over else 1 - over
        return probability
    if line is None and (label in {"evet", "hayir", "yes", "no"}):
        return None
    return None


def render_model_forecast(row: dict[str, Any], history: pd.DataFrame, markets: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    forecast = model_match_forecast(row, history)
    if forecast is None:
        st.info("Bu ligde model tahmini üretmek için yeterli maç geçmişi yok.")
        return None, None
    corners = model_corner_forecast(row, history)
    # Index quoted selections by market category and line for inline display
    # within the corresponding model tables below.
    quote_index: dict[tuple[str, str, str], dict[str, Any]] = {}
    half_quote_index: dict[tuple[int, str, str, str], dict[str, Any]] = {}
    for market in markets:
        for selection in market["selections"]:
            probability = quote_selection_model_probability(
                market, selection["name"], forecast, corners
            )
            market_name = str(market.get("source_name") or market.get("name") or "").casefold()
            label = selection["name"].casefold().strip()
            for source, target in (("ı", "i"), ("ş", "s"), ("ü", "u"), ("ö", "o"), ("ç", "c"), ("ğ", "g")):
                market_name = market_name.replace(source, target)
                label = label.replace(source, target)
            compact_label = label.replace(" ", "")
            label = {"0": "x", "1ve0": "1x", "0ve1": "1x", "0ve2": "x2", "2ve0": "x2", "1ve2": "12", "2ve1": "12"}.get(compact_label, label)
            line_value = market.get("line")
            if line_value in (None, "", 0, "0"):
                match = re.search(r"\d+(?:[.,]\d+)?", f"{market.get('name') or ''} {label}")
                line_value = match.group(0) if match else ""
            line_key = str(line_value).replace(",", ".")
            market_context = f"{market_name} {label}"
            sport_type = str(market.get("sport_type") or "")
            market_id = str(market.get("market_id") or "")
            second_half = any(term in market_context for term in ("2. yari", "2 yari", "ikinci yari", "second half")) or (sport_type == "4" and market_id == "321")
            first_half = any(term in market_context for term in ("1. yari", "1 yari", "ilk yari", "first half", "yari"))
            half_number = 2 if second_half else 1 if first_half or (sport_type == "2" and market_id in {"36", "60", "722", "723"}) else None
            half_category = (
                "result" if market_id in {"36", "321"} or (half_number == 2 and "sonuc" in market_context)
                else "total" if market_id == "60" or (half_number == 2 and any(term in market_context for term in ("alt/ust", "over/under", "toplam")))
                else "home_goals" if market_id == "722"
                else "away_goals" if market_id == "723"
                else None
            )
            if half_number and half_category:
                if half_category == "result":
                    half_outcome = (
                        "1" if label in {"1", "1.", "ev sahibi", "ev sahibi kazanir", "home"}
                        else "x" if label in {"0", "x", "eşit", "esit", "beraberlik", "draw"}
                        else "2" if label in {"2", "2.", "deplasman", "deplasman kazanir", "away"}
                        else label
                    )
                    line_key = ""
                else:
                    half_outcome = "x" if label == "0" else label
                if half_category != "result" and any(term in label for term in ("ust", "over")):
                    half_outcome = "over"
                elif half_category != "result" and any(term in label for term in ("alt", "under")):
                    half_outcome = "under"
                half_quote_index[(half_number, half_category, line_key, half_outcome)] = {
                    "odds": selection["odds"],
                    "market_probability": selection["probability"],
                    "market": market["name"],
                }
                continue
            if (sport_type, market_id) in {("4", "203"), ("2", "603")}:
                category = "home_goals"
            elif (sport_type, market_id) in {("4", "207"), ("2", "604")}:
                category = "away_goals"
            elif any(term in market_context for term in ("korner", "corner")):
                category = "home_corners" if any(term in market_context for term in ("ev sahibi", "ev korner", "home team", "home corner")) else "away_corners" if any(term in market_context for term in ("deplasman", "dep korner", "away team", "away corner")) else "corners"
            elif any(term in market_context for term in ("ev sahibi", "home team", "ev gol")):
                category = "home_goals"
            elif any(term in market_context for term in ("deplasman", "away team", "dep gol")):
                category = "away_goals"
            elif any(term in market_context for term in ("karsilikli", "both teams", "kg")):
                category = "btts"
            elif re.search(r"\d+\s*[-–:]\s*\d+", label):
                category, line_key = "score", ""
                outcome = re.sub(r"\s*[-–:]\s*", "-", label)
                quote_index[(category, line_key, outcome)] = {
                    "odds": selection["odds"],
                    "market_probability": selection["probability"],
                    "model_probability": probability,
                    "market": market["name"],
                }
                continue
            elif any(term in label.replace(" ", "") for term in ("1x", "x2", "12")):
                category, line_key = "double_chance", label.replace(" ", "")
            elif label in {"1", "x", "2"}:
                category, line_key = "result", label
            elif any(term in market_context for term in ("gol", "goal", "alt/ust", "over/under", "toplam")):
                category = "total_goals"
            else:
                continue
            outcome = "over" if any(term in label for term in ("ust", "over")) else "under" if any(term in label for term in ("alt", "under")) else "yes" if any(term in label for term in ("var", "evet", "yes")) else "no" if any(term in label for term in ("yok", "hayir", "no")) else label
            quote_index[(category, line_key, outcome)] = {
                "odds": selection["odds"],
                "market_probability": selection["probability"],
                "model_probability": probability,
                "market": market["name"],
            }

    def quote_columns(category: str, line: float | str, outcome: str) -> dict[str, str]:
        quote = quote_index.get((category, str(line), outcome))
        if not quote:
            return {"Oran": "—", "Piyasa %": "—"}
        return {
            "Oran": f"{quote['odds']:.2f}",
            "Piyasa %": f"%{quote['market_probability'] * 100:.1f}",
        }

    total_table, team_table, outcome_table = market_probability_tables(forecast)
    with st.expander("🤖 Tüm maç tahminleri ve karşılık gelen oranlar", expanded=True):
        st.markdown("#### Model tahmini · Form Poisson")
        st.metric("En olası skor", forecast["most_likely_score"])
        if forecast["tuning_matches"] and forecast["tuning_log_loss"] is not None:
            tuning_note = (
                f"Form ağırlığı {forecast['tuning_selection_matches']} eski maçta seçildi; "
                f"sonraki {forecast['tuning_matches']} maçta 1X2 log-loss "
                f"{forecast['tuning_log_loss']:.3f} (sabit ağırlık: "
                f"{forecast['tuning_baseline_log_loss']:.3f}; düşük değer daha iyi). "
                f"Uygulanan önsel: {forecast['recent_form_prior']:g}. "
            )
        else:
            tuning_note = "Ayrı değerlendirme için yeterli geçmiş maç bulunmadığından varsayılan form ağırlığı kullanılıyor. "
        st.caption(
            f"Bağımsız Poisson modeli · lig ortalaması ve takımların aynı ligdeki son maçları. "
            f"Lig örneklemi: {forecast['history_count']} maç; takım örneklemi: "
            f"{forecast['home_sample']} / {forecast['away_sample']} maç. "
            f"Beklenen gol: {forecast['expected_home']:.2f}–{forecast['expected_away']:.2f}. "
            f"{tuning_note}"
            "Oranlardan türetilmemiştir; olasılıklar istatistiksel tahmindir."
        )
        if forecast["tuning_market_metrics"]:
            market_labels = {
                "result": "Maç sonucu (1X2)",
                "over_2_5": "Toplam gol 2.5 üst",
                "btts": "Karşılıklı gol var",
            }
            evaluation_rows = []
            for market, metrics in forecast["tuning_market_metrics"].items():
                evaluation_rows.append({
                    "Market": market_labels[market],
                    "Örnek": metrics["sample_size"],
                    "Model log-loss": metrics["model_log_loss"],
                    "Sabit log-loss": metrics["baseline_log_loss"],
                    "Model Brier": metrics["model_brier_score"],
                    "Sabit Brier": metrics["baseline_brier_score"],
                    "Model kalibrasyon": metrics["model_calibration_error"],
                    "Sabit kalibrasyon": metrics["baseline_calibration_error"],
                })
            with st.expander("Geçmiş maçlarda zaman sıralı model değerlendirmesi"):
                st.caption(
                    "Ağırlıklar erken dönemden seçildi; aşağıdaki log-loss daha sonraki, "
                    "seçimde kullanılmayan maçlarda hesaplandı. Brier ve 10 dilimli kalibrasyon "
                    "hatası da gösterilir (düşük değer daha iyi). Lig bazlıdır; küçük örneklem "
                    "tek başına modelin gelecekte daha başarılı olacağını göstermez."
                )
                st.dataframe(
                    pd.DataFrame(evaluation_rows),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Model log-loss": st.column_config.NumberColumn(format="%.3f"),
                        "Sabit log-loss": st.column_config.NumberColumn(format="%.3f"),
                        "Model Brier": st.column_config.NumberColumn(format="%.3f"),
                        "Sabit Brier": st.column_config.NumberColumn(format="%.3f"),
                        "Model kalibrasyon": st.column_config.NumberColumn(format="%.3f"),
                        "Sabit kalibrasyon": st.column_config.NumberColumn(format="%.3f"),
                    },
                )
        st.markdown("#### Tahminler ve karşılık gelen oranlar")
        st.markdown("**Tüm tahminler**")
        total_rows = []
        for line in (0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5):
            over = sum(forecast["score_matrix"][h][a] for h in range(len(forecast["score_matrix"])) for a in range(len(forecast["score_matrix"][h])) if h + a > line)
            for outcome, probability in (("over", over), ("under", 1 - over)):
                total_rows.append({"Pazar": f"Maç Sonucu Alt/Üst · {line}", "Seçim": "Üst" if outcome == "over" else "Alt", "Model tahmini": probability, **quote_columns("total_goals", line, outcome)})
        for key, label in (("1", "1 · Ev sahibi"), ("x", "X · Beraberlik"), ("2", "2 · Deplasman")):
            probability = {"1": forecast["home_win"], "x": forecast["draw"], "2": forecast["away_win"]}[key]
            total_rows.append({"Pazar": "Maç Sonucu", "Seçim": label, "Model tahmini": probability, **quote_columns("result", key, key)})
        for key, probability in (("1x", forecast["home_win"] + forecast["draw"]), ("x2", forecast["draw"] + forecast["away_win"]), ("12", forecast["home_win"] + forecast["away_win"])):
            total_rows.append({"Pazar": "Çifte Şans", "Seçim": key.upper(), "Model tahmini": probability, **quote_columns("double_chance", key, key)})
        for outcome, probability in (("yes", forecast["btts"]), ("no", 1 - forecast["btts"])):
            total_rows.append({"Pazar": "Karşılıklı Gol", "Seçim": "Var" if outcome == "yes" else "Yok", "Model tahmini": probability, **quote_columns("btts", "", outcome)})
        def render_half_table(half: int, half_label: str) -> None:
            half_forecast = model_half_forecast(row, history, half=half)
            if half_forecast is None:
                st.caption(f"{half_label}: en az 20 geçmiş devre skoru olmadığı için tahmin üretilemedi.")
                return
            st.caption(
                f"Yarıya özel Poisson modeli · {half_forecast['history_count']} lig maçı; "
                f"beklenen gol {half_forecast['expected_home']:.2f}–{half_forecast['expected_away']:.2f}."
            )
            def add_row(category: str, line: str, outcome: str, label: str, probability: float) -> None:
                quote = half_quote_index.get((half, category, line, outcome))
                total_rows.append({
                    "Pazar": f"{half_label} · {label}",
                    "Seçim": "Üst" if outcome == "over" else "Alt" if outcome == "under" else {"1": "1 · Ev sahibi", "x": "X · Beraberlik", "2": "2 · Deplasman"}.get(outcome, outcome),
                    "Model tahmini": probability,
                    "Oran": f"{quote['odds']:.2f}" if quote else "—",
                    "Piyasa %": f"%{quote['market_probability'] * 100:.1f}" if quote else "—",
                })

            matrix = half_forecast["score_matrix"]
            for key, label, probability in (
                ("1", "Yarı sonucu", half_forecast["home_win"]),
                ("x", "Yarı sonucu", half_forecast["draw"]),
                ("2", "Yarı sonucu", half_forecast["away_win"]),
            ):
                add_row("result", "", key, label, probability)
            for line in (0.5, 1.5, 2.5):
                over = sum(matrix[h][a] for h in range(len(matrix)) for a in range(len(matrix[h])) if h + a > line)
                for outcome, probability in (("over", over), ("under", 1 - over)):
                    add_row("total", str(line), outcome, f"Yarı toplam gol · {line}", probability)
            for line in (0.5, 1.5):
                for side, category, prefix in (("home", "home_goals", "Ev"), ("away", "away_goals", "Dep")):
                    over = sum(matrix[h][a] for h in range(len(matrix)) for a in range(len(matrix[h])) if (h if side == "home" else a) > line)
                    for outcome, probability in (("over", over), ("under", 1 - over)):
                        add_row(category, str(line), outcome, f"Takım golü · {line} · {prefix}", probability)

        render_half_table(1, "1. Yarı")
        render_half_table(2, "2. Yarı")

        matrix = forecast["score_matrix"]
        for line in (0.5, 1.5, 2.5, 3.5):
            for side, prefix in (("home", "Ev"), ("away", "Dep")):
                over = sum(matrix[h][a] for h in range(len(matrix)) for a in range(len(matrix[h])) if (h if side == "home" else a) > line)
                category = "home_goals" if side == "home" else "away_goals"
                for outcome, probability in (("over", over), ("under", 1 - over)):
                    total_rows.append({"Pazar": f"Takım golü · {line} · {prefix}", "Seçim": "Üst" if outcome == "over" else "Alt", "Model tahmini": probability, **quote_columns(category, line, outcome)})

        if corners is None:
            st.caption("Korner tahmini için bu ligde yeterli tamamlanmış korner istatistiği bulunmuyor.")
            for (category, line, outcome), quote in quote_index.items():
                if category not in {"corners", "home_corners", "away_corners"}:
                    continue
                total_rows.append({
                    "Pazar": f"{quote['market']} · {line}",
                    "Seçim": "Üst" if outcome == "over" else "Alt" if outcome == "under" else outcome,
                    "Model tahmini": None, "Oran": f"{quote['odds']:.2f}",
                    "Piyasa %": f"%{quote['market_probability'] * 100:.1f}",
                })
        else:
            for line in (6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5):
                over = sum(corners["score_matrix"][h][a] for h in range(len(corners["score_matrix"])) for a in range(len(corners["score_matrix"][h])) if h + a > line)
                for outcome, probability in (("over", over), ("under", 1 - over)):
                    total_rows.append({"Pazar": f"Toplam korner · {line}", "Seçim": "Üst" if outcome == "over" else "Alt", "Model tahmini": probability, **quote_columns("corners", line, outcome)})
            for line in (2.5, 3.5, 4.5, 5.5, 6.5):
                for side, prefix in (("home", "Ev"), ("away", "Dep")):
                    over = sum(corners["score_matrix"][h][a] for h in range(len(corners["score_matrix"])) for a in range(len(corners["score_matrix"][h])) if (h if side == "home" else a) > line)
                    for outcome, probability in (("over", over), ("under", 1 - over)):
                        total_rows.append({"Pazar": f"Takım korneri · {line} · {prefix}", "Seçim": "Üst" if outcome == "over" else "Alt", "Model tahmini": probability, **quote_columns(f"{side}_corners", line, outcome)})
            st.caption(f"Korner Poisson modeli · {corners['history_count']} lig maçı; beklenen korner: {corners['home_expected']:.2f}–{corners['away_expected']:.2f}.")

        matrix = forecast["score_matrix"]
        scores = sorted(((matrix[h][a], h, a) for h in range(len(matrix)) for a in range(len(matrix[h]))), reverse=True)[:5]
        for probability, home, away in scores:
            score_key = f"{home}-{away}"
            score_quote = quote_index.get(("score", "", score_key), {})
            total_rows.append({"Pazar": "Doğru skor", "Seçim": score_key.replace("-", "–"), "Model tahmini": probability, "Oran": f"{score_quote['odds']:.2f}" if score_quote else "—", "Piyasa %": f"%{score_quote['market_probability'] * 100:.1f}" if score_quote else "—"})

        st.dataframe(
            pd.DataFrame(total_rows),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Model tahmini": st.column_config.NumberColumn(
                    "Model tahmini",
                    format="percent",
                )
            },
        )

        st.markdown("**Veri kapsamı**")
        st.info("Maç sonu ve yarı tahminleri geçmiş skor dağılımlarından hesaplanır. Yarı modelleri, ilgili ligde en az 20 devre skoru kaydı oluşunca devreye girer; kart istatistikleri arşivde bulunmadığı için kart tahmini üretilmez.")
    return forecast, corners


def render_recent_form(
    row: dict[str, Any], history: pd.DataFrame, team_names: dict[Any, str]
) -> None:
    """Show recent form and match logs for both teams without implying a forecast."""
    kickoff = row["kickoff"]
    home_id, away_id = row["home_team_id"], row["away_team_id"]
    home_recent = team_recent_matches(history, home_id, kickoff, row["league_id"])
    away_recent = team_recent_matches(history, away_id, kickoff, row["league_id"])
    home_stats, away_stats = summarize_team_form(home_recent, home_id), summarize_team_form(away_recent, away_id)
    st.markdown("#### Maç öncesi form · son 5 tamamlanmış maç")
    st.caption("Özet yalnızca başlama saatinden önce tamamlanmış maçları kullanır. Bu bölüm geçmiş performansı gösterir, tahmin değildir.")
    home_col, away_col = st.columns(2)
    for col, name, stats, recent, team_id in (
        (home_col, row["home_name"], home_stats, home_recent, home_id),
        (away_col, row["away_name"], away_stats, away_recent, away_id),
    ):
        with col:
            st.markdown(f"**{name}**")
            if recent.empty:
                st.info("Bu takım için maç öncesi tamamlanmış geçmiş maç kaydı yok.")
                continue
            st.metric("Form (yeniden eskiye)", stats["form"], f"{len(recent)} maç")
            a, b = st.columns(2)
            a.metric("Maç başı gol", f"{stats['goals_for']:.2f} attı · {stats['goals_against']:.2f} yedi")
            b.metric("Maç başı puan", f"{stats['points']:.2f} / 3")
            if stats["xg_for"] is not None:
                c, d = st.columns(2)
                c.metric("Ortalama xG", f"{stats['xg_for']:.2f}")
                d.metric("Rakibe verilen xG", f"{stats['xg_against']:.2f}" if stats["xg_against"] is not None else "Veri yok")
            log = []
            for match in recent.to_dict("records"):
                is_home = int(match["home_team_id"]) == int(team_id)
                opponent_id = match["away_team_id"] if is_home else match["home_team_id"]
                scored = match["home_score"] if is_home else match["away_score"]
                conceded = match["away_score"] if is_home else match["home_score"]
                result = "G" if scored > conceded else "B" if scored == conceded else "M"
                match_date = pd.to_datetime(match["match_date"], utc=True, errors="coerce")
                log.append({
                    "Tarih": match_date.tz_convert(ISTANBUL).strftime("%d.%m.%Y") if pd.notna(match_date) else "—",
                    "Saha": "Ev" if is_home else "Dep",
                    "Rakip": team_names.get(opponent_id, "Rakip"),
                    "Skor": f"{scored}–{conceded}",
                    "Sonuç": result,
                })
            with st.expander(f"{name}: son maçların dökümü"):
                st.dataframe(pd.DataFrame(log), use_container_width=True, hide_index=True)


def render_analysis(matches_data: list[dict[str, Any]], quotes_data: list[dict[str, Any]], history_data: list[dict[str, Any]], teams: pd.DataFrame, leagues: pd.DataFrame, match_day: date, market_labels: dict[str, str]) -> None:
    matches = pd.DataFrame(matches_data)
    if matches.empty:
        st.info(f"{match_day:%d.%m.%Y} için kayıtlı maç bulunamadı.")
        return
    team_names = dict(zip(teams.get("id", []), teams.get("name", []))) if not teams.empty else {}
    league_names = dict(zip(leagues.get("id", []), leagues.get("name", []))) if not leagues.empty else {}
    league_countries = dict(zip(leagues.get("id", []), leagues.get("country", []))) if not leagues.empty else {}
    latest: dict[int, dict[str, Any]] = {}
    for quote in sorted(quotes_data, key=lambda q: str(q.get("captured_at", ""))):
        try:
            latest[int(quote["match_id"])] = quote
        except (KeyError, TypeError, ValueError):
            continue

    matches["home_name"] = matches["home_team_id"].map(team_names).fillna("Ev sahibi")
    matches["away_name"] = matches["away_team_id"].map(team_names).fillna("Deplasman")
    matches["league_name"] = matches["league_id"].map(league_names).fillna("Lig bilgisi yok")
    matches["league_name"] = matches["league_name"].map(lambda name: LEAGUE_NAMES_TR.get(name, name))
    matches["country_name"] = matches["league_id"].map(league_countries).fillna("").map(lambda country: COUNTRY_NAMES_TR.get(country, country))
    matches["league_label"] = matches.apply(lambda row: f"{row['country_name']} · {row['league_name']}" if row["country_name"] else row["league_name"], axis=1)
    matches["kickoff"] = pd.to_datetime(matches["match_date"], utc=True, errors="coerce").dt.tz_convert(ISTANBUL)
    now = pd.Timestamp.now(tz=ISTANBUL)
    live_or_finished = matches["status"].fillna("").astype(str).str.casefold().isin({"live", "finished"})
    kickoff_reached = matches["kickoff"].notna() & (matches["kickoff"] <= now)
    matches = matches[~(live_or_finished | kickoff_reached)].copy()
    history = pd.DataFrame(history_data)
    if not history.empty:
        history["match_date"] = pd.to_datetime(history["match_date"], utc=True, errors="coerce")
        for column in ("id", "league_id", "home_team_id", "away_team_id", "home_score", "away_score", "home_halftime_score", "away_halftime_score", "home_corners", "away_corners"):
            history[column] = pd.to_numeric(history[column], errors="coerce")

    left, right, third = st.columns([1, 1, 2])
    with left:
        league_filter = st.selectbox("Lig", ["Tüm ligler", *sorted(matches["league_label"].unique())])
    with right:
        coverage = st.selectbox("Oran durumu", ["Tüm maçlar", "Oranı olanlar", "Oranı olmayanlar"])
    with third:
        search = st.text_input("Takım ara", placeholder="Takım adı yazın")
    filtered = matches.copy()
    if league_filter != "Tüm ligler":
        filtered = filtered[filtered["league_label"] == league_filter]
    if search.strip():
        q = search.strip().casefold()
        filtered = filtered[filtered["home_name"].str.casefold().str.contains(q) | filtered["away_name"].str.casefold().str.contains(q)]
    if coverage == "Oranı olanlar":
        filtered = filtered[filtered["id"].map(lambda x: int(x) in latest)]
    elif coverage == "Oranı olmayanlar":
        filtered = filtered[~filtered["id"].map(lambda x: int(x) in latest)]

    daily_match_ids = {int(match_id) for match_id in matches["id"]}
    quoted_ids = {int(row["match_id"]) for row in quotes_data if row.get("match_id") is not None} & daily_match_ids
    market_count = sum(
        len(unpack_payload(quote.get("odds")).get("markets", []))
        for match_id, quote in latest.items()
        if match_id in daily_match_ids
    )
    a, b, c, d = st.columns(4)
    a.metric("Listelenen maç", f"{len(filtered)}")
    b.metric("Oranı bulunan maç", f"{len(quoted_ids)} / {len(matches)}")
    c.metric("Oran kaydı olmayan maç", f"{len(matches) - len(quoted_ids)}")
    d.metric("Mevcut market satırı", f"{market_count}")
    capture_times = pd.to_datetime(
        [quote.get("captured_at") for quote in quotes_data], utc=True, errors="coerce"
    ).dropna()
    if not capture_times.empty:
        last_capture = capture_times.max().tz_convert(ISTANBUL).strftime("%d.%m.%Y %H:%M")
        st.caption(
            f"Seçili gün için son oran kaydı: {last_capture} (İstanbul). "
            "Oran kaydı olmayan maç kaynakta bulunmuyor veya henüz eşleşmemiş olabilir."
        )
    else:
        st.caption("Seçili gün için henüz oran kaydı yok; maçlar kaynakta bulunmuyor veya eşleşmemiş olabilir.")
    st.caption("Piyasa yüzdeleri, market içindeki IdDaa oranlarının marjdan arındırılmış karşılığıdır; bağımsız model tahmini değildir.")
    st.divider()

    if filtered.empty:
        st.info("Bu filtrelerle eşleşen maç yok.")
        return
    for row in filtered.sort_values("kickoff", na_position="last").to_dict("records"):
        match_id = int(row["id"])
        quote = latest.get(match_id)
        kickoff = row["kickoff"]
        time_text = kickoff.strftime("%H:%M") if pd.notna(kickoff) else "Saat bilgisi yok"
        with st.expander(f"{time_text}  ·  {row['home_name']} — {row['away_name']}  ·  {row['league_label']}", expanded=False):
            meta, status_col = st.columns([4, 1])
            meta.caption(row["league_name"])
            status_col.caption(str(row.get("status", "")).replace("_", " ").title())
            payload = unpack_payload(quote.get("odds")) if quote else {}
            markets = market_rows(payload, market_labels) if quote else []
            model_forecast, corner_forecast = render_model_forecast(row, history, markets)
            st.divider()
            render_recent_form(row, history, team_names)
            st.divider()
            if not quote:
                st.warning("Bu maç için oran kaydı yok. Kaynakta bulunmuyor veya maç eşleşmesi henüz yapılamamış olabilir.")
                continue
            if not markets:
                st.warning("Oran kaydı var ancak gösterilebilir tam market bulunamadı.")
                continue
            st.caption(f"Son oran kaydı: {quote.get('captured_at', 'bilinmiyor')}")


def render_archive() -> None:
    try:
        matches = pd.DataFrame(load_completed_match_history())
        teams, leagues = load_reference_catalog()
    except Exception as exc:
        st.error(f"Maç istatistikleri şu anda yüklenemedi: {exc}")
        return
    if matches.empty:
        st.info("Kayıtlı tamamlanmış maç bulunamadı.")
        return
    home = teams.rename(columns={"id": "home_team_id", "name": "Ev sahibi"})[["home_team_id", "Ev sahibi"]]
    away = teams.rename(columns={"id": "away_team_id", "name": "Deplasman"})[["away_team_id", "Deplasman"]]
    league_labels = leagues.rename(columns={"id": "league_id", "name": "Lig", "country": "Ülke"})[["league_id", "Lig", "Ülke"]]
    matches = matches.merge(home, on="home_team_id", how="left").merge(away, on="away_team_id", how="left").merge(league_labels, on="league_id", how="left")
    matches["Tarih"] = pd.to_datetime(matches["match_date"], utc=True, errors="coerce").dt.tz_convert(ISTANBUL)
    matches["Skor"] = matches["home_score"].astype("Int64").astype(str) + " - " + matches["away_score"].astype("Int64").astype(str)
    c1, c2 = st.columns(2)
    with c1:
        selected = st.multiselect("Lig", sorted(matches["Lig"].dropna().unique()))
    with c2:
        query = st.text_input("Takım ara", key="archive_search", placeholder="Takım adı")
    if selected:
        matches = matches[matches["Lig"].isin(selected)]
    if query.strip():
        q = query.casefold().strip()
        matches = matches[matches["Ev sahibi"].fillna("").str.casefold().str.contains(q) | matches["Deplasman"].fillna("").str.casefold().str.contains(q)]
    columns = ["Tarih", "Ülke", "Lig", "Ev sahibi", "Skor", "Deplasman", "home_xg", "away_xg", "home_xa", "away_xa", "home_corners", "away_corners"]
    labels = {"home_xg": "Ev xG", "away_xg": "Dep xG", "home_xa": "Ev xA", "away_xa": "Dep xA", "home_corners": "Ev korner", "away_corners": "Dep korner"}
    st.dataframe(matches.sort_values("Tarih", ascending=False)[columns].rename(columns=labels), use_container_width=True, hide_index=True)


st.markdown("<section class='hero'><div class='hero-kicker'>Günlük karşılaşmalar · iddaa marketleri</div><h1>Maç Merkezi</h1><p>Maçları inceleyin, marketleri karşılaştırın ve maç istatistikleri arşivine ulaşın.</p></section>", unsafe_allow_html=True)
today = datetime.now(ISTANBUL).date()
col, refresh_col, _ = st.columns([1, 1, 2])
with col:
    selected_day = st.date_input("Maç günü", value=today, format="DD.MM.YYYY")
with refresh_col:
    st.write("")
    if st.button("Verileri şimdi yenile", use_container_width=True):
        load_reference_catalog.clear()
        load_match_day.clear()
        load_completed_match_history.clear()
        load_iddaa_market_labels.clear()
        st.rerun()
tab_analysis, tab_archive = st.tabs(["⚽  Maç Analizleri", "📊  İstatistik Arşivi"])
try:
    teams, leagues = load_reference_catalog()
except Exception as exc:
    st.error(f"Takım ve lig bilgileri yüklenemedi: {exc}")
    teams, leagues = pd.DataFrame(), pd.DataFrame()
with tab_analysis:
    try:
        match_records, quote_records = load_match_day(selected_day)
        try:
            analysis_records = load_completed_match_history()
        except Exception as exc:
            analysis_records = []
            st.warning(f"Geçmiş maçlar yüklenemedi; model tahmini ve form analizi gösterilemiyor: {exc}")
        try:
            market_labels = load_iddaa_market_labels()
        except Exception:
            market_labels = {}
        render_analysis(match_records, quote_records, analysis_records, teams, leagues, selected_day, market_labels)
    except Exception as exc:
        st.error(f"Maç ve oran verileri yüklenemedi: {exc}")
with tab_archive:
    render_archive()
