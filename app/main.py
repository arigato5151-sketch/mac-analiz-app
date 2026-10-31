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

from app.odds_quality import total_market_key
from app.market_mapping import is_compound_selection, standard_market_price, total_goals_outcome
from app.ai_match_analysis import (
    MatchAnalysisAIError,
    build_match_analysis_context,
    generate_match_analysis,
    generate_match_risk_recommendations,
)
from config.settings import get_optional_setting
from app.components.data import (
    load_completed_match_history,
    get_db,
    load_league_match_history,
    load_match_day,
    load_recent_team_matches,
    load_reference_catalog,
)

ISTANBUL = ZoneInfo("Europe/Istanbul")
# Bump this when forecast logic or the persisted analysis JSON contract changes.
ANALYSIS_MODEL_VERSION = "form_poisson_ai_v2"
NON_PLAYABLE_MATCH_STATUSES = {"postponed", "cancelled", "unlisted"}

COUNTRY_NAMES_TR = {
    "Argentina": "Arjantin", "Austria": "Avusturya", "Belgium": "Belçika",
    "Brazil": "Brezilya", "England": "İngiltere", "France": "Fransa",
    "Germany": "Almanya", "Italy": "İtalya", "Netherlands": "Hollanda",
    "Portugal": "Portekiz", "Russia": "Rusya", "Saudi-Arabia": "Suudi Arabistan",
    "Spain": "İspanya", "Switzerland": "İsviçre", "Turkey": "Türkiye",
    "USA": "ABD", "World": "Uluslararası",
}
LEAGUE_NAMES_TR = {
    "Premier League": "Premier Lig", "La Liga": "LaLiga", "Serie A": "Serie A",
    "Bundesliga": "Bundesliga", "Ligue 1": "Ligue 1", "UEFA Champions League": "UEFA Şampiyonlar Ligi",
    "UEFA Europa League": "UEFA Avrupa Ligi", "UEFA Europa Conference League": "UEFA Konferans Ligi",
    "Süper Lig": "Süper Lig", "Primeira Liga": "Portekiz Ligi", "Eredivisie": "Hollanda Ligi",
    "Jupiler Pro League": "Belçika Pro Ligi", "Championship": "İngiltere Championship",
    "Pro League": "Suudi Arabistan Pro Ligi", "Major League Soccer": "MLS",
    "Premier Liga": "Rusya Premier Ligi", "Liga Profesional": "Arjantin Profesyonel Ligi",
    "Swiss Super League": "İsviçre Süper Ligi",
    "FIFA World Cup": "FIFA Dünya Kupası", "Euro Championship": "Avrupa Şampiyonası",
    "UEFA Nations League": "UEFA Uluslar Ligi", "Africa Cup of Nations": "Afrika Uluslar Kupası",
    "Asian Cup": "Asya Kupası", "Copa America": "Copa América", "CONCACAF Gold Cup": "CONCACAF Altın Kupa",
}

if __name__ == "__main__":
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


@st.cache_data(ttl=900, show_spinner="Tahmin performansı yükleniyor…")
def load_evaluated_prediction_results() -> list[dict[str, Any]]:
    """Read evaluated forecasts without requiring a newly imported helper module."""
    return get_db().select_all(
        "evaluated_prediction_results",
        columns=(
            "match_id,match_date,predicted_at,model_version,actual_result,was_correct,brier_score,"
            "prob_home_win,prob_draw,prob_away_win,prob_over_2_5,over_2_5_actual,"
            "over_2_5_was_correct,over_2_5_brier_score,prob_btts,btts_actual,"
            "btts_was_correct,btts_brier_score,market_probabilities,market_performance"
        ),
        order="match_date.asc,match_id.asc",
    )


def market_rows(payload: dict[str, Any], labels: dict[str, str]) -> list[dict[str, Any]]:
    rows = []
    for market in payload.get("markets", []):
        selections = []
        for selection in market.get("selections", []):
            try:
                odd = standard_market_price(selection)
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


def all_market_quote_rows(
    payload: dict[str, Any],
    labels: dict[str, str],
    inconsistent_keys: set[tuple[str, str, str]] | None = None,
) -> list[dict[str, Any]]:
    """Build a complete IdDaa odds table, including markets outside model coverage."""
    rows: list[dict[str, Any]] = []
    for market in payload.get("markets", []):
        code = market.get("market_id")
        sport_type = market.get("sport_type")
        key = f"{sport_type if sport_type is not None else ''}:{code if code is not None else ''}"
        name = str(labels.get(key) or labels.get(str(code or "")) or market.get("market_name") or f"Market {code or ''}")
        if market.get("line") not in (None, "", 0, "0"):
            name = f"{name} · {market['line']}"
        selections = []
        for selection in market.get("selections", []):
            try:
                odd = standard_market_price(selection)
            except (TypeError, ValueError):
                continue
            if odd > 1:
                selections.append((str(selection.get("name") or selection.get("id") or "Seçim"), odd))
        implied = [1 / odd for _, odd in selections]
        total = sum(implied)
        for (selection_name, odd), probability in zip(selections, implied):
            row = {
                "Market": name,
                "Seçim": selection_name,
                "Oran": f"{odd:.2f}",
                "Piyasa %": f"%{probability / total * 100:.1f}" if len(selections) > 1 and total else "—",
            }
            if total_market_key(market) in (inconsistent_keys or set()):
                row["Veri kontrolü"] = "Çizgilerle tutarsız; doğrulanana kadar kullanmayın"
            rows.append(row)
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


def render_model_forecast(
    row: dict[str, Any],
    history: pd.DataFrame,
    markets: list[dict[str, Any]],
    *,
    precomputed: dict[str, Any] | None = None,
    show_details: bool = True,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    forecast = (
        precomputed.get("forecast")
        if precomputed is not None
        else model_match_forecast(row, history)
    )
    if forecast is None:
        st.info("Bu ligde model tahmini üretmek için yeterli maç geçmişi yok.")
        return None, None
    corners = (
        precomputed.get("corners")
        if precomputed is not None
        else model_corner_forecast(row, history)
    )
    if not show_details:
        return forecast, corners
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
            if is_compound_selection(label):
                # A combined outcome such as "Üst ve Yok" is not the pure
                # "Üst" selection and must never overwrite its quote.
                continue
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
                if half_category != "result" and label in {"ust", "over"}:
                    half_outcome = "over"
                elif half_category != "result" and label in {"alt", "under"}:
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
            elif (
                any(term in market_context for term in ("gol", "goal", "alt/ust", "over/under", "toplam"))
                and total_goals_outcome(label) is not None
            ):
                category = "total_goals"
            else:
                continue
            outcome = (
                total_goals_outcome(label)
                or ("yes" if label in {"var", "evet", "yes"}
                else "no" if label in {"yok", "hayir", "no"}
                else label)
            )
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
                    width="stretch",
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
                total_rows.append({"Pazar": f"Toplam Gol Alt/Üst · {line}", "Seçim": "Üst" if outcome == "over" else "Alt", "Model tahmini": probability, **quote_columns("total_goals", line, outcome)})
        for key, label in (("1", "1 · Ev sahibi"), ("x", "X · Beraberlik"), ("2", "2 · Deplasman")):
            probability = {"1": forecast["home_win"], "x": forecast["draw"], "2": forecast["away_win"]}[key]
            total_rows.append({"Pazar": "Maç Sonucu", "Seçim": label, "Model tahmini": probability, **quote_columns("result", key, key)})
        for key, probability in (("1x", forecast["home_win"] + forecast["draw"]), ("x2", forecast["draw"] + forecast["away_win"]), ("12", forecast["home_win"] + forecast["away_win"])):
            total_rows.append({"Pazar": "Çifte Şans", "Seçim": key.upper(), "Model tahmini": probability, **quote_columns("double_chance", key, key)})
        for outcome, probability in (("yes", forecast["btts"]), ("no", 1 - forecast["btts"])):
            total_rows.append({"Pazar": "Karşılıklı Gol", "Seçim": "Var" if outcome == "yes" else "Yok", "Model tahmini": probability, **quote_columns("btts", "", outcome)})
        def render_half_table(half: int, half_label: str) -> None:
            half_forecast = (
                precomputed.get("halves", {}).get(str(half))
                if precomputed is not None
                else model_half_forecast(row, history, half=half)
            )
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
            width="stretch",
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
                st.dataframe(pd.DataFrame(log), width="stretch", hide_index=True)


def build_precomputed_match_analysis(
    row: dict[str, Any],
    history: pd.DataFrame,
    recent_form_tuning: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build display-ready match analysis without Streamlit rendering calls."""
    kickoff = row["kickoff"]
    league_id = int(row["league_id"])
    home_recent = team_recent_matches(history, row["home_team_id"], kickoff, league_id)
    away_recent = team_recent_matches(history, row["away_team_id"], kickoff, league_id)
    history_by_id = {
        int(match["id"]): match
        for match in (*home_recent.to_dict("records"), *away_recent.to_dict("records"))
    }
    return {
        "history": list(history_by_id.values()),
        "forecast": model_match_forecast(row, history, recent_form_tuning),
        "corners": model_corner_forecast(row, history),
        "halves": {
            str(half): model_half_forecast(row, history, half=half)
            for half in (1, 2)
        },
    }


def _cached_analysis_for_match(
    record: dict[str, Any] | None, row: dict[str, Any]
) -> dict[str, Any] | None:
    if not record or record.get("model_version") != ANALYSIS_MODEL_VERSION:
        return None
    try:
        if any(
            int(record.get(key, -1)) != int(row[key])
            for key in ("league_id", "home_team_id", "away_team_id")
        ):
            return None
        cached_date = pd.to_datetime(record.get("match_date"), utc=True, errors="coerce")
        fixture_date = pd.to_datetime(row.get("match_date"), utc=True, errors="coerce")
        if pd.isna(cached_date) or pd.isna(fixture_date) or cached_date != fixture_date:
            return None
    except (KeyError, TypeError, ValueError):
        return None
    analysis = record.get("analysis_json")
    if (
        not isinstance(analysis, dict)
        or not isinstance(analysis.get("history"), list)
        or not isinstance(analysis.get("halves"), dict)
    ):
        return None
    return analysis


def render_tactical_match_readout(
    row: dict[str, Any], history: pd.DataFrame, forecast: dict[str, Any] | None,
    precomputed_risks: dict[str, Any] | None = None,
) -> None:
    """Summarize each fixture from its pre-match form and independent goal model."""
    st.markdown("#### Taktik eşleşme ve senaryolar")
    if forecast is None:
        st.info("Bu eşleşme için yeterli maç geçmişi bulunmadığından veriye dayalı senaryo üretilemedi.")
        return

    home_id, away_id = row["home_team_id"], row["away_team_id"]
    home_recent = team_recent_matches(history, home_id, row["kickoff"], row["league_id"])
    away_recent = team_recent_matches(history, away_id, row["kickoff"], row["league_id"])
    home_form = summarize_team_form(home_recent, home_id)
    away_form = summarize_team_form(away_recent, away_id)
    home_xg, away_xg = forecast["expected_home"], forecast["expected_away"]
    home_p, draw_p, away_p = forecast["home_win"], forecast["draw"], forecast["away_win"]
    over_15 = sum(
        forecast["score_matrix"][h][a]
        for h in range(len(forecast["score_matrix"]))
        for a in range(len(forecast["score_matrix"][h]))
        if h + a >= 2
    )
    btts_no = 1 - forecast["btts"]

    home_name, away_name = row["home_name"], row["away_name"]
    recent_goal_diff = (
        (home_form["goals_for"] - home_form["goals_against"])
        - (away_form["goals_for"] - away_form["goals_against"])
        if home_form["goals_for"] is not None and away_form["goals_for"] is not None else None
    )
    if home_form["points"] is not None and away_form["points"] is not None:
        points_gap = home_form["points"] - away_form["points"]
        goal_balance = (
            f"Gol ortalamaları {home_name} için {home_form['goals_for']:.1f} attı / {home_form['goals_against']:.1f} yedi, "
            f"{away_name} için {away_form['goals_for']:.1f} attı / {away_form['goals_against']:.1f} yedi."
        )
        if points_gap >= 0.7:
            form_read = f"Son 5 maçın puan ortalaması {home_name} lehine belirgin: {home_form['points']:.2f}'e {away_form['points']:.2f}. {goal_balance}"
        elif points_gap <= -0.7:
            form_read = f"Yakın dönem sonuçları {away_name} lehine: maç başına {away_form['points']:.2f} puan, {home_name} için {home_form['points']:.2f}. {goal_balance} Form üstünlüğüne rağmen ev sahibinin saha avantajı model tahmininde ayrıca hesaba katılıyor."
        else:
            form_read = f"Son 5 maçın puan temposu birbirine yakın ({home_name} {home_form['points']:.2f}, {away_name} {away_form['points']:.2f} puan/maç). {goal_balance} Ayrımı sonuç serisinden çok savunma direnci ve saha avantajı yapabilir."
        form_read += f" Son maç form dizileri: {home_name} {home_form['form']}; {away_name} {away_form['form']} (yeniden eskiye)."
        if home_form["xg_for"] is not None and away_form["xg_for"] is not None:
            form_read += f" Mevcut xG kayıtlarında maç başı üretim {home_name} {home_form['xg_for']:.2f}, {away_name} {away_form['xg_for']:.2f}."
    else:
        form_read = "Takımlardan en az biri için yeterli yakın dönem maç kaydı bulunmadığından form karşılaştırması sınırlı; aşağıdaki senaryolar modelin mevcut lig örneklemine dayanıyor."

    total_xg = home_xg + away_xg
    if total_xg < 2.15:
        tempo_read = f"Beklenen toplam gol {total_xg:.2f}: model düşük skorlu ve sabırlı bir karşılaşmaya daha yakın. İlk golün değeri artıyor; geriye düşen tarafın risk alması maçın son bölümünü belirgin biçimde açabilir."
    elif total_xg > 3.15:
        tempo_read = f"Beklenen toplam gol {total_xg:.2f} ile açık oyun ihtimali öne çıkıyor. İki tarafın da hücum katkısı var; savunma arkası koşular ve top kaybı sonrası geçişler skorun yönünü sık değiştirebilir."
    else:
        tempo_read = f"Beklenen toplam gol {total_xg:.2f}: model dengeli bir gol bandına işaret ediyor. Maçın tamamen açılması için erken gol veya savunma hatası gerekebilir; aksi durumda skor uzun süre tek farklı kalabilir."
    if forecast["btts"] >= 0.62:
        tempo_read += f" Karşılıklı gol olasılığının %{forecast['btts'] * 100:.0f} olması, iki takımın da skor katkısı vermesini makul kılıyor."
    elif forecast["btts"] <= 0.38:
        tempo_read += f" KG Var olasılığı yalnızca %{forecast['btts'] * 100:.0f}; model gol tehdidinin en az bir tarafta sınırlı kalabileceğini düşünüyor."

    if draw_p >= max(home_p, away_p) - 0.06:
        edge_read = f"Üç sonuç olasılığı birbirine yakın; beraberlik %{draw_p:.0%} ile güçlü bir seçenek. {home_name} için {home_xg:.2f}, {away_name} için {away_xg:.2f} beklenen gol farkı küçük olduğundan duran toplar ve ceza sahası içindeki verim öne çıkabilir."
    elif home_p > away_p:
        edge_read = f"Model {home_name} tarafını {home_p:.0%} galibiyet olasılığıyla önde tutuyor. Ev sahibinin beklenen gol üstünlüğü {home_xg - away_xg:+.2f}; {away_name} için en gerçekçi karşılık, savunma arkasına hızlı çıkışlar ve geçişlerde sayısal dengeyi korumak."
    else:
        edge_read = f"Deplasman tarafı {away_name}, {away_p:.0%} galibiyet olasılığıyla modelde öne çıkıyor. Bu tablo saha avantajının tek başına yeterli olmadığını gösteriyor; {home_name} için geçiş savunması ve erken baskıdan çıkış kalitesi kritik."
    if recent_goal_diff is not None and abs(recent_goal_diff) >= 0.7:
        stronger = home_name if recent_goal_diff > 0 else away_name
        edge_read += f" Son maçlardaki gol atma/yeme farkı da {stronger} lehine ayrışıyor."

    st.markdown(f"**Form ve üretim.** {form_read}")
    st.markdown(f"**Beklenen oyun akışı.** {tempo_read}")
    st.markdown(f"**Maçın belirleyici eşleşmesi.** {edge_read}")
    st.caption(
        f"Model görünümü: {row['home_name']} {home_p:.0%} · Beraberlik {draw_p:.0%} · "
        f"{row['away_name']} {away_p:.0%} | KG Yok {btts_no:.0%} | 1,5 gol üst {over_15:.0%} | "
        f"En olası skor {forecast['most_likely_score']}. Olasılıklar geçmiş lig ve takım maçlarından hesaplanan model çıktılarıdır."
    )
    matrix = forecast["score_matrix"]
    def probability(predicate: Any) -> float:
        return sum(
            matrix[h][a]
            for h in range(len(matrix))
            for a in range(len(matrix[h]))
            if predicate(h, a)
        )

    candidates = [
        ("1X", home_p + draw_p, "Ev sahibi kazanır veya maç berabere biter."),
        ("X2", away_p + draw_p, "Deplasman kazanır veya maç berabere biter."),
        ("12", home_p + away_p, "Maçta kazanan çıkar; beraberlik olmaz."),
        (f"Maç sonucu 1 · {home_name}", home_p, f"Model ev sahibine {home_p:.0%} galibiyet olasılığı veriyor."),
        ("Beraberlik", draw_p, f"Beraberlik olasılığı {draw_p:.0%}; iki takımın beklenen golleri {home_xg:.2f}–{away_xg:.2f}."),
        (f"Maç sonucu 2 · {away_name}", away_p, f"Model deplasman galibiyetini {away_p:.0%} hesaplıyor."),
        ("1,5 gol üst", probability(lambda h, a: h + a >= 2), f"İki veya daha fazla gol gerekir; beklenen toplam gol {total_xg:.2f}."),
        ("2,5 gol üst", probability(lambda h, a: h + a >= 3), f"En az üç gol gerekir; beklenen toplam gol {total_xg:.2f}."),
        ("2,5 gol alt", probability(lambda h, a: h + a <= 2), "En fazla iki gol çıkması gerekir."),
        ("3,5 gol alt", probability(lambda h, a: h + a <= 3), "Dört veya daha fazla gol çıkmaması gerekir."),
        ("Karşılıklı gol · Var", forecast["btts"], f"İki takımın da gol bulma olasılığı {forecast['btts']:.0%}."),
        ("Karşılıklı gol · Yok", btts_no, f"En az bir takımın gol bulamama olasılığı {btts_no:.0%}."),
        (f"{home_name} 0,5 gol üst", probability(lambda h, a: h >= 1), f"Ev sahibinin en az bir gol bulma ihtimali; beklenen gol {home_xg:.2f}."),
        (f"{away_name} 0,5 gol üst", probability(lambda h, a: a >= 1), f"Deplasmanın en az bir gol bulma ihtimali; beklenen gol {away_xg:.2f}."),
        (f"{home_name} kazanır ve KG Yok", probability(lambda h, a: h > a and a == 0), "Ev sahibinin galibiyetiyle birlikte deplasmanın gol atamaması gerekir."),
        (f"{away_name} kazanır ve KG Yok", probability(lambda h, a: a > h and h == 0), "Deplasman galibiyetiyle birlikte ev sahibinin gol atamaması gerekir."),
        (f"{home_name} kazanır ve 2,5 üst", probability(lambda h, a: h > a and h + a >= 3), "Ev sahibi kazanmalı ve maçta en az üç gol çıkmalı."),
        (f"{away_name} kazanır ve 2,5 üst", probability(lambda h, a: a > h and h + a >= 3), "Deplasman kazanmalı ve maçta en az üç gol çıkmalı."),
    ]
    candidates.extend([
        (f"Tam skor {forecast['most_likely_score']}", probability(
            lambda h, a: f"{h}-{a}" == str(forecast["most_likely_score"]).replace("–", "-").replace(":", "-")
        ), "Model matrisinde en yüksek olasılığa sahip tek skor."),
    ])
    candidate_probabilities = {label: chance for label, chance, _ in candidates}
    risk_state_key = f"ai_risk_selections_{int(row['id'])}"
    if precomputed_risks is None and st.button(
        "🧠 Risk seçimlerini yapay zekâ belirlesin",
        key=f"generate_ai_risks_{int(row['id'])}",
        help="Gemini maç verisini ve model olasılıklarını değerlendirerek üç risk profiline seçim yapar.",
    ):
        api_key = get_optional_setting("GOOGLE_API_KEY")
        if not api_key:
            st.warning("AI risk seçimi için GOOGLE_API_KEY ayarlanmalı (.env veya Streamlit secrets).")
        else:
            model = get_optional_setting("GOOGLE_MODEL", "gemini-3.8-flash")
            context = build_match_analysis_context(row, history, forecast)
            context["model"]["candidate_markets"] = {
                label: float(chance) for label, chance, _ in candidates
            }
            with st.spinner("Yapay zekâ maç senaryolarını değerlendiriyor…"):
                try:
                    st.session_state[risk_state_key] = generate_match_risk_recommendations(
                        context, api_key=api_key, model=model
                    )
                except MatchAnalysisAIError as exc:
                    st.error(str(exc))
    ai_risks = (
        precomputed_risks
        if precomputed_risks is not None
        else st.session_state.get(risk_state_key, {})
    )
    low, medium, high = st.columns(3)
    for column, title, profile, explanation in (
        (low, "Görece daha olası", "low", "Yapay zekânın temkinli bulduğu, model olasılığı yüksek seçimler:"),
        (medium, "Dengeli risk", "medium", "Yapay zekânın olasılık ve maç senaryosu bakımından dengeli bulduğu seçimler:"),
        (high, "Daha yüksek risk", "high", "Yapay zekânın daha koşullu ve riskli bulduğu seçimler:"),
    ):
        with column:
            st.markdown(f"**{title}**")
            st.caption(explanation)
            selections = ai_risks.get(profile, [])
            if not selections:
                st.caption("Bu maç için henüz yapay zekâ seçimi oluşturulmadı.")
            for selection in selections:
                label = selection["market"]
                chance = candidate_probabilities.get(label)
                if chance is not None:
                    st.markdown(f"**{label} · %{chance * 100:.1f}**  \n{selection['reason']}")
    st.caption("Senaryolar tahmin garantisi veya bahis tavsiyesi değildir. Birleşik koşulların gerçekleşme olasılığı daha düşüktür; maç önü kadro ve sakatlık bilgileri bu model özetinde yer almaz.")


def render_analysis(
    matches_data: list[dict[str, Any]],
    quotes_data: list[dict[str, Any]],
    analyses_data: list[dict[str, Any]],
    teams: pd.DataFrame,
    leagues: pd.DataFrame,
    match_day: date,
) -> None:
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
    analyses_by_id = {
        int(record["match_id"]): record
        for record in analyses_data
        if record.get("match_id") is not None
    }

    matches["home_name"] = matches["home_team_id"].map(team_names).fillna("Ev sahibi")
    matches["away_name"] = matches["away_team_id"].map(team_names).fillna("Deplasman")
    matches["league_name"] = matches["league_id"].map(league_names).fillna("Lig bilgisi yok")
    matches["league_name"] = matches["league_name"].map(lambda name: LEAGUE_NAMES_TR.get(name, name))
    matches["country_name"] = matches["league_id"].map(league_countries).fillna("").map(lambda country: COUNTRY_NAMES_TR.get(country, country))
    matches["league_label"] = matches.apply(lambda row: f"{row['country_name']} · {row['league_name']}" if row["country_name"] else row["league_name"], axis=1)
    matches["kickoff"] = pd.to_datetime(matches["match_date"], utc=True, errors="coerce").dt.tz_convert(ISTANBUL)
    now = pd.Timestamp.now(tz=ISTANBUL)
    live_or_finished = matches["status"].fillna("").astype(str).str.casefold().isin(
        {"live", "finished", *NON_PLAYABLE_MATCH_STATUSES}
    )
    kickoff_reached = matches["kickoff"].notna() & (matches["kickoff"] <= now)
    matches = matches[~(live_or_finished | kickoff_reached)].copy()
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
    a.metric("Filtreye uyan maç", f"{len(filtered)}")
    b.metric("Gün toplamı · oranlı", f"{len(quoted_ids)} / {len(matches)}", help="Oran kapsamı, seçili filtreye değil günün tüm maçlarına göredir.")
    c.metric("Gün toplamı · oransız", f"{len(matches) - len(quoted_ids)}", help="Oran kapsamı, seçili filtreye değil günün tüm maçlarına göredir.")
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
        kickoff = row["kickoff"]
        time_text = kickoff.strftime("%H:%M") if pd.notna(kickoff) else "Saat bilgisi yok"
        match_expander = st.expander(
            f"{time_text}  ·  {row['home_name']} — {row['away_name']}  ·  {row['league_label']}",
            expanded=False,
            key=f"match_{match_id}",
            on_change="rerun",
        )
        if not match_expander.open:
            continue
        with match_expander:
            cached_record = analyses_by_id.get(match_id)
            cached_analysis = _cached_analysis_for_match(cached_record, row)
            history = pd.DataFrame()
            if cached_analysis is not None:
                history = pd.DataFrame(cached_analysis["history"])
                if not history.empty:
                    history["match_date"] = pd.to_datetime(history["match_date"], utc=True, errors="coerce")
                    for column in ("id", "league_id", "home_team_id", "away_team_id", "home_score", "away_score", "home_halftime_score", "away_halftime_score", "home_corners", "away_corners"):
                        if column in history:
                            history[column] = pd.to_numeric(history[column], errors="coerce")
                st.caption(f"Analiz gece önceden hesaplandı · {cached_record.get('calculated_at', '')}")
            else:
                try:
                    history_data = load_league_match_history(int(row["league_id"]))
                    team_history = load_recent_team_matches(
                        int(row["league_id"]),
                        (int(row["home_team_id"]), int(row["away_team_id"])),
                        row["kickoff"].isoformat(),
                    )
                    history_by_id = {int(match["id"]): match for match in history_data}
                    history_by_id.update(
                        (int(match["id"]), match) for match in team_history
                    )
                    history = pd.DataFrame(history_by_id.values())
                    if not history.empty:
                        history["match_date"] = pd.to_datetime(history["match_date"], utc=True, errors="coerce")
                        history = history[history["match_date"] < row["kickoff"]].copy()
                        for column in ("id", "league_id", "home_team_id", "away_team_id", "home_score", "away_score", "home_halftime_score", "away_halftime_score", "home_corners", "away_corners"):
                            if column in history:
                                history[column] = pd.to_numeric(history[column], errors="coerce")
                except Exception as exc:
                    st.warning(f"Geçmiş maçlar yüklenemedi; model tahmini ve form analizi gösterilemiyor: {exc}")
                st.caption("Bu maçın gece analiz önbelleği yok; geçmiş ve tahmin şimdi hazırlanıyor.")
            meta, status_col = st.columns([4, 1])
            meta.caption(row["league_name"])
            status_col.caption(str(row.get("status", "")).replace("_", " ").title())
            model_forecast, _ = render_model_forecast(
                row, history, [], precomputed=cached_analysis, show_details=False
            )
            st.divider()
            render_tactical_match_readout(
                row,
                history,
                model_forecast,
                precomputed_risks=(cached_analysis or {}).get("ai_risk_selections"),
            )
            if model_forecast is not None:
                render_ai_match_analysis(
                    row,
                    history,
                    model_forecast,
                    match_id,
                    precomputed_analysis=(cached_analysis or {}).get("ai_analysis"),
                )
            st.divider()
            render_recent_form(row, history, team_names)
            st.divider()


def render_ai_match_analysis(
    row: dict[str, Any],
    history: pd.DataFrame,
    forecast: dict[str, Any],
    match_id: int,
    precomputed_analysis: str | None = None,
) -> None:
    """Generate an optional, session-cached AI interpretation for one match."""
    home_probability = round(float(forecast["home_win"]), 3)
    away_probability = round(float(forecast["away_win"]), 3)
    state_key = (
        f"ai_match_analysis_{match_id}_"
        f"{round(home_probability * 1000)}_{round(away_probability * 1000)}"
    )
    st.markdown("#### Yapay zekâ destekli detaylı yorum")
    st.caption(
        "Google Gemini, uygulamanın maç geçmişi ve Poisson olasılıklarını yorumlar; "
        "hesaplanmış model tahminlerini değiştirmez. Kadro bilgisi yoksa oyuncu veya sakatlık iddiası üretmez."
    )
    if precomputed_analysis:
        st.caption("Yapay zekâ destekli maç yorumu gece önceden hazırlandı.")
        st.markdown(precomputed_analysis)
        return
    if st.button(
        "🧠 Yapay zekâ analizi oluştur",
        key=f"generate_ai_analysis_{match_id}",
        help="Bu maç için bir Google Gemini API isteği gönderir.",
    ):
        api_key = get_optional_setting("GOOGLE_API_KEY")
        if not api_key:
            st.warning(
                "AI yorumu için GOOGLE_API_KEY ayarlanmalı (.env veya Streamlit secrets)."
            )
        else:
            context = build_match_analysis_context(row, history, forecast)
            model = get_optional_setting("GOOGLE_MODEL", "gemini-3.8-flash")
            with st.spinner("Maç verileri yorumlanıyor…"):
                try:
                    st.session_state[state_key] = generate_match_analysis(
                        context, api_key=api_key, model=model
                    )
                except MatchAnalysisAIError as exc:
                    st.error(str(exc))
    analysis = st.session_state.get(state_key)
    if analysis:
        st.markdown(analysis)


def coupon_market_candidates(row: dict[str, Any], forecast: dict[str, Any]) -> list[dict[str, Any]]:
    """Build model-derived single-match selections for the automatic coupon."""
    matrix = forecast["score_matrix"]
    home, away = row["home_name"], row["away_name"]

    def chance(predicate: Any) -> float:
        return sum(
            matrix[h][a]
            for h in range(len(matrix))
            for a in range(len(matrix[h]))
            if predicate(h, a)
        )

    candidates = [
        ("1X", "Çifte şans", chance(lambda h, a: h >= a), "Ev sahibi yenilmez."),
        ("X2", "Çifte şans", chance(lambda h, a: a >= h), "Deplasman yenilmez."),
        ("12", "Çifte şans", chance(lambda h, a: h != a), "Maç berabere bitmez."),
        (f"1 · {home}", "Maç sonucu", forecast["home_win"], f"{home} galibiyeti."),
        ("X · Beraberlik", "Maç sonucu", forecast["draw"], "Maç berabere biter."),
        (f"2 · {away}", "Maç sonucu", forecast["away_win"], f"{away} galibiyeti."),
        ("1,5 gol üst", "Toplam gol", chance(lambda h, a: h + a >= 2), "Maçta en az iki gol çıkar."),
        ("2,5 gol üst", "Toplam gol", chance(lambda h, a: h + a >= 3), "Maçta en az üç gol çıkar."),
        ("2,5 gol alt", "Toplam gol", chance(lambda h, a: h + a <= 2), "Maçta en fazla iki gol çıkar."),
        ("3,5 gol alt", "Toplam gol", chance(lambda h, a: h + a <= 3), "Maçta en fazla üç gol çıkar."),
        ("Karşılıklı gol var", "Karşılıklı gol", forecast["btts"], "İki takım da gol atar."),
        ("Karşılıklı gol yok", "Karşılıklı gol", 1 - forecast["btts"], "Takımlardan en az biri gol atamaz."),
        (f"{home} 0,5 gol üst", "Takım golü", chance(lambda h, a: h >= 1), f"{home} en az bir gol atar."),
        (f"{away} 0,5 gol üst", "Takım golü", chance(lambda h, a: a >= 1), f"{away} en az bir gol atar."),
        (f"{home} 1,5 gol üst", "Takım golü", chance(lambda h, a: h >= 2), f"{home} en az iki gol atar."),
        (f"{away} 1,5 gol üst", "Takım golü", chance(lambda h, a: a >= 2), f"{away} en az iki gol atar."),
        (f"{home} kazanır · KG yok", "Birleşik seçim", chance(lambda h, a: h > a and a == 0), f"{home} kazanır ve {away} gol atamaz."),
        (f"{away} kazanır · KG yok", "Birleşik seçim", chance(lambda h, a: a > h and h == 0), f"{away} kazanır ve {home} gol atamaz."),
        (f"{home} kazanır · 2,5 üst", "Birleşik seçim", chance(lambda h, a: h > a and h + a >= 3), f"{home} kazanır ve toplam gol en az üç olur."),
        (f"{away} kazanır · 2,5 üst", "Birleşik seçim", chance(lambda h, a: a > h and h + a >= 3), f"{away} kazanır ve toplam gol en az üç olur."),
    ]
    for score_h in range(len(matrix)):
        for score_a in range(len(matrix[score_h])):
            candidates.append((
                f"Tam skor {score_h}–{score_a}", "Tam skor", matrix[score_h][score_a],
                f"Modelin bu kesin skora verdiği olasılık {matrix[score_h][score_a]:.1%}.",
            ))
    return [
        {"selection": label, "market": market, "probability": float(probability), "reason": reason}
        for label, market, probability, reason in candidates
        if math.isfinite(float(probability)) and 0 < float(probability) < 1
    ]


def render_auto_coupon(
    matches_data: list[dict[str, Any]],
    analyses_data: list[dict[str, Any]],
    teams: pd.DataFrame,
    selected_day: date,
) -> None:
    """Generate a sample coupon from the selected day's pre-match model forecasts."""
    st.markdown(
        "<section class='hero'><div class='hero-kicker'>Model olasılıklarına göre otomatik seçim</div>"
        "<h1>Otomatik Kupon Oluşturucu</h1><p>Risk profilini ve maç sayısını seç; sistem uygun maç senaryolarını sıralasın.</p></section>",
        unsafe_allow_html=True,
    )
    st.info("Bu bölüm model olasılıklarına göre örnek seçim listesi üretir; bahis sitesine bağlanmaz, gerçek oran veya kazanç vaadi içermez.")
    matches = pd.DataFrame(matches_data)
    if matches.empty:
        st.info(f"{selected_day:%d.%m.%Y} için fikstür bulunamadı.")
        return

    team_names = dict(zip(teams.get("id", []), teams.get("name", []))) if not teams.empty else {}
    matches["home_name"] = matches["home_team_id"].map(team_names).fillna("Ev sahibi")
    matches["away_name"] = matches["away_team_id"].map(team_names).fillna("Deplasman")
    matches["kickoff"] = pd.to_datetime(matches["match_date"], utc=True, errors="coerce").dt.tz_convert(ISTANBUL)
    now = pd.Timestamp.now(tz=ISTANBUL)
    status_done = matches["status"].fillna("").astype(str).str.casefold().isin(
        {"live", "finished", *NON_PLAYABLE_MATCH_STATUSES}
    )
    upcoming = matches[
        ~status_done
        & matches["kickoff"].notna()
        & (matches["kickoff"] > now)
    ].copy()
    if upcoming.empty:
        st.info("Seçili tarihte henüz başlamamış maç bulunmuyor.")
        return

    c1, c2 = st.columns(2)
    with c1:
        risk = st.selectbox("Risk profili", ["Daha temkinli", "Dengeli", "Yüksek riskli"], index=1)
    with c2:
        legs = st.number_input("Kupondaki maç sayısı", min_value=2, max_value=10, value=5, step=1)

    analyses_by_id = {
        int(record["match_id"]): record
        for record in analyses_data
        if record.get("match_id") is not None
    }
    league_history: dict[int, pd.DataFrame] = {}
    fixture_candidates: list[dict[str, Any]] = []
    for row in upcoming.sort_values("kickoff").to_dict("records"):
        match_id = int(row["id"])
        cached = _cached_analysis_for_match(analyses_by_id.get(match_id), row)
        forecast = cached.get("forecast") if cached else None
        if forecast is None:
            league_id = int(row["league_id"])
            if league_id not in league_history:
                try:
                    history = pd.DataFrame(load_league_match_history(league_id))
                except Exception:
                    history = pd.DataFrame()
                if not history.empty:
                    history["match_date"] = pd.to_datetime(history["match_date"], utc=True, errors="coerce")
                    for column in ("id", "league_id", "home_team_id", "away_team_id", "home_score", "away_score"):
                        if column in history:
                            history[column] = pd.to_numeric(history[column], errors="coerce")
                league_history[league_id] = history
            forecast = model_match_forecast(row, league_history[league_id])
        if forecast is None:
            continue
        for candidate in coupon_market_candidates(row, forecast):
            p = candidate["probability"]
            market = candidate["market"]
            if risk == "Daha temkinli":
                eligible = p >= 0.65 and market not in {"Birleşik seçim", "Tam skor"}
                score = p
            elif risk == "Dengeli":
                eligible = 0.40 <= p <= 0.72 and market != "Tam skor"
                score = 1 - abs(p - 0.56)
            else:
                eligible = 0.12 <= p <= 0.42 and market in {"Maç sonucu", "Toplam gol", "Karşılıklı gol", "Takım golü", "Birleşik seçim", "Tam skor"}
                score = 1 - abs(p - 0.28) + (0.04 if market in {"Birleşik seçim", "Tam skor"} else 0)
            if eligible:
                fixture_candidates.append({
                    "match_id": match_id,
                    "match": f"{row['home_name']} – {row['away_name']}",
                    "kickoff": row["kickoff"],
                    "score": score,
                    **candidate,
                })

    if not fixture_candidates:
        st.warning(f"{risk} profilinde uygun tahmin üretilemedi. Başka bir risk profili seçmeyi deneyin.")
        return

    # Choose greedily across all outcomes, allowing at most one pick per match
    # and mildly favoring market diversity when alternatives are comparable.
    choices = fixture_candidates.copy()
    selected: list[dict[str, Any]] = []
    category_counts: dict[str, int] = {}
    used_matches: set[int] = set()
    while choices and len(selected) < int(legs):
        eligible_choices = [item for item in choices if item["match_id"] not in used_matches]
        if not eligible_choices:
            break
        best = max(
            eligible_choices,
            key=lambda item: item["score"] - 0.035 * category_counts.get(item["market"], 0),
        )
        choices.remove(best)
        selected.append(best)
        used_matches.add(best["match_id"])
        category_counts[best["market"]] = category_counts.get(best["market"], 0) + 1

    if len(selected) < int(legs):
        st.warning(f"Bu risk profiline uygun {len(selected)} maç bulundu; kupon bu sayıyla oluşturuldu.")
    combined_probability = math.prod(item["probability"] for item in selected)
    a, b, c = st.columns(3)
    a.metric("Seçilen maç", f"{len(selected)}")
    b.metric("Ortalama seçim olasılığı", f"%{sum(item['probability'] for item in selected) / len(selected) * 100:.1f}" if selected else "—")
    c.metric("Birleşik model olasılığı", f"%{combined_probability * 100:.3f}" if selected else "—")
    st.dataframe(
        pd.DataFrame([
            {
                "Başlama": item["kickoff"].strftime("%H:%M"),
                "Maç": item["match"],
                "Seçim": item["selection"],
                "Model %": item["probability"] * 100,
                "Gerekçe": item["reason"],
            }
            for item in selected
        ]),
        width="stretch",
        hide_index=True,
        column_config={"Model %": st.column_config.NumberColumn(format="%.1f%%")},
    )
    st.caption("Birleşik olasılık, maç sonuçlarının birbirinden bağımsız olduğunu varsayan yaklaşık model çarpımıdır; gerçek oran veya kupon getirisi değildir. Her maçtan en fazla bir seçim alınır.")


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
    matches["Lig"] = matches["Lig"].map(lambda name: LEAGUE_NAMES_TR.get(name, name))
    matches["Ülke"] = matches["Ülke"].map(lambda country: COUNTRY_NAMES_TR.get(country, country))
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
    matches = matches.sort_values("Tarih", ascending=False)
    page_size = 200
    page_count = max(1, math.ceil(len(matches) / page_size))
    page = int(st.number_input("Sayfa", min_value=1, max_value=page_count, value=1, step=1))
    st.caption(f"{len(matches):,} maç · {page} / {page_count} sayfa")
    start = (page - 1) * page_size
    st.dataframe(
        matches.iloc[start : start + page_size][columns].rename(columns=labels),
        width="stretch",
        hide_index=True,
    )


def market_performance_rows(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Summarize accuracy and probability quality for each evaluated market."""
    observations: dict[str, list[tuple[float, bool, float | None, float | None]]] = {}

    for record in records:
        result_probs = [record.get("prob_home_win"), record.get("prob_draw"), record.get("prob_away_win")]
        result_keys = ("home_win", "draw", "away_win")
        try:
            result_probs = [float(probability) for probability in result_probs]
        except (TypeError, ValueError):
            result_probs = []
        actual_result = str(record.get("actual_result") or "").casefold()
        result_aliases = {"home": "home_win", "1": "home_win", "away": "away_win", "2": "away_win", "x": "draw"}
        actual_result = result_aliases.get(actual_result, actual_result)
        if len(result_probs) == 3 and all(math.isfinite(p) and 0 <= p <= 1 for p in result_probs) and actual_result in result_keys:
            actual_index = result_keys.index(actual_result)
            confidence = max(result_probs)
            correct = result_keys[result_probs.index(confidence)] == actual_result
            brier = sum((p - float(index == actual_index)) ** 2 for index, p in enumerate(result_probs))
            loss = -math.log(max(1e-15, result_probs[actual_index]))
            observations.setdefault("Maç sonucu (1X2)", []).append((confidence, correct, brier, loss))

        for prefix, label in (("over_2_5", "Toplam gol 2,5 üst"), ("btts", "Karşılıklı gol (KG Var)")):
            p, actual, correct, stored_brier = (
                record.get(f"prob_{prefix}"), record.get(f"{prefix}_actual"),
                record.get(f"{prefix}_was_correct"), record.get(f"{prefix}_brier_score"),
            )
            try:
                p = float(p)
                brier = float(stored_brier) if stored_brier is not None else None
            except (TypeError, ValueError):
                continue
            if isinstance(actual, bool) and isinstance(correct, bool) and math.isfinite(p) and 0 <= p <= 1:
                loss = -math.log(max(1e-15, p if actual else 1 - p))
                observations.setdefault(label, []).append((max(p, 1 - p), correct, brier if brier is not None else (p - float(actual)) ** 2, loss))

        performance = record.get("market_performance") or {}
        if not isinstance(performance, dict):
            continue
        total_goals = performance.get("total_goals") or {}
        if isinstance(total_goals, dict):
            by_line: dict[str, dict[str, dict[str, Any]]] = {}
            for key, value in total_goals.items():
                parts = key.split("_", 1)
                if len(parts) == 2 and parts[0] in {"over", "under"} and isinstance(value, dict):
                    by_line.setdefault(parts[1], {})[parts[0]] = value
            for line, sides in by_line.items():
                over, under = sides.get("over"), sides.get("under")
                try:
                    p_over = float(over["probability"]) if over else 1 - float(under["probability"])
                    actual_over = over.get("actual") if over else not under.get("actual")
                except (TypeError, ValueError, KeyError):
                    continue
                if line == "2_5" or not isinstance(actual_over, bool) or not math.isfinite(p_over) or not 0 <= p_over <= 1:
                    continue
                correct = (p_over >= 0.5) == actual_over
                p_actual = p_over if actual_over else 1 - p_over
                brier = (p_over - float(actual_over)) ** 2
                observations.setdefault(f"Toplam gol {line.replace('_', ',')} çizgisi", []).append((max(p_over, 1 - p_over), correct, brier, -math.log(max(1e-15, p_actual))))

        team_goals = performance.get("team_goals") or {}
        if isinstance(team_goals, dict):
            for key, value in team_goals.items():
                if not isinstance(value, dict):
                    continue
                parts = key.split("_over_", 1)
                if len(parts) != 2 or parts[0] not in {"home", "away"}:
                    continue
                try:
                    p = float(value["probability"])
                    actual = value["actual"]
                except (TypeError, ValueError, KeyError):
                    continue
                if not isinstance(actual, bool) or not math.isfinite(p) or not 0 <= p <= 1:
                    continue
                correct = (p >= 0.5) == actual
                label = "Ev sahibi" if parts[0] == "home" else "Deplasman"
                line = parts[1].replace("_", ",")
                observations.setdefault(f"{label} gol {line} üst", []).append((max(p, 1 - p), correct, (p - float(actual)) ** 2, -math.log(max(1e-15, p if actual else 1 - p))))

        double_chance = performance.get("double_chance") or {}
        if isinstance(double_chance, dict):
            choices = [value for value in double_chance.values() if isinstance(value, dict) and isinstance(value.get("actual"), bool)]
            try:
                choice = max(choices, key=lambda value: float(value["probability"]))
                p, actual = float(choice["probability"]), choice["actual"]
            except (ValueError, TypeError, KeyError):
                p = None
            if p is not None and 0 <= p <= 1:
                observations.setdefault("Çifte şans", []).append((p, actual, (p - float(actual)) ** 2, -math.log(max(1e-15, p if actual else 1 - p))))

        correct_score = performance.get("correct_score") or {}
        scores = (record.get("market_probabilities") or {}).get("correct_scores", [])
        if isinstance(correct_score, dict) and isinstance(correct_score.get("top_3_hit"), bool) and isinstance(scores, list):
            confidence = sum(float(score.get("probability", 0)) for score in scores if isinstance(score, dict))
            hit = correct_score["top_3_hit"]
            if math.isfinite(confidence) and 0 <= confidence <= 1:
                observations.setdefault("Doğru skor (ilk 3)", []).append((confidence, hit, (confidence - float(hit)) ** 2, -math.log(max(1e-15, confidence if hit else 1 - confidence))))

    rows = []
    for market, values in observations.items():
        if not values:
            continue
        count = len(values)
        pairs = [(confidence, float(hit)) for confidence, hit, _, _ in values]
        rows.append({
            "Pazar": market,
            "Örnek": count,
            "Doğru": sum(hit for _, hit, _, _ in values),
            "Başarı %": 100 * sum(hit for _, hit, _, _ in values) / count,
            "Brier": sum(brier for _, _, brier, _ in values if brier is not None) / count,
            "Log-loss": sum(loss for _, _, _, loss in values if loss is not None) / count,
            "Kalibrasyon hatası": _expected_calibration_error(pairs),
        })
    return pd.DataFrame(rows).sort_values("Pazar").reset_index(drop=True) if rows else pd.DataFrame()


def render_performance() -> None:
    try:
        records = load_evaluated_prediction_results()
    except Exception as exc:
        st.error(f"Tahmin performansı yüklenemedi: {exc}")
        return
    if not records:
        st.info("Henüz değerlendirilmiş tahmin bulunmuyor.")
        return
    dates = pd.to_datetime([row.get("match_date") for row in records], utc=True, errors="coerce").dropna().tz_convert(ISTANBUL)
    if dates.empty:
        st.info("Tahmin kayıtlarında geçerli maç tarihi bulunamadı.")
        return
    versions = sorted({str(row.get("model_version")) for row in records if row.get("model_version")}, reverse=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        start_day = st.date_input("Başlangıç", value=dates.min().date(), format="DD.MM.YYYY", key="performance_start")
    with c2:
        end_day = st.date_input("Bitiş", value=dates.max().date(), format="DD.MM.YYYY", key="performance_end")
    with c3:
        version_filter = st.selectbox("Model sürümü", ["Tüm sürümler", *versions])
    if start_day > end_day:
        st.warning("Başlangıç tarihi bitiş tarihinden sonra olamaz.")
        return
    start_ts = pd.Timestamp(start_day, tz=ISTANBUL)
    end_ts = pd.Timestamp(end_day, tz=ISTANBUL) + pd.Timedelta(days=1)
    selected = [
        row for row in records
        if pd.notna(pd.to_datetime(row.get("match_date"), utc=True, errors="coerce"))
        and start_ts <= pd.to_datetime(row["match_date"], utc=True).tz_convert(ISTANBUL) < end_ts
        and (version_filter == "Tüm sürümler" or str(row.get("model_version")) == version_filter)
    ]
    table = market_performance_rows(selected)
    if table.empty:
        st.info("Seçilen dönemde ölçülebilir pazar tahmini bulunamadı.")
        return
    total = int(table["Örnek"].sum())
    correct = int(table["Doğru"].sum())
    a, b, c = st.columns(3)
    a.metric("Değerlendirilen maç", f"{len(selected):,}")
    b.metric("Tüm pazar kararları", f"{correct:,} / {total:,} · %{100 * correct / total:.1f}")
    c.metric("Pazar sayısı", f"{len(table)}")
    st.caption(
        f"Dönem: {start_day:%d.%m.%Y}–{end_day:%d.%m.%Y}. Her pazarın örnek sayısı farklıdır; "
        "toplu oran karar başına eşit ağırlıklıdır ve bahis getirisi değildir. "
        "Kârlılık hesabı için bu maçlarla eşleştirilmiş tarihsel oran kaydı bulunmadığından ROI gösterilmiyor."
    )
    st.dataframe(
        table,
        width="stretch",
        hide_index=True,
        column_config={
            "Başarı %": st.column_config.NumberColumn(format="%.1f%%"),
            "Brier": st.column_config.NumberColumn(format="%.3f"),
            "Log-loss": st.column_config.NumberColumn(format="%.3f"),
            "Kalibrasyon hatası": st.column_config.NumberColumn(format="%.3f"),
        },
    )
    st.caption("Brier, log-loss ve 10 dilimli kalibrasyon hatasında düşük değer daha iyidir. Kalibrasyon, seçilen sonucun güven olasılığını gerçekleşen isabetle karşılaştırır.")


if __name__ == '__main__':
    st.markdown("<section class='hero'><div class='hero-kicker'>Günlük karşılaşmalar · iddaa marketleri</div><h1>Maç Merkezi</h1><p>Maçları inceleyin, marketleri karşılaştırın ve maç istatistikleri arşivine ulaşın.</p></section>", unsafe_allow_html=True)
    today = datetime.now(ISTANBUL).date()
    col, refresh_col, _ = st.columns([1, 1, 2])
    with col:
        selected_day = st.date_input("Maç günü", value=today, format="DD.MM.YYYY")
    with refresh_col:
        st.write("")
        if st.button("Verileri şimdi yenile", width="stretch"):
            load_reference_catalog.clear()
            load_match_day.clear()
            load_completed_match_history.clear()
            load_evaluated_prediction_results.clear()
            load_league_match_history.clear()
            load_recent_team_matches.clear()
            st.rerun()
    try:
        teams, leagues = load_reference_catalog()
    except Exception as exc:
        st.error(f"Takım ve lig bilgileri yüklenemedi: {exc}")
        teams, leagues = pd.DataFrame(), pd.DataFrame()
    section = st.radio(
        "Bölüm",
        ["⚽  Maç Analizleri", "🧠  Günlük Taktik Analiz", "🎟️  Kupon Oluşturucu", "📊  İstatistik Arşivi", "📈  Tahmin Performansı"],
        horizontal=True,
        label_visibility="collapsed",
    )
    if section in {"⚽  Maç Analizleri", "🧠  Günlük Taktik Analiz"}:
        try:
            match_records, quote_records, analyses = load_match_day(selected_day)
            render_analysis(
                match_records, quote_records, analyses, teams, leagues, selected_day
            )
        except Exception as exc:
            st.error(f"Maç ve oran verileri yüklenemedi: {exc}")
    elif section == "🎟️  Kupon Oluşturucu":
        try:
            match_records, _, analyses = load_match_day(selected_day)
            render_auto_coupon(match_records, analyses, teams, selected_day)
        except Exception as exc:
            st.error(f"Otomatik kupon verileri hazırlanamadı: {exc}")
    elif section == "📊  İstatistik Arşivi":
        render_archive()
    else:
        render_performance()
