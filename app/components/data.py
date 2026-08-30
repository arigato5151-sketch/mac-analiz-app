"""Cached data access tailored for Streamlit pages."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from config.settings import PROJECT_ROOT, get_public_supabase_settings
from db.db_client import PublicSupabaseRestClient
from models.feature_engineering import CausalFeatureState
from models.train_model import load_completed_matches


@st.cache_resource
def get_db() -> PublicSupabaseRestClient:
    settings = get_public_supabase_settings()
    return PublicSupabaseRestClient(settings.supabase_url, settings.supabase_anon_key)


@st.cache_data(ttl=300, show_spinner=False)
def load_upcoming_dashboard(horizon_days: int = 3) -> pd.DataFrame:
    db = get_db()
    now = datetime.now(timezone.utc)
    end = now + timedelta(days=horizon_days)
    matches = db.select_all(
        "matches",
        columns="id,league_id,home_team_id,away_team_id,match_date,status",
        filters={
            "status": "eq.scheduled",
            "and": f"(match_date.gte.{now.isoformat()},match_date.lte.{end.isoformat()})",
        },
        order="match_date.asc,id.asc",
    )
    if not matches:
        return pd.DataFrame()

    teams = pd.DataFrame(db.select_all("teams", columns="id,name,logo_url"))
    leagues = pd.DataFrame(db.select_all("leagues", columns="id,name,country"))
    predictions = pd.DataFrame(
        db.select_all(
            "predictions",
            columns=(
                "match_id,model_version,prob_home_win,prob_draw,prob_away_win,"
                "prob_over_2_5,prob_btts,predicted_at"
            ),
            order="predicted_at.desc",
        )
    )
    frame = pd.DataFrame(matches)
    if not predictions.empty:
        predictions = predictions.drop_duplicates("match_id", keep="first")
        frame = frame.merge(predictions, left_on="id", right_on="match_id", how="left")

    home = teams.rename(columns={"id": "home_team_id", "name": "home_team"})[
        ["home_team_id", "home_team"]
    ]
    away = teams.rename(columns={"id": "away_team_id", "name": "away_team"})[
        ["away_team_id", "away_team"]
    ]
    league_names = leagues.rename(
        columns={"id": "league_id", "name": "league_name"}
    )[["league_id", "league_name", "country"]]
    frame = frame.merge(home, on="home_team_id", how="left")
    frame = frame.merge(away, on="away_team_id", how="left")
    frame = frame.merge(league_names, on="league_id", how="left")
    frame["match_date"] = pd.to_datetime(frame["match_date"], utc=True).dt.tz_convert(
        "Europe/Istanbul"
    )
    return frame.sort_values("match_date").reset_index(drop=True)


@st.cache_data(ttl=900, show_spinner="Poisson baseline hazırlanıyor...")
def load_match_baseline(match_id: int) -> dict[str, Any]:
    db = get_db()
    upcoming = db.select(
        "matches",
        columns=(
            "id,league_id,home_team_id,away_team_id,match_date,status,"
            "home_score,away_score"
        ),
        filters={"id": f"eq.{match_id}"},
        limit=1,
    )
    if not upcoming:
        raise ValueError("Maç bulunamadı")
    history = load_completed_matches(db)
    state = CausalFeatureState()
    for row in history:
        state.update(row)
    match = upcoming[0]
    baseline = state.poisson_baseline(match)
    return {
        "prediction": baseline,
        "home_state": state.states[int(match["home_team_id"])],
        "away_state": state.states[int(match["away_team_id"])],
        "history": history,
        "match": match,
    }


@st.cache_data(ttl=300, show_spinner=False)
def load_prediction_performance() -> pd.DataFrame:
    db = get_db()
    rows = db.select_all(
        "live_prediction_performance",
        columns=(
            "prediction_id,match_id,actual_result,was_correct,brier_score,"
            "evaluated_at,over_2_5_actual,over_2_5_was_correct,over_2_5_brier_score,"
            "btts_actual,btts_was_correct,btts_brier_score"
        ),
        order="evaluated_at.asc",
    )
    return pd.DataFrame(rows)


@st.cache_data(ttl=300, show_spinner=False)
def load_match_availability(
    home_team_id: int, away_team_id: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the latest public squad context for the two teams in a fixture."""
    team_filter = f"(team_id.eq.{home_team_id},team_id.eq.{away_team_id})"
    db = get_db()
    players = pd.DataFrame(
        db.select_all(
            "player_availability",
            columns="team_id,player_name,status,updated_at",
            filters={"or": team_filter},
            order="updated_at.desc",
        )
    )
    snapshots = pd.DataFrame(
        db.select_all(
            "team_availability_status",
            columns="team_id,refreshed_at,available_count",
            filters={"or": team_filter},
        )
    )
    return players, snapshots


@st.cache_data(ttl=300, show_spinner=False)
def load_evaluated_predictions(limit: int = 250) -> pd.DataFrame:
    """Load evaluated predictions from the RLS-protected database view."""
    if limit < 1 or limit > 1_000:
        raise ValueError("limit must be between 1 and 1000")

    db = get_db()
    rows = db.select(
        "evaluated_prediction_results",
        columns=(
            "prediction_id,match_id,actual_result,was_correct,brier_score,evaluated_at,"
            "league_id,match_date,home_score,away_score,prob_home_win,prob_draw,"
            "prob_away_win,prob_over_2_5,prob_btts,model_version,predicted_at,"
            "home_team,away_team,league_name,over_2_5_actual,over_2_5_was_correct,"
            "over_2_5_brier_score,btts_actual,btts_was_correct,btts_brier_score"
        ),
        limit=limit,
        order="evaluated_at.desc",
    )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    frame["match_date"] = pd.to_datetime(frame["match_date"], utc=True).dt.tz_convert(
        "Europe/Istanbul"
    )
    frame["evaluated_at"] = pd.to_datetime(frame["evaluated_at"], utc=True).dt.tz_convert(
        "Europe/Istanbul"
    )
    return frame.sort_values("evaluated_at", ascending=False).reset_index(drop=True)


@st.cache_data(ttl=3600, show_spinner=False)
def load_latest_model_metadata() -> dict[str, Any] | None:
    model_dir = PROJECT_ROOT / "models" / "saved_models"
    files = sorted(model_dir.glob("model_v*.json"))
    if not files:
        return None
    return json.loads(files[-1].read_text(encoding="utf-8"))


def clear_app_cache() -> None:
    st.cache_data.clear()
