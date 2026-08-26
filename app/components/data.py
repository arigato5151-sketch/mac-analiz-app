"""Cached data access tailored for Streamlit pages."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st

from config.settings import PROJECT_ROOT, get_settings
from db.db_client import SupabaseRestClient
from models.feature_engineering import CausalFeatureState
from models.train_model import load_completed_matches


@st.cache_resource
def get_db() -> SupabaseRestClient:
    settings = get_settings()
    return SupabaseRestClient(settings.supabase_url, settings.supabase_service_role_key)


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
        "prediction_performance",
        columns=(
            "id,prediction_id,match_id,actual_result,was_correct,brier_score,"
            "evaluated_at"
        ),
        order="evaluated_at.asc",
    )
    return pd.DataFrame(rows)


@st.cache_data(ttl=3600, show_spinner=False)
def load_latest_model_metadata() -> dict[str, Any] | None:
    model_dir = PROJECT_ROOT / "models" / "saved_models"
    files = sorted(model_dir.glob("model_v*.json"))
    if not files:
        return None
    return json.loads(files[-1].read_text(encoding="utf-8"))


def fetch_api_status() -> dict[str, Any]:
    settings = get_settings()
    response = requests.get(
        "https://v3.football.api-sports.io/status",
        headers={
            "x-apisports-key": settings.api_football_key,
            "User-Agent": "mac-analiz-app/1.0",
        },
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()["response"]
    return {
        "plan": payload["subscription"]["plan"],
        "active": payload["subscription"]["active"],
        "used": payload["requests"]["current"],
        "limit": payload["requests"]["limit_day"],
    }


def clear_app_cache() -> None:
    st.cache_data.clear()
