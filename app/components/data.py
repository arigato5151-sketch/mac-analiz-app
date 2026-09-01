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


LIVE_DATA_TTL_SECONDS = 300
MATCH_DETAIL_TTL_SECONDS = 900
HISTORY_TTL_SECONDS = 900
REFERENCE_DATA_TTL_SECONDS = 21_600
MODEL_METADATA_TTL_SECONDS = 3_600


@st.cache_resource(show_spinner=False)
def get_db() -> PublicSupabaseRestClient:
    """Keep one read-only HTTP client per Streamlit process."""
    settings = get_public_supabase_settings()
    return PublicSupabaseRestClient(settings.supabase_url, settings.supabase_anon_key)


@st.cache_data(ttl=REFERENCE_DATA_TTL_SECONDS, show_spinner=False)
def load_reference_catalog() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cache slowly changing team and league labels separately from live fixtures."""
    db = get_db()
    return (
        pd.DataFrame(db.select_all("teams", columns="id,name,logo_url")),
        pd.DataFrame(db.select_all("leagues", columns="id,name,country")),
    )


@st.cache_data(ttl=HISTORY_TTL_SECONDS, show_spinner=False)
def load_completed_match_history() -> list[dict[str, Any]]:
    """Share the expensive chronological history used by all match detail pages."""
    return load_completed_matches(get_db())


@st.cache_data(ttl=LIVE_DATA_TTL_SECONDS, show_spinner=False)
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

    match_ids = ",".join(str(int(match["id"])) for match in matches)
    predictions = pd.DataFrame(
        db.select_all(
            "predictions",
            columns=(
                "match_id,model_version,prob_home_win,prob_draw,prob_away_win,"
                "prob_over_2_5,prob_btts,predicted_at"
            ),
            filters={"match_id": f"in.({match_ids})"},
            order="predicted_at.desc",
        )
    )
    teams, leagues = load_reference_catalog()
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


@st.cache_data(ttl=MATCH_DETAIL_TTL_SECONDS, show_spinner="Poisson baseline hazırlanıyor...")
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
    history = load_completed_match_history()
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


@st.cache_data(ttl=LIVE_DATA_TTL_SECONDS, show_spinner=False)
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


@st.cache_data(ttl=LIVE_DATA_TTL_SECONDS, show_spinner=False)
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


@st.cache_data(ttl=LIVE_DATA_TTL_SECONDS, show_spinner=False)
def load_confirmed_lineups(match_id: int) -> pd.DataFrame:
    """Load official XIs when both teams have been published by the provider."""
    rows = get_db().select_all(
        "fixture_lineups",
        columns="team_id,formation,coach_name,starters,substitutes,confirmed_at",
        filters={"match_id": f"eq.{match_id}"},
    )
    return pd.DataFrame(rows)


@st.cache_data(ttl=LIVE_DATA_TTL_SECONDS, show_spinner=False)
def load_odds_history(match_id: int) -> pd.DataFrame:
    rows = get_db().select_all(
        "odds_quote_history",
        columns="bookmaker,odds,captured_at,is_notification_reference",
        filters={"match_id": f"eq.{match_id}"}, order="captured_at.asc",
    )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["captured_at"] = pd.to_datetime(frame["captured_at"], utc=True).dt.tz_convert("Europe/Istanbul")
    return frame


@st.cache_data(ttl=LIVE_DATA_TTL_SECONDS, show_spinner=False)
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


@st.cache_data(ttl=MODEL_METADATA_TTL_SECONDS, show_spinner=False)
def load_latest_model_metadata() -> dict[str, Any] | None:
    model_dir = PROJECT_ROOT / "models" / "saved_models"
    files = sorted(model_dir.glob("model_v*.json"))
    if not files:
        return None
    return json.loads(files[-1].read_text(encoding="utf-8"))


def clear_app_cache() -> None:
    """Clear cached values and the read-only client after a configuration refresh."""
    st.cache_data.clear()
    st.cache_resource.clear()
