"""Cached data access tailored for Streamlit pages."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import joblib
import streamlit as st

from config.settings import PROJECT_ROOT, get_public_supabase_settings
from db.db_client import PublicSupabaseRestClient
from models.feature_engineering import CausalFeatureState


LOGGER = logging.getLogger(__name__)


LIVE_DATA_TTL_SECONDS = 300
MATCH_DETAIL_TTL_SECONDS = 900
HISTORY_TTL_SECONDS = 900
# Team/league rows are updated by the fixture sync. Keep this short enough
# that newly synced national-team fixtures never remain as ``nan`` in the UI
# because an old reference catalog is still cached.
REFERENCE_DATA_TTL_SECONDS = 300
MODEL_METADATA_TTL_SECONDS = 3_600
UPCOMING_HORIZON_DAYS = 7


def load_historical_matches(db: PublicSupabaseRestClient) -> list[dict[str, Any]]:
    """Load training history without importing the training stack at startup."""
    from models.train_model import load_historical_matches as _load_historical_matches

    return _load_historical_matches(db)


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
    """Load only the public match history required by the Poisson UI state."""
    return load_historical_matches(get_db())


@st.cache_data(ttl=LIVE_DATA_TTL_SECONDS, show_spinner=False)
def load_upcoming_dashboard(
    horizon_days: int = UPCOMING_HORIZON_DAYS,
) -> pd.DataFrame:
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
                "prob_over_2_5,prob_btts,market_probabilities,predicted_at"
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
    if "predicted_at" in frame:
        frame["predicted_at"] = pd.to_datetime(
            frame["predicted_at"], utc=True, errors="coerce"
        ).dt.tz_convert("Europe/Istanbul")
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
        "evaluated_prediction_results",
        columns=(
            "prediction_id,match_id,model_version,actual_result,was_correct,brier_score,"
            "prob_home_win,prob_draw,prob_away_win,"
            "evaluated_at,over_2_5_actual,over_2_5_was_correct,over_2_5_brier_score,"
            "btts_actual,btts_was_correct,btts_brier_score,"
            "market_probabilities,market_performance,league_name"
        ),
        order="evaluated_at.asc",
    )
    return pd.DataFrame(rows)


@st.cache_data(ttl=LIVE_DATA_TTL_SECONDS, show_spinner=False)
def load_shadow_model_status() -> pd.DataFrame:
    """Load sanitized, same-match shadow/production aggregate metrics."""
    rows = get_db().select_all(
        "shadow_model_status",
        columns=(
            "model_version,status,registered_at,promoted_at,prediction_count,"
            "evaluated_matches,paired_matches,candidate_accuracy,candidate_brier,"
            "production_accuracy,production_brier,offline_log_loss,"
            "offline_raw_log_loss,offline_baseline_log_loss,offline_ece"
        ),
        order="registered_at.desc",
    )
    return pd.DataFrame(rows)


@st.cache_data(ttl=LIVE_DATA_TTL_SECONDS, show_spinner=False)
def load_match_availability(
    home_team_id: int, away_team_id: int, match_id: int | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the latest public squad context for the two teams in a fixture.

    Player rows are deduplicated to the newest record per player so historical
    availability rows never mix into the current squad view. When the team
    snapshot table is not exposed by the environment, the freshness reference
    is derived from the newest per-team row timestamp instead of reporting
    every team as unknown.
    """
    team_filter = f"(team_id.eq.{home_team_id},team_id.eq.{away_team_id})"
    player_filters = {"or": team_filter}
    if match_id is not None:
        player_filters["match_id"] = f"eq.{match_id}"
    db = get_db()
    try:
        players = pd.DataFrame(
            db.select_all(
                "player_availability",
                columns="match_id,team_id,player_name,status,updated_at",
                filters=player_filters,
                order="updated_at.desc",
            )
        )
    except Exception as exc:
        LOGGER.warning(
            "Fixture-scoped player availability unavailable: %s", type(exc).__name__
        )
        players = pd.DataFrame()
    snapshots = pd.DataFrame()
    try:
        snapshots = pd.DataFrame(
            db.select_all(
                "team_availability_status",
                columns="team_id,refreshed_at,available_count,unavailable_count",
                filters={"or": team_filter},
            )
        )
    except Exception as exc:
        LOGGER.warning(
            "team_availability_status unavailable, deriving freshness from rows: %s",
            exc,
        )
    if not players.empty:
        if "match_id" not in players:
            players["match_id"] = None
        if match_id is not None:
            players = players[
                pd.to_numeric(players["match_id"], errors="coerce") == match_id
            ].copy()
        if players.empty:
            return players, snapshots
        players["updated_at"] = pd.to_datetime(
            players["updated_at"], utc=True, errors="coerce"
        )
        players = (
            players.dropna(subset=["updated_at"])
            .sort_values("updated_at", ascending=False)
            .drop_duplicates(
                subset=["match_id", "team_id", "player_name"], keep="first"
            )
        )
        if snapshots.empty:
            derived = (
                players.assign(
                    refreshed_at=pd.to_datetime(
                        players["updated_at"], utc=True, errors="coerce"
                    )
                )
                .dropna(subset=["refreshed_at"])
                .groupby("team_id", as_index=False)["refreshed_at"]
                .max()
            )
            snapshots = derived
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
def load_recent_odds_for_matches(match_ids: tuple[int, ...]) -> pd.DataFrame:
    """Latest odds quote per match for a small, visible match window."""
    if not match_ids:
        return pd.DataFrame()
    ids_filter = "in.(" + ",".join(str(int(match_id)) for match_id in match_ids) + ")"
    rows = get_db().select_all(
        "odds_quote_history",
        columns="match_id,bookmaker,odds,captured_at",
        filters={"match_id": ids_filter},
        order="captured_at.desc",
    )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["captured_at"] = pd.to_datetime(frame["captured_at"], utc=True).dt.tz_convert(
        "Europe/Istanbul"
    )
    return frame.drop_duplicates("match_id", keep="first")


@st.cache_data(ttl=LIVE_DATA_TTL_SECONDS, show_spinner=False)
def load_evaluated_predictions(limit: int = 1_000) -> pd.DataFrame:
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
            "over_2_5_brier_score,btts_actual,btts_was_correct,btts_brier_score,"
            "market_probabilities,market_performance"
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
    latest_path = model_dir / "latest.joblib"
    if not latest_path.is_file():
        return None
    bundle = joblib.load(latest_path)
    model_version = str(bundle.get("model_version", "")).strip()
    if not model_version:
        return None
    metadata_path = model_dir / f"{model_version}.json"
    if not metadata_path.is_file():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["active_model_version"] = model_version
    return metadata


def clear_app_cache() -> None:
    """Clear cached values and the read-only client after a configuration refresh."""
    st.cache_data.clear()
    st.cache_resource.clear()
