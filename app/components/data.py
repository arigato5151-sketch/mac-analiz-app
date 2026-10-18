"""Read-only data access for the match statistics screen."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from config.settings import get_public_supabase_settings
from db.db_client import PublicSupabaseRestClient


@st.cache_resource(show_spinner=False)
def get_db() -> PublicSupabaseRestClient:
    """Create a cached client that cannot write to Supabase."""
    settings = get_public_supabase_settings()
    return PublicSupabaseRestClient(settings.supabase_url, settings.supabase_anon_key)


@st.cache_data(ttl=900, show_spinner="Maç istatistikleri yükleniyor…")
def load_completed_match_history() -> list[dict[str, Any]]:
    """Read finished matches with recorded scores and available statistics."""
    return get_db().select_all(
        "matches",
        columns=(
            "id,league_id,home_team_id,away_team_id,match_date,status,"
            "home_score,away_score,home_xg,away_xg,home_xa,away_xa,"
            "home_corners,away_corners"
        ),
        filters={
            "status": "eq.finished",
            "home_score": "not.is.null",
            "away_score": "not.is.null",
        },
        order="match_date.desc,id.desc",
    )


@st.cache_data(ttl=900, show_spinner=False)
def load_reference_catalog() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read team and league names used to label stored match statistics."""
    db = get_db()
    return (
        pd.DataFrame(db.select_all("teams", columns="id,name")),
        pd.DataFrame(db.select_all("leagues", columns="id,name,country")),
    )
