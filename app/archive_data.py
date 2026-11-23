"""Paged, read-only data access for the completed-match archive."""

from __future__ import annotations

from typing import Any

import requests
import streamlit as st

from app.components.data import get_db
from config.settings import get_public_supabase_settings
from db.db_client import DatabaseError


def _completed_match_filters(
    league_ids: tuple[int, ...] | None, team_ids: tuple[int, ...] | None
) -> dict[str, str | int]:
    filters: dict[str, str | int] = {
        "status": "eq.finished",
        "home_score": "not.is.null",
        "away_score": "not.is.null",
    }
    if league_ids is not None:
        normalized_leagues = tuple(sorted({int(value) for value in league_ids}))
        if not normalized_leagues:
            raise ValueError("An empty league filter cannot be converted to a database query")
        filters["league_id"] = (
            f"eq.{normalized_leagues[0]}" if len(normalized_leagues) == 1
            else f"in.({','.join(map(str, normalized_leagues))})"
        )
    if team_ids is not None:
        normalized_teams = tuple(sorted({int(value) for value in team_ids}))
        if not normalized_teams:
            raise ValueError("An empty team filter cannot be converted to a database query")
        ids = ",".join(map(str, normalized_teams))
        filters["or"] = f"(home_team_id.in.({ids}),away_team_id.in.({ids}))"
    return filters


@st.cache_data(ttl=900, show_spinner=False)
def count_completed_match_history(
    league_ids: tuple[int, ...] | None = None, team_ids: tuple[int, ...] | None = None
) -> int:
    """Count archive rows without downloading them, including during hot reloads."""
    if league_ids is not None and not league_ids or team_ids is not None and not team_ids:
        return 0
    filters = _completed_match_filters(league_ids, team_ids)
    client = get_db()
    count_method = getattr(client, "count", None)
    if callable(count_method):
        return int(count_method("matches", filters=filters))

    # Streamlit may retain an older imported client class during a hot reload.
    # Keep the archive usable until the process restarts with the current client.
    settings = get_public_supabase_settings()
    headers = {
        "apikey": settings.supabase_anon_key,
        "Prefer": "count=exact",
        "Range": "0-0",
    }
    if settings.supabase_anon_key.count(".") == 2:
        headers["Authorization"] = f"Bearer {settings.supabase_anon_key}"
    try:
        response = requests.get(
            f"{settings.supabase_url}/rest/v1/matches",
            params={"select": "id", **filters},
            headers=headers,
            timeout=settings.request_timeout_seconds,
        )
    except requests.RequestException as exc:
        raise DatabaseError("Completed match count could not be loaded.") from exc
    if not response.ok:
        raise DatabaseError(f"Completed match count failed (HTTP {response.status_code}).")
    _, separator, total = response.headers.get("Content-Range", "").rpartition("/")
    if not separator or not total.isdigit():
        raise DatabaseError("Completed match count response was invalid.")
    return int(total)


@st.cache_data(ttl=900, show_spinner="Maç istatistikleri yükleniyor…")
def load_completed_match_history(
    *,
    offset: int = 0,
    limit: int = 200,
    league_ids: tuple[int, ...] | None = None,
    team_ids: tuple[int, ...] | None = None,
) -> list[dict[str, Any]]:
    """Read one filtered page of completed match history."""
    if offset < 0 or limit < 1 or limit > 1_000:
        raise ValueError("Archive page bounds are invalid")
    if league_ids is not None and not league_ids or team_ids is not None and not team_ids:
        return []
    return get_db().select(
        "matches",
        columns=(
            "id,league_id,home_team_id,away_team_id,match_date,status,"
            "home_score,away_score,home_halftime_score,away_halftime_score,home_xg,away_xg,home_xa,away_xa,"
            "home_corners,away_corners,home_shots,away_shots,home_shots_on_target,away_shots_on_target,"
            "home_possession,away_possession"
        ),
        filters=_completed_match_filters(league_ids, team_ids),
        limit=limit,
        offset=offset,
        order="match_date.desc,id.desc",
    )
