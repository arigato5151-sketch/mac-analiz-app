"""Statistics archive screen: server-paged match history and filters."""

from __future__ import annotations

import math
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from app.archive_data import (
    count_completed_match_history,
    load_completed_match_history,
)
from app.components.data import load_reference_catalog
from app.localization import COUNTRY_NAMES_TR, LEAGUE_NAMES_TR

ISTANBUL = ZoneInfo("Europe/Istanbul")


def _reset_archive_page() -> None:
    st.session_state["archive_page"] = 1


def render_archive() -> None:
    """Render the completed-match archive using server-side filters and paging."""
    try:
        teams, leagues = load_reference_catalog()
    except Exception as exc:
        st.error(f"Maç istatistikleri şu anda yüklenemedi: {exc}")
        return
    team_names = {
        int(row.id): str(row.name)
        for row in teams.itertuples(index=False)
        if pd.notna(row.id) and pd.notna(row.name)
    } if not teams.empty else {}
    league_names = {
        int(row.id): LEAGUE_NAMES_TR.get(str(row.name), str(row.name))
        for row in leagues.itertuples(index=False)
        if pd.notna(row.id) and pd.notna(row.name)
    } if not leagues.empty else {}
    league_countries = {
        int(row.id): COUNTRY_NAMES_TR.get(str(row.country), str(row.country))
        for row in leagues.itertuples(index=False)
        if pd.notna(row.id) and pd.notna(row.country)
    } if not leagues.empty else {}
    c1, c2 = st.columns(2)
    with c1:
        selected = st.multiselect(
            "Lig", sorted(set(league_names.values())), key="archive_leagues",
            on_change=_reset_archive_page,
        )
    with c2:
        query = st.text_input(
            "Takım ara", key="archive_search", placeholder="Takım adı",
            on_change=_reset_archive_page,
        )
    selected_names = set(selected)
    league_ids = tuple(sorted(
        league_id for league_id, name in league_names.items() if name in selected_names
    )) if selected else None
    team_ids: tuple[int, ...] | None = None
    if query.strip():
        q = query.casefold().strip()
        team_ids = tuple(
            sorted(team_id for team_id, name in team_names.items() if q in name.casefold())
        )
    try:
        total = count_completed_match_history(league_ids=league_ids, team_ids=team_ids)
    except Exception as exc:
        st.error(f"Arşiv sonuçları sayılamadı: {exc}")
        return
    if not total:
        st.info("Filtrelere uyan tamamlanmış maç bulunamadı.")
        return
    page_size = 200
    page_count = max(1, math.ceil(total / page_size))
    page = min(max(1, int(st.session_state.get("archive_page", 1))), page_count)
    st.session_state["archive_page"] = page
    page = int(st.number_input(
        "Sayfa", min_value=1, max_value=page_count, value=page, step=1, key="archive_page"
    ))
    try:
        page_rows = load_completed_match_history(
            offset=(page - 1) * page_size,
            limit=page_size,
            league_ids=league_ids,
            team_ids=team_ids,
        )
    except Exception as exc:
        st.error(f"Arşiv sayfası yüklenemedi: {exc}")
        return
    matches = pd.DataFrame(page_rows)
    if matches.empty:
        st.info("Bu sayfada maç bulunamadı; veriler yenilenmiş olabilir.")
        return
    matches["Ev sahibi"] = matches["home_team_id"].map(team_names).fillna("Bilinmeyen takım")
    matches["Deplasman"] = matches["away_team_id"].map(team_names).fillna("Bilinmeyen takım")
    matches["Lig"] = matches["league_id"].map(league_names).fillna("Bilinmeyen lig")
    matches["Ülke"] = matches["league_id"].map(league_countries).fillna("—")
    matches["Tarih"] = (
        pd.to_datetime(matches["match_date"], utc=True, errors="coerce")
        .dt.tz_convert(ISTANBUL)
    )
    matches["Skor"] = (
        matches["home_score"].astype("Int64").astype(str)
        + " - "
        + matches["away_score"].astype("Int64").astype(str)
    )
    columns = [
        "Tarih", "Ülke", "Lig", "Ev sahibi", "Skor", "Deplasman", "home_xg", "away_xg",
        "home_xa", "away_xa", "home_corners", "away_corners", "home_shots", "away_shots",
        "home_shots_on_target", "away_shots_on_target", "home_possession", "away_possession",
    ]
    labels = {
        "home_xg": "Ev xG", "away_xg": "Dep xG", "home_xa": "Ev xA", "away_xa": "Dep xA",
        "home_corners": "Ev korner", "away_corners": "Dep korner", "home_shots": "Ev şut",
        "away_shots": "Dep şut", "home_shots_on_target": "Ev isabetli şut",
        "away_shots_on_target": "Dep isabetli şut", "home_possession": "Ev topa sahip olma %",
        "away_possession": "Dep topa sahip olma %",
    }
    st.caption(f"{total:,} maç · {page} / {page_count} sayfa")
    st.dataframe(matches[columns].rename(columns=labels), width="stretch", hide_index=True)
