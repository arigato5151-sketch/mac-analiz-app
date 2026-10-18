"""Minimal match statistics application entry point."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.components.data import load_completed_match_history, load_reference_catalog

st.set_page_config(page_title="Maç İstatistikleri", page_icon="⚽", layout="wide")
st.markdown(
    "<style>.block-container{max-width:1320px;padding-top:1.5rem}</style>",
    unsafe_allow_html=True,
)

st.markdown(
    """
    <section class="app-hero">
      <div class="app-eyebrow">Arşiv</div>
      <h1>Maç istatistikleri</h1>
      <p>Tamamlanmış maçların sonuç ve saha içi verileri.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

try:
    matches = pd.DataFrame(load_completed_match_history())
    teams, leagues = load_reference_catalog()
except Exception as exc:
    st.error(f"Maç istatistikleri şu anda yüklenemedi: {exc}")
    st.stop()

if matches.empty:
    st.info("Kayıtlı tamamlanmış maç bulunamadı.")
    st.stop()

home = teams.rename(columns={"id": "home_team_id", "name": "Ev sahibi"})[
    ["home_team_id", "Ev sahibi"]
]
away = teams.rename(columns={"id": "away_team_id", "name": "Deplasman"})[
    ["away_team_id", "Deplasman"]
]
league_labels = leagues.rename(
    columns={"id": "league_id", "name": "Lig", "country": "Ülke"}
)[["league_id", "Lig", "Ülke"]]
matches = matches.merge(home, on="home_team_id", how="left")
matches = matches.merge(away, on="away_team_id", how="left")
matches = matches.merge(league_labels, on="league_id", how="left")
matches["Tarih"] = pd.to_datetime(
    matches["match_date"], utc=True, errors="coerce"
).dt.tz_convert("Europe/Istanbul")
matches["Skor"] = (
    matches["home_score"].astype("Int64").astype(str)
    + " - "
    + matches["away_score"].astype("Int64").astype(str)
)

left, right = st.columns(2)
with left:
    league_options = sorted(matches["Lig"].dropna().unique().tolist())
    selected_leagues = st.multiselect("Lig", league_options)
with right:
    search = st.text_input("Takım ara", placeholder="Takım adı")

filtered = matches.copy()
if selected_leagues:
    filtered = filtered[filtered["Lig"].isin(selected_leagues)]
if search.strip():
    query = search.strip().casefold()
    filtered = filtered[
        filtered["Ev sahibi"].fillna("").str.casefold().str.contains(query)
        | filtered["Deplasman"].fillna("").str.casefold().str.contains(query)
    ]

st.metric("Gösterilen maç", f"{len(filtered):,}")
columns = [
    "Tarih", "Ülke", "Lig", "Ev sahibi", "Skor", "Deplasman",
    "home_xg", "away_xg", "home_xa", "away_xa", "home_corners", "away_corners",
]
labels = {
    "home_xg": "Ev xG",
    "away_xg": "Dep xG",
    "home_xa": "Ev xA",
    "away_xa": "Dep xA",
    "home_corners": "Ev korner",
    "away_corners": "Dep korner",
}
display = filtered.sort_values("Tarih", ascending=False)[columns].rename(columns=labels)
st.dataframe(display, use_container_width=True, hide_index=True)
