"""Tracked leagues, quota status, and manual data refresh."""

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from app.components.data import clear_app_cache, fetch_api_status, get_db
from app.components.ui import configure_page
from config.leagues import TRACKED_LEAGUES
from config.settings import get_settings
from data_pipeline.api_client import ApiFootballClient
from data_pipeline.fetch_fixtures import sync_fixtures


configure_page("Ayarlar")
st.title("Ayarlar ve Veri Durumu")

st.subheader("Takip edilen ligler")
st.dataframe(
    pd.DataFrame(
        [
            {"ID": league.id, "Lig": league.name, "Ülke": league.country, "Sezon": league.season}
            for league in TRACKED_LEAGUES
        ]
    ),
    hide_index=True,
    use_container_width=True,
    height=420,
)

st.subheader("API-Football kullanımı")
if st.button("Kullanımı kontrol et"):
    try:
        status = fetch_api_status()
        cols = st.columns(3)
        cols[0].metric("Plan", status["plan"])
        cols[1].metric("Bugünkü kullanım", f"{status['used']:,}")
        cols[2].metric("Günlük limit", f"{status['limit']:,}")
        st.progress(min(status["used"] / status["limit"], 1.0))
    except Exception as exc:
        st.error(f"API durumu alınamadı: {exc}")

st.subheader("Manuel veri yenileme")
st.caption("Bugün ve sonraki iki günün fikstürlerini yeniden senkronize eder.")
if st.button("Fikstürleri şimdi yenile", type="primary"):
    try:
        settings = get_settings()
        today = date.today()
        with st.spinner("API-Football ve Supabase senkronize ediliyor..."):
            summary = sync_fixtures(
                ApiFootballClient(settings.api_football_key),
                get_db(),
                start_date=today,
                end_date=today + timedelta(days=2),
            )
        clear_app_cache()
        st.success(
            f"Tamamlandı: {summary.tracked_fixtures} maç, "
            f"{summary.teams_upserted} takım güncellendi."
        )
    except Exception as exc:
        st.error(f"Yenileme başarısız: {exc}")
