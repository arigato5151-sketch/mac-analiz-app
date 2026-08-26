"""Tracked leagues and public data refresh status."""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.components.ui import configure_page
from config.leagues import TRACKED_LEAGUES


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

st.subheader("Veri güncelleme")
st.caption(
    "Fikstür, form, sakatlık ve tahminler GitHub Actions üzerinden otomatik "
    "güncellenir. Bu arayüz yalnızca sınırlı, salt-okunur Supabase erişimi kullanır."
)
