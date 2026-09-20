"""Tracked leagues and public data refresh status."""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.components.ui import configure_page, page_header, section_intro
from config.leagues import TRACKED_LEAGUES


configure_page("Ayarlar")
page_header(
    "Ayarlar ve veri kapsamı",
    "Takip edilen ligleri ve uygulamanın veriyi nasıl güncellediğini görüntüleyin.",
    eyebrow="SİSTEM BİLGİSİ",
)

status_columns = st.columns(3)
status_columns[0].metric("Takip edilen lig", len(TRACKED_LEAGUES))
status_columns[1].metric("Arayüz erişimi", "Salt okunur")
status_columns[2].metric("Güncelleme", "Otomatik")

st.subheader("Takip edilen ligler")
section_intro("Tahmin ve sonuç ekranlarında yalnızca aşağıdaki ligler izlenir.")
st.dataframe(
    pd.DataFrame(
        [
            {"ID": league.id, "Lig": league.name, "Ülke": league.country, "Sezon": league.season}
            for league in TRACKED_LEAGUES
        ]
    ),
    hide_index=True,
    width="stretch",
    height=420,
)

st.subheader("Veri güncelleme")
with st.container(border=True):
    st.markdown(
        "**Otomatik veri akışı**\n\n"
        "Fikstür, takım formu, kadro uygunluğu ve tahminler zamanlanmış işler üzerinden "
        "yenilenir. Tamamlanan maçlar ayrıca performans ölçümüne eklenir."
    )
with st.container(border=True):
    st.markdown(
        "**Güvenli arayüz erişimi**\n\n"
        "Bu ekran yalnızca sınırlı ve salt-okunur Supabase erişimi kullanır; "
        "veri ekleyemez, değiştiremez veya silemez."
    )
