"""Streamlit entry point and compact upcoming-match dashboard."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Streamlit Cloud starts with ``app/`` on the import path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.components.data import load_latest_model_metadata, load_upcoming_dashboard
from app.components.ui import (
    compact_dashboard_display,
    configure_page,
    dashboard_display,
    disclaimer,
    page_header,
    section_intro,
)
from config.settings import UPCOMING_HORIZON_DAYS


configure_page("Ana Sayfa")
page_header(
    "Maçları tek bakışta değerlendirin",
    "Yaklaşan karşılaşmaları, model olasılıklarını ve öne çıkan sinyalleri sade bir görünümde inceleyin.",
    eyebrow="25 LİG · GÜNCEL TAHMİNLER",
)
disclaimer()

try:
    matches = load_upcoming_dashboard()
    metadata = load_latest_model_metadata()
except Exception as exc:  # Streamlit must remain usable during upstream outages.
    st.error(f"Veriler şu anda yüklenemedi: {exc}")
    st.stop()

col1, col2, col3 = st.columns(3)
ready_count = (
    int(matches["model_version"].notna().sum())
    if not matches.empty and "model_version" in matches
    else 0
)
col1.metric(
    "Yaklaşan maç",
    len(matches),
    help=f"Önümüzdeki {UPCOMING_HORIZON_DAYS} gündeki maç sayısı",
)
col2.metric(
    "Analiz hazır",
    f"{ready_count}/{len(matches)}",
    help="Model olasılıkları hazırlanmış maçlar",
)
col3.metric(
    "Offline Brier",
    f"{metadata['metrics']['brier_score']:.3f}" if metadata else "—",
    help="Daha düşük değer, olasılık tahminlerinin daha iyi olduğunu gösterir.",
)

if not matches.empty and "model_version" in matches:
    active_versions = matches["model_version"].dropna().astype(str)
    if not active_versions.empty:
        st.caption(f"Yaklaşan maçlarda kullanılan model: `{active_versions.iloc[0]}`")

st.subheader("Yaklaşan maçlar")
section_intro("Önce temel olasılıkları inceleyin; alternatif pazarları gerektiğinde açın.")
if matches.empty:
    st.info(
        f"Seçili liglerde önümüzdeki {UPCOMING_HORIZON_DAYS} gün için "
        "planlanmış maç bulunamadı."
    )
else:
    leagues = ["Tümü", *sorted(matches["league_name"].dropna().unique())]
    selected = st.selectbox("Lig filtresi", leagues)
    filtered = matches if selected == "Tümü" else matches[matches["league_name"] == selected]
    overview_tab, markets_tab = st.tabs(["Hızlı görünüm", "Tüm pazarlar"])
    with overview_tab:
        st.dataframe(
            compact_dashboard_display(filtered),
            hide_index=True,
            width="stretch",
            height=min(680, 40 + 35 * len(filtered)),
        )
    with markets_tab:
        st.dataframe(
            dashboard_display(filtered),
            hide_index=True,
            width="stretch",
            height=min(680, 40 + 35 * len(filtered)),
        )
    st.caption(f"{len(filtered)} maç gösteriliyor.")

navigation = st.columns(3)
navigation[0].page_link("pages/1_bugunun_maclari.py", label="Tüm maçları filtrele", icon="📅")
navigation[1].page_link("pages/2_mac_detay.py", label="Maç detayını aç", icon="🔎")
navigation[2].page_link("pages/3_model_performans.py", label="Model performansı", icon="📈")
