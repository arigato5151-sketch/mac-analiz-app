"""Upcoming fixtures with league and day filters."""

from datetime import datetime
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import streamlit as st
from st_aggrid import AgGrid

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.components.data import load_upcoming_dashboard
from app.components.grid import build_read_only_grid_options
from app.components.ui import (
    compact_dashboard_display,
    configure_page,
    dashboard_display,
    disclaimer,
    page_header,
    section_intro,
)


configure_page("Bugünün Maçları")
page_header(
    "Bugün ve yaklaşan maçlar",
    "Lig ve gün filtreleriyle programı daraltın, temel olasılıkları karşılaştırın ve istediğiniz maça odaklanın.",
    eyebrow="MAÇ MERKEZİ",
)
disclaimer()

try:
    matches = load_upcoming_dashboard(3)
except Exception as exc:
    st.error(f"Maçlar yüklenemedi: {exc}")
    st.stop()

if matches.empty:
    st.info("Yaklaşan maç bulunamadı.")
    st.stop()

with st.container(border=True):
    st.markdown("**Maçları filtrele**")
    left, right = st.columns(2)
    league = left.multiselect(
        "Ligler", sorted(matches["league_name"].dropna().unique()), placeholder="Tüm ligler"
    )
    available_days = sorted(matches["match_date"].dt.date.unique())
    istanbul_today = datetime.now(ZoneInfo("Europe/Istanbul")).date()
    days = right.multiselect(
        "Günler",
        available_days,
        default=available_days,
        format_func=lambda value: "Bugün" if value == istanbul_today else value.strftime("%d.%m.%Y"),
    )
filtered = matches[matches["match_date"].dt.date.isin(days)]
if league:
    filtered = filtered[filtered["league_name"].isin(league)]

st.subheader(f"Maçlar · {len(filtered)} sonuç")
section_intro("Hızlı görünüm karar için gereken temel alanları, ayrıntılı görünüm tüm pazarları gösterir.")
if filtered.empty:
    st.info("Seçtiğiniz filtrelerle eşleşen yaklaşan maç bulunamadı.")
else:
    overview_tab, markets_tab = st.tabs(["Hızlı görünüm", "Ayrıntılı pazarlar"])
    with overview_tab:
        overview_df = compact_dashboard_display(filtered)
        AgGrid(
            overview_df,
            gridOptions=build_read_only_grid_options(overview_df),
            fit_columns_on_grid_load=True,
            theme="streamlit",
            enable_enterprise_modules=False,
            height=640,
        )
    with markets_tab:
        display_df = dashboard_display(filtered)
        AgGrid(
            display_df,
            gridOptions=build_read_only_grid_options(display_df),
            fit_columns_on_grid_load=True,
            theme="streamlit",
            enable_enterprise_modules=False,
            height=640,
        )

st.caption(
    "Öne çıkan sinyal, 1-X-2, Üst 2.5 ve KG Var olasılıklarının en yükseğidir. "
    "%60 ve üzeri güçlü, %55–59.9 orta sinyaldir; kesinlik değildir."
)
