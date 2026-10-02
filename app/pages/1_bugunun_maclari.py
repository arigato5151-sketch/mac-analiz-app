"""Upcoming fixtures with league and day filters."""

import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import streamlit as st
from st_aggrid import AgGrid

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.components.data import load_upcoming_dashboard
from app.components.grid import (
    build_read_only_grid_options,
    fit_match_dashboard_columns,
)
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
    matches = load_upcoming_dashboard()
except Exception:
    st.error("Maçlar şu anda yüklenemiyor. Birkaç dakika sonra tekrar deneyin.")
    st.stop()

if matches.empty:
    st.info(
        "Yaklaşan maç bulunamadı. Takip edilen liglerde veri henüz yenilenmemiş "
        "veya seçili dönem boş olabilir."
    )
    st.stop()

with st.container(border=True):
    st.markdown("**Maçları filtrele**")
    if st.button("Lig filtresini sıfırla", key="reset_fixture_filters"):
        st.session_state.pop("fixture_leagues", None)
        st.rerun()
    left, right = st.columns(2)
    league_options = sorted(matches["league_name"].dropna().unique())
    league = left.multiselect(
        "Ligler",
        league_options,
        placeholder="Tüm ligler",
        key="fixture_leagues",
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
    st.info(
        "Bu filtrelerle maç bulunamadı. Lig veya gün seçimini genişletip tekrar deneyin."
    )
else:
    overview_tab, markets_tab = st.tabs(["Hızlı görünüm", "Ayrıntılı pazarlar"])
    with overview_tab:
        overview_df = compact_dashboard_display(filtered)
        AgGrid(
            overview_df,
            gridOptions=fit_match_dashboard_columns(
                build_read_only_grid_options(overview_df)
            ),
            fit_columns_on_grid_load=True,
            reload_data=True,
            theme="streamlit",
            enable_enterprise_modules=False,
            height=640,
        )
    with markets_tab:
        display_df = dashboard_display(filtered)
        AgGrid(
            display_df,
            gridOptions=fit_match_dashboard_columns(
                build_read_only_grid_options(display_df)
            ),
            fit_columns_on_grid_load=True,
            reload_data=True,
            theme="streamlit",
            enable_enterprise_modules=False,
            height=640,
        )

st.caption(
    "Öne çıkan sinyal, 1-X-2, Üst 2.5 ve KG Var olasılıklarının en yükseğidir. "
    "%60 ve üzeri güçlü, %55–59.9 orta sinyaldir; kesinlik değildir."
)
