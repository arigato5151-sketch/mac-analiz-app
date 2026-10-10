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
    fit_read_only_grid_columns,
)
from app.components.ui import (
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
section_intro("Tüm maçlar ve ayrıntılı pazar olasılıkları.")
if filtered.empty:
    st.info(
        "Bu filtrelerle maç bulunamadı. Lig veya gün seçimini genişletip tekrar deneyin."
    )
else:
    display_df = dashboard_display(filtered)
    compact_columns = [
        column for column in display_df.columns if column not in {"Tarih", "Lig", "Maç"}
    ]
    for column in compact_columns:
        display_df[column] = display_df[column].str.replace(" · ", " ", regex=False)
    AgGrid(
        display_df,
        gridOptions=fit_read_only_grid_columns(
            build_read_only_grid_options(display_df)
        ),
        fit_columns_on_grid_load=False,
        reload_data=True,
        theme="streamlit",
        enable_enterprise_modules=False,
        height=640,
    )

st.caption(
    "Öne çıkan sinyal, Maç Sonucu, 2,5 Gol Alt/Üst ve Karşılıklı Gol pazarlarındaki "
    "en yüksek olasılığı gösterir. "
    "%60 ve üzeri güçlü, %55–59.9 orta sinyaldir; kesinlik değildir."
)
