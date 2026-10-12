"""Upcoming fixtures with league and day filters."""

import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.components.data import load_upcoming_dashboard
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
    "Lig ve gün filtreleriyle maçları bulun, seçtiğiniz maçın analizini açın.",
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
        default=[istanbul_today] if istanbul_today in available_days else [],
        format_func=lambda value: "Bugün" if value == istanbul_today else value.strftime("%d.%m.%Y"),
    )
filtered = matches[matches["match_date"].dt.date.isin(days)]
if league:
    filtered = filtered[filtered["league_name"].isin(league)]

st.subheader(f"Maçlar · {len(filtered)} sonuç")
section_intro("Bir maçı seçerek detaylı analizini açın.")
if filtered.empty:
    st.info(
        "Bu filtrelerle maç bulunamadı. Lig veya gün seçimini genişletip tekrar deneyin."
    )
else:
    display_df = dashboard_display(filtered)[["Tarih", "Lig", "Maç"]].copy()
    display_df["match_id"] = filtered["id"].astype(int).to_numpy()
    builder = GridOptionsBuilder.from_dataframe(display_df)
    builder.configure_default_column(
        filter=False,
        sortable=True,
        resizable=True,
        wrapHeaderText=True,
        autoHeaderHeight=True,
    )
    builder.configure_column("match_id", hide=True)
    builder.configure_column("Maç", flex=1, minWidth=220)
    builder.configure_selection(selection_mode="single", use_checkbox=False)
    grid_options = builder.build()
    grid_options["pagination"] = True
    grid_options["paginationPageSize"] = 25
    grid_response = AgGrid(
        display_df,
        gridOptions=grid_options,
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        update_on=["selectionChanged"],
        fit_columns_on_grid_load=True,
        reload_data=True,
        theme="streamlit",
        enable_enterprise_modules=False,
        height=640,
        key="fixture_match_list",
    )
    selected_rows = grid_response.get("selected_rows")
    if isinstance(selected_rows, pd.DataFrame) and not selected_rows.empty:
        selected_match_id = selected_rows.iloc[0].get("match_id")
    elif isinstance(selected_rows, list) and selected_rows:
        selected_match_id = selected_rows[0].get("match_id")
    else:
        selected_match_id = None
    if selected_match_id is not None:
        st.session_state["selected_match_id"] = int(selected_match_id)
        st.switch_page("pages/2_mac_detay.py")
