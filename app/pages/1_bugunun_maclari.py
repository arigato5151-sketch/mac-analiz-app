"""Upcoming fixtures with league and day filters."""

from datetime import date

import streamlit as st

from app.components.data import load_upcoming_dashboard
from app.components.ui import configure_page, dashboard_display, disclaimer


configure_page("Bugünün Maçları")
st.title("Bugün ve Yaklaşan Maçlar")
disclaimer()

try:
    matches = load_upcoming_dashboard(3)
except Exception as exc:
    st.error(f"Maçlar yüklenemedi: {exc}")
    st.stop()

if matches.empty:
    st.info("Yaklaşan maç bulunamadı.")
    st.stop()

left, right = st.columns(2)
league = left.multiselect(
    "Ligler", sorted(matches["league_name"].dropna().unique()), placeholder="Tümü"
)
available_days = sorted(matches["match_date"].dt.date.unique())
days = right.multiselect(
    "Günler",
    available_days,
    default=available_days,
    format_func=lambda value: "Bugün" if value == date.today() else value.strftime("%d.%m.%Y"),
)
filtered = matches[matches["match_date"].dt.date.isin(days)]
if league:
    filtered = filtered[filtered["league_name"].isin(league)]

st.dataframe(
    dashboard_display(filtered), hide_index=True, use_container_width=True, height=680
)
st.caption(f"{len(filtered)} maç gösteriliyor. %60 üzerindeki değerler güçlü olasılık sinyali olarak yorumlanabilir; kesinlik değildir.")
