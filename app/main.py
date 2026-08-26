"""Streamlit entry point and compact upcoming-match dashboard."""

from __future__ import annotations

import streamlit as st

from app.components.data import load_latest_model_metadata, load_upcoming_dashboard
from app.components.ui import configure_page, dashboard_display, disclaimer


configure_page("Ana Sayfa")
st.title("⚽ Maç Analiz ve Tahmin")
st.caption("25 lig · Poisson baseline · XGBoost · şeffaf performans takibi")
disclaimer()

try:
    matches = load_upcoming_dashboard(3)
    metadata = load_latest_model_metadata()
except Exception as exc:  # Streamlit must remain usable during upstream outages.
    st.error(f"Veriler şu anda yüklenemedi: {exc}")
    st.stop()

col1, col2, col3 = st.columns(3)
col1.metric("Önümüzdeki 3 gün", len(matches))
col2.metric("Tahmin hazır", int(matches.get("model_version", []).notna().sum()) if not matches.empty and "model_version" in matches else 0)
col3.metric("Model test Log Loss", f"{metadata['metrics']['log_loss']:.3f}" if metadata else "—")

st.subheader("Yaklaşan maçlar")
if matches.empty:
    st.info("Seçili liglerde önümüzdeki üç gün için planlanmış maç bulunamadı.")
else:
    leagues = ["Tümü", *sorted(matches["league_name"].dropna().unique())]
    selected = st.selectbox("Lig", leagues)
    filtered = matches if selected == "Tümü" else matches[matches["league_name"] == selected]
    st.dataframe(
        dashboard_display(filtered),
        hide_index=True,
        use_container_width=True,
        height=min(700, 40 + 35 * len(filtered)),
    )

st.caption("Detaylı analiz için sol menüden Maç Detay sayfasını açın.")
