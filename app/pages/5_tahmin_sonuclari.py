"""Auditable live prediction results for completed fixtures."""

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.components.data import load_evaluated_predictions
from app.components.ui import configure_page, disclaimer, evaluated_result_display


configure_page("Tahmin Sonuçları")
st.title("Tahmin Sonuçları")
disclaimer()
st.caption("Yalnızca tamamlanmış ve otomatik değerlendirilmiş maçlar gösterilir.")

try:
    evaluations = load_evaluated_predictions()
except Exception as exc:
    st.error(f"Tahmin sonuçları yüklenemedi: {exc}")
    st.stop()

if evaluations.empty:
    st.info("Henüz değerlendirilmiş tahmin yok. Maçlar tamamlandıkça burada görünür.")
    st.stop()

filters = st.columns(2)
leagues = ["Tümü", *sorted(evaluations["league_name"].dropna().unique())]
selected_league = filters[0].selectbox("Lig", leagues)
status = filters[1].selectbox("Sonuç", ["Tümü", "Doğru", "Yanlış"])

filtered = evaluations.copy()
if selected_league != "Tümü":
    filtered = filtered[filtered["league_name"] == selected_league]
if status == "Doğru":
    filtered = filtered[filtered["was_correct"]]
elif status == "Yanlış":
    filtered = filtered[~filtered["was_correct"]]

summary = st.columns(3)
summary[0].metric("Değerlendirilen maç", len(filtered))
summary[1].metric("İsabet", f"%{filtered['was_correct'].mean() * 100:.1f}")
summary[2].metric("Ortalama Brier", f"{filtered['brier_score'].astype(float).mean():.3f}")

st.dataframe(evaluated_result_display(filtered), hide_index=True, use_container_width=True, height=680)
st.caption("Brier skoru düştükçe olasılık tahmininin kalitesi artar; 1-X-2 sonuç olasılıkları üzerinden hesaplanır.")
