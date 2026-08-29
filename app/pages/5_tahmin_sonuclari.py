"""Auditable live prediction results for completed fixtures."""

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.components.data import load_evaluated_predictions
from app.components.ui import configure_page, disclaimer, evaluated_result_display, outcome_prediction_signal


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

date_bounds = evaluations["match_date"].dt.date
filters = st.columns(4)
leagues = ["Tümü", *sorted(evaluations["league_name"].dropna().unique())]
selected_league = filters[0].selectbox("Lig", leagues)
status = filters[1].selectbox("Sonuç", ["Tümü", "Doğru", "Yanlış"])
confidence = filters[2].selectbox("1-X-2 güveni", ["Tümü", "Güçlü", "Orta", "Düşük"])
selected_dates = filters[3].date_input(
    "Maç tarihi",
    value=(date_bounds.min(), date_bounds.max()),
    min_value=date_bounds.min(),
    max_value=date_bounds.max(),
)

filtered = evaluations.copy()
filtered["_confidence"] = [
    signal_confidence for _, _, signal_confidence in filtered.apply(outcome_prediction_signal, axis=1)
]
if len(selected_dates) == 2:
    start_date, end_date = selected_dates
    filtered = filtered[
        filtered["match_date"].dt.date.between(start_date, end_date)
    ]
if selected_league != "Tümü":
    filtered = filtered[filtered["league_name"] == selected_league]
if status == "Doğru":
    filtered = filtered[filtered["was_correct"]]
elif status == "Yanlış":
    filtered = filtered[~filtered["was_correct"]]
if confidence != "Tümü":
    filtered = filtered[filtered["_confidence"] == confidence]

summary = st.columns(4)
summary[0].metric("Değerlendirilen maç", len(filtered))
if filtered.empty:
    summary[1].metric("İsabet", "—")
    summary[2].metric("Ortalama Brier", "—")
    summary[3].metric("Güçlü 1-X-2 sinyali", "0")
    st.info("Bu filtrelerle eşleşen değerlendirilmiş tahmin bulunamadı.")
    st.stop()

summary[1].metric("İsabet", f"%{filtered['was_correct'].mean() * 100:.1f}")
summary[2].metric("Ortalama Brier", f"{filtered['brier_score'].astype(float).mean():.3f}")
summary[3].metric("Güçlü 1-X-2 sinyali", int((filtered["_confidence"] == "Güçlü").sum()))

st.dataframe(evaluated_result_display(filtered), hide_index=True, use_container_width=True, height=680)
st.caption(
    "Güven etiketi yalnızca değerlendirilen 1-X-2 tahminine aittir. Brier skoru düştükçe "
    "olasılık tahmininin kalitesi artar; 1-X-2 sonuç olasılıkları üzerinden hesaplanır."
)
