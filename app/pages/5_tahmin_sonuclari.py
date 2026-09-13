"""Auditable live prediction results for completed fixtures."""

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.components.data import load_evaluated_predictions
from app.components.metrics import format_accuracy, summarize_binary_accuracy
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
status = filters[1].selectbox("1-X-2 sonucu", ["Tümü", "Doğru", "Yanlış"])
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

summary = st.columns(5)
summary[0].metric("Değerlendirilen maç", len(filtered))
if filtered.empty:
    for column, label in zip(
        summary[1:],
        [
            "1-X-2 isabet",
            "Üst/Alt 2.5 isabet",
            "KG Var/Yok isabet",
            "Ortalama Brier",
        ],
    ):
        column.metric(label, "—")
    st.info("Bu filtrelerle eşleşen değerlendirilmiş tahmin bulunamadı.")
    st.stop()

outcome_accuracy = summarize_binary_accuracy(filtered["was_correct"])
over_accuracy = summarize_binary_accuracy(filtered["over_2_5_was_correct"])
btts_accuracy = summarize_binary_accuracy(filtered["btts_was_correct"])
brier_scores = filtered["brier_score"].dropna().astype(float)

summary[1].metric(
    "1-X-2 isabet",
    format_accuracy(outcome_accuracy),
    delta=f"n={outcome_accuracy.sample_size}",
    delta_color="off",
)
summary[2].metric(
    "Üst/Alt 2.5 isabet",
    format_accuracy(over_accuracy),
    delta=f"n={over_accuracy.sample_size}",
    delta_color="off",
)
summary[3].metric(
    "KG Var/Yok isabet",
    format_accuracy(btts_accuracy),
    delta=f"n={btts_accuracy.sample_size}",
    delta_color="off",
)
summary[4].metric(
    "Ortalama Brier",
    f"{brier_scores.mean():.3f}" if not brier_scores.empty else "—",
    delta=f"n={len(brier_scores)}",
    delta_color="off",
)

st.dataframe(
    evaluated_result_display(filtered),
    hide_index=True,
    width="stretch",
    height=680,
)
st.caption(
    "1-X-2, Üst/Alt 2.5 ve KG Var/Yok sonuçları %50 sınıflandırma eşiğiyle ayrı ayrı "
    "değerlendirilir. Güven etiketi ve Brier skoru yalnızca 1-X-2 tahminine aittir."
)
