"""Auditable live prediction results for completed fixtures."""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.components.data import load_evaluated_predictions
from app.components.metrics import format_accuracy, summarize_binary_accuracy
from app.components.model_registry import (
    active_production_version,
    version_label,
)
from app.components.ui import (
    configure_page,
    disclaimer,
    outcome_prediction_signal,
    page_header,
    section_intro,
)
configure_page("Tahmin Sonuçları")
page_header(
    "Tahmin geçmişi",
    "Geçmiş maçlarda tahminlerin ne kadar isabetli olduğunu ve hangi koşullarda yanıldığını inceleyin.",
    eyebrow="TAHMİN GEÇMİŞİ",
)
disclaimer()
section_intro("Yalnızca maç başlamadan önce kaydedilmiş ve otomatik değerlendirilmiş tahminler gösterilir.")

production_version = active_production_version()
if production_version:
    st.caption(f"Üretim modeli: {version_label(production_version)}")
else:
    st.caption("Üretim modeli bilgisi doğrulanamadı.")

try:
    evaluations = load_evaluated_predictions(limit=1_000)
except Exception:
    st.error("Tahmin geçmişi şu anda kullanılamıyor. Birkaç dakika sonra tekrar deneyin.")
    st.stop()

if evaluations.empty:
    st.info(
        "Henüz değerlendirilmiş tahmin yok. Maç tamamlandıktan sonra sonuçlar "
        "otomatik olarak burada görünür."
    )
    st.stop()

if len(evaluations) >= 1_000:
    st.caption(
        "Denetim listesi en güncel 1.000 değerlendirmeyi gösterir; daha eski "
        "kayıtlar bu ekranda yer almaz."
    )

date_bounds = evaluations["match_date"].dt.date
with st.container(border=True):
    st.markdown("**Sonuçları filtrele**")
    first_row = st.columns(2)
    leagues = ["Tümü", *sorted(evaluations["league_name"].dropna().unique())]
    selected_league = first_row[0].selectbox("Lig", leagues)
    status = first_row[1].selectbox("Maç Sonucu", ["Tümü", "Doğru", "Yanlış"])
    second_row = st.columns(2)
    confidence = second_row[0].selectbox(
        "Maç Sonucu güveni", ["Tümü", "Güçlü", "Orta", "Düşük"]
    )
    selected_dates = second_row[1].date_input(
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
            "Maç Sonucu isabet",
            "2,5 Gol Alt/Üst isabet",
            "Karşılıklı Gol isabet",
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
    "Maç Sonucu isabet",
    format_accuracy(outcome_accuracy),
    delta=f"n={outcome_accuracy.sample_size}",
    delta_color="off",
)
summary[2].metric(
    "2,5 Gol Alt/Üst isabet",
    format_accuracy(over_accuracy),
    delta=f"n={over_accuracy.sample_size}",
    delta_color="off",
)
summary[3].metric(
    "Karşılıklı Gol isabet",
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
if len(filtered) < 100:
    st.info(
        f"Bu filtrede yalnızca {len(filtered)} maç var. Başarı oranını kesin bir "
        "model performansı olarak yorumlamak için en az 100 maç bekleyin."
    )

st.subheader("Sonuç mesajları")
st.caption("En yeni değerlendirmeler burada gösterilir.")
for _, result_row in filtered.head(8).iterrows():
    is_correct = bool(result_row["was_correct"])
    status_label = "✓ Doğru" if is_correct else "✕ Yanlış"
    status_color = "#35C48D" if is_correct else "#FF6B6B"
    st.markdown(
        (
            '<div style="border:1px solid rgba(120,120,120,.22);border-radius:.8rem;'
            'padding:.7rem .8rem;margin-bottom:.65rem;background:rgba(23,28,36,.72);">'
            f'<div style="font-size:.76rem;color:rgba(242,245,247,.62);">{result_row["match_date"].strftime("%d.%m.%Y %H:%M")}</div>'
            f'<div style="font-weight:700;margin:.25rem 0;">{result_row["home_team"]} — {result_row["away_team"]}</div>'
            f'<div style="font-size:1.15rem;font-weight:800;">{int(result_row["home_score"])} — {int(result_row["away_score"])} '
            f'<span style="float:right;color:{status_color};font-size:.82rem;">{status_label}</span></div>'
            f'<div style="font-size:.78rem;color:rgba(242,245,247,.62);">{result_row["league_name"]}</div>'
            '</div>'
        ),
        unsafe_allow_html=True,
    )

st.caption(
    "Maç Sonucu, 2,5 Gol Alt/Üst ve Karşılıklı Gol sonuçları %50 sınıflandırma eşiğiyle ayrı ayrı "
    "değerlendirilir. Güven etiketi ve Brier skoru yalnızca Maç Sonucu tahminine aittir."
)
