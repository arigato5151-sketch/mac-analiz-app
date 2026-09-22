"""Auditable live prediction results for completed fixtures."""

import sys
from pathlib import Path

import streamlit as st
from st_aggrid import AgGrid

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.components.data import load_evaluated_predictions
from app.components.grid import build_read_only_grid_options
from app.components.metrics import format_accuracy, summarize_binary_accuracy
from app.components.model_registry import (
    active_production_version,
    production_version_for,
    version_label,
)
from app.components.ui import (
    configure_page,
    disclaimer,
    evaluated_result_display,
    outcome_prediction_signal,
    page_header,
    section_intro,
)


configure_page("Tahmin Sonuçları")
page_header(
    "Tahmin sonuçları",
    "Tamamlanan maçlarda modelin ne kadar isabetli olduğunu sürüm, lig, tarih ve güven seviyesine göre denetleyin.",
    eyebrow="ŞEFFAF PERFORMANS",
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
except Exception as exc:
    st.error(f"Tahmin sonuçları yüklenemedi: {exc}")
    st.stop()

if evaluations.empty:
    st.info("Henüz değerlendirilmiş tahmin yok. Maçlar tamamlandıkça burada görünür.")
    st.stop()

if len(evaluations) >= 1_000:
    st.caption(
        "Denetim listesi en güncel 1.000 değerlendirmeyi gösterir; daha eski "
        "kayıtlar bu ekranda yer almaz."
    )

date_bounds = evaluations["match_date"].dt.date
with st.container(border=True):
    st.markdown("**Sonuçları filtrele**")
    first_row = st.columns(3)
    model_versions = evaluations["model_version"].dropna().astype(str).unique().tolist()
    sample_sizes = evaluations.groupby("model_version").size().to_dict()
    default_version = production_version_for(model_versions)
    selected_model = first_row[0].selectbox(
        "Model sürümü",
        ["Tümü", *model_versions],
        index=(["Tümü", *model_versions].index(default_version) if default_version else 0),
        format_func=lambda value: (
            value
            if value == "Tümü"
            else version_label(value, sample_size=sample_sizes.get(value, 0))
        ),
        help="Varsayılan, üretim modelidir; her sürümün rolü ve örneklemi etikette görünür.",
    )
    leagues = ["Tümü", *sorted(evaluations["league_name"].dropna().unique())]
    selected_league = first_row[1].selectbox("Lig", leagues)
    status = first_row[2].selectbox("1-X-2 sonucu", ["Tümü", "Doğru", "Yanlış"])
    second_row = st.columns(2)
    confidence = second_row[0].selectbox(
        "1-X-2 güveni", ["Tümü", "Güçlü", "Orta", "Düşük"]
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
if selected_model != "Tümü":
    filtered = filtered[filtered["model_version"].astype(str) == selected_model]
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
if len(filtered) < 100:
    st.info(
        f"Bu filtrede yalnızca {len(filtered)} maç var. Başarı oranını kesin bir "
        "model performansı olarak yorumlamak için en az 100 maç bekleyin."
    )

display_df = evaluated_result_display(filtered)
grid_options = build_read_only_grid_options(display_df)

st.subheader("Maç bazlı sonuçlar")
section_intro("✓ doğru, ✕ yanlış tahmini gösterir; Brier skorunda daha düşük değer daha iyidir.")
AgGrid(
    display_df,
    gridOptions=grid_options,
    fit_columns_on_grid_load=True,
    theme="streamlit",
    enable_enterprise_modules=False,
    height=680,
)
st.caption(
    "1-X-2, Üst/Alt 2.5 ve KG Var/Yok sonuçları %50 sınıflandırma eşiğiyle ayrı ayrı "
    "değerlendirilir. Güven etiketi ve Brier skoru yalnızca 1-X-2 tahminine aittir."
)
