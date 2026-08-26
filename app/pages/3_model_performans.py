"""Transparent offline and live model performance metrics."""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.components.data import load_latest_model_metadata, load_prediction_performance
from app.components.ui import configure_page, disclaimer
from config.leagues import LEAGUES_BY_ID


configure_page("Model Performansı")
st.title("Model Performansı")
disclaimer()

metadata = load_latest_model_metadata()
if metadata:
    metrics = metadata["metrics"]
    cols = st.columns(4)
    cols[0].metric(
        "Kalibre Log Loss",
        f"{metrics['log_loss']:.3f}",
        delta=f"{metrics.get('raw_log_loss', metrics['log_loss']) - metrics['log_loss']:.3f}",
    )
    cols[1].metric("Baseline Log Loss", f"{metrics['baseline_log_loss']:.3f}")
    cols[2].metric("Brier Score", f"{metrics['brier_score']:.3f}")
    cols[3].metric("Accuracy", f"%{metrics['accuracy'] * 100:.1f}")
    st.caption(
        f"Kronolojik test: {metrics['test_start'][:10]} – {metrics['test_end'][:10]} · "
        f"{metrics['test_size']:,} maç"
    )
    if "expected_calibration_error" in metrics:
        calibration = metadata.get("calibration", {})
        st.caption(
            "Kalibrasyon: ayrı kronolojik dilim "
            f"({metrics['calibration_size']:,} maç) · ECE "
            f"{metrics['expected_calibration_error']:.3f} "
            f"(ham {metrics['raw_expected_calibration_error']:.3f}) · "
            f"365 gün yarı ömürlü zaman ağırlığı"
        )

    league_rows = metadata.get("league_metrics", [])
    if league_rows:
        st.subheader("Lig bazlı kronolojik backtest")
        league_frame = pd.DataFrame(league_rows)
        league_frame["Lig"] = league_frame["league_id"].map(
            lambda league_id: LEAGUES_BY_ID.get(int(league_id)).name
            if int(league_id) in LEAGUES_BY_ID
            else str(league_id)
        )
        league_frame["Doğruluk"] = league_frame["accuracy"].map(
            lambda value: f"%{value * 100:.1f}"
        )
        st.dataframe(
            league_frame[["Lig", "matches", "log_loss", "brier_score", "ece", "Doğruluk"]]
            .rename(
                columns={
                    "matches": "Maç",
                    "log_loss": "Log Loss",
                    "brier_score": "Brier",
                    "ece": "ECE",
                }
            )
            .sort_values("Log Loss"),
            hide_index=True,
            use_container_width=True,
        )
else:
    st.warning("Kaydedilmiş model değerlendirme metadatası bulunamadı.")

st.subheader("Canlı tahmin takibi")
performance = load_prediction_performance()
if performance.empty:
    st.info(
        "Henüz sonuçlanıp değerlendirilen canlı tahmin yok. Maçlar tamamlandıkça "
        "isabet ve Brier trendi burada görünecek."
    )
else:
    performance["evaluated_at"] = pd.to_datetime(performance["evaluated_at"], utc=True)
    performance["rolling_accuracy"] = performance["was_correct"].astype(float).rolling(30, min_periods=1).mean()
    performance["rolling_brier"] = performance["brier_score"].astype(float).rolling(30, min_periods=1).mean()
    chart_data = performance.melt(
        id_vars="evaluated_at",
        value_vars=["rolling_accuracy", "rolling_brier"],
        var_name="Metrik",
        value_name="Değer",
    )
    st.plotly_chart(
        px.line(chart_data, x="evaluated_at", y="Değer", color="Metrik"),
        use_container_width=True,
    )
    last_30 = performance.tail(30)
    cols = st.columns(2)
    cols[0].metric("Son 30 isabet", f"%{last_30['was_correct'].mean() * 100:.1f}")
    cols[1].metric("Son 30 Brier", f"{last_30['brier_score'].astype(float).mean():.3f}")
