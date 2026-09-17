"""Transparent offline and live model performance metrics."""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.components.data import load_latest_model_metadata, load_prediction_performance, load_upcoming_dashboard
from app.components.live_performance import summarize_live_performance
from app.components.market_performance import summarize_diversified_market_performance
from app.components.monte_carlo import simulate_top_pick_accuracy
from app.components.ui import configure_page, disclaimer
from config.leagues import LEAGUES_BY_ID
from models.decision_policy import MINIMUM_ACTIONABLE_1X2_CONFIDENCE
from monitoring.drift_service import DriftMonitoringService


configure_page("Model Performansı")
st.title("Model Performansı")
disclaimer()

metadata = load_latest_model_metadata()
if metadata:
    metrics = metadata["metrics"]
    st.caption(f"Aktif model: {metadata['active_model_version']}")
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
            width="stretch",
        )

    confidence_rows = metadata.get("confidence_coverage", [])
    if confidence_rows:
        st.subheader("1-X-2 güven ve kapsama dengesi")
        confidence_frame = pd.DataFrame(confidence_rows)
        confidence_frame["Güven eşiği"] = confidence_frame["threshold"].map(
            lambda value: f"%{value * 100:.0f}"
        )
        confidence_frame["Kapsama"] = confidence_frame["coverage"].map(
            lambda value: f"%{value * 100:.1f}"
        )
        confidence_frame["Doğruluk"] = confidence_frame["accuracy"].map(
            lambda value: f"%{value * 100:.1f}" if pd.notna(value) else "—"
        )
        st.dataframe(
            confidence_frame[["Güven eşiği", "matches", "Kapsama", "Doğruluk"]]
            .rename(columns={"matches": "Maç"}),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "Daha yüksek eşik daha az maça tahmin verir. Eşik seçimi yalnızca "
            "kronolojik testte yeterli örneklem bıraktığında yapılmalıdır."
        )
else:
    st.warning("Kaydedilmiş model değerlendirme metadatası bulunamadı.")

st.subheader("Model ve Veri Sapma Analizi (Drift & Kalite)")
try:
    drift_service = DriftMonitoringService()
    upcoming_data = load_upcoming_dashboard(7)
    perf_data = load_prediction_performance()

    from models.feature_engineering import FEATURE_COLUMNS
    feature_cols = list(FEATURE_COLUMNS)

    if upcoming_data.empty and perf_data.empty:
        st.info("Drift analizi için yeterli tahmin veya değerlendirme verisi bulunmuyor.")
    else:
        ref_metrics = metadata.get("metrics") if metadata else None
        drift_report = drift_service.generate_report(
            reference_features=pd.DataFrame(columns=feature_cols),
            current_features=pd.DataFrame(columns=feature_cols),
            feature_columns=feature_cols,
            reference_predictions=perf_data if not perf_data.empty else None,
            current_predictions=upcoming_data if not upcoming_data.empty else None,
            performance_evaluations=perf_data if not perf_data.empty else None,
            reference_metrics=ref_metrics,
        )

        status_colors = {
            "OK": ("green", "✅ Normal (Sapma Yok)"),
            "WARNING": ("orange", "⚠️ Dikkat (Hafif Sapma)"),
            "CRITICAL": ("red", "🚨 Kritik Sapma"),
        }
        color, status_text = status_colors.get(drift_report.status, ("gray", drift_report.status))
        st.markdown(f"**Drift Durumu:** :{color}[{status_text}]")

        drift_cols = st.columns(4)
        drift_cols[0].metric(
            "Sapan Özellik Oranı",
            f"%{drift_report.drifted_features_share * 100:.1f}",
            f"{drift_report.drifted_features_count}/{drift_report.total_features_count} özellik",
        )
        conf_mean = drift_report.confidence_summary.get("mean_confidence")
        drift_cols[1].metric(
            "Ortalama Güven",
            f"%{conf_mean * 100:.1f}" if conf_mean is not None else "—",
        )
        low_conf = drift_report.confidence_summary.get("low_confidence_share")
        drift_cols[2].metric(
            "Düşük Güven Payı (<%40)",
            f"%{low_conf * 100:.1f}" if low_conf is not None else "—",
        )
        cur_dist = drift_report.class_distribution_shift.get("current", {})
        if cur_dist:
            dist_str = f"Ev %{cur_dist.get('home_win', 0)*100:.0f} | B %{cur_dist.get('draw', 0)*100:.0f} | Dep %{cur_dist.get('away_win', 0)*100:.0f}"
        else:
            dist_str = "—"
        drift_cols[3].metric("Tahmin Dağılımı", dist_str)

        if drift_report.critical_alerts:
            for alert in drift_report.critical_alerts:
                st.error(f"🚨 {alert}")
except Exception as drift_exc:
    st.info(f"Drift analizi şu anda yüklenemedi: {drift_exc}")

st.subheader("Monte Carlo tahmin belirsizliği")
try:
    upcoming = load_upcoming_dashboard(3)
    probability_columns = ["prob_home_win", "prob_draw", "prob_away_win"]
    simulation_input = upcoming.dropna(subset=probability_columns)
    if simulation_input.empty:
        st.info("Yaklaşan maç tahminleri hazır olduğunda Monte Carlo dağılımı gösterilecek.")
    else:
        simulation = simulate_top_pick_accuracy(simulation_input)
        lower, upper = simulation["Tahmini isabet"].quantile([0.1, 0.9])
        mean_accuracy = simulation["Tahmini isabet"].mean()
        histogram = px.histogram(
            simulation,
            x="Tahmini isabet",
            histnorm="probability",
            nbins=30,
            labels={"Tahmini isabet": "En güçlü 1-X-2 seçiminin isabet oranı", "probability": "Senaryo payı"},
        )
        histogram.add_vline(
            x=mean_accuracy,
            line_dash="dash",
            annotation_text=f"Ortalama %{mean_accuracy * 100:.1f}",
        )
        st.plotly_chart(histogram, width="stretch")
        st.caption(
            f"Önümüzdeki üç gündeki {len(simulation_input)} maç için 10.000 senaryo. "
            f"Merkez %80 aralığı: %{lower * 100:.1f} – %{upper * 100:.1f}. "
            "Bu dağılım gerçekleşmiş performans değil, model olasılıklarındaki belirsizliktir."
        )
except Exception as exc:
    st.warning(f"Monte Carlo dağılımı şu anda hazırlanamadı: {exc}")

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
    reference_brier = metrics.get("brier_score") if metadata else None
    reference_accuracy = metrics.get("accuracy") if metadata else None
    summary = summarize_live_performance(
        performance["was_correct"].tolist(),
        performance["brier_score"].tolist(),
        reference_brier_score=reference_brier,
        reference_accuracy=reference_accuracy,
    )

    result_confidence = performance[
        ["prob_home_win", "prob_draw", "prob_away_win"]
    ].astype(float).max(axis=1)
    actionable = performance[result_confidence >= MINIMUM_ACTIONABLE_1X2_CONFIDENCE]
    actionable_accuracy = (
        actionable["was_correct"].astype(bool).mean() if not actionable.empty else None
    )

    summary_columns = st.columns(6)
    summary_columns[0].metric("Değerlendirilen maç", str(summary.sample_size))
    summary_columns[1].metric("Tüm 1X2 isabet", f"%{summary.accuracy * 100:.1f}")
    summary_columns[2].metric(
        "%95 güven aralığı",
        f"%{summary.accuracy_lower * 100:.1f} – %{summary.accuracy_upper * 100:.1f}",
    )
    summary_columns[3].metric(
        "Güvenli 1X2 isabet",
        f"%{actionable_accuracy * 100:.1f}" if actionable_accuracy is not None else "—",
    )
    summary_columns[4].metric(
        "Güvenli 1X2 kapsama",
        f"%{len(actionable) / len(performance) * 100:.1f}",
    )
    summary_columns[5].metric("Canlı Brier", f"{summary.brier_score:.3f}")
    st.caption(
        f"Güvenli 1X2, en yüksek sonuç olasılığı en az "
        f"%{MINIMUM_ACTIONABLE_1X2_CONFIDENCE * 100:.0f} olan maçları kapsar; "
        "diğer maçlar Telegram'da Pas olarak işaretlenir."
    )

    market_rows = [
        {
            "Pazar": "1-X-2",
            "Örneklem": len(performance),
            "İsabet": summary.accuracy,
            "Brier": summary.brier_score,
        }
    ]
    for label, correct_column, brier_column in (
        ("Üst/Alt 2.5", "over_2_5_was_correct", "over_2_5_brier_score"),
        ("KG Var/Yok", "btts_was_correct", "btts_brier_score"),
    ):
        market = performance.dropna(subset=[correct_column, brier_column])
        if not market.empty:
            market_rows.append(
                {
                    "Pazar": label,
                    "Örneklem": len(market),
                    "İsabet": market[correct_column].astype(bool).mean(),
                    "Brier": market[brier_column].astype(float).mean(),
                }
            )
    market_rows.extend(
        summarize_diversified_market_performance(performance.to_dict("records"))
    )
    market_frame = pd.DataFrame(market_rows)
    market_frame["İsabet"] = market_frame["İsabet"].map(lambda value: f"%{value * 100:.1f}")
    market_frame["Brier"] = market_frame["Brier"].map(
        lambda value: f"{value:.3f}" if pd.notna(value) else "—"
    )
    st.subheader("Pazar bazlı canlı performans")
    st.dataframe(market_frame, hide_index=True, width="stretch")

    if summary.status == "Yetersiz örneklem":
        st.info(
            f"Canlı örneklem henüz {summary.sample_size}/100 maç. Güven aralığı, "
            "mevcut isabet oranının belirsizliğini gösterir; sonuçları kesin performans "
            "olarak yorumlamayın."
        )
    elif summary.status == "İzlenmeli":
        st.warning(
            "Canlı performans, kronolojik test referansının güven aralığının dışında "
            "zayıfladı veya Brier skoru %5'ten fazla kötüleşti."
        )
    elif summary.status == "İyileşti":
        st.success("Canlı performans, test referansını güven aralığıyla aşıyor.")
    else:
        st.info("Canlı sonuçlar referansla uyumlu görünüyor; üstünlük kanıtlanmış değil.")

    if summary.reference_brier_score is not None:
        st.caption(
            f"Referans: son kronolojik test Brier {summary.reference_brier_score:.3f}. "
            "Sapma uyarısı, en az 100 benzersiz maçta Brier'in %5 kötüleşmesi veya "
            "referans isabetin güven aralığının üstünde kalmasıyla verilir."
        )

    chart_data = performance.melt(
        id_vars="evaluated_at",
        value_vars=["rolling_accuracy", "rolling_brier"],
        var_name="Metrik",
        value_name="Değer",
    )
    st.plotly_chart(
        px.line(chart_data, x="evaluated_at", y="Değer", color="Metrik"),
        width="stretch",
    )
    last_30 = performance.tail(30)
    st.caption(
        f"Grafik, son {len(last_30)} değerlendirmelik kayan ortalamayı gösterir."
    )
