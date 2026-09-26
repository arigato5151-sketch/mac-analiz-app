"""Auditable live prediction results for completed fixtures."""

import sys
from statistics import median
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from st_aggrid import AgGrid

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.components.data import load_evaluated_predictions, load_odds_history_for_matches
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
from models.value_analysis import (
    MIN_VALUE_EV,
    MIN_VALUE_SAMPLE,
    assess_market_value,
    evaluate_flat_stakes,
    fractional_kelly_stake,
    value_confidence_status,
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

st.subheader("Geçmiş value performansı")
st.caption(
    "Yalnızca maç öncesi kaydedilmiş oranı ve model olasılığı bulunan seçimler dahil edilir. "
    "Hesaplama 1 birim sabit bahis ve pozitif EV eşiğiyle yapılır; bu gerçekleşmiş yatırım getirisi değildir."
)
value_filter_columns = st.columns(2)
selected_ev_threshold = value_filter_columns[0].number_input(
    "Minimum EV (%)",
    min_value=0.0,
    max_value=50.0,
    value=MIN_VALUE_EV * 100,
    step=0.5,
    help="Yalnızca bu eşik veya üzerindeki model-oran farkları value olarak değerlendirilir.",
) / 100
selected_min_sample = value_filter_columns[1].number_input(
    "Minimum örneklem",
    min_value=1,
    max_value=500,
    value=MIN_VALUE_SAMPLE,
    step=1,
    help="Bu sayının altındaki gruplar istatistiksel olarak güvenilir kabul edilmez.",
)
try:
    odds = load_odds_history_for_matches(tuple(int(value) for value in filtered["match_id"].tolist()))
    odds_by_match = {}
    for match_id, group in odds.groupby("match_id") if not odds.empty else []:
        valid_quotes = [item for item in group.sort_values("captured_at")["odds"] if isinstance(item, dict)]
        if valid_quotes:
            odds_by_match[int(match_id)] = (valid_quotes[0], valid_quotes[-1])
    market_labels = {
        "home_win": "1", "draw": "X", "away_win": "2",
        "over_2_5": "Üst 2.5", "under_2_5": "Alt 2.5",
        "btts_yes": "KG Var", "btts_no": "KG Yok",
    }
    historical_bets: dict[tuple[str, str], list[tuple[object, bool]]] = {}
    historical_clv: dict[tuple[str, str], list[float]] = {}
    historical_stakes: dict[tuple[str, str], list[float]] = {}
    historical_events: list[dict[str, object]] = []
    skipped_quality = 0
    for _, row in filtered.iterrows():
        match_quotes = odds.loc[odds["match_id"] == row["match_id"]].sort_values("captured_at")
        pre_match_quotes = match_quotes[match_quotes["captured_at"] < row["match_date"]]
        valid_quotes = [item for item in pre_match_quotes["odds"] if isinstance(item, dict)]
        if not valid_quotes:
            skipped_quality += 1
            continue
        raw_odds, closing_odds = valid_quotes[0], valid_quotes[-1]
        over_probability = row.get("prob_over_2_5")
        btts_probability = row.get("prob_btts")
        model_probabilities = {
            "home_win": row.get("prob_home_win"),
            "draw": row.get("prob_draw"),
            "away_win": row.get("prob_away_win"),
            "over_2_5": over_probability,
            "under_2_5": 1 - float(over_probability) if pd.notna(over_probability) else None,
            "btts_yes": btts_probability,
            "btts_no": 1 - float(btts_probability) if pd.notna(btts_probability) else None,
        }
        actual_result = "1" if row["home_score"] > row["away_score"] else "X" if row["home_score"] == row["away_score"] else "2"
        actuals = {
            "home_win": actual_result == "1", "draw": actual_result == "X", "away_win": actual_result == "2",
            "over_2_5": row["home_score"] + row["away_score"] >= 3,
            "under_2_5": row["home_score"] + row["away_score"] < 3,
            "btts_yes": row["home_score"] > 0 and row["away_score"] > 0,
            "btts_no": row["home_score"] == 0 or row["away_score"] == 0,
        }
        for assessment in assess_market_value(model_probabilities, raw_odds):
            if assessment.expected_value >= selected_ev_threshold:
                group_key = (str(row.get("league_name") or "Bilinmeyen lig"), assessment.key)
                historical_bets.setdefault(group_key, []).append((assessment, actuals[assessment.key]))
                historical_stakes.setdefault(group_key, []).append(fractional_kelly_stake(assessment))
                historical_events.append({
                    "Tarih": row["match_date"],
                    "Pazar": market_labels.get(assessment.key, assessment.key),
                    "Kâr": assessment.odds - 1.0 if actuals[assessment.key] else -1.0,
                })
                try:
                    closing_price = float(closing_odds[assessment.key])
                    if closing_price > 1:
                        historical_clv.setdefault(group_key, []).append(
                            closing_price / assessment.odds - 1.0
                        )
                except (KeyError, TypeError, ValueError):
                    pass

    performance_rows = []
    for (league_name, key), bets in historical_bets.items():
        performance = evaluate_flat_stakes(bets)
        clv_values = historical_clv.get((league_name, key), [])
        median_line_move = median(clv_values) if clv_values else None
        status_label = value_confidence_status(
            performance,
            median_line_move=median_line_move,
            minimum_sample=selected_min_sample,
        )
        stake_values = historical_stakes.get((league_name, key), [])
        show_stake = status_label == "İzlenebilir value"
        performance_rows.append({
            "Lig": league_name,
            "Pazar": market_labels.get(key, key),
            "Bahis": performance.bets,
            "Kazanan": performance.wins,
            "İsabet": f"%{performance.strike_rate * 100:.1f}",
            "Kâr/Zarar": f"{performance.profit:+.2f}",
            "ROI": f"%{performance.roi * 100:+.1f}",
            "Ort. çeyrek Kelly": (
                f"%{sum(stake_values) / len(stake_values) * 100:.2f}"
                if show_stake and stake_values
                else "—"
            ),
            "Güven durumu": status_label,
            "Medyan oran hareketi": (
                f"%{median_line_move * 100:+.1f}"
                if median_line_move is not None
                else "—"
            ),
        })
    if performance_rows:
        performance_frame = pd.DataFrame(performance_rows)
        st.dataframe(performance_frame, hide_index=True, width="stretch")
        if historical_events:
            curve = pd.DataFrame(historical_events).sort_values("Tarih").reset_index(drop=True)
            curve["Kümülatif Kâr"] = curve["Kâr"].cumsum()
            curve["Zirve"] = curve["Kümülatif Kâr"].cummax()
            curve["Drawdown"] = curve["Kümülatif Kâr"] - curve["Zirve"]
            st.subheader("Kümülatif kâr ve düşüş")
            st.plotly_chart(
                px.line(
                    curve,
                    x="Tarih",
                    y="Kümülatif Kâr",
                    color="Pazar",
                    labels={"Kümülatif Kâr": "Birim", "Tarih": "Maç tarihi"},
                ),
                width="stretch",
            )
            st.caption(f"Maksimum toplam düşüş: {curve['Drawdown'].min():+.2f} birim.")
        total_bets = sum(row["Bahis"] for row in performance_rows)
        if total_bets < selected_min_sample:
            st.warning(
                f"Yalnızca {total_bets} value bahsi bulundu. Minimum {selected_min_sample} bahis olmadan "
                "ROI ve isabet oranı istatistiksel olarak güvenilir kabul edilmemeli."
            )
        st.caption(
            f"Value eşiği: EV ≥ %{selected_ev_threshold * 100:.1f}. Sabit bahis varsayımı: 1 birim. "
            "Medyan oran hareketi, ilk kaydedilen orandan son kaydedilen orana değişimi gösterir; "
            "pozitif değer seçimin piyasa tarafından kısaldığını belirtir. Çeyrek Kelly üst sınırı bankroll'un %5'idir."
        )
        if skipped_quality:
            st.caption(
                f"Veri kalitesi filtresi {skipped_quality} maçı dışarıda bıraktı: "
                "maç öncesi geçerli oran kaydı bulunamadı."
            )
    else:
        st.info("Filtrelenen sonuçlarda value performansı için eşleşen oran verisi bulunamadı.")
except Exception as exc:
    st.warning(f"Geçmiş value performansı hazırlanamadı: {exc}")
st.caption(
    "1-X-2, Üst/Alt 2.5 ve KG Var/Yok sonuçları %50 sınıflandırma eşiğiyle ayrı ayrı "
    "değerlendirilir. Güven etiketi ve Brier skoru yalnızca 1-X-2 tahminine aittir."
)
