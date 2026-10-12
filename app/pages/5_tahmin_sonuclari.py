"""Auditable live prediction results for completed fixtures."""

import json
import sys
from html import escape
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
    probability_percent,
    section_intro,
    binary_market_evaluation,
)


def _market_result_lines(row: pd.Series) -> tuple[list[str], list[str]]:
    """Build visible primary-market outcomes and saved secondary-market results."""
    result_names = {
        "home_win": "Maç Sonucu 1",
        "draw": "Maç Sonucu X",
        "away_win": "Maç Sonucu 2",
    }
    predicted_result, probability, _ = outcome_prediction_signal(row)
    actual_result = result_names.get(str(row.get("actual_result")), "Bilinmiyor")
    outcome_correct = bool(row.get("was_correct"))
    primary = [
        (
            f"Maç Sonucu: tahmin {predicted_result}"
            f" ({probability_percent(probability)}) · gerçekleşen {actual_result} · "
            f"{'✓ Doğru' if outcome_correct else '✕ Yanlış'}"
        )
    ]

    for market_name, probability_key, actual_key, positive_label, negative_label in (
        ("2,5 Gol Alt/Üst", "prob_over_2_5", "over_2_5_actual", "Üst", "Alt"),
        ("Karşılıklı Gol", "prob_btts", "btts_actual", "Var", "Yok"),
    ):
        probability_value = row.get(probability_key)
        actual_value = row.get(actual_key)
        if pd.isna(probability_value) or pd.isna(actual_value):
            primary.append(f"{market_name}: sonuç verisi yok")
            continue
        prediction, actual, correct = binary_market_evaluation(
            probability_value,
            actual_positive=bool(actual_value),
            positive_label=positive_label,
            negative_label=negative_label,
        )
        status = "✓ Doğru" if correct else "✕ Yanlış"
        primary.append(f"{market_name}: tahmin {prediction} · gerçekleşen {actual} · {status}")

    performance = row.get("market_performance")
    if performance is None or (isinstance(performance, float) and pd.isna(performance)):
        performance = {}
    if isinstance(performance, str):
        try:
            performance = json.loads(performance)
        except json.JSONDecodeError:
            performance = {}
    if not isinstance(performance, dict):
        performance = {}

    secondary: list[str] = []
    for group, markets in performance.items():
        if group == "correct_score" and isinstance(markets, dict):
            scores = row.get("market_probabilities")
            if scores is None or (isinstance(scores, float) and pd.isna(scores)):
                scores = {}
            if isinstance(scores, str):
                try:
                    scores = json.loads(scores)
                except json.JSONDecodeError:
                    scores = {}
            predicted_scores = (scores.get("correct_scores") or []) if isinstance(scores, dict) else []
            score_text = " · ".join(
                f"{item.get('score')} ({float(item.get('probability', 0)) * 100:.1f}%)"
                for item in predicted_scores if isinstance(item, dict)
            ) or "Tahmin yok"
            hit = bool(markets.get("top_3_hit"))
            secondary.append(
                f"Doğru Skor İlk 3: tahmin {score_text} · gerçekleşen "
                f"{markets.get('actual', '—')} · {'✓ Tuttu' if hit else '✕ Tutmadı'}"
            )
            continue
        if not isinstance(markets, dict):
            continue
        for key, item in markets.items():
            if not isinstance(item, dict):
                continue
            if item.get("probability") is None or item.get("actual") is None:
                continue
            probability_value = float(item["probability"])
            actual_value = bool(item["actual"])
            predicted_value = probability_value >= 0.5
            result = "✓ Doğru" if predicted_value == actual_value else "✕ Yanlış"
            label = _market_label(group, key)
            selection = _selection_label(key, predicted_value)
            actual_selection = _selection_label(key, actual_value)
            secondary.append(
                f"{label}: tahmin {selection} "
                f"(%{probability_value * 100:.1f}) · gerçekleşen "
                f"{actual_selection} · {result}"
            )
    return primary, secondary


def _market_label(group: str, key: str) -> str:
    """Translate stored market keys to readable labels, including new market keys."""
    known = {
        ("double_chance", "1X"): "Çifte Şans 1X",
        ("double_chance", "X2"): "Çifte Şans X2",
        ("double_chance", "12"): "Çifte Şans 12",
        ("correct_score", "top_3_hit"): "Doğru Skor İlk 3",
        ("total_goals", "over_1_5"): "Toplam Gol 1,5 Üst",
        ("total_goals", "under_1_5"): "Toplam Gol 1,5 Alt",
        ("total_goals", "over_3_5"): "Toplam Gol 3,5 Üst",
        ("total_goals", "under_3_5"): "Toplam Gol 3,5 Alt",
        ("team_goals", "home_over_0_5"): "Ev Sahibi Gol 0,5 Üst",
        ("team_goals", "away_over_0_5"): "Deplasman Gol 0,5 Üst",
        ("team_goals", "home_over_1_5"): "Ev Sahibi Gol 1,5 Üst",
        ("team_goals", "away_over_1_5"): "Deplasman Gol 1,5 Üst",
    }
    if (group, key) in known:
        return known[(group, key)]

    group_names = {
        "total_goals": "Toplam Gol",
        "team_goals": "Takım Golü",
        "half_time": "İlk Yarı",
        "full_time": "Maç Sonucu",
        "result": "Maç Sonucu",
        "btts": "Karşılıklı Gol",
        "correct_score": "Doğru Skor",
    }
    key_names = {
        "home": "Ev Sahibi",
        "away": "Deplasman",
        "over": "Üst",
        "under": "Alt",
        "yes": "Var",
        "no": "Yok",
        "1": "1",
        "x": "X",
        "2": "2",
    }
    tokens = [key_names.get(token.lower(), token.replace("_", " ")) for token in key.split("_")]
    return " ".join([group_names.get(group, group.replace("_", " ").title()), *tokens])


def _selection_label(key: str, positive: bool) -> str:
    """Describe a boolean market selection and its opposite in Turkish."""
    lowered = key.lower()
    if lowered in {"1x", "x2", "12"}:
        return key if positive else "Diğer sonuç"
    if "over" in lowered:
        return "Üst" if positive else "Alt"
    if "under" in lowered:
        return "Alt" if positive else "Üst"
    if "yes" in lowered or "btts" in lowered:
        return "Var" if positive else "Yok"
    if "no" in lowered:
        return "Yok" if positive else "Var"
    return "Evet" if positive else "Hayır"


def _market_accuracy(evaluations: pd.DataFrame) -> list[tuple[str, int, float]]:
    """Summarize accuracy for each market that has stored predictions and outcomes."""
    correct_by_market: dict[str, list[bool]] = {}

    def add(label: str, correct: object) -> None:
        if pd.notna(correct):
            correct_by_market.setdefault(label, []).append(bool(correct))

    for _, row in evaluations.iterrows():
        add("Maç Sonucu", row.get("was_correct"))
        add("2,5 Gol Alt/Üst", row.get("over_2_5_was_correct"))
        add("Karşılıklı Gol", row.get("btts_was_correct"))

        performance = row.get("market_performance")
        if isinstance(performance, str):
            try:
                performance = json.loads(performance)
            except json.JSONDecodeError:
                performance = {}
        if not isinstance(performance, dict):
            continue

        for group, markets in performance.items():
            if group == "correct_score" and isinstance(markets, dict):
                add("Doğru Skor İlk 3", markets.get("top_3_hit"))
                continue
            if not isinstance(markets, dict):
                continue
            for key, market in markets.items():
                if not isinstance(market, dict):
                    continue
                probability = market.get("probability")
                actual = market.get("actual")
                if probability is None or actual is None:
                    continue
                add(
                    _market_label(group, key),
                    (float(probability) >= 0.5) == bool(actual),
                )

    return [
        (label, len(values), sum(values) / len(values))
        for label, values in sorted(correct_by_market.items())
        if values
    ]


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
    for column, label in zip(summary[1:5], [
        "Maç Sonucu isabet", "2,5 Gol Alt/Üst isabet",
        "Karşılıklı Gol isabet", "Ortalama Brier",
    ]):
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
market_accuracy = _market_accuracy(filtered)
if market_accuracy:
    st.subheader("Pazar başarı oranları")
    st.caption(
        "Tahmin ve gerçekleşen sonucu kaydedilmiş tüm pazarlar listelenir. "
        "Yüzde filtrelenmiş maçlardan hesaplanır; n değerlendirme sayısıdır."
    )
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Pazar": market_label,
                    "Başarı oranı": f"%{accuracy * 100:.1f}",
                    "Değerlendirme sayısı": sample_size,
                }
                for market_label, sample_size, accuracy in market_accuracy
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )
if len(filtered) < 100:
    st.info(
        f"Bu filtrede yalnızca {len(filtered)} maç var. Başarı oranını kesin bir "
        "model performansı olarak yorumlamak için en az 100 maç bekleyin."
    )

st.subheader("Sonuç mesajları")
st.caption("Seçili filtrelere uyan tüm maç sonuçları gösterilir.")
for _, result_row in filtered.iterrows():
    primary_markets, secondary_markets = _market_result_lines(result_row)
    is_correct = bool(result_row["was_correct"])
    status_label = "✓ Doğru" if is_correct else "✕ Yanlış"
    status_color = "#35C48D" if is_correct else "#FF6B6B"
    market_lines = primary_markets + secondary_markets
    market_html = "".join(f"<li>{escape(line)}</li>" for line in market_lines)
    st.markdown(
        (
            '<div style="border:1px solid rgba(120,120,120,.22);border-radius:.8rem;'
            'padding:.7rem .8rem;margin-bottom:.65rem;background:rgba(23,28,36,.72);">'
            f'<div style="font-size:.76rem;color:rgba(242,245,247,.62);">{escape(result_row["match_date"].strftime("%d.%m.%Y %H:%M"))}</div>'
            f'<div style="font-weight:700;margin:.25rem 0;">{escape(str(result_row["home_team"]))} — {escape(str(result_row["away_team"]))}</div>'
            f'<div style="font-size:1.15rem;font-weight:800;">{int(result_row["home_score"])} — {int(result_row["away_score"])} '
            f'<span style="float:right;color:{status_color};font-size:.82rem;">{status_label}</span></div>'
            f'<div style="font-size:.78rem;color:rgba(242,245,247,.62);">{escape(str(result_row["league_name"]))}</div>'
            '<div style="margin-top:.7rem;padding-top:.55rem;border-top:1px solid rgba(120,120,120,.18);">'
            '<div style="font-weight:700;margin-bottom:.25rem;">Pazar sonuçları</div>'
            f'<ul style="margin:.2rem 0 0;padding-left:1.2rem;">{market_html}</ul>'
            '</div>'
            '</div>'
        ),
        unsafe_allow_html=True,
    )

st.caption(
    "Maç Sonucu, 2,5 Gol Alt/Üst ve Karşılıklı Gol sonuçları %50 sınıflandırma eşiğiyle ayrı ayrı "
    "değerlendirilir. Güven etiketi ve Brier skoru yalnızca Maç Sonucu tahminine aittir."
)
