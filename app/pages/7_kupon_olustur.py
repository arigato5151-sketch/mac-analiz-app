"""Daily coupon builder based on current predictions and captured odds."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.components.data import (
    load_recent_nesine_markets_for_matches,
    load_prediction_performance,
    load_recent_odds_for_matches,
    load_upcoming_dashboard,
)
from app.components.freshness import odds_are_current
from app.components.ui import configure_page, disclaimer, page_header
from models.coupon_policy import diversified_coupon_rows
from models.value_analysis import (
    MAX_PUBLISHABLE_EXPECTED_VALUE,
    ValueAssessment,
    assess_market_value,
    derive_combo_probabilities,
)


def fails_prediction_sanity_check(probabilities: dict[str, object]) -> bool:
    """Reject internally contradictory extreme-goal predictions from coupons."""
    market_probabilities = probabilities.get("market_probabilities")
    if not isinstance(market_probabilities, dict):
        return False
    expected_goals = market_probabilities.get("expected_goals")
    if not isinstance(expected_goals, dict):
        return False
    try:
        home_goals = float(expected_goals.get("home"))
        away_goals = float(expected_goals.get("away"))
        home_win = float(probabilities["home_win"])
        away_win = float(probabilities["away_win"])
    except (KeyError, TypeError, ValueError):
        return False
    # Extreme expected goals must agree with a strong 1X2 signal.
    total_goals = home_goals + away_goals
    return (
        (away_goals >= 3.5 and away_win < 0.55)
        or (home_goals >= 3.5 and home_win < 0.55)
        or (total_goals >= 4.5 and float(probabilities.get("over_2_5", 0)) < 0.65)
    )


configure_page("Günlük kupon")
page_header(
    "Günlük kupon oluştur",
    "Yalnızca günün kalan maçları, mevcut model tahmini ve kaydedilmiş oranlarla kupon üretir.",
    eyebrow="KUPON ÇALIŞMA ALANI",
)
disclaimer()

if st.button("Verileri yenile", key="refresh_daily_coupon"):
    st.cache_data.clear()
    st.rerun()

matches = load_upcoming_dashboard(7)
selected_coupon_date = None
if not matches.empty:
    today = datetime.now(timezone.utc).astimezone(ZoneInfo("Europe/Istanbul")).date()
    available_dates = sorted(matches["match_date"].dt.date.dropna().unique())
    selected_coupon_date = today if today in available_dates else (available_dates[0] if available_dates else None)
    if selected_coupon_date is not None:
        matches = matches[matches["match_date"].dt.date == selected_coupon_date].reset_index(drop=True)
if matches.empty:
    st.info("Bugünün kalan planlanmış maçı bulunamadı.")
    st.stop()

if selected_coupon_date != today:
    st.warning(
        f"Bugün ({today:%d.%m.%Y}) kalan maç yok. Kupon ilk sonraki maç gününe "
        f"({selected_coupon_date:%d.%m.%Y}) göre oluşturuldu."
    )

odds = load_recent_odds_for_matches(tuple(int(value) for value in matches["id"]))
odds_by_match = {
    int(row["match_id"]): row["odds"]
    for _, row in odds.iterrows()
    if isinstance(row.get("odds"), dict)
}
odds_source_by_match = {
    int(row["match_id"]): str(row.get("bookmaker") or "Bilinmiyor")
    for _, row in odds.iterrows()
}
odds_captured_by_match = {
    int(row["match_id"]): row.get("captured_at")
    for _, row in odds.iterrows()
}
nesine_market_quotes = load_recent_nesine_markets_for_matches(
    tuple(int(value) for value in matches["id"])
)
nesine_odds_by_match: dict[int, dict[str, float]] = {}
if not nesine_market_quotes.empty:
    for match_id, group in nesine_market_quotes.groupby("match_id"):
        nesine_odds_by_match[int(match_id)] = {
            str(row["market_key"]): float(row["odd"])
            for _, row in group.iterrows()
            if row.get("market_key") and pd.notna(row.get("odd"))
        }
    # Prefer today's complete Nesine snapshot, while retaining other-bookmaker
    # markets where Nesine has no captured selection.
    for match_id, prices in nesine_odds_by_match.items():
        if match_id in odds_by_match:
            odds_by_match[match_id] = {**odds_by_match[match_id], **prices}
        else:
            odds_by_match[match_id] = prices
            latest_quote = nesine_market_quotes.loc[
                nesine_market_quotes["match_id"] == match_id
            ].iloc[0]
            odds_source_by_match[match_id] = "Nesine"
            odds_captured_by_match[match_id] = latest_quote["captured_at"]

rows: list[dict[str, object]] = []
filter_counts = {"no_odds": 0, "stale_odds": 0, "sanity": 0, "below_probability": 0}
odds_now = datetime.now(timezone.utc)
for _, match in matches.iterrows():
    match_id = int(match["id"])
    raw_odds = odds_by_match.get(match_id)
    if not isinstance(raw_odds, dict):
        filter_counts["no_odds"] += 1
        continue
    if not odds_are_current(odds_captured_by_match.get(match_id), now=odds_now):
        filter_counts["stale_odds"] += 1
        continue
    probabilities = {
        "home_win": match.get("prob_home_win"),
        "draw": match.get("prob_draw"),
        "away_win": match.get("prob_away_win"),
        "over_2_5": match.get("prob_over_2_5"),
        "btts_yes": match.get("prob_btts"),
    }
    # Half-time probabilities are persisted in market_probabilities by the
    # prediction job; only use them when the bookmaker supplies matching odds.
    persisted_markets = match.get("market_probabilities")
    if isinstance(persisted_markets, dict):
        probabilities.update(persisted_markets)
    market_quotes = nesine_market_quotes.loc[
        nesine_market_quotes["match_id"] == match_id
    ] if not nesine_market_quotes.empty else pd.DataFrame()
    market_captured_at = market_quotes["captured_at"].max() if not market_quotes.empty else None
    if market_captured_at is not None and odds_are_current(market_captured_at, now=odds_now):
        raw_odds = {**raw_odds, **nesine_odds_by_match.get(match_id, {})}
    if fails_prediction_sanity_check(probabilities):
        filter_counts["sanity"] += 1
        continue
    probabilities.update(
        derive_combo_probabilities(
            probabilities,
            score_matrix=(persisted_markets or {}).get("score_matrix")
            if isinstance(persisted_markets, dict)
            else None,
        )
    )
    assessments = assess_market_value(probabilities, raw_odds)
    # Never publish selections below the user's minimum 50% model probability.
    assessments = [
        item
        for item in assessments
        if item.model_probability >= (0.55 if item.key in {"btts_yes", "btts_no"} else 0.50)
        and item.odds <= 20.0
        and item.expected_value <= MAX_PUBLISHABLE_EXPECTED_VALUE
    ]
    if not assessments:
        filter_counts["below_probability"] += 1
        continue
    best = max(assessments, key=lambda item: item.expected_value)
    rows.append({
        "match_id": match_id,
        "match": f"{match['home_team']} — {match['away_team']}",
        "time": match["match_date"].strftime("%H:%M"),
        "league": match["league_name"],
        "assessments": assessments,
        "best": best,
        "probability": best.model_probability,
        "bookmaker": odds_source_by_match.get(match_id, "Bilinmiyor"),
        "odds_captured_at": odds_captured_by_match.get(match_id),
    })

rows.sort(key=lambda row: (float(row["best"].expected_value), float(row["probability"])), reverse=True)

with st.expander("Filtreleme özeti"):
    st.write(
        f"İncelenen maç: {len(matches)} · Oranı olmayan: {filter_counts['no_odds']} · "
        f"Güncel olmayan oran: {filter_counts['stale_odds']} · "
        f"Model tutarsızlığı: {filter_counts['sanity']} · %50 altı: {filter_counts['below_probability']}"
    )

try:
    performance = load_prediction_performance()
except Exception:
    performance = pd.DataFrame()
if not performance.empty:
    performance_rows = []
    for label_name, correct_column in (
        ("Maç Sonucu", "was_correct"),
        ("2,5 Gol Alt/Üst", "over_2_5_was_correct"),
        ("Karşılıklı Gol", "btts_was_correct"),
    ):
        values = performance[correct_column].dropna().astype(bool)
        if not values.empty:
            performance_rows.append(
                {
                    "Market": label_name,
                    "Örneklem": len(values),
                    "Geçmiş isabet": f"%{values.mean() * 100:.1f}",
                }
            )
    if performance_rows:
        with st.expander("Geçmiş market performansı"):
            st.dataframe(pd.DataFrame(performance_rows), hide_index=True, width="stretch")

def label(item: ValueAssessment) -> str:
    names = {
        "home_win": "Maç Sonucu · 1", "draw": "Maç Sonucu · X", "away_win": "Maç Sonucu · 2",
        "over_2_5": "2,5 Gol Alt/Üst · Üst", "btts_yes": "Karşılıklı Gol · Var",
    }
    market_name, selection_name = _coupon_market_label(item.key)
    market_label = names.get(item.key, f"{market_name} · {selection_name}")
    return f"{market_label} · %{item.model_probability * 100:.0f} · oran {item.odds:.2f}"


def _coupon_market_label(key: str) -> tuple[str, str]:
    from app.components.nesine_labels import nesine_market_name, nesine_selection_name

    return nesine_market_name(key), nesine_selection_name(key)

def render_coupon(title: str, selected: list[dict[str, object]], color: str) -> None:
    st.markdown(f"### {color} {title}")
    if not selected:
        st.info("Bu risk profiline uyan yeterli value seçimi yok.")
        return
    total = 1.0
    for row in selected:
        item = row["best"]
        total *= item.odds
        st.write(f"- **{row['time']} · {row['match']}** — {label(item)}")
    st.caption(f"Yaklaşık toplam oran: **{total:.2f}** · Garanti değildir.")

eligible_rows = diversified_coupon_rows(rows, min_probability=0.55, min_ev=1e-9)
st.caption(f"Bugünün kupona uygun value seçimi: {len(eligible_rows)}")
render_coupon(
    "Düşük riskli kupon",
    diversified_coupon_rows(rows, max_items=None, min_probability=0.55, min_ev=1e-9),
    "🟢",
)
render_coupon("Dengeli kupon", diversified_coupon_rows(rows, max_items=None, min_ev=0.03), "🟡")
render_coupon("Yüksek oranlı kupon", diversified_coupon_rows(rows, max_items=None, min_ev=0.03, high_odds=True), "🔴")

st.divider()
st.subheader("Kupona uygun maçların value tablosu")
if not eligible_rows:
    st.info("Güncel oran ve pozitif value koşullarını karşılayan seçim yok.")
else:
    st.dataframe(
        pd.DataFrame([
            {
                "Saat": row["time"], "Karşılaşma": row["match"], "Seçim": label(row["best"]),
                "Value": f"%{row['best'].expected_value * 100:+.1f}",
                "Oran kaynağı": row["bookmaker"],
                "Oran güncelleme": (
                    row["odds_captured_at"].strftime("%d.%m.%Y %H:%M")
                    if hasattr(row.get("odds_captured_at"), "strftime")
                    else "Bilinmiyor"
                ),
                "Lig": row["league"],
            }
            for row in eligible_rows
        ]),
        hide_index=True,
        width="stretch",
    )
