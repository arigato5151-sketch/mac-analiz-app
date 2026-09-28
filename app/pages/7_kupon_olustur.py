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

from app.components.data import load_recent_odds_for_matches, load_upcoming_dashboard
from app.components.ui import configure_page, disclaimer, page_header
from models.value_analysis import ValueAssessment, assess_market_value
from models.coupon_policy import diversified_coupon_rows


def derive_combo_probabilities(probabilities: dict[str, object]) -> dict[str, float]:
    """Keep the coupon page compatible with older deployed model modules."""
    combos = {
        "home_win_over_2_5": ("home_win", "over_2_5"),
        "home_win_under_2_5": ("home_win", "under_2_5"),
        "draw_over_2_5": ("draw", "over_2_5"),
        "draw_under_2_5": ("draw", "under_2_5"),
        "away_win_over_2_5": ("away_win", "over_2_5"),
        "away_win_under_2_5": ("away_win", "under_2_5"),
        "home_win_btts_yes": ("home_win", "btts_yes"),
        "draw_btts_yes": ("draw", "btts_yes"),
        "away_win_btts_yes": ("away_win", "btts_yes"),
    }
    result: dict[str, float] = {}
    for key, (left, right) in combos.items():
        try:
            left_value, right_value = float(probabilities[left]), float(probabilities[right])
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= left_value <= 1 and 0 <= right_value <= 1:
            result[key] = left_value * right_value
    return result


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

rows: list[dict[str, object]] = []
for _, match in matches.iterrows():
    match_id = int(match["id"])
    raw_odds = odds_by_match.get(match_id)
    if not isinstance(raw_odds, dict):
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
    if fails_prediction_sanity_check(probabilities):
        continue
    probabilities.update(derive_combo_probabilities(probabilities))
    assessments = assess_market_value(probabilities, raw_odds)
    # Do not publish extreme longshots that the model itself rates as weak.
    assessments = [
        item
        for item in assessments
        if not (item.odds >= 20.0 and item.model_probability < 0.35)
    ]
    if not assessments:
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
    })

if not rows:
    st.warning("Bugünün maçlarında hem oran hem de geçerli model tahmini eşleşmedi.")
    st.stop()

rows.sort(key=lambda row: (float(row["best"].expected_value), float(row["probability"])), reverse=True)

def label(item: ValueAssessment) -> str:
    names = {
        "home_win": "MS 1", "draw": "MS X", "away_win": "MS 2",
        "over_2_5": "Üst 2.5", "btts_yes": "KG Var",
    }
    return f"{names.get(item.key, item.key)} · %{item.model_probability * 100:.0f} · oran {item.odds:.2f}"

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

st.caption(f"Bugünün oranlı ve tahminli maç sayısı: {len(rows)}")
render_coupon("Düşük riskli kupon", diversified_coupon_rows(rows, max_items=None, min_probability=0.55, min_ev=0.0), "🟢")
render_coupon("Dengeli kupon", diversified_coupon_rows(rows, max_items=None, min_ev=0.03), "🟡")
render_coupon("Yüksek oranlı kupon", diversified_coupon_rows(rows, max_items=None, min_ev=0.03, high_odds=True), "🔴")

st.divider()
st.subheader("Kupona uygun maçların value tablosu")
st.dataframe(
    pd.DataFrame([
        {
            "Saat": row["time"], "Karşılaşma": row["match"], "Seçim": label(row["best"]),
            "Value": f"%{row['best'].expected_value * 100:+.1f}", "Lig": row["league"],
        }
        for row in rows
    ]),
    hide_index=True,
    width="stretch",
)
