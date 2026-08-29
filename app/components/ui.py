"""Shared Streamlit presentation helpers."""

from __future__ import annotations

import pandas as pd
import streamlit as st


OUTCOME_COLUMNS: tuple[tuple[str, str], ...] = (
    ("prob_home_win", "Ev kazanır"),
    ("prob_draw", "Beraberlik"),
    ("prob_away_win", "Deplasman kazanır"),
    ("prob_over_2_5", "Üst 2.5"),
    ("prob_btts", "KG Var"),
)


def configure_page(title: str) -> None:
    st.set_page_config(
        page_title=f"{title} · Maç Analiz",
        page_icon="⚽",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1280px;}
        [data-testid="stMetric"] {background: rgba(120,120,120,.07); border: 1px solid rgba(120,120,120,.18); padding: .8rem; border-radius: .8rem;}
        .disclaimer {font-size: .84rem; opacity: .72; border-left: 3px solid #f0b429; padding-left: .8rem;}
        @media (max-width: 700px) {.block-container {padding-left: .8rem; padding-right: .8rem;}}
        </style>
        """,
        unsafe_allow_html=True,
    )


def disclaimer() -> None:
    st.markdown(
        '<p class="disclaimer">Tahminler istatistiksel olasılıktır; kesin sonuç veya bahis tavsiyesi değildir.</p>',
        unsafe_allow_html=True,
    )


def probability_percent(value: object) -> str:
    if pd.isna(value):
        return "—"
    return f"%{float(value) * 100:.1f}"


def prediction_signal(row: pd.Series) -> tuple[str, float | None, str]:
    """Return the strongest available market and a deliberately cautious label."""
    candidates = [
        (label, float(row[column]))
        for column, label in OUTCOME_COLUMNS
        if column in row and pd.notna(row[column])
    ]
    if not candidates:
        return "Tahmin bekleniyor", None, "—"

    market, probability = max(candidates, key=lambda item: item[1])
    confidence = "Güçlü" if probability >= 0.60 else "Orta" if probability >= 0.50 else "Düşük"
    return market, probability, confidence


def prediction_signal_text(row: pd.Series) -> str:
    market, probability, confidence = prediction_signal(row)
    if probability is None:
        return market
    return f"{market} · {probability_percent(probability)} · {confidence}"


def evaluated_result_display(frame: pd.DataFrame) -> pd.DataFrame:
    """Format completed-match evaluations for the user-facing audit table."""
    result_labels = {
        "home_win": "Ev kazandı",
        "draw": "Beraberlik",
        "away_win": "Deplasman kazandı",
    }
    rows = frame.copy()
    signals = rows.apply(prediction_signal, axis=1)
    rows["Model tahmini"] = [
        f"{market} ({probability_percent(probability)})"
        if probability is not None
        else market
        for market, probability, _ in signals
    ]
    rows["Sonuç"] = rows["actual_result"].map(result_labels).fillna("—")
    rows["Durum"] = rows["was_correct"].map({True: "✓ Doğru", False: "✗ Yanlış"})
    rows["Skor"] = rows.apply(
        lambda row: f"{int(row['home_score'])} – {int(row['away_score'])}", axis=1
    )
    rows["Maç"] = rows["home_team"] + " — " + rows["away_team"]
    rows["Tarih"] = rows["match_date"].dt.strftime("%d.%m.%Y %H:%M")
    rows["Brier"] = rows["brier_score"].astype(float).map(lambda value: f"{value:.3f}")
    return rows[
        ["Tarih", "league_name", "Maç", "Skor", "Sonuç", "Model tahmini", "Durum", "Brier"]
    ].rename(columns={"league_name": "Lig"})

def dashboard_display(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Tarih": frame["match_date"].dt.strftime("%d.%m %H:%M"),
            "Lig": frame["league_name"],
            "Maç": frame["home_team"] + " — " + frame["away_team"],
            "1": frame.get("prob_home_win", pd.Series(index=frame.index)).map(
                probability_percent
            ),
            "X": frame.get("prob_draw", pd.Series(index=frame.index)).map(
                probability_percent
            ),
            "2": frame.get("prob_away_win", pd.Series(index=frame.index)).map(
                probability_percent
            ),
            "Üst 2.5": frame.get("prob_over_2_5", pd.Series(index=frame.index)).map(
                probability_percent
            ),
            "KG Var": frame.get("prob_btts", pd.Series(index=frame.index)).map(
                probability_percent
            ),
            "En güçlü sinyal": frame.apply(prediction_signal_text, axis=1),
        }
    )
