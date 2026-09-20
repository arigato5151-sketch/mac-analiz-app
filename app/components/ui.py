"""Shared Streamlit presentation helpers."""

from __future__ import annotations

from html import escape
import json

import pandas as pd
import streamlit as st

from models.market_forecast import (
    MINIMUM_DOUBLE_CHANCE_CONFIDENCE,
    MINIMUM_GOAL_MARKET_CONFIDENCE,
)
from models.decision_policy import MINIMUM_ACTIONABLE_1X2_CONFIDENCE


OUTCOME_COLUMNS: tuple[tuple[str, str], ...] = (
    ("prob_home_win", "Ev kazanır"),
    ("prob_draw", "Beraberlik"),
    ("prob_away_win", "Deplasman kazanır"),
    ("prob_over_2_5", "Üst 2.5"),
    ("prob_btts", "KG Var"),
)
RESULT_COLUMNS = OUTCOME_COLUMNS[:3]


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
        .block-container {padding-top: 1.25rem; padding-bottom: 3rem; max-width: 1320px;}
        [data-testid="stMetric"] {
            background: linear-gradient(145deg, rgba(53,196,141,.10), rgba(23,28,36,.88));
            border: 1px solid rgba(53,196,141,.22);
            padding: 1rem; border-radius: 1rem;
        }
        [data-testid="stMetricLabel"] {font-weight: 650;}
        [data-testid="stMetricValue"] {letter-spacing: -.035em;}
        [data-testid="stDataFrame"], [data-testid="stTable"] {
            border: 1px solid rgba(120,120,120,.20); border-radius: .9rem; overflow: hidden;
        }
        .app-hero {
            padding: 1.25rem 1.35rem; margin: 0 0 1rem;
            border-radius: 1.15rem; border: 1px solid rgba(53,196,141,.24);
            background: radial-gradient(circle at top right, rgba(53,196,141,.18), transparent 42%),
                        linear-gradient(135deg, rgba(23,28,36,.96), rgba(14,17,23,.96));
        }
        .app-eyebrow {color: #35C48D; font-size: .76rem; font-weight: 750; letter-spacing: .08em; text-transform: uppercase;}
        .app-hero h1 {font-size: clamp(1.65rem, 4vw, 2.45rem); line-height: 1.12; margin: .35rem 0 .45rem;}
        .app-hero p {max-width: 760px; margin: 0; color: rgba(242,245,247,.74); font-size: .98rem;}
        .section-note {color: rgba(242,245,247,.68); margin-top: -.35rem; margin-bottom: .9rem;}
        .disclaimer {
            font-size: .84rem; opacity: .82; border: 1px solid rgba(240,180,41,.24);
            background: rgba(240,180,41,.07); padding: .7rem .85rem; border-radius: .75rem;
        }
        div[data-testid="stExpander"] {border-radius: .9rem; border-color: rgba(120,120,120,.20);}
        div[data-testid="stTabs"] button {font-weight: 650;}
        @media (max-width: 700px) {
            .block-container {padding: .75rem .7rem 2rem;}
            .app-hero {padding: 1rem; border-radius: .9rem;}
            [data-testid="stMetric"] {padding: .75rem;}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(
    title: str,
    description: str,
    *,
    eyebrow: str = "MAÇ ANALİZ",
) -> None:
    """Render a consistent, accessible heading without trusting dynamic HTML."""
    st.markdown(
        (
            '<section class="app-hero">'
            f'<div class="app-eyebrow">{escape(eyebrow)}</div>'
            f'<h1>{escape(title)}</h1>'
            f'<p>{escape(description)}</p>'
            "</section>"
        ),
        unsafe_allow_html=True,
    )


def disclaimer() -> None:
    st.markdown(
        '<p class="disclaimer">Tahminler istatistiksel olasılıktır; kesin sonuç veya bahis tavsiyesi değildir.</p>',
        unsafe_allow_html=True,
    )


def confidence_label(probability: float) -> str:
    """Map a probability to the same confidence bands used for publishing."""
    if probability >= 0.60:
        return "Güçlü"
    if probability >= MINIMUM_ACTIONABLE_1X2_CONFIDENCE:
        return "Orta"
    return "Düşük"


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
    return market, probability, confidence_label(probability)


def prediction_signal_text(row: pd.Series) -> str:
    market, probability, confidence = prediction_signal(row)
    if probability is None:
        return market
    return f"{market} · {probability_percent(probability)} · {confidence}"


SECONDARY_MARKET_COLUMNS: tuple[str, ...] = (
    "Çifte şans",
    "Üst 1.5",
    "Alt 3.5",
    "Ev 0.5 Üst",
    "Dep. 0.5 Üst",
    "Ev 1.5 Üst",
    "Dep. 1.5 Üst",
    "Skor 1",
    "Skor 2",
    "Skor 3",
)


def diversified_prediction_cells(row: pd.Series) -> dict[str, str]:
    """Return one dashboard cell per published secondary prediction."""
    cells = {column: "—" for column in SECONDARY_MARKET_COLUMNS}
    raw_markets = row.get("market_probabilities")
    if isinstance(raw_markets, str):
        try:
            raw_markets = json.loads(raw_markets)
        except json.JSONDecodeError:
            return cells
    if not isinstance(raw_markets, dict) or not raw_markets:
        return cells

    try:
        double_chance = raw_markets.get("double_chance") or {}
        if double_chance:
            label, probability = max(
                ((str(key), float(value)) for key, value in double_chance.items()),
                key=lambda item: item[1],
            )
            if probability >= MINIMUM_DOUBLE_CHANCE_CONFIDENCE:
                cells["Çifte şans"] = f"{label} · {probability_percent(probability)}"

        for group, key, column in (
            ("total_goals", "over_1_5", "Üst 1.5"),
            ("total_goals", "under_3_5", "Alt 3.5"),
            ("team_goals", "home_over_0_5", "Ev 0.5 Üst"),
            ("team_goals", "away_over_0_5", "Dep. 0.5 Üst"),
            ("team_goals", "home_over_1_5", "Ev 1.5 Üst"),
            ("team_goals", "away_over_1_5", "Dep. 1.5 Üst"),
        ):
            probability = float((raw_markets.get(group) or {}).get(key, 0.0))
            if probability >= MINIMUM_GOAL_MARKET_CONFIDENCE:
                cells[column] = probability_percent(probability)

        for index, score in enumerate((raw_markets.get("correct_scores") or [])[:3], 1):
            cells[f"Skor {index}"] = (
                f"{str(score['score'])} · {probability_percent(float(score['probability']))}"
            )
    except (KeyError, TypeError, ValueError):
        # A malformed historical row must not break the entire dashboard.
        return {column: "—" for column in SECONDARY_MARKET_COLUMNS}
    return cells


def outcome_prediction_signal(row: pd.Series) -> tuple[str, float | None, str]:
    """Return the strongest 1-X-2 prediction used by the evaluated outcome."""
    candidates = [
        (label, float(row[column]))
        for column, label in RESULT_COLUMNS
        if column in row and pd.notna(row[column])
    ]
    if not candidates:
        return "Tahmin bekleniyor", None, "—"

    market, probability = max(candidates, key=lambda item: item[1])
    confidence = "Güçlü" if probability >= 0.60 else "Orta" if probability >= 0.50 else "Düşük"
    return market, probability, confidence_label(probability)


def binary_market_evaluation(
    probability: object,
    *,
    actual_positive: bool,
    positive_label: str,
    negative_label: str,
) -> tuple[str, str, bool | None]:
    """Format one binary market and evaluate its >=50% classification."""
    if pd.isna(probability):
        return "Tahmin yok", "—", None

    positive_probability = float(probability)
    predicted_positive = positive_probability >= 0.5
    selected_label = positive_label if predicted_positive else negative_label
    selected_probability = (
        positive_probability if predicted_positive else 1 - positive_probability
    )
    actual_label = positive_label if actual_positive else negative_label
    return (
        f"{selected_label} ({probability_percent(selected_probability)})",
        actual_label,
        predicted_positive == actual_positive,
    )


def section_intro(text: str) -> None:
    """Add short context below a section heading."""
    st.markdown(f'<p class="section-note">{escape(text)}</p>', unsafe_allow_html=True)


def evaluated_result_display(frame: pd.DataFrame) -> pd.DataFrame:
    """Format completed-match evaluations for the user-facing audit table."""
    result_labels = {
        "home_win": "Ev kazandı",
        "draw": "Beraberlik",
        "away_win": "Deplasman kazandı",
    }
    rows = frame.copy()
    # ``was_correct`` evaluates the 1-X-2 outcome, so do not mix in goal markets here.
    signals = rows.apply(outcome_prediction_signal, axis=1)
    rows["Model tahmini"] = [
        f"{market} ({probability_percent(probability)})"
        if probability is not None
        else market
        for market, probability, _ in signals
    ]
    rows["Sonuç"] = rows["actual_result"].map(result_labels).fillna("—")
    rows["Durum"] = rows["was_correct"].map({True: "✓ Doğru", False: "✗ Yanlış"})
    rows["Güven"] = [confidence for _, _, confidence in signals]
    over_evaluations = [
        binary_market_evaluation(
            row.get("prob_over_2_5"), actual_positive=bool(row["over_2_5_actual"]),
            positive_label="Üst", negative_label="Alt"
        )
        for _, row in rows.iterrows()
    ]
    btts_evaluations = [
        binary_market_evaluation(
            row.get("prob_btts"), actual_positive=bool(row["btts_actual"]),
            positive_label="KG Var", negative_label="KG Yok"
        )
        for _, row in rows.iterrows()
    ]
    rows["Üst 2.5 tahmini"] = [prediction for prediction, _, _ in over_evaluations]
    rows["Üst 2.5 sonucu"] = [actual for _, actual, _ in over_evaluations]
    rows["Üst 2.5 durum"] = rows["over_2_5_was_correct"].map({True: "✓ Doğru", False: "✗ Yanlış"}).fillna("—")
    rows["KG tahmini"] = [prediction for prediction, _, _ in btts_evaluations]
    rows["KG sonucu"] = [actual for _, actual, _ in btts_evaluations]
    rows["KG durum"] = rows["btts_was_correct"].map({True: "✓ Doğru", False: "✗ Yanlış"}).fillna("—")
    rows["Skor"] = rows.apply(
        lambda row: f"{int(row['home_score'])} – {int(row['away_score'])}", axis=1
    )
    rows["Maç"] = rows["home_team"] + " — " + rows["away_team"]
    rows["Tarih"] = rows["match_date"].dt.strftime("%d.%m.%Y %H:%M")
    rows["Brier"] = rows["brier_score"].astype(float).map(lambda value: f"{value:.3f}")
    rows["Üst Brier"] = rows["over_2_5_brier_score"].map(
        lambda value: f"{float(value):.3f}" if pd.notna(value) else "—"
    )
    rows["KG Brier"] = rows["btts_brier_score"].map(
        lambda value: f"{float(value):.3f}" if pd.notna(value) else "—"
    )
    return rows[
        [
            "Tarih", "league_name", "Maç", "Skor", "Sonuç", "Model tahmini", "Güven", "Durum",
            "Üst 2.5 tahmini", "Üst 2.5 sonucu", "Üst 2.5 durum",
            "KG tahmini", "KG sonucu", "KG durum", "Brier", "Üst Brier", "KG Brier",
        ]
    ].rename(columns={"league_name": "Lig"})

def dashboard_display(frame: pd.DataFrame) -> pd.DataFrame:
    secondary_rows = [
        diversified_prediction_cells(row) for _, row in frame.iterrows()
    ]
    return pd.DataFrame(
        {
            "Tarih": frame["match_date"].dt.strftime("%d.%m %H:%M"),
            "Lig": frame["league_name"],
            "Maç": frame["home_team"] + " — " + frame["away_team"],
            **{
                column: [cells[column] for cells in secondary_rows]
                for column in SECONDARY_MARKET_COLUMNS
            },
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


def compact_dashboard_display(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the decision-focused fixture view; advanced markets stay optional."""
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
            "Üst 2.5": frame.get(
                "prob_over_2_5", pd.Series(index=frame.index)
            ).map(probability_percent),
            "KG Var": frame.get("prob_btts", pd.Series(index=frame.index)).map(
                probability_percent
            ),
            "Öne çıkan": frame.apply(prediction_signal_text, axis=1),
        }
    )


def diversified_dashboard_display(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep older deployed entry points compatible during rolling updates."""
    secondary_rows = [
        diversified_prediction_cells(row) for _, row in frame.iterrows()
    ]
    rows = pd.DataFrame(
        {
            "Tarih": frame["match_date"].dt.strftime("%d.%m %H:%M"),
            "Maç": frame["home_team"] + " — " + frame["away_team"],
            **{
                column: [cells[column] for cells in secondary_rows]
                for column in SECONDARY_MARKET_COLUMNS
            },
        }
    )
    if rows.empty:
        return rows
    has_prediction = rows[list(SECONDARY_MARKET_COLUMNS)].ne("—").any(axis=1)
    return rows.loc[has_prediction].reset_index(drop=True)
