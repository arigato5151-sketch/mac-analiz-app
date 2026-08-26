"""Shared Streamlit presentation helpers."""

from __future__ import annotations

import pandas as pd
import streamlit as st


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
        }
    )
