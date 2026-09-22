"""Streamlit entry point: a prioritized decision screen over upcoming fixtures."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st

# Streamlit Cloud starts with ``app/`` on the import path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.components.data import (
    UPCOMING_HORIZON_DAYS,
    load_latest_model_metadata,
    load_recent_odds_for_matches,
    load_upcoming_dashboard,
)
from app.components.decision_board import build_match_decisions
from app.components.model_registry import active_production_version, version_label
from app.components.ui import (
    compact_dashboard_display,
    configure_page,
    dashboard_display,
    disclaimer,
    page_header,
    section_intro,
)
from models.baselines import detect_upcoming_collapse


configure_page("Ana Sayfa")
page_header(
    "Maçları tek bakışta değerlendirin",
    "Bugün ve yarının önceliklendirilmiş karar özeti; tüm program ikincil görünümdedir.",
    eyebrow="25 LİG · GÜNCEL TAHMİNLER",
)
disclaimer()

try:
    matches = load_upcoming_dashboard()
    metadata = load_latest_model_metadata()
except Exception as exc:  # Streamlit must remain usable during upstream outages.
    st.error(f"Veriler şu anda yüklenemedi: {exc}")
    st.stop()

col1, col2, col3 = st.columns(3)
ready_count = (
    int(matches["model_version"].notna().sum())
    if not matches.empty and "model_version" in matches
    else 0
)
col1.metric(
    "Yaklaşan maç",
    len(matches),
    help=f"Önümüzdeki {UPCOMING_HORIZON_DAYS} gündeki maç sayısı",
)
col2.metric(
    "Analiz hazır",
    f"{ready_count}/{len(matches)}",
    help="Model olasılıkları hazırlanmış maçlar",
)
col3.metric(
    "Offline Brier",
    f"{metadata['metrics']['brier_score']:.3f}" if metadata else "—",
    help="Daha düşük değer, olasılık tahminlerinin daha iyi olduğunu gösterir.",
)

if not matches.empty and "model_version" in matches:
    predicted_versions = matches["model_version"].dropna().astype(str).unique().tolist()
    production_version = active_production_version()
    if not production_version:
        st.caption("Üretim modeli bilgisi doğrulanamadı.")
    elif set(predicted_versions) - {production_version}:
        st.warning(
            "Yaklaşan maç tahminleri üretim modeliyle eşleşmiyor: "
            f"{', '.join(sorted(predicted_versions))} (üretim: {production_version}). "
            "Tutarsızlık giderilene kadar bu tahminleri doğrulanmış kabul etmeyin."
        )
    else:
        st.caption(
            f"Yaklaşan maçlarda kullanılan model: {version_label(production_version)}"
        )
    prob_matrix = (
        matches[["prob_home_win", "prob_draw", "prob_away_win"]]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(float)
    )
    collapse_warning = detect_upcoming_collapse(
        prob_matrix[~np.isnan(prob_matrix).any(axis=1)]
    )
    if collapse_warning:
        st.warning(f"{collapse_warning} Tahminleri doğrulanmış kabul etmeyin.")

    # Son veri güncelleme yalnızca doğrulanmış bir zaman damgası varsa gösterilir.
    latest_prediction = (
        matches["predicted_at"].dropna().max() if "predicted_at" in matches else None
    )
    if latest_prediction is not None:
        st.caption(
            f"Son tahmin güncellemesi: {latest_prediction:%d.%m %H:%M} (Europe/Istanbul)"
        )


def open_detail(match_id: int) -> None:
    # Streamlit sayfa geçişinde query parametrelerini temizler; oturum durumu
    # seçimi korur. Doğrudan paylaşılabilir URL'ler detay sayfasında ayrıca okunur.
    st.session_state["selected_match_id"] = match_id
    st.switch_page("pages/2_mac_detay.py")


istanbul_now = datetime.now(ZoneInfo("Europe/Istanbul"))
if matches.empty:
    st.info(
        f"Seçili liglerde önümüzdeki {UPCOMING_HORIZON_DAYS} gün için "
        "planlanmış maç bulunamadı."
    )
else:
    tomorrow = istanbul_now.date() + timedelta(days=1)
    window = matches[
        matches["match_date"].dt.date.isin({istanbul_now.date(), tomorrow})
    ]

    window_title = "Bugün ve yarın"
    if window.empty:
        first_date = matches["match_date"].dt.date.min()
        window = matches[matches["match_date"].dt.date == first_date]
        window_title = f"Sıradaki maçlar · {first_date:%d.%m.%Y}"

    st.subheader(window_title)
    section_intro(
        "Başlama saatine kalan süre, model güveni ve veri eksiksizliğiyle "
        "önceliklendirilmiştir; tüm program aşağıdaki ikincil görünümdedir."
    )
    if not window.empty:
        odds_frame = load_recent_odds_for_matches(
            tuple(int(match_id) for match_id in window["id"])
        )
        odds_by_match = (
            {int(row["match_id"]): row.to_dict() for _, row in odds_frame.iterrows()}
            if not odds_frame.empty
            else {}
        )
        decisions = build_match_decisions(window, odds_by_match, now=istanbul_now)

        featured = sorted(
            (decision for decision in decisions if decision.featured),
            key=lambda decision: decision.probability or 0.0,
            reverse=True,
        )
        st.markdown("**En güçlü doğrulanmış sinyaller**")
        if not featured:
            st.info(
                "Güven eşiğini geçen doğrulanmış sinyal yok; maçları aşağıdaki "
                "listeden inceleyin."
            )
        for decision in featured[:5]:
            with st.container(border=True):
                cols = st.columns([3, 1, 2])
                cols[0].markdown(f"**{decision.label}**")
                cols[0].caption(
                    f"{decision.countdown} · güven: {decision.confidence} · "
                    f"{decision.data_status}"
                )
                cols[1].metric(
                    decision.market, f"%{(decision.probability or 0.0) * 100:.1f}"
                )
                if decision.market_gap:
                    cols[2].caption(decision.market_gap)
                cols[2].button(
                    "Detayı aç",
                    key=f"detail_{decision.match_id}",
                    on_click=open_detail,
                    args=(decision.match_id,),
                )

        others = [decision for decision in decisions if not decision.featured]
        if others:
            st.markdown("**Diğer maçlar ve Pas gerekçeleri**")
            for decision in others:
                with st.container(border=True):
                    cols = st.columns([3, 2])
                    signal_text = f"Pas · {decision.pas_reason}"
                    cols[0].markdown(f"**{decision.label}**")
                    cols[0].caption(
                        f"{decision.countdown} · {signal_text} · {decision.data_status}"
                    )
                    if decision.market_gap:
                        cols[1].caption(decision.market_gap)
                    cols[1].button(
                        "Detayı aç",
                        key=f"detail_{decision.match_id}",
                        on_click=open_detail,
                        args=(decision.match_id,),
                    )

st.subheader("Tüm maç programı")
section_intro("Önce temel olasılıkları inceleyin; alternatif pazarları gerektiğinde açın.")
if matches.empty:
    st.info(
        f"Seçili liglerde önümüzdeki {UPCOMING_HORIZON_DAYS} gün için "
        "planlanmış maç bulunamadı."
    )
else:
    leagues = ["Tümü", *sorted(matches["league_name"].dropna().unique())]
    selected = st.selectbox("Lig filtresi", leagues)
    filtered = matches if selected == "Tümü" else matches[matches["league_name"] == selected]
    overview_tab, markets_tab = st.tabs(["Hızlı görünüm", "Tüm pazarlar"])
    with overview_tab:
        st.dataframe(
            compact_dashboard_display(filtered),
            hide_index=True,
            width="stretch",
            height=min(680, 40 + 35 * len(filtered)),
        )
    with markets_tab:
        st.dataframe(
            dashboard_display(filtered),
            hide_index=True,
            width="stretch",
            height=min(680, 40 + 35 * len(filtered)),
        )
    st.caption(f"{len(filtered)} maç gösteriliyor.")

navigation = st.columns(3)
navigation[0].page_link("pages/1_bugunun_maclari.py", label="Tüm maçları filtrele", icon="📅")
navigation[1].page_link("pages/2_mac_detay.py", label="Maç detayını aç", icon="🔎")
navigation[2].page_link("pages/3_model_performans.py", label="Model performansı", icon="📈")
