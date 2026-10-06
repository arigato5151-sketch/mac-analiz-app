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

from app.components.auth import current_user_id, get_user_db, render_auth_panel
from app.components.data import (
    UPCOMING_HORIZON_DAYS,
    load_recent_odds_for_matches,
    load_upcoming_dashboard,
)
from app.components.decision_board import build_match_decisions
from app.components.freshness import odds_are_current
from app.components.model_registry import active_production_version, version_label
from app.components.ui import (
    compact_dashboard_display,
    configure_page,
    disclaimer,
    page_header,
    section_intro,
)
from models.baselines import detect_upcoming_collapse
from models.value_analysis import (
    MAX_PUBLISHABLE_EXPECTED_VALUE,
    MIN_VALUE_EV,
    assess_market_value,
    best_value_assessment,
)

configure_page("Ana Sayfa")
page_header(
    "Maçları tek bakışta değerlendirin",
    "Bugün ve yarının önceliklendirilmiş karar özeti; tüm program ikincil görünümdedir.",
    eyebrow="25 LİG · MAÇ TAHMİNLERİ",
)
disclaimer()
auth_enabled = render_auth_panel()

try:
    matches = load_upcoming_dashboard()
except Exception:  # Streamlit must remain usable during upstream outages.
    st.error("Maç verileri şu anda kullanılamıyor. Birkaç dakika sonra tekrar deneyin.")
    st.stop()

st.caption(
    "Analiz ekranı yalnızca incelemeye değer maçları öne çıkarır. "
    "Olasılıklar kesin sonuç değildir."
)

if not matches.empty and "model_version" in matches:
    predicted_versions = matches["model_version"].dropna().astype(str).unique().tolist()
    production_version = active_production_version()
    with st.expander("Veri ve model durumu"):
        if not production_version:
            st.caption("Üretim modeli bilgisi doğrulanamadı.")
        elif set(predicted_versions) - {production_version}:
            st.warning(
                "Yaklaşan maç tahminleri aktif modelle eşleşmiyor: "
                f"{', '.join(sorted(predicted_versions))} (aktif: {production_version}). "
                "Bu durum düzelene kadar tahminleri temkinli yorumlayın."
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
        with st.expander("Tahmin kalitesi uyarısı"):
            st.warning(f"{collapse_warning} Tahminleri temkinli yorumlayın.")

    # Use the whole prediction table, not only fixtures still shown as upcoming.
    latest_prediction = matches["latest_prediction_at"].dropna().max()
    if latest_prediction is not None:
        st.caption(
            f"Son tahmin güncellemesi: {latest_prediction:%d.%m %H:%M} (Europe/Istanbul)"
        )
        prediction_age = datetime.now(ZoneInfo("Europe/Istanbul")) - latest_prediction.to_pydatetime()
        if prediction_age > timedelta(hours=24):
            st.warning(
                "Tahminler son 24 saatte yenilenmedi; maç seçmeden önce "
                "veri güncelliğini doğrulayın."
            )


def open_detail(match_id: int) -> None:
    # Streamlit sayfa geçişinde query parametrelerini temizler; oturum durumu
    # seçimi korur. Doğrudan paylaşılabilir URL'ler detay sayfasında ayrıca okunur.
    st.session_state["selected_match_id"] = match_id
    st.switch_page("pages/2_mac_detay.py")


def toggle_watch(match_id: int) -> None:
    """Keep a lightweight personal watchlist for the current session."""
    watched = st.session_state.setdefault("watched_match_ids", set())
    if match_id in watched:
        watched.remove(match_id)
    else:
        watched.add(match_id)
    db = get_user_db()
    if db:
        db.upsert(
            "personal_match_workspace",
            [{"user_id": current_user_id(), "match_id": match_id, "is_watched": match_id in watched}],
            on_conflict="user_id,match_id",
        )


if auth_enabled:
    try:
        rows = get_user_db().select_all("personal_match_workspace", columns="match_id,is_watched")
        st.session_state["watched_match_ids"] = {
            int(row["match_id"]) for row in rows if row.get("is_watched")
        }
    except Exception:
        st.warning("Kalıcı takip listesi yüklenemedi; geçici liste kullanılıyor.")

istanbul_now = datetime.now(ZoneInfo("Europe/Istanbul"))
if matches.empty:
    st.info(
        f"Seçili liglerde önümüzdeki {UPCOMING_HORIZON_DAYS} gün için "
        "planlanmış maç bulunamadı. Veri akışı henüz yenilenmemiş olabilir; daha sonra tekrar kontrol edin."
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

        value_only = st.checkbox(
            "Yalnızca avantajlı oran bulunan maçları göster",
            help=f"Model olasılığı ile oran arasındaki farkı en az %{MIN_VALUE_EV * 100:.1f} olan maçları filtreler.",
        )
        watched_only = st.checkbox("Yalnızca takip ettiklerimi göster")
        show_passes = st.checkbox("Aksiyon sinyali olmayan maçları göster")
        watched_count = len(st.session_state.setdefault("watched_match_ids", set()))
        st.caption(f"Takipteki maç: {watched_count}")
        st.caption(
            "Takip listesi Supabase hesabına kaydedilir."
            if auth_enabled
            else "Kalıcı kayıt için Supabase hesabıyla giriş yapın."
        )
        if value_only:
            value_match_ids = set()
            for _, match_row in window.iterrows():
                odds_row = odds_by_match.get(int(match_row["id"]), {})
                if not odds_are_current(odds_row.get("captured_at"), now=istanbul_now):
                    continue
                raw_odds = odds_row.get("odds") if isinstance(odds_row, dict) else None
                if not isinstance(raw_odds, dict):
                    continue
                assessments = assess_market_value(
                    {
                        "home_win": match_row.get("prob_home_win"),
                        "draw": match_row.get("prob_draw"),
                        "away_win": match_row.get("prob_away_win"),
                    },
                    raw_odds,
                )
                if any(
                    MIN_VALUE_EV <= item.expected_value <= MAX_PUBLISHABLE_EXPECTED_VALUE
                    for item in assessments
                ):
                    value_match_ids.add(int(match_row["id"]))
            decisions = [decision for decision in decisions if decision.match_id in value_match_ids]
        if watched_only:
            watched_ids = st.session_state.setdefault("watched_match_ids", set())
            decisions = [decision for decision in decisions if decision.match_id in watched_ids]
            if not decisions:
                st.info("Takip listenizde bu zaman aralığına uyan maç yok.")

        featured = sorted(
            (decision for decision in decisions if decision.featured),
            key=lambda decision: decision.probability or 0.0,
            reverse=True,
        )
        st.markdown("**Aksiyon e\u015fi\u011fini ge\u00e7en ma\u00e7lar**")
        if not featured:
            st.info(
                "Bu aralıkta aksiyon eşiğini geçen sinyal yok. Aşağıdaki maçlar "
                "yalnızca inceleme amaçlıdır."
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
                odds_row = odds_by_match.get(int(decision.match_id), {})
                match_row = window.loc[window["id"] == decision.match_id].iloc[0]
                model_odds_keys = {
                    "home_win": "prob_home_win",
                    "draw": "prob_draw",
                    "away_win": "prob_away_win",
                }
                raw_odds = odds_row.get("odds") if isinstance(odds_row, dict) else None
                if isinstance(raw_odds, dict) and odds_are_current(
                    odds_row.get("captured_at"), now=istanbul_now
                ):
                    value = best_value_assessment(
                        {
                            odds_key: match_row.get(probability_key)
                            for odds_key, probability_key in model_odds_keys.items()
                        },
                        raw_odds,
                    )
                    if value is not None and value.expected_value >= MIN_VALUE_EV:
                        cols[2].caption(
                            f"Oran avantajı: {value.key} · fark %{value.expected_value * 100:.1f} · "
                            f"oran {value.odds:.2f}"
                        )
                    elif value is not None:
                        cols[2].caption(
                            f"Oran avantajı eşiği aşılmadı · minimum %{MIN_VALUE_EV * 100:.1f}"
                        )
                    else:
                        cols[2].caption("Oran avantajı hesaplanamadı · geçerli oran yok")
                watched = st.session_state.setdefault("watched_match_ids", set())
                cols[2].button(
                    "Takipten çıkar" if decision.match_id in watched else "Takibe al",
                    key=f"watch_{decision.match_id}",
                    on_click=toggle_watch,
                    args=(decision.match_id,),
                )
                cols[2].button(
                    "Detayı aç",
                    key=f"detail_{decision.match_id}",
                    on_click=open_detail,
                    args=(decision.match_id,),
                )

        others = [decision for decision in decisions if not decision.featured]
        if others and show_passes:
            st.markdown("**Aksiyon sinyali olmayan maçlar**")
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
                    odds_row = odds_by_match.get(int(decision.match_id), {})
                    raw_odds = odds_row.get("odds") if isinstance(odds_row, dict) else None
                    if isinstance(raw_odds, dict) and odds_are_current(
                        odds_row.get("captured_at"), now=istanbul_now
                    ):
                        match_row = window.loc[window["id"] == decision.match_id].iloc[0]
                        value = best_value_assessment(
                            {
                                "home_win": match_row.get("prob_home_win"),
                                "draw": match_row.get("prob_draw"),
                                "away_win": match_row.get("prob_away_win"),
                            },
                            raw_odds,
                        )
                        if value is None:
                            cols[1].caption("Oran avantajı hesaplanamadı · geçerli oran yok")
                        elif value.expected_value >= MIN_VALUE_EV:
                            cols[1].caption(
                                f"Oran avantajı bulundu · {value.key} · fark %{value.expected_value * 100:.1f}"
                            )
                        else:
                            cols[1].caption(
                                f"Avantaj eşiği aşılmadı · minimum %{MIN_VALUE_EV * 100:.1f}"
                            )
                    else:
                        cols[1].caption("Oran avantajı hesaplanamadı · oran yok")
                    watched = st.session_state.setdefault("watched_match_ids", set())
                    cols[1].button(
                        "Takipten çıkar" if decision.match_id in watched else "Takibe al",
                        key=f"watch_{decision.match_id}",
                        on_click=toggle_watch,
                        args=(decision.match_id,),
                    )
                    cols[1].button(
                        "Detayı aç",
                        key=f"detail_{decision.match_id}",
                        on_click=open_detail,
                        args=(decision.match_id,),
                    )

st.subheader("Maç programı")
section_intro("Ana akışta yalnızca karar için gerekli özet gösterilir. Ayrıntılı tabloyu gerektiğinde açın.")
if matches.empty:
    st.info(
        f"Seçili liglerde önümüzdeki {UPCOMING_HORIZON_DAYS} gün için "
        "planlanmış maç bulunamadı. Filtreyi genişletin veya veri akışının yenilenmesini bekleyin."
    )
else:
    leagues = ["Tümü", *sorted(matches["league_name"].dropna().unique())]
    if st.button("Filtreleri sıfırla", key="reset_dashboard_filters"):
        st.session_state.pop("dashboard_league", None)
        st.rerun()
    saved_league = st.session_state.get("dashboard_league", "Tümü")
    selected = st.selectbox(
        "Lig filtresi",
        leagues,
        index=leagues.index(saved_league) if saved_league in leagues else 0,
        key="dashboard_league",
    )
    filtered = matches if selected == "Tümü" else matches[matches["league_name"] == selected]
    with st.expander("Tüm maçları tablo olarak göster"):
        st.dataframe(
            compact_dashboard_display(filtered),
            hide_index=True,
            width="stretch",
            height=min(680, 40 + 35 * len(filtered)),
        )
    st.caption(f"{len(filtered)} maç gösteriliyor.")

navigation = st.columns(2)
navigation[0].page_link("pages/1_bugunun_maclari.py", label="Tüm maçları filtrele", icon="📅")
navigation[1].page_link("pages/2_mac_detay.py", label="Maç detayını aç", icon="🔎")
st.page_link("pages/6_kisisel_rapor.py", label="Kişisel rapor", icon="📊")
st.page_link("pages/7_kupon_olustur.py", label="Günlük kupon oluştur", icon="🎟️")
