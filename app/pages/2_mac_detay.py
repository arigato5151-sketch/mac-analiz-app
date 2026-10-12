"""Compact match forecast and personal decision notes."""

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.components.auth import current_user_id, get_user_db, render_auth_panel
from app.components.data import load_upcoming_dashboard
from app.components.decision_board import requested_match_id
from app.components.model_registry import active_production_version, version_label
from app.components.ui import (
    configure_page,
    disclaimer,
    page_header,
    prediction_signal,
    probability_percent,
)
from models.decision_policy import minimum_confidence_for_league

configure_page("Maç Detay")
page_header(
    "Maç detay analizi",
    "Maç tahminini görüntüleyin ve kişisel karar notunuzu kaydedin.",
    eyebrow="DERİNLEMESİNE ANALİZ",
)
disclaimer()
auth_enabled = render_auth_panel()

try:
    matches = load_upcoming_dashboard(7)
except Exception:
    st.error("Maç listesi şu anda kullanılamıyor. Birkaç dakika sonra tekrar deneyin.")
    st.stop()

if matches.empty:
    st.info("Detayı gösterilecek yaklaşan maç bulunamadı.")
    st.stop()

matches = matches.copy()
matches["label"] = (
    matches["match_date"].dt.strftime("%d.%m %H:%M")
    + " · "
    + matches["home_team"]
    + " — "
    + matches["away_team"]
)
deep_linked_id = requested_match_id(st.query_params.to_dict(), matches)
if deep_linked_id is None:
    deep_linked_id = requested_match_id(
        {"match_id": st.session_state.pop("selected_match_id", None)}, matches
    )
match_ids = matches["id"].tolist()
default_index = match_ids.index(deep_linked_id) if deep_linked_id in match_ids else 0
selected_id = st.selectbox(
    "İncelenecek maç",
    match_ids,
    index=default_index,
    format_func=lambda match_id: matches.loc[matches["id"] == match_id, "label"].iloc[0],
)
selected = matches.loc[matches["id"] == selected_id].iloc[0]
st.subheader(f"{selected['home_team']} — {selected['away_team']}")
st.caption(f"{selected['league_name']} · {selected['match_date'].strftime('%d.%m.%Y %H:%M')}")
st.markdown("### Karar özeti")

market, confidence_probability, confidence = prediction_signal(selected)
if confidence_probability is None:
    st.info("Bu maç için model tahmini henüz hazırlanmadı.")
else:
    threshold = minimum_confidence_for_league(selected.get("league_id"))
    if confidence_probability >= threshold:
        st.success(
            f"Aksiyon eşiğini geçen sinyal: **{market}** "
            f"({probability_percent(confidence_probability)}) · güven: **{confidence}**."
        )
    else:
        st.info(
            f"Aksiyon sinyali yok. En yüksek olasılık **{market}** "
            f"({probability_percent(confidence_probability)}); gerekli eşik "
            f"%{threshold * 100:.0f}."
        )

production_version = active_production_version()
selected_version = str(selected.get("model_version") or "").strip()
with st.expander("Model ve veri teknik bilgileri"):
    if not production_version:
        st.caption("Üretim modeli bilgisi doğrulanamadı.")
    elif selected_version and selected_version != production_version:
        st.warning(
            f"Bu maçın tahmini `{selected_version}` eski bir model sürümünden üretildi. "
            f"Aktif sürüm: {production_version}."
        )
    else:
        st.caption(f"Üretim modeli: {version_label(production_version)}")

note_key = f"match_note_{int(selected_id)}"
if auth_enabled:
    try:
        saved_note = get_user_db().select(
            "personal_match_workspace",
            columns="note",
            filters={"match_id": f"eq.{int(selected_id)}"},
            limit=1,
        )
        if saved_note and note_key not in st.session_state:
            st.session_state[note_key] = saved_note[0].get("note") or ""
    except Exception:
        st.warning("Kişisel not yüklenemedi; bu oturumda düzenleyebilirsiniz.")
st.text_area(
    "Kişisel maç notu",
    key=note_key,
    placeholder="Bu maçı neden takip ettiğini veya hangi riski gördüğünü yaz...",
    height=100,
)
if auth_enabled and st.button("Notu kaydet", key=f"save_note_{int(selected_id)}"):
    try:
        get_user_db().upsert(
            "personal_match_workspace",
            [{"user_id": current_user_id(), "match_id": int(selected_id), "note": st.session_state.get(note_key, "")}],
            on_conflict="user_id,match_id",
        )
        st.success("Not kaydedildi.")
    except Exception:
        st.error("Not kaydedilemedi.")
else:
    st.caption("Notu kalıcılaştırmak için Supabase hesabıyla giriş yapın.")

st.markdown("### Karar günlüğü")
if not auth_enabled:
    st.info("Karar kaydetmek için Supabase hesabıyla giriş yapın.")
else:
    with st.form(f"decision_form_{int(selected_id)}"):
        decision = st.radio(
            "Bu maç için kararınız",
            ["played", "passed", "ignored"],
            format_func={
                "played": "Oyna",
                "passed": "Pas geç",
                "ignored": "Daha sonra incele",
            }.get,
            horizontal=True,
        )
        decision_market = st.text_input(
            "Seçilen piyasa", value=str(market or "") if confidence_probability is not None else ""
        )
        decision_odds = st.number_input(
            "Karar anındaki oran", min_value=0.0, value=0.0, step=0.01, format="%.2f"
        )
        if st.form_submit_button("Kararı kaydet"):
            try:
                get_user_db().insert(
                    "personal_match_decisions",
                    [
                        {
                            "user_id": current_user_id(),
                            "match_id": int(selected_id),
                            "decision": decision,
                            "selected_market": decision_market.strip() or None,
                            "selected_odds": decision_odds or None,
                            "model_probability": confidence_probability,
                            "model_threshold": threshold if confidence_probability is not None else None,
                            "note": st.session_state.get(note_key, ""),
                        }
                    ],
                )
                st.success("Karar günlüğüne kaydedildi.")
            except Exception:
                st.error("Karar kaydedilemedi.")

metrics = st.columns(5)
for column, label, key in zip(
    metrics,
    ["Maç Sonucu 1", "Maç Sonucu X", "Maç Sonucu 2", "2,5 Gol Üst", "Karşılıklı Gol Var"],
    ["prob_home_win", "prob_draw", "prob_away_win", "prob_over_2_5", "prob_btts"],
):
    column.metric(label, probability_percent(selected.get(key)))
