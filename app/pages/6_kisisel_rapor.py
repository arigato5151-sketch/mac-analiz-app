"""Authenticated weekly decision report."""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.components.auth import get_user_db, render_auth_panel
from app.components.personal_report import summarize_decisions
from app.components.ui import configure_page, disclaimer, page_header


configure_page("Kişisel Rapor")
page_header(
    "Kişisel karar raporu",
    "Son yedi gündeki kararlarını, sonuçlarını ve tekrar eden hatalarını gösterir.",
    eyebrow="KARAR GÜNLÜĞÜ",
)
disclaimer()

if not render_auth_panel():
    st.info("Kişisel raporu görmek için Supabase hesabıyla giriş yapın.")
    st.stop()

try:
    rows = get_user_db().select_all(
        "personal_match_decisions",
        columns=(
            "id,match_id,decision,selected_market,selected_odds,model_probability,"
            "model_threshold,note,match_result,was_correct,created_at"
        ),
        order="created_at.desc",
    )
except Exception:
    st.error("Karar günlüğü yüklenemedi.")
    st.stop()

summary = summarize_decisions(rows)
metrics = st.columns(5)
metrics[0].metric("Son 7 gün karar", summary["decision_count"])
metrics[1].metric("Oynanan", summary["played_count"])
metrics[2].metric("Pas geçilen", summary["passed_count"])
metrics[3].metric("Sonuçlanan", summary["settled_count"])
metrics[4].metric(
    "Karar isabeti",
    f"%{summary['accuracy'] * 100:.1f}" if summary["accuracy"] is not None else "—",
)

if not summary["recent"]:
    st.info("Son yedi gün içinde kayıtlı karar yok.")
else:
    frame = pd.DataFrame(summary["recent"])
    frame["Karar"] = frame["decision"].map(
        {"played": "Oyna", "passed": "Pas geç", "ignored": "Daha sonra"}
    )
    frame["Olasılık"] = pd.to_numeric(frame["model_probability"], errors="coerce").map(
        lambda value: f"%{value * 100:.1f}" if pd.notna(value) else "—"
    )
    frame["Sonuç"] = frame["was_correct"].map(
        {True: "Doğru", False: "Yanlış"}
    ).fillna("Bekliyor")
    st.dataframe(
        frame[["created_at", "match_id", "Karar", "selected_market", "Olasılık", "Sonuç"]],
        hide_index=True,
        width="stretch",
    )
