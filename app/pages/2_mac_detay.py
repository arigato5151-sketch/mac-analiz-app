"""Selected match analysis and Poisson score heatmap."""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.components.availability import summarize_availability
from app.components.data import load_match_availability, load_match_baseline, load_upcoming_dashboard
from app.components.ui import (
    configure_page,
    disclaimer,
    prediction_signal,
    probability_percent,
)


configure_page("Maç Detay")
st.title("Maç Detay Analizi")
disclaimer()

try:
    matches = load_upcoming_dashboard(7)
except Exception as exc:
    st.error(f"Maç listesi yüklenemedi: {exc}")
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
selected_id = st.selectbox(
    "Maç seçin",
    matches["id"].tolist(),
    format_func=lambda match_id: matches.loc[matches["id"] == match_id, "label"].iloc[0],
)
selected = matches.loc[matches["id"] == selected_id].iloc[0]

st.subheader(f"{selected['home_team']} — {selected['away_team']}")
st.caption(f"{selected['league_name']} · {selected['match_date'].strftime('%d.%m.%Y %H:%M')}")

market, confidence_probability, confidence = prediction_signal(selected)
if confidence_probability is None:
    st.info("Bu maç için model tahmini henüz hazırlanmadı.")
else:
    st.info(
        f"En güçlü model sinyali: **{market}** ({probability_percent(confidence_probability)}) · "
        f"güven seviyesi: **{confidence}**. Bu, kesin sonuç değil istatistiksel olasılıktır."
    )

metrics = st.columns(5)
for column, label, key in zip(
    metrics,
    ["Ev", "Beraberlik", "Deplasman", "Üst 2.5", "KG Var"],
    ["prob_home_win", "prob_draw", "prob_away_win", "prob_over_2_5", "prob_btts"],
):
    column.metric(label, probability_percent(selected.get(key)))

probability_frame = pd.DataFrame(
    {
        "Sonuç": ["Ev", "Beraberlik", "Deplasman"],
        "Olasılık": [
            selected.get("prob_home_win", 0),
            selected.get("prob_draw", 0),
            selected.get("prob_away_win", 0),
        ],
    }
)
st.plotly_chart(
    px.bar(
        probability_frame,
        x="Sonuç",
        y="Olasılık",
        range_y=[0, 1],
        text_auto=".1%",
        color="Sonuç",
    ).update_layout(showlegend=False),
    use_container_width=True,
)

st.subheader("Kadro durumu")
try:
    availability_rows, availability_snapshots = load_match_availability(
        int(selected["home_team_id"]), int(selected["away_team_id"])
    )
    snapshots_by_team = {
        int(row["team_id"]): row.to_dict()
        for _, row in availability_snapshots.iterrows()
    }
    availability_data = availability_rows.to_dict("records")
    availability_now = pd.Timestamp.now(tz="UTC").to_pydatetime()
    availability_columns = st.columns(2)
    for column, team_id, team_name in (
        (availability_columns[0], int(selected["home_team_id"]), selected["home_team"]),
        (availability_columns[1], int(selected["away_team_id"]), selected["away_team"]),
    ):
        summary = summarize_availability(
            availability_data,
            snapshots_by_team.get(team_id),
            team_id=team_id,
            now=availability_now,
        )
        column.markdown(f"**{team_name}** · {summary.status}")
        if summary.status == "Güncel":
            column.caption(
                f"Sakat: {summary.injured} · Cezalı: {summary.suspended} · Şüpheli: {summary.doubtful}"
            )
        elif summary.status == "Güncel değil":
            column.warning("Kadro verisi 30 saati geçti; tahmine ek bağlam olarak kullanmayın.")
        else:
            column.info("Bu takım için doğrulanmış güncel kadro verisi yok.")
    if not availability_rows.empty:
        display_availability = availability_rows.copy()
        display_availability["Durum"] = display_availability["status"].map(
            {"injured": "Sakat", "suspended": "Cezalı", "doubtful": "Şüpheli"}
        )
        display_availability["Takım"] = display_availability["team_id"].map(
            {
                int(selected["home_team_id"]): selected["home_team"],
                int(selected["away_team_id"]): selected["away_team"],
            }
        )
        st.dataframe(
            display_availability[["Takım", "player_name", "Durum"]].rename(
                columns={"player_name": "Oyuncu"}
            ),
            hide_index=True,
            use_container_width=True,
        )
    st.caption(
        "Kadro verisi doğrulanmış güncellik bağlamıdır; tarihsel oyuncu erişilebilirliği olmadığı için "
        "kalibre edilmiş model olasılıklarına doğrudan ağırlık uygulanmaz."
    )
except Exception as exc:
    st.warning(f"Kadro durumu şu anda yüklenemedi: {exc}")

try:
    detail = load_match_baseline(int(selected_id))
    baseline = detail["prediction"]
    st.subheader("Poisson skor matrisi")
    matrix = baseline.score_matrix[:7, :7]
    heatmap = px.imshow(
        matrix,
        labels={"x": "Deplasman golü", "y": "Ev golü", "color": "Olasılık"},
        x=list(range(7)),
        y=list(range(7)),
        text_auto=".1%",
        aspect="auto",
        color_continuous_scale="YlGnBu",
    )
    st.plotly_chart(heatmap, use_container_width=True)
    home_state = detail["home_state"]
    away_state = detail["away_state"]
    elo_cols = st.columns(3)
    elo_cols[0].metric(f"{selected['home_team']} Elo", f"{home_state.elo:.0f}")
    elo_cols[1].metric("Elo farkı", f"{home_state.elo - away_state.elo:+.0f}")
    elo_cols[2].metric(f"{selected['away_team']} Elo", f"{away_state.elo:.0f}")
    st.caption(
        f"Baseline beklenen goller: {baseline.home_expected_goals:.2f} — "
        f"{baseline.away_expected_goals:.2f}; en olası skor {baseline.most_likely_score[0]}-"
        f"{baseline.most_likely_score[1]}. XGBoost; form, Elo, dinlenme, H2H ve bu "
        "Poisson sinyallerini birlikte kullanır."
    )
except Exception as exc:
    st.warning(f"Poisson detayları şu anda hazırlanamadı: {exc}")
