"""Selected match analysis and Poisson score heatmap."""

import pandas as pd
import plotly.express as px
import streamlit as st

from app.components.data import load_match_baseline, load_upcoming_dashboard
from app.components.ui import configure_page, disclaimer, probability_percent


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
