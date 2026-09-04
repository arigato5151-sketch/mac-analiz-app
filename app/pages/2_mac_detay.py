"""Selected match analysis and Poisson score heatmap."""

import logging
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.components.availability import summarize_availability
from app.components.commentary import summarize_absences, summarize_form
from app.components.data import load_confirmed_lineups, load_match_availability, load_match_baseline, load_odds_history, load_upcoming_dashboard
from app.components.match_visuals import build_form_comparison, build_radar_comparison
from app.components.ui import (
    configure_page,
    disclaimer,
    prediction_signal,
    probability_percent,
)
from data_pipeline.match_commentary import MatchCommentaryError, generate_match_commentary


LOGGER = logging.getLogger(__name__)


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
availability_data: list[dict[str, object]] = []

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

st.subheader("Onaylı ilk 11")
try:
    lineups = load_confirmed_lineups(int(selected_id))
    if len(lineups) < 2:
        st.info("Resmî ilk 11 henüz açıklanmadı. API-Football çoğu ligde kadroyu başlama saatinden 20–40 dakika önce yayınlar.")
    else:
        lineup_columns = st.columns(2)
        for column, team_id, team_name in (
            (lineup_columns[0], int(selected["home_team_id"]), selected["home_team"]),
            (lineup_columns[1], int(selected["away_team_id"]), selected["away_team"]),
        ):
            lineup = lineups.loc[lineups["team_id"].astype(int) == team_id].iloc[0]
            column.markdown(f"**{team_name}** · {lineup.get('formation') or 'Diziliş bilinmiyor'}")
            if lineup.get("coach_name"):
                column.caption(f"Teknik direktör: {lineup['coach_name']}")
            starters = lineup.get("starters") or []
            column.dataframe(
                pd.DataFrame(starters)[["name", "pos"]].rename(columns={"name": "Oyuncu", "pos": "Pozisyon"}),
                hide_index=True,
                use_container_width=True,
            )
        st.caption("Bu alan yalnızca API’nin onaylı ilk 11 kaydı geldiğinde görünür. Mevcut model, tarihsel ilk-11 verisi henüz olmadığı için olasılıkları sonradan yapay olarak değiştirmez.")
except Exception as exc:
    st.warning(f"Onaylı ilk 11 şu anda yüklenemedi: {exc}")

st.subheader("Oran hareketi")
try:
    odds_history = load_odds_history(int(selected_id))
    if odds_history.empty:
        st.info("Bu maç için henüz kaydedilmiş oran hareketi yok.")
    else:
        odds_rows = odds_history.copy()
        odds_rows["Zaman"] = odds_rows["captured_at"].dt.strftime("%d.%m %H:%M")
        odds_rows["Referans"] = odds_rows["is_notification_reference"].map({True: "Bildirim anı", False: "Piyasa güncellemesi"})
        odds_rows["1"] = odds_rows["odds"].map(lambda item: (item or {}).get("home_win", "—"))
        odds_rows["X"] = odds_rows["odds"].map(lambda item: (item or {}).get("draw", "—"))
        odds_rows["2"] = odds_rows["odds"].map(lambda item: (item or {}).get("away_win", "—"))
        st.dataframe(odds_rows[["Zaman", "bookmaker", "1", "X", "2", "Referans"]].rename(columns={"bookmaker": "Sağlayıcı"}), hide_index=True, use_container_width=True)
        st.caption("Kapanış oranı, maç başlangıcından önce yakalanabilen son sağlayıcı kotasyonudur. Veri yoksa CLV hesaplanmaz.")
except Exception as exc:
    st.warning(f"Oran geçmişi şu anda yüklenemedi: {exc}")

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

    st.subheader("Yapay Zeka Maç Yorumu")
    st.caption(
        "Yorum; mevcut model olasılıkları, Poisson beklenen golü, son form ve doğrulanmış "
        "kadro verisinden üretilir. Kesin sonuç veya bahis tavsiyesi değildir."
    )
    commentary_key = f"gemini_commentary:{int(selected_id)}:{selected.get('model_version', 'pending')}"
    probabilities = [
        selected.get("prob_home_win"),
        selected.get("prob_draw"),
        selected.get("prob_away_win"),
    ]
    has_probabilities = all(pd.notna(probability) for probability in probabilities)
    if not has_probabilities:
        st.info("Yapay zeka yorumu için önce bu maçın 1X2 model olasılıkları hazırlanmalıdır.")
    elif st.button("Yorumu oluştur", key=f"generate_commentary:{int(selected_id)}"):
        try:
            with st.spinner("Maç yorumu hazırlanıyor..."):
                st.session_state[commentary_key] = generate_match_commentary(
                    home_team=str(selected["home_team"]),
                    away_team=str(selected["away_team"]),
                    home_xg=float(baseline.home_expected_goals),
                    away_xg=float(baseline.away_expected_goals),
                    home_absences=summarize_absences(
                        availability_data, team_id=int(selected["home_team_id"])
                    ),
                    away_absences=summarize_absences(
                        availability_data, team_id=int(selected["away_team_id"])
                    ),
                    home_form=summarize_form(home_state),
                    away_form=summarize_form(away_state),
                    home_win_probability=float(probabilities[0]),
                    draw_probability=float(probabilities[1]),
                    away_win_probability=float(probabilities[2]),
                )
        except MatchCommentaryError as error:
            # Keep provider details and credentials out of both the UI and application logs.
            LOGGER.warning("Gemini commentary unavailable: reason=%s", error.reason)
            user_messages = {
                "configuration": "Yapay zeka yorum ayarı eksik. GEMINI_API_KEY ve google-genai kurulumu kontrol edilmelidir.",
                "authentication": "Yapay zekâ yorum servisi doğrulanamadı. Uygulama yöneticisi anahtar ayarını kontrol etmelidir.",
                "quota": "Yapay zekâ yorum kotası geçici olarak dolu. Birkaç dakika sonra tekrar deneyin.",
                "timeout": "Yapay zekâ yorum servisi zaman aşımına uğradı. Lütfen tekrar deneyin.",
                "provider": "Yapay zekâ yorum servisi bu isteği işleyemedi. GEMINI_MODEL ayarını ve API erişimini kontrol edin.",
            }
            st.error(user_messages.get(error.reason, "Yorum şu anda üretilemedi. Lütfen tekrar deneyin."))
        except Exception:
            LOGGER.exception("Unexpected error while generating Gemini commentary")
            st.error(
                "Yorum hazırlanırken beklenmeyen bir hata oluştu. "
                "Uygulama logunda ayrıntılı hata kaydı oluşturuldu."
            )

    commentary = st.session_state.get(commentary_key)
    if commentary:
        st.info(commentary)

    st.subheader("Takım karşılaştırması")
    radar = build_radar_comparison(
        home_state,
        away_state,
        match_at=selected["match_date"].to_pydatetime(),
    )
    form = build_form_comparison(home_state, away_state)
    radar_figure = go.Figure()
    for column, team_name in (("Ev", selected["home_team"]), ("Deplasman", selected["away_team"])):
        values = radar[column].tolist()
        radar_figure.add_trace(
            go.Scatterpolar(
                r=[*values, values[0]],
                theta=[*radar["Metrik"].tolist(), radar["Metrik"].iloc[0]],
                fill="toself",
                name=team_name,
            )
        )
    radar_figure.update_layout(
        polar={"radialaxis": {"visible": True, "range": [0, 100]}},
        margin={"l": 30, "r": 30, "t": 30, "b": 30},
        legend={"orientation": "h", "y": -0.15},
    )
    form_figure = px.line(
        form,
        x="Maç",
        y="Puan",
        color="Takım",
        markers=True,
        category_orders={"Maç": ["M-5", "M-4", "M-3", "M-2", "M-1"]},
    ).update_layout(
        yaxis={"range": [0, 3], "dtick": 1, "title": "Maç puanı"},
        xaxis_title="Son beş maç · eski → yeni",
        margin={"l": 30, "r": 20, "t": 30, "b": 30},
        legend={"orientation": "h", "y": -0.2},
    )
    comparison_columns = st.columns(2)
    comparison_columns[0].plotly_chart(radar_figure, use_container_width=True)
    comparison_columns[1].plotly_chart(form_figure, use_container_width=True)
    st.caption("Radar 0–100 ölçeğinde normalize edilmiştir; form grafiği son beş maçın gerçek 0/1/3 puanlarını gösterir.")

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
