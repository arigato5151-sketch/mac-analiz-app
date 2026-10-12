import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.components.data import _nesine_market_key, get_db, load_upcoming_dashboard
from app.components.decision_board import requested_match_id
from app.components.nesine_labels import nesine_market_name, nesine_selection_name
from app.components.ui import (
    configure_page,
    disclaimer,
    page_header,
    probability_percent,
)

COUNTRY_NAMES_TR = {
    "austria": "Avusturya",
    "belgium": "Belçika",
    "brazil": "Brezilya",
    "england": "İngiltere",
    "france": "Fransa",
    "germany": "Almanya",
    "greece": "Yunanistan",
    "italy": "İtalya",
    "mexico": "Meksika",
    "netherlands": "Hollanda",
    "portugal": "Portekiz",
    "saudi-arabia": "Suudi Arabistan",
    "saudi arabia": "Suudi Arabistan",
    "scotland": "İskoçya",
    "spain": "İspanya",
    "switzerland": "İsviçre",
    "turkey": "Türkiye",
    "usa": "ABD",
    "united states": "ABD",
    "world": "Uluslararası",
}
LEAGUE_NAMES_TR = {
    ("england", "premier league"): "Premier Lig",
    ("england", "championship"): "İngiltere Championship",
    ("spain", "la liga"): "La Liga",
    ("spain", "segunda división"): "İspanya 2. Ligi",
    ("italy", "serie a"): "Serie A",
    ("italy", "serie b"): "Serie B",
    ("brazil", "serie a"): "Série A",
    ("germany", "bundesliga"): "Bundesliga",
    ("france", "ligue 1"): "Ligue 1",
    ("france", "ligue 2"): "Ligue 2",
    ("portugal", "primeira liga"): "Portekiz Premier Ligi",
    ("netherlands", "eredivisie"): "Hollanda Eredivisie",
    ("belgium", "jupiler pro league"): "Belçika Pro Ligi",
    ("saudi-arabia", "pro league"): "Suudi Arabistan Pro Ligi",
    ("saudi arabia", "pro league"): "Suudi Arabistan Pro Ligi",
    ("usa", "major league soccer"): "ABD MLS",
    ("united states", "major league soccer"): "ABD MLS",
    ("mexico", "liga mx"): "Meksika Liga MX",
    ("scotland", "premiership"): "İskoçya Premiership",
    ("austria", "bundesliga"): "Avusturya Bundesliga",
    ("switzerland", "super league"): "İsviçre Süper Ligi",
    ("greece", "super league 1"): "Yunanistan Süper Ligi",
    ("world", "uefa champions league"): "UEFA Şampiyonlar Ligi",
    ("world", "uefa europa league"): "UEFA Avrupa Ligi",
    ("world", "uefa europa conference league"): "UEFA Konferans Ligi",
    ("world", "conmebol libertadores"): "Libertadores Kupası",
    ("world", "fifa world cup"): "Dünya Kupası",
    ("world", "euro championship"): "Avrupa Şampiyonası",
    ("world", "uefa nations league"): "UEFA Uluslar Ligi",
    ("world", "africa cup of nations"): "Afrika Uluslar Kupası",
    ("world", "asian cup"): "Asya Kupası",
    ("world", "copa america"): "Amerika Kupası",
    ("world", "concacaf gold cup"): "CONCACAF Altın Kupa",
}


def _all_market_display_rows(
    probabilities: dict[str, float], quotes: list[dict[str, object]]
) -> list[dict[str, str]]:
    """Combine supported model markets with the latest matching Nesine quotes."""
    quote_by_key: dict[str, float] = {}
    for quote in quotes:
        key = _nesine_market_key(quote)
        try:
            odd = float(quote["odd"])
        except (KeyError, TypeError, ValueError):
            continue
        if key and odd > 1:
            quote_by_key[key] = odd

    labels: dict[str, tuple[str, str]] = {}

    def add(key: str, market: str, selection: str) -> None:
        labels[key] = (market, selection)

    for key, selection in (("home_win", "1"), ("draw", "X"), ("away_win", "2")):
        add(key, "Maç sonucu", selection)
    for token in ("1x", "x2", "12"):
        add(f"double_chance_{token}", "Çifte Şans", token.upper())
    for period, period_label in (("first_half", "1. yarı"), ("second_half", "2. yarı")):
        for key, selection in (("home_win", "1"), ("draw", "X"), ("away_win", "2")):
            add(f"{period}_{key}", f"{period_label} maç sonucu", selection)
    for line in ("1_5", "2_5", "3_5", "4_5"):
        for direction in ("over", "under"):
            add(f"{direction}_{line}", f"{line.replace('_', ',')} Gol Alt/Üst", f"{'Üst' if direction == 'over' else 'Alt'}")
            for result, result_label in (("home_win", "1"), ("draw", "X"), ("away_win", "2")):
                add(f"{result}_{direction}_{line}", f"Maç sonucu ve {line.replace('_', '.')} gol", f"{result_label} + {'Üst' if direction == 'over' else 'Alt'}")
    for line in ("0_5", "1_5", "2_5", "3_5", "4_5"):
        for direction in ("over", "under"):
            for team, team_label in (("home", "Ev sahibi"), ("away", "Deplasman")):
                add(f"{team}_{direction}_{line}", f"{team_label} gol", f"{'Üst' if direction == 'over' else 'Alt'} {line.replace('_', '.')}")
    for result, result_label in (("home_win", "1"), ("draw", "X"), ("away_win", "2")):
        for outcome, outcome_label in (("yes", "Var"), ("no", "Yok")):
            add(f"{result}_btts_{outcome}", "Maç sonucu ve KG", f"{result_label} + {outcome_label}")
    add("btts_yes", "Karşılıklı Gol", "Var")
    add("btts_no", "Karşılıklı Gol", "Yok")
    for team, team_label in (("home", "Ev sahibi"), ("away", "Deplasman")):
        for period, period_label in (("first_half", "1. yarı"), ("second_half", "2. yarı")):
            for line in ("0_5", "1_5"):
                for direction in ("over", "under"):
                    add(
                        f"{team}_{period}_{direction}_{line}",
                        f"{team_label} {period_label} gol",
                        f"{'Üst' if direction == 'over' else 'Alt'} {line.replace('_', '.')}",
                    )
    for line in ("7_5", "8_5", "9_5", "10_5", "11_5"):
        for direction in ("over", "under"):
            add(f"corners_{direction}_{line}", f"{line.replace('_', ',')} Korner Alt/Üst", f"{'Üst' if direction == 'over' else 'Alt'} {line.replace('_', ',')}")

    rows = []
    for key, (market, selection) in labels.items():
        probability = probabilities.get(key)
        odd = quote_by_key.get(key)
        rows.append({
            "Pazar": nesine_market_name(key),
            "Seçim": nesine_selection_name(key),
            "Model olasılığı": f"%{float(probability) * 100:.1f}" if probability is not None else "Tahmin yok",
            "Nesine oranı": f"{odd:.2f}" if odd is not None else "—",
        })
    return rows


configure_page("Maç Detay")
page_header(
    "Maç detay analizi",
    "Maç olasılıklarını ve Nesine pazar oranlarını görüntüleyin.",
    eyebrow="DERİNLEMESİNE ANALİZ",
)
disclaimer()

try:
    matches = load_upcoming_dashboard(7)
except Exception:
    st.error("Maç listesi şu anda kullanılamıyor. Birkaç dakika sonra tekrar deneyin.")
    st.stop()

if matches.empty:
    st.info("Detayı gösterilecek yaklaşan maç bulunamadı.")
    st.stop()

matches = matches.copy()
def _match_label(match: pd.Series) -> str:
    country = match.get("country")
    league = match.get("league_name")
    country = country.strip() if isinstance(country, str) and country.strip() else "Ülke bilinmiyor"
    league = league.strip() if isinstance(league, str) and league.strip() else "Lig bilinmiyor"
    country_key = country.casefold()
    league = LEAGUE_NAMES_TR.get((country_key, league.casefold()), league)
    country = COUNTRY_NAMES_TR.get(country_key, country)
    kickoff = match["match_date"].strftime("%d.%m %H:%M")
    return (
        f"{kickoff} · {country} · {league} · "
        f"{match['home_team']} — {match['away_team']}"
    )


matches["label"] = matches.apply(_match_label, axis=1)
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
# Market probabilities are already part of the saved prediction. Read only this
# match's captured Nesine prices; do not rebuild the historical Poisson baseline.
raw_market_probabilities = selected.get("market_probabilities") or {}
if isinstance(raw_market_probabilities, str):
    try:
        raw_market_probabilities = json.loads(raw_market_probabilities)
    except json.JSONDecodeError:
        raw_market_probabilities = {}
market_probabilities: dict[str, float] = {
    "home_win": selected.get("prob_home_win"),
    "draw": selected.get("prob_draw"),
    "away_win": selected.get("prob_away_win"),
    "over_2_5": selected.get("prob_over_2_5"),
    "under_2_5": (
        1 - float(selected.get("prob_over_2_5"))
        if pd.notna(selected.get("prob_over_2_5"))
        else None
    ),
    "btts_yes": selected.get("prob_btts"),
    "btts_no": (
        1 - float(selected.get("prob_btts"))
        if pd.notna(selected.get("prob_btts"))
        else None
    ),
}
if isinstance(raw_market_probabilities, dict):
    double_chance = raw_market_probabilities.get("double_chance") or {}
    if isinstance(double_chance, dict):
        for selection, probability in double_chance.items():
            market_probabilities[f"double_chance_{str(selection).lower()}"] = probability
    for group in ("total_goals", "team_goals"):
        values = raw_market_probabilities.get(group) or {}
        if isinstance(values, dict):
            market_probabilities.update(values)
    for key, value in raw_market_probabilities.items():
        if key not in {"score_matrix", "correct_scores", "expected_goals", "double_chance", "total_goals", "team_goals"}:
            if isinstance(value, (int, float)):
                market_probabilities[key] = float(value)

requested_market_rows: list[dict[str, object]] = []
try:
    detail_rows = get_db().select_all(
        "nesine_market_quotes",
        columns="market_code,market_name,selection_code,selection_name,odd,captured_at",
        filters={"match_id": f"eq.{int(selected_id)}"},
        order="captured_at.desc",
    )
    seen_markets: set[tuple[str, str]] = set()
    for row in detail_rows:
        row_key = (str(row.get("market_code", "")), str(row.get("selection_code", "")))
        if row_key in seen_markets:
            continue
        seen_markets.add(row_key)
        requested_market_rows.append(row)
except Exception:
    requested_market_rows = []

st.subheader("Pazar tahminleri")
st.caption(
    "Model tahminleri ve yakalanan Nesine oranları. Oran olmayan pazarlarda —, "
    "model tahmini olmayanlarda Tahmin yok gösterilir."
)
st.dataframe(
    pd.DataFrame(_all_market_display_rows(market_probabilities, requested_market_rows)),
    hide_index=True,
    width="stretch",
)
