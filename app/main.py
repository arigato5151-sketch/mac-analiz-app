"""Streamlit match center and completed-match statistics archive."""

from __future__ import annotations

import json
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.components.data import (
    load_completed_match_history,
    load_iddaa_market_labels,
    load_match_day,
    load_reference_catalog,
)

ISTANBUL = ZoneInfo("Europe/Istanbul")

st.set_page_config(page_title="Maç Merkezi", page_icon="⚽", layout="wide")
st.markdown(
    """
    <style>
    .block-container{max-width:1440px;padding-top:1.6rem;padding-bottom:3rem}
    .hero{padding:1.5rem 1.8rem;border:1px solid rgba(140,160,190,.18);border-radius:20px;
      background:linear-gradient(120deg,#101c31 0%,#152b3c 55%,#153a36 100%);margin-bottom:1.2rem}
    .hero-kicker{color:#84e0be;text-transform:uppercase;letter-spacing:.12em;font-size:.78rem;font-weight:700}
    .hero h1{margin:.4rem 0 .2rem;font-size:2.2rem}.hero p{color:#c1cedb;margin:0}
    .match-title{font-size:1.25rem;font-weight:700;margin:.2rem 0 .7rem}
    .market-head{font-size:1rem;font-weight:700;margin:.4rem 0 .2rem}
    [data-testid="stMetric"]{background:rgba(128,150,180,.08);padding:.8rem 1rem;border-radius:14px}
    </style>
    """,
    unsafe_allow_html=True,
)


def unpack_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def market_rows(payload: dict[str, Any], labels: dict[str, str]) -> list[dict[str, Any]]:
    rows = []
    for market in payload.get("markets", []):
        selections = []
        for selection in market.get("selections", []):
            try:
                odd = float(selection.get("odds"))
            except (TypeError, ValueError):
                continue
            if odd > 1:
                selections.append((str(selection.get("name") or selection.get("id") or "Seçim"), odd))
        if len(selections) < 2:
            continue
        implied = [1 / odd for _, odd in selections]
        total = sum(implied)
        line = market.get("line")
        market_code = market.get("market_id")
        sport_type = market.get("sport_type")
        market_key = f"{sport_type if sport_type is not None else ''}:{market_code if market_code is not None else ''}"
        name = str(
            market.get("market_name")
            or labels.get(market_key)
            or labels.get(str(market_code if market_code is not None else ""))
            or _infer_legacy_market_name(selections)
            or (f"Market {market_code}" if market_code is not None else "Diğer market")
        )
        if line not in (None, "", 0, "0"):
            name = f"{name} · {line}"
        rows.append({
            "name": name,
            "selections": [
                {"name": item[0], "odds": item[1], "probability": prob / total}
                for item, prob in zip(selections, implied)
            ],
            "overround": total - 1,
        })
    return rows


def _infer_legacy_market_name(selections: list[tuple[str, float]]) -> str | None:
    """Give old code-less market rows a useful label from their outcomes."""
    names = [name.strip().casefold() for name, _ in selections]
    normalized = {" ".join(name.replace("-", " ").split()) for name in names}
    if normalized.issubset({"1", "x", "2"}) and len(normalized) >= 2:
        return "Maç Sonucu"
    if normalized.issubset({"ev sahibi", "beraberlik", "deplasman"}) and len(normalized) >= 2:
        return "Maç Sonucu"
    if any("üst" in name or "alt" in name or "over" in name or "under" in name for name in names):
        return "Alt/Üst"
    if any("karşılıklı" in name or "both teams" in name or name in {"kg var", "kg yok"} for name in names):
        return "Karşılıklı Gol"
    if normalized.issubset({"evet", "hayır", "yes", "no"}):
        return "Evet / Hayır"
    return None


def team_recent_matches(history: pd.DataFrame, team_id: Any, before: pd.Timestamp, limit: int = 5) -> pd.DataFrame:
    """Return a team's latest finished matches strictly before kickoff."""
    if history.empty or pd.isna(before):
        return pd.DataFrame()
    team = pd.to_numeric(pd.Series([team_id]), errors="coerce").iloc[0]
    if pd.isna(team):
        return pd.DataFrame()
    prior = history[
        (history["match_date"] < before)
        & ((history["home_team_id"] == int(team)) | (history["away_team_id"] == int(team)))
    ].copy()
    return prior.sort_values(["match_date", "id"], ascending=False).head(limit)


def summarize_team_form(recent: pd.DataFrame, team_id: Any) -> dict[str, Any]:
    """Summarize recent results from the team's point of view."""
    summary: dict[str, Any] = {"form": "—", "points": None, "goals_for": None, "goals_against": None, "xg_for": None, "xg_against": None}
    if recent.empty:
        return summary
    team = int(team_id)
    form, points, goals_for, goals_against = [], 0, [], []
    xg_for, xg_against = [], []
    for match in recent.to_dict("records"):
        home = int(match["home_team_id"]) == team
        scored = match["home_score"] if home else match["away_score"]
        conceded = match["away_score"] if home else match["home_score"]
        scored, conceded = int(scored), int(conceded)
        goals_for.append(scored)
        goals_against.append(conceded)
        if scored > conceded:
            form.append("G")
            points += 3
        elif scored == conceded:
            form.append("B")
            points += 1
        else:
            form.append("M")
        for target, key in ((xg_for, "home_xg" if home else "away_xg"), (xg_against, "away_xg" if home else "home_xg")):
            try:
                value = float(match[key])
                if pd.notna(value):
                    target.append(value)
            except (TypeError, ValueError):
                pass
    summary.update({
        "form": " · ".join(form),
        "points": points / len(recent),
        "goals_for": sum(goals_for) / len(goals_for),
        "goals_against": sum(goals_against) / len(goals_against),
        "xg_for": sum(xg_for) / len(xg_for) if xg_for else None,
        "xg_against": sum(xg_against) / len(xg_against) if xg_against else None,
    })
    return summary


def render_recent_form(
    row: dict[str, Any], history: pd.DataFrame, team_names: dict[Any, str]
) -> None:
    """Show recent form and match logs for both teams without implying a forecast."""
    kickoff = row["kickoff"]
    home_id, away_id = row["home_team_id"], row["away_team_id"]
    home_recent = team_recent_matches(history, home_id, kickoff)
    away_recent = team_recent_matches(history, away_id, kickoff)
    home_stats, away_stats = summarize_team_form(home_recent, home_id), summarize_team_form(away_recent, away_id)
    st.markdown("#### Maç öncesi form · son 5 tamamlanmış maç")
    st.caption("Özet yalnızca başlama saatinden önce tamamlanmış maçları kullanır. Bu bölüm geçmiş performansı gösterir, tahmin değildir.")
    home_col, away_col = st.columns(2)
    for col, name, stats, recent, team_id in (
        (home_col, row["home_name"], home_stats, home_recent, home_id),
        (away_col, row["away_name"], away_stats, away_recent, away_id),
    ):
        with col:
            st.markdown(f"**{name}**")
            if recent.empty:
                st.info("Bu takım için maç öncesi tamamlanmış geçmiş maç kaydı yok.")
                continue
            st.metric("Form (yeniden eskiye)", stats["form"], f"{len(recent)} maç")
            a, b = st.columns(2)
            a.metric("Maç başı gol", f"{stats['goals_for']:.2f} attı · {stats['goals_against']:.2f} yedi")
            b.metric("Maç başı puan", f"{stats['points']:.2f} / 3")
            if stats["xg_for"] is not None:
                c, d = st.columns(2)
                c.metric("Ortalama xG", f"{stats['xg_for']:.2f}")
                d.metric("Rakibe verilen xG", f"{stats['xg_against']:.2f}" if stats["xg_against"] is not None else "Veri yok")
            log = []
            for match in recent.to_dict("records"):
                is_home = int(match["home_team_id"]) == int(team_id)
                opponent_id = match["away_team_id"] if is_home else match["home_team_id"]
                scored = match["home_score"] if is_home else match["away_score"]
                conceded = match["away_score"] if is_home else match["home_score"]
                result = "G" if scored > conceded else "B" if scored == conceded else "M"
                match_date = pd.to_datetime(match["match_date"], utc=True, errors="coerce")
                log.append({
                    "Tarih": match_date.tz_convert(ISTANBUL).strftime("%d.%m.%Y") if pd.notna(match_date) else "—",
                    "Saha": "Ev" if is_home else "Dep",
                    "Rakip": team_names.get(opponent_id, "Rakip"),
                    "Skor": f"{scored}–{conceded}",
                    "Sonuç": result,
                })
            with st.expander(f"{name}: son maçların dökümü"):
                st.dataframe(pd.DataFrame(log), use_container_width=True, hide_index=True)


def render_analysis(matches_data: list[dict[str, Any]], quotes_data: list[dict[str, Any]], history_data: list[dict[str, Any]], teams: pd.DataFrame, leagues: pd.DataFrame, match_day: date, market_labels: dict[str, str]) -> None:
    matches = pd.DataFrame(matches_data)
    if matches.empty:
        st.info(f"{match_day:%d.%m.%Y} için kayıtlı maç bulunamadı.")
        return
    team_names = dict(zip(teams.get("id", []), teams.get("name", []))) if not teams.empty else {}
    league_names = dict(zip(leagues.get("id", []), leagues.get("name", []))) if not leagues.empty else {}
    latest: dict[int, dict[str, Any]] = {}
    for quote in sorted(quotes_data, key=lambda q: str(q.get("captured_at", ""))):
        try:
            latest[int(quote["match_id"])] = quote
        except (KeyError, TypeError, ValueError):
            continue

    matches["home_name"] = matches["home_team_id"].map(team_names).fillna("Ev sahibi")
    matches["away_name"] = matches["away_team_id"].map(team_names).fillna("Deplasman")
    matches["league_name"] = matches["league_id"].map(league_names).fillna("Lig bilgisi yok")
    matches["kickoff"] = pd.to_datetime(matches["match_date"], utc=True, errors="coerce").dt.tz_convert(ISTANBUL)
    history = pd.DataFrame(history_data)
    if not history.empty:
        history["match_date"] = pd.to_datetime(history["match_date"], utc=True, errors="coerce")
        for column in ("id", "home_team_id", "away_team_id", "home_score", "away_score"):
            history[column] = pd.to_numeric(history[column], errors="coerce")

    left, right, third = st.columns([1, 1, 2])
    with left:
        league_filter = st.selectbox("Lig", ["Tüm ligler", *sorted(matches["league_name"].unique())])
    with right:
        coverage = st.selectbox("Oran durumu", ["Tüm maçlar", "Oranı olanlar", "Oranı olmayanlar"])
    with third:
        search = st.text_input("Takım ara", placeholder="Takım adı yazın")
    filtered = matches.copy()
    if league_filter != "Tüm ligler":
        filtered = filtered[filtered["league_name"] == league_filter]
    if search.strip():
        q = search.strip().casefold()
        filtered = filtered[filtered["home_name"].str.casefold().str.contains(q) | filtered["away_name"].str.casefold().str.contains(q)]
    if coverage == "Oranı olanlar":
        filtered = filtered[filtered["id"].map(lambda x: int(x) in latest)]
    elif coverage == "Oranı olmayanlar":
        filtered = filtered[~filtered["id"].map(lambda x: int(x) in latest)]

    quoted_ids = {int(row["match_id"]) for row in quotes_data if row.get("match_id") is not None}
    market_count = sum(len(unpack_payload(row.get("odds")).get("markets", [])) for row in latest.values())
    a, b, c = st.columns(3)
    a.metric("Listelenen maç", f"{len(filtered)}")
    b.metric("Oranı bulunan maç", f"{len(quoted_ids)} / {len(matches)}")
    c.metric("Mevcut market satırı", f"{market_count}")
    st.caption("Piyasa yüzdeleri, market içindeki iddaa oranlarının marjdan arındırılmış karşılığıdır; bağımsız model tahmini değildir.")
    st.divider()

    if filtered.empty:
        st.info("Bu filtrelerle eşleşen maç yok.")
        return
    for row in filtered.sort_values("kickoff", na_position="last").to_dict("records"):
        match_id = int(row["id"])
        quote = latest.get(match_id)
        kickoff = row["kickoff"]
        time_text = kickoff.strftime("%H:%M") if pd.notna(kickoff) else "Saat bilgisi yok"
        with st.container(border=True):
            meta, status_col = st.columns([4, 1])
            meta.caption(f"{row['league_name']}  ·  {time_text}")
            status_col.caption(str(row.get("status", "")).replace("_", " ").title())
            st.markdown(f"<div class='match-title'>{row['home_name']} <span style='color:#91a2b4'>—</span> {row['away_name']}</div>", unsafe_allow_html=True)
            render_recent_form(row, history, team_names)
            st.divider()
            if not quote:
                st.warning("Bu maç için seçilen günde iddaa oran kaydı bulunamadı.")
                continue
            payload = unpack_payload(quote.get("odds"))
            markets = market_rows(payload, market_labels)
            if not markets:
                st.warning("Oran kaydı var ancak gösterilebilir tam market bulunamadı.")
                continue
            st.caption(f"Son oran kaydı: {quote.get('captured_at', 'bilinmiyor')}")
            tabs = st.tabs([m["name"] for m in markets])
            for tab, market in zip(tabs, markets):
                with tab:
                    cols = st.columns(min(3, len(market["selections"])))
                    for idx, selection in enumerate(market["selections"]):
                        with cols[idx % len(cols)]:
                            st.markdown(f"**{selection['name']}**")
                            st.metric("Oran", f"{selection['odds']:.2f}", f"Piyasa %{selection['probability'] * 100:.1f}")
                            st.progress(min(1.0, max(0.0, selection["probability"])))
                    st.caption(f"Ham market marjı: %{market['overround'] * 100:.1f}")


def render_archive() -> None:
    try:
        matches = pd.DataFrame(load_completed_match_history())
        teams, leagues = load_reference_catalog()
    except Exception as exc:
        st.error(f"Maç istatistikleri şu anda yüklenemedi: {exc}")
        return
    if matches.empty:
        st.info("Kayıtlı tamamlanmış maç bulunamadı.")
        return
    home = teams.rename(columns={"id": "home_team_id", "name": "Ev sahibi"})[["home_team_id", "Ev sahibi"]]
    away = teams.rename(columns={"id": "away_team_id", "name": "Deplasman"})[["away_team_id", "Deplasman"]]
    league_labels = leagues.rename(columns={"id": "league_id", "name": "Lig", "country": "Ülke"})[["league_id", "Lig", "Ülke"]]
    matches = matches.merge(home, on="home_team_id", how="left").merge(away, on="away_team_id", how="left").merge(league_labels, on="league_id", how="left")
    matches["Tarih"] = pd.to_datetime(matches["match_date"], utc=True, errors="coerce").dt.tz_convert(ISTANBUL)
    matches["Skor"] = matches["home_score"].astype("Int64").astype(str) + " - " + matches["away_score"].astype("Int64").astype(str)
    c1, c2 = st.columns(2)
    with c1:
        selected = st.multiselect("Lig", sorted(matches["Lig"].dropna().unique()))
    with c2:
        query = st.text_input("Takım ara", key="archive_search", placeholder="Takım adı")
    if selected:
        matches = matches[matches["Lig"].isin(selected)]
    if query.strip():
        q = query.casefold().strip()
        matches = matches[matches["Ev sahibi"].fillna("").str.casefold().str.contains(q) | matches["Deplasman"].fillna("").str.casefold().str.contains(q)]
    columns = ["Tarih", "Ülke", "Lig", "Ev sahibi", "Skor", "Deplasman", "home_xg", "away_xg", "home_xa", "away_xa", "home_corners", "away_corners"]
    labels = {"home_xg": "Ev xG", "away_xg": "Dep xG", "home_xa": "Ev xA", "away_xa": "Dep xA", "home_corners": "Ev korner", "away_corners": "Dep korner"}
    st.dataframe(matches.sort_values("Tarih", ascending=False)[columns].rename(columns=labels), use_container_width=True, hide_index=True)


st.markdown("<section class='hero'><div class='hero-kicker'>Günlük karşılaşmalar · iddaa marketleri</div><h1>Maç Merkezi</h1><p>Maçları inceleyin, marketleri karşılaştırın ve maç istatistikleri arşivine ulaşın.</p></section>", unsafe_allow_html=True)
today = datetime.now(ISTANBUL).date()
col, _ = st.columns([1, 3])
with col:
    selected_day = st.date_input("Maç günü", value=today, format="DD.MM.YYYY")
tab_analysis, tab_archive = st.tabs(["⚽  Maç Analizleri", "📊  İstatistik Arşivi"])
try:
    teams, leagues = load_reference_catalog()
except Exception as exc:
    st.error(f"Takım ve lig bilgileri yüklenemedi: {exc}")
    teams, leagues = pd.DataFrame(), pd.DataFrame()
with tab_analysis:
    try:
        match_records, quote_records = load_match_day(selected_day)
        analysis_records = load_completed_match_history()
        try:
            market_labels = load_iddaa_market_labels()
        except Exception:
            market_labels = {}
        render_analysis(match_records, quote_records, analysis_records, teams, leagues, selected_day, market_labels)
    except Exception as exc:
        st.error(f"Maç ve oran verileri yüklenemedi: {exc}")
with tab_archive:
    render_archive()
