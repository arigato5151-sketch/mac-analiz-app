"""Streamlit match center and completed-match statistics archive."""

from __future__ import annotations

import json
import math
import re
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
            "source_name": str(market.get("market_name") or name),
            "line": line,
            "market_id": market_code,
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


def team_recent_matches(history: pd.DataFrame, team_id: Any, before: pd.Timestamp, league_id: Any, limit: int = 5) -> pd.DataFrame:
    """Return a team's latest finished matches strictly before kickoff."""
    if history.empty or pd.isna(before):
        return pd.DataFrame()
    team = pd.to_numeric(pd.Series([team_id]), errors="coerce").iloc[0]
    league = pd.to_numeric(pd.Series([league_id]), errors="coerce").iloc[0]
    if pd.isna(team) or pd.isna(league):
        return pd.DataFrame()
    prior = history[
        (history["match_date"] < before)
        & (history["league_id"] == int(league))
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


def poisson_probabilities(mean: float, max_goals: int = 12) -> list[float]:
    """Return a normalized Poisson goal distribution, including its tail."""
    mean = max(0.05, float(mean))
    probabilities = [math.exp(-mean) * mean**goals / math.factorial(goals) for goals in range(max_goals)]
    probabilities.append(max(0.0, 1.0 - sum(probabilities)))
    total = sum(probabilities)
    return [probability / total for probability in probabilities]


def model_match_forecast(row: dict[str, Any], history: pd.DataFrame) -> dict[str, Any] | None:
    """Fit a simple independent Poisson goals model from pre-match league history."""
    kickoff = row["kickoff"]
    if history.empty or pd.isna(kickoff):
        return None
    league_id = pd.to_numeric(pd.Series([row.get("league_id")]), errors="coerce").iloc[0]
    if pd.isna(league_id):
        return None
    league_matches = history[
        (history["league_id"] == int(league_id))
        & (history["match_date"] < kickoff)
    ]
    if len(league_matches) < 10:
        return None
    home_mean = float(league_matches["home_score"].mean())
    away_mean = float(league_matches["away_score"].mean())
    baseline = (home_mean + away_mean) / 2
    if baseline <= 0:
        return None

    home_recent = team_recent_matches(history, row["home_team_id"], kickoff, league_id)
    away_recent = team_recent_matches(history, row["away_team_id"], kickoff, league_id)

    def strengths(recent: pd.DataFrame, team_id: Any) -> tuple[float, float]:
        if recent.empty:
            return 1.0, 1.0
        team = int(team_id)
        home_mask = recent["home_team_id"] == team
        goals_for = [int(match["home_score"] if home else match["away_score"]) for match, home in zip(recent.to_dict("records"), home_mask)]
        goals_against = [int(match["away_score"] if home else match["home_score"]) for match, home in zip(recent.to_dict("records"), home_mask)]
        shrink = len(recent) / (len(recent) + 5)
        attack = 1 + shrink * (sum(goals_for) / len(goals_for) / baseline - 1)
        defense_weakness = 1 + shrink * (sum(goals_against) / len(goals_against) / baseline - 1)
        return max(0.25, attack), max(0.25, defense_weakness)

    home_attack, home_defense = strengths(home_recent, row["home_team_id"])
    away_attack, away_defense = strengths(away_recent, row["away_team_id"])
    expected_home = home_mean * math.sqrt(home_attack * away_defense)
    expected_away = away_mean * math.sqrt(away_attack * home_defense)
    home_goals, away_goals = poisson_probabilities(expected_home), poisson_probabilities(expected_away)
    score_probabilities = [
        (home_count, away_count, home_probability * away_probability)
        for home_count, home_probability in enumerate(home_goals)
        for away_count, away_probability in enumerate(away_goals)
    ]
    total = sum(probability for _, _, probability in score_probabilities)
    home_win = sum(probability for home_count, away_count, probability in score_probabilities if home_count > away_count) / total
    draw = sum(probability for home_count, away_count, probability in score_probabilities if home_count == away_count) / total
    away_win = sum(probability for home_count, away_count, probability in score_probabilities if home_count < away_count) / total
    over_2_5 = sum(probability for home_count, away_count, probability in score_probabilities if home_count + away_count >= 3) / total
    btts = sum(probability for home_count, away_count, probability in score_probabilities if home_count > 0 and away_count > 0) / total
    most_likely = max(score_probabilities, key=lambda item: item[2])
    score_matrix = [
        [home_probability * away_probability for away_probability in away_goals]
        for home_probability in home_goals
    ]
    return {
        "home_win": home_win,
        "draw": draw,
        "away_win": away_win,
        "over_2_5": over_2_5,
        "btts": btts,
        "expected_home": expected_home,
        "expected_away": expected_away,
        "most_likely_score": f"{most_likely[0]}–{most_likely[1]}",
        "score_matrix": score_matrix,
        "history_count": len(league_matches),
        "home_sample": len(home_recent),
        "away_sample": len(away_recent),
    }


def model_corner_forecast(row: dict[str, Any], history: pd.DataFrame) -> dict[str, Any] | None:
    """Estimate full-time corners from historical corner counts in this league."""
    kickoff = row["kickoff"]
    if history.empty or pd.isna(kickoff):
        return None
    league_id = pd.to_numeric(pd.Series([row.get("league_id")]), errors="coerce").iloc[0]
    if pd.isna(league_id):
        return None
    league = history[
        (history["league_id"] == int(league_id)) & (history["match_date"] < kickoff)
    ].dropna(subset=["home_corners", "away_corners"])
    if len(league) < 20:
        return None
    home_mean, away_mean = float(league["home_corners"].mean()), float(league["away_corners"].mean())
    baseline = (home_mean + away_mean) / 2
    if baseline <= 0:
        return None

    def recent_corner_strength(team_id: Any) -> tuple[float, float, int]:
        team = int(team_id)
        recent = history[
            (history["league_id"] == int(league_id))
            & (history["match_date"] < kickoff)
            & ((history["home_team_id"] == team) | (history["away_team_id"] == team))
        ].dropna(subset=["home_corners", "away_corners"])
        recent = recent.sort_values(["match_date", "id"], ascending=False).head(5)
        if recent.empty:
            return 1.0, 1.0, 0
        for_values, against_values = [], []
        for match in recent.to_dict("records"):
            is_home = int(match["home_team_id"]) == team
            for_values.append(float(match["home_corners"] if is_home else match["away_corners"]))
            against_values.append(float(match["away_corners"] if is_home else match["home_corners"]))
        shrink = len(recent) / (len(recent) + 5)
        attack = max(0.25, 1 + shrink * (sum(for_values) / len(for_values) / baseline - 1))
        weakness = max(0.25, 1 + shrink * (sum(against_values) / len(against_values) / baseline - 1))
        return attack, weakness, len(recent)

    home_attack, home_weakness, home_n = recent_corner_strength(row["home_team_id"])
    away_attack, away_weakness, away_n = recent_corner_strength(row["away_team_id"])
    expected_home = home_mean * math.sqrt(home_attack * away_weakness)
    expected_away = away_mean * math.sqrt(away_attack * home_weakness)
    home_dist, away_dist = poisson_probabilities(expected_home), poisson_probabilities(expected_away)
    matrix = [[home_probability * away_probability for away_probability in away_dist] for home_probability in home_dist]
    return {
        "home_expected": expected_home,
        "away_expected": expected_away,
        "score_matrix": matrix,
        "history_count": len(league),
        "home_sample": home_n,
        "away_sample": away_n,
    }


def market_probability_tables(forecast: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Derive standard goal and team-goal markets from the model score matrix."""
    matrix = forecast["score_matrix"]
    total_goals = []
    for line in (0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5):
        over = sum(matrix[h][a] for h in range(len(matrix)) for a in range(len(matrix[h])) if h + a > line)
        total_goals.append({"Gol çizgisi": line, "Üst": f"%{over * 100:.1f}", "Alt": f"%{(1 - over) * 100:.1f}"})

    team_goals = []
    for line in (0.5, 1.5, 2.5, 3.5):
        home_over = sum(matrix[h][a] for h in range(len(matrix)) for a in range(len(matrix[h])) if h > line)
        away_over = sum(matrix[h][a] for h in range(len(matrix)) for a in range(len(matrix[h])) if a > line)
        team_goals.append({
            "Takım gol çizgisi": line,
            "Ev Üst": f"%{home_over * 100:.1f}",
            "Ev Alt": f"%{(1 - home_over) * 100:.1f}",
            "Dep Üst": f"%{away_over * 100:.1f}",
            "Dep Alt": f"%{(1 - away_over) * 100:.1f}",
        })

    home_win, draw, away_win = forecast["home_win"], forecast["draw"], forecast["away_win"]
    btts = forecast["btts"]
    outcomes = pd.DataFrame([
        {"Pazar": "Çifte şans 1X", "Olasılık": f"%{(home_win + draw) * 100:.1f}"},
        {"Pazar": "Çifte şans X2", "Olasılık": f"%{(draw + away_win) * 100:.1f}"},
        {"Pazar": "Çifte şans 12", "Olasılık": f"%{(home_win + away_win) * 100:.1f}"},
        {"Pazar": "Karşılıklı gol Var", "Olasılık": f"%{btts * 100:.1f}"},
        {"Pazar": "Karşılıklı gol Yok", "Olasılık": f"%{(1 - btts) * 100:.1f}"},
    ])
    return pd.DataFrame(total_goals), pd.DataFrame(team_goals), outcomes


def quote_selection_model_probability(
    market: dict[str, Any], selection_name: str, forecast: dict[str, Any] | None,
    corners: dict[str, Any] | None,
) -> float | None:
    """Match a quoted selection to an outcome computed by the statistical model."""
    if forecast is None:
        return None
    label = selection_name.casefold().strip().replace("ı", "i").replace("ş", "s").replace("ü", "u").replace("ö", "o").replace("ç", "c").replace("ğ", "g")
    market_name = str(market.get("source_name") or market.get("name") or "").casefold()
    market_name = market_name.replace("ı", "i").replace("ş", "s").replace("ü", "u").replace("ö", "o").replace("ç", "c").replace("ğ", "g")
    if any(word in market_name or word in label for word in ("kart", "card", "sari", "kirmizi", "yari", "half time", "ilk yari")):
        return None
    if any(word in market_name for word in ("korner", "corner")):
        matrix = corners["score_matrix"] if corners else None
    else:
        matrix = forecast["score_matrix"]
    if matrix is None:
        return None

    line_value = market.get("line")
    if line_value in (None, "", 0, "0"):
        match = re.search(r"\d+(?:[.,]\d+)?", str(market.get("name") or ""))
        line_value = match.group(0) if match else None
    try:
        line = float(str(line_value).replace(",", ".")) if line_value is not None else None
    except (TypeError, ValueError):
        line = None

    if any(word in market_name for word in ("karsilikli", "both teams", "kg")):
        probability = forecast["btts"]
        return probability if any(word in label for word in ("var", "evet", "yes")) else 1 - probability if any(word in label for word in ("yok", "hayir", "no")) else None
    if any(word in label.replace(" ", "") for word in ("1x", "x2", "12")):
        code = label.replace(" ", "")
        return {"1x": forecast["home_win"] + forecast["draw"], "x2": forecast["draw"] + forecast["away_win"], "12": forecast["home_win"] + forecast["away_win"]}.get(code)
    outcomes = {
        str(item.get("name", "")).casefold().strip().replace("ı", "i").replace("ş", "s").replace("ü", "u").replace("ö", "o").replace("ç", "c").replace("ğ", "g")
        for item in market.get("selections", [])
    }
    if outcomes.issubset({"1", "x", "2"}) and len(outcomes) >= 2 and label in {"1", "x", "2"}:
        return {"1": forecast["home_win"], "x": forecast["draw"], "2": forecast["away_win"]}[label]

    selection_line = re.search(r"(\d+)\s*[-–:]\s*(\d+)", label)
    if selection_line:
        home_score, away_score = map(int, selection_line.groups())
        return matrix[home_score][away_score] if home_score < len(matrix) and away_score < len(matrix[home_score]) else 0.0

    is_over = any(word in label for word in ("ust", "over"))
    is_under = any(word in label for word in ("alt", "under"))
    if line is not None and (is_over or is_under):
        is_corner_market = any(word in market_name for word in ("korner", "corner"))
        team_side = "home" if any(word in market_name for word in ("ev sahibi", "home team", "ev gol")) else "away" if any(word in market_name for word in ("deplasman", "away team", "dep gol")) else None
        has_goal_context = any(word in market_name for word in ("gol", "goal", "alt/ust", "over/under", "toplam"))
        if not is_corner_market and team_side is None and not has_goal_context:
            return None
        over = sum(
            matrix[h][a]
            for h in range(len(matrix)) for a in range(len(matrix[h]))
            if (h if team_side == "home" else a if team_side == "away" else h + a) > line
        )
        probability = over if is_over else 1 - over
        return probability
    if line is None and (label in {"evet", "hayir", "yes", "no"}):
        return None
    return None


def render_model_forecast(row: dict[str, Any], history: pd.DataFrame) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    forecast = model_match_forecast(row, history)
    st.markdown("#### 🤖 Model tahmini · Form Poisson")
    if forecast is None:
        st.info("Bu ligde model tahmini üretmek için yeterli maç geçmişi yok.")
        return None, None
    top, details = st.columns([1, 3])
    with top:
        st.metric("En olası skor", forecast["most_likely_score"])
    with details:
        st.markdown("**1X2 olasılıkları**")
        cols = st.columns(3)
        for col, label, key in zip(cols, ("1 · Ev sahibi", "X · Beraberlik", "2 · Deplasman"), ("home_win", "draw", "away_win")):
            col.metric(label, f"%{forecast[key] * 100:.1f}")
        more = st.columns(2)
        more[0].metric("2.5 Üst", f"%{forecast['over_2_5'] * 100:.1f}")
        more[1].metric("Karşılıklı gol", f"%{forecast['btts'] * 100:.1f}")
    st.caption(
        f"Bağımsız Poisson modeli · lig ortalaması ve takımların aynı ligdeki son maçları. "
        f"Lig örneklemi: {forecast['history_count']} maç; takım örneklemi: "
        f"{forecast['home_sample']} / {forecast['away_sample']} maç. "
        f"Beklenen gol: {forecast['expected_home']:.2f}–{forecast['expected_away']:.2f}. "
        "Oranlardan türetilmemiştir; olasılıklar istatistiksel tahmindir."
    )
    total_table, team_table, outcome_table = market_probability_tables(forecast)
    corners = model_corner_forecast(row, history)
    with st.expander("Tüm hesaplanabilir model marketleri"):
        odds_tab, team_tab, corner_tab, score_tab, data_tab = st.tabs(["Gol & sonuç", "Takım golleri", "Korner", "Doğru skor", "Veri kapsamı"])
        with odds_tab:
            st.dataframe(total_table, use_container_width=True, hide_index=True)
            st.dataframe(outcome_table, use_container_width=True, hide_index=True)
        with team_tab:
            st.dataframe(team_table, use_container_width=True, hide_index=True)
        with corner_tab:
            if corners is None:
                st.info("Bu ligde en az 20 tamamlanmış maçın korner istatistiği bulununca korner tahmini açılır.")
            else:
                corner_lines = []
                for line in (6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5):
                    over = sum(corners["score_matrix"][h][a] for h in range(len(corners["score_matrix"])) for a in range(len(corners["score_matrix"][h])) if h + a > line)
                    corner_lines.append({"Korner çizgisi": line, "Üst": f"%{over * 100:.1f}", "Alt": f"%{(1 - over) * 100:.1f}"})
                st.dataframe(pd.DataFrame(corner_lines), use_container_width=True, hide_index=True)
                team_corner_lines = []
                for line in (2.5, 3.5, 4.5, 5.5, 6.5):
                    home_over = sum(corners["score_matrix"][h][a] for h in range(len(corners["score_matrix"])) for a in range(len(corners["score_matrix"][h])) if h > line)
                    away_over = sum(corners["score_matrix"][h][a] for h in range(len(corners["score_matrix"])) for a in range(len(corners["score_matrix"][h])) if a > line)
                    team_corner_lines.append({"Takım korner çizgisi": line, "Ev Üst": f"%{home_over * 100:.1f}", "Ev Alt": f"%{(1 - home_over) * 100:.1f}", "Dep Üst": f"%{away_over * 100:.1f}", "Dep Alt": f"%{(1 - away_over) * 100:.1f}"})
                st.dataframe(pd.DataFrame(team_corner_lines), use_container_width=True, hide_index=True)
                st.caption(
                    f"Bağımsız korner Poisson modeli · {corners['history_count']} lig maçı; "
                    f"takımlar için korner örneklemi {corners['home_sample']} / {corners['away_sample']}. "
                    f"Beklenen korner: {corners['home_expected']:.2f}–{corners['away_expected']:.2f}."
                )
        with score_tab:
            matrix = forecast["score_matrix"]
            scores = sorted(
                ((matrix[h][a], h, a) for h in range(len(matrix)) for a in range(len(matrix[h]))),
                reverse=True,
            )[:5]
            st.dataframe(
                pd.DataFrame([{"Skor": f"{home}–{away}", "Olasılık": f"%{probability * 100:.1f}"} for probability, home, away in scores]),
                use_container_width=True,
                hide_index=True,
            )
        with data_tab:
            st.info("Gol, sonuç, takım golü ve doğru skor olasılıkları geçmiş tam maç skorlarından hesaplanır. Devre sonuçları ile sarı/kırmızı kart verileri arşivde tutulmadığı için bu pazarlara tahmin üretilemez.")
            if corners is None:
                st.info("Korner tahmini için bu ligde en az 20 maçın tamamlanmış korner istatistiği gerekir.")
            else:
                st.success(f"Korner modeli kullanılabilir: maç öncesi {corners['history_count']} lig korner kaydı.")
    return forecast, corners


def render_recent_form(
    row: dict[str, Any], history: pd.DataFrame, team_names: dict[Any, str]
) -> None:
    """Show recent form and match logs for both teams without implying a forecast."""
    kickoff = row["kickoff"]
    home_id, away_id = row["home_team_id"], row["away_team_id"]
    home_recent = team_recent_matches(history, home_id, kickoff, row["league_id"])
    away_recent = team_recent_matches(history, away_id, kickoff, row["league_id"])
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
        for column in ("id", "league_id", "home_team_id", "away_team_id", "home_score", "away_score", "home_corners", "away_corners"):
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
            model_forecast, corner_forecast = render_model_forecast(row, history)
            st.divider()
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
                            st.metric("Oran", f"{selection['odds']:.2f}")
                            model_probability = quote_selection_model_probability(
                                market, selection["name"], model_forecast, corner_forecast
                            )
                            model_text = f"Model %{model_probability * 100:.1f}" if model_probability is not None else "Model tahmini yok"
                            st.caption(f"Piyasa %{selection['probability'] * 100:.1f} · {model_text}")
                            st.progress(min(1.0, max(0.0, model_probability if model_probability is not None else selection["probability"])))
                    st.caption(f"Ham market marjı: %{market['overround'] * 100:.1f}")
            st.caption("Model olasılığı, etiketi ve tarihsel sonucu modele uygun market seçimlerinde gösterilir. Desteklenmeyen marketlerde oran yanında ‘Model tahmini yok’ görünür.")


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
        try:
            analysis_records = load_completed_match_history()
        except Exception as exc:
            analysis_records = []
            st.warning(f"Geçmiş maçlar yüklenemedi; model tahmini ve form analizi gösterilemiyor: {exc}")
        try:
            market_labels = load_iddaa_market_labels()
        except Exception:
            market_labels = {}
        render_analysis(match_records, quote_records, analysis_records, teams, leagues, selected_day, market_labels)
    except Exception as exc:
        st.error(f"Maç ve oran verileri yüklenemedi: {exc}")
with tab_archive:
    render_archive()
