"""Streamlit match center and completed-match statistics archive."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.odds_quality import total_market_key
from app.archive_data import count_completed_match_history, load_completed_match_history
from app.archive import render_archive
from app.forecast import (
    market_probability_tables,
    model_corner_forecast,
    model_half_forecast,
    model_match_forecast,
    summarize_team_form,
    team_recent_matches,
)
from app.performance import load_evaluated_prediction_results, render_performance
from app.localization import COUNTRY_NAMES_TR, LEAGUE_NAMES_TR
from app.market_mapping import is_compound_selection, standard_market_price, total_goals_outcome
from app.ai_match_analysis import (
    confirmed_lineup_context,
    previous_lineup_context,
    previous_match_ids_for_fixture,
)
from app.components.data import (
    load_league_match_history,
    load_match_statistics_check_times,
    load_match_day,
    load_iddaa_market_labels,
    load_recent_team_matches,
    load_fixture_lineups_for_matches,
    load_previous_team_match_history,
    load_reference_catalog,
    submit_ai_analysis_feedback,
)

ISTANBUL = ZoneInfo("Europe/Istanbul")
# Bump this when forecast logic or the persisted analysis JSON contract changes.
ANALYSIS_MODEL_VERSION = "form_poisson_ai_v20"
NON_PLAYABLE_MATCH_STATUSES = {"postponed", "cancelled", "unlisted"}

if __name__ == "__main__":
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
                odd = standard_market_price(selection)
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
        configured_name = (
            labels.get(market_key)
            or labels.get(str(market_code if market_code is not None else ""))
        )
        # The event feed stores a generic market_name for team-goal codes.
        # Preserve the code meaning across the sport-type variants in stored quotes.
        if str(sport_type) in {"2", "4"}:
            team_goal_name = (
                {(2, 603): "Ev Sahibi Gol Alt/Üst", (2, 604): "Deplasman Gol Alt/Üst",
                 (4, 203): "Ev Sahibi Gol Alt/Üst", (4, 207): "Deplasman Gol Alt/Üst"}
                .get((int(sport_type), int(market_code))) if market_code is not None else None
            )
            if team_goal_name and str(configured_name or "").casefold() in {"", "maç sonucu alt/üst", "mac sonucu alt/ust"}:
                configured_name = team_goal_name
        name = str(
            configured_name
            or market.get("market_name")
            or _infer_legacy_market_name(selections)
            or (f"Market {market_code}" if market_code is not None else "Diğer market")
        )
        if line not in (None, "", 0, "0"):
            name = f"{name} · {line}"
        rows.append({
            "name": name,
            "source_name": str(configured_name or market.get("market_name") or name),
            "line": line,
            "market_id": market_code,
            "sport_type": sport_type,
            "selections": [
                {"name": item[0], "odds": item[1], "probability": prob / total}
                for item, prob in zip(selections, implied)
            ],
            "overround": total - 1,
        })
    return rows


def all_market_quote_rows(
    payload: dict[str, Any],
    labels: dict[str, str],
    inconsistent_keys: set[tuple[str, str, str]] | None = None,
) -> list[dict[str, Any]]:
    """Build a complete IdDaa odds table, including markets outside model coverage."""
    rows: list[dict[str, Any]] = []
    for market in payload.get("markets", []):
        code = market.get("market_id")
        sport_type = market.get("sport_type")
        key = f"{sport_type if sport_type is not None else ''}:{code if code is not None else ''}"
        name = str(labels.get(key) or labels.get(str(code or "")) or market.get("market_name") or f"Market {code or ''}")
        if market.get("line") not in (None, "", 0, "0"):
            name = f"{name} · {market['line']}"
        selections = []
        for selection in market.get("selections", []):
            try:
                odd = standard_market_price(selection)
            except (TypeError, ValueError):
                continue
            if odd > 1:
                selections.append((str(selection.get("name") or selection.get("id") or "Seçim"), odd))
        implied = [1 / odd for _, odd in selections]
        total = sum(implied)
        for (selection_name, odd), probability in zip(selections, implied):
            row = {
                "Market": name,
                "Seçim": selection_name,
                "Oran": f"{odd:.2f}",
                "Piyasa %": f"%{probability / total * 100:.1f}" if len(selections) > 1 and total else "—",
            }
            if total_market_key(market) in (inconsistent_keys or set()):
                row["Veri kontrolü"] = "Çizgilerle tutarsız; doğrulanana kadar kullanmayın"
            rows.append(row)
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


























def quote_selection_model_probability(
    market: dict[str, Any], selection_name: str, forecast: dict[str, Any] | None,
    corners: dict[str, Any] | None,
) -> float | None:
    """Match a quoted selection to an outcome computed by the statistical model."""
    if forecast is None:
        return None
    label = selection_name.casefold().strip().replace("ı", "i").replace("ş", "s").replace("ü", "u").replace("ö", "o").replace("ç", "c").replace("ğ", "g")
    if label == "0":
        label = "x"
    if label.replace(" ", "") in {"1ve0", "0ve1"}:
        label = "1x"
    elif label.replace(" ", "") in {"0ve2", "2ve0"}:
        label = "x2"
    elif label.replace(" ", "") in {"1ve2", "2ve1"}:
        label = "12"
    market_name = str(market.get("source_name") or market.get("name") or "").casefold()
    market_name = market_name.replace("ı", "i").replace("ş", "s").replace("ü", "u").replace("ö", "o").replace("ç", "c").replace("ğ", "g")
    market_context = f"{market_name} {label}"
    if any(word in market_name or word in label for word in ("kart", "card", "sari", "kirmizi", "yari", "half time", "ilk yari")):
        return None
    if any(word in market_context for word in ("korner", "corner")):
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
        ("x" if str(item.get("name", "")).casefold().strip() == "0" else str(item.get("name", "")).casefold().strip().replace("ı", "i").replace("ş", "s").replace("ü", "u").replace("ö", "o").replace("ç", "c").replace("ğ", "g"))
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
        team_side = "home" if any(word in market_context for word in ("ev sahibi", "home team", "ev gol", "ev korner", "home corner")) else "away" if any(word in market_context for word in ("deplasman", "away team", "dep gol", "dep korner", "away corner")) else None
        has_goal_context = any(word in market_context for word in ("gol", "goal", "alt/ust", "over/under", "toplam", "takim", "team"))
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


def render_model_forecast(
    row: dict[str, Any],
    history: pd.DataFrame,
    markets: list[dict[str, Any]],
    *,
    precomputed: dict[str, Any] | None = None,
    show_details: bool = True,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    forecast = (
        precomputed.get("forecast")
        if precomputed is not None
        else model_match_forecast(row, history)
    )
    if forecast is None:
        st.info("Bu ligde model tahmini üretmek için yeterli maç geçmişi yok.")
        return None, None
    corners = (
        precomputed.get("corners")
        if precomputed is not None
        else model_corner_forecast(row, history)
    )
    if not show_details:
        return forecast, corners
    # Index quoted selections by market category and line for inline display
    # within the corresponding model tables below.
    quote_index: dict[tuple[str, str, str], dict[str, Any]] = {}
    half_quote_index: dict[tuple[int, str, str, str], dict[str, Any]] = {}
    for market in markets:
        for selection in market["selections"]:
            probability = quote_selection_model_probability(
                market, selection["name"], forecast, corners
            )
            market_name = str(market.get("source_name") or market.get("name") or "").casefold()
            label = selection["name"].casefold().strip()
            for source, target in (("ı", "i"), ("ş", "s"), ("ü", "u"), ("ö", "o"), ("ç", "c"), ("ğ", "g")):
                market_name = market_name.replace(source, target)
                label = label.replace(source, target)
            compact_label = label.replace(" ", "")
            label = {"0": "x", "1ve0": "1x", "0ve1": "1x", "0ve2": "x2", "2ve0": "x2", "1ve2": "12", "2ve1": "12"}.get(compact_label, label)
            if is_compound_selection(label):
                # A combined outcome such as "Üst ve Yok" is not the pure
                # "Üst" selection and must never overwrite its quote.
                continue
            line_value = market.get("line")
            if line_value in (None, "", 0, "0"):
                match = re.search(r"\d+(?:[.,]\d+)?", f"{market.get('name') or ''} {label}")
                line_value = match.group(0) if match else ""
            line_key = str(line_value).replace(",", ".")
            market_context = f"{market_name} {label}"
            sport_type = str(market.get("sport_type") or "")
            market_id = str(market.get("market_id") or "")
            second_half = any(term in market_context for term in ("2. yari", "2 yari", "ikinci yari", "second half")) or (sport_type == "4" and market_id == "321")
            first_half = any(term in market_context for term in ("1. yari", "1 yari", "ilk yari", "first half", "yari"))
            half_number = 2 if second_half else 1 if first_half or (sport_type == "2" and market_id in {"36", "60", "722", "723"}) else None
            half_category = (
                "result" if market_id in {"36", "321"} or (half_number == 2 and "sonuc" in market_context)
                else "total" if market_id == "60" or (half_number == 2 and any(term in market_context for term in ("alt/ust", "over/under", "toplam")))
                else "home_goals" if market_id == "722"
                else "away_goals" if market_id == "723"
                else None
            )
            if half_number and half_category:
                if half_category == "result":
                    half_outcome = (
                        "1" if label in {"1", "1.", "ev sahibi", "ev sahibi kazanir", "home"}
                        else "x" if label in {"0", "x", "eşit", "esit", "beraberlik", "draw"}
                        else "2" if label in {"2", "2.", "deplasman", "deplasman kazanir", "away"}
                        else label
                    )
                    line_key = ""
                else:
                    half_outcome = "x" if label == "0" else label
                if half_category != "result" and label in {"ust", "over"}:
                    half_outcome = "over"
                elif half_category != "result" and label in {"alt", "under"}:
                    half_outcome = "under"
                half_quote_index[(half_number, half_category, line_key, half_outcome)] = {
                    "odds": selection["odds"],
                    "market_probability": selection["probability"],
                    "market": market["name"],
                }
                continue
            if (sport_type, market_id) in {("4", "203"), ("2", "603")}:
                category = "home_goals"
            elif (sport_type, market_id) in {("4", "207"), ("2", "604")}:
                category = "away_goals"
            elif any(term in market_context for term in ("korner", "corner")):
                category = "home_corners" if any(term in market_context for term in ("ev sahibi", "ev korner", "home team", "home corner")) else "away_corners" if any(term in market_context for term in ("deplasman", "dep korner", "away team", "away corner")) else "corners"
            elif any(term in market_context for term in ("ev sahibi", "home team", "ev gol")):
                category = "home_goals"
            elif any(term in market_context for term in ("deplasman", "away team", "dep gol")):
                category = "away_goals"
            elif any(term in market_context for term in ("karsilikli", "both teams", "kg")):
                category = "btts"
            elif re.search(r"\d+\s*[-–:]\s*\d+", label):
                category, line_key = "score", ""
                outcome = re.sub(r"\s*[-–:]\s*", "-", label)
                quote_index[(category, line_key, outcome)] = {
                    "odds": selection["odds"],
                    "market_probability": selection["probability"],
                    "model_probability": probability,
                    "market": market["name"],
                }
                continue
            elif any(term in label.replace(" ", "") for term in ("1x", "x2", "12")):
                category, line_key = "double_chance", label.replace(" ", "")
            elif label in {"1", "x", "2"}:
                category, line_key = "result", label
            elif (
                any(term in market_context for term in ("gol", "goal", "alt/ust", "over/under", "toplam"))
                and total_goals_outcome(label) is not None
            ):
                category = "total_goals"
            else:
                continue
            outcome = (
                total_goals_outcome(label)
                or ("yes" if label in {"var", "evet", "yes"}
                else "no" if label in {"yok", "hayir", "no"}
                else label)
            )
            quote_index[(category, line_key, outcome)] = {
                "odds": selection["odds"],
                "market_probability": selection["probability"],
                "model_probability": probability,
                "market": market["name"],
            }

    def quote_columns(category: str, line: float | str, outcome: str) -> dict[str, str]:
        quote = quote_index.get((category, str(line), outcome))
        if not quote:
            return {"Oran": "—", "Piyasa %": "—"}
        return {
            "Oran": f"{quote['odds']:.2f}",
            "Piyasa %": f"%{quote['market_probability'] * 100:.1f}",
        }

    total_table, team_table, outcome_table = market_probability_tables(forecast)
    with st.expander("🤖 Tüm maç tahminleri ve karşılık gelen oranlar", expanded=True):
        st.markdown("#### Model tahmini · Form Poisson")
        st.metric("En olası skor", forecast["most_likely_score"])
        if forecast["tuning_matches"] and forecast["tuning_log_loss"] is not None:
            tuning_note = (
                f"Form ağırlığı {forecast['tuning_selection_matches']} eski maçta seçildi; "
                f"sonraki {forecast['tuning_matches']} maçta 1X2 log-loss "
                f"{forecast['tuning_log_loss']:.3f} (sabit ağırlık: "
                f"{forecast['tuning_baseline_log_loss']:.3f}; düşük değer daha iyi). "
                f"Uygulanan önsel: {forecast['recent_form_prior']:g}. "
            )
        else:
            tuning_note = "Ayrı değerlendirme için yeterli geçmiş maç bulunmadığından varsayılan form ağırlığı kullanılıyor. "
        st.caption(
            f"Bağımsız Poisson modeli · lig ortalaması ve takımların aynı ligdeki son maçları. "
            f"Lig örneklemi: {forecast['history_count']} maç; takım örneklemi: "
            f"{forecast['home_sample']} / {forecast['away_sample']} maç. "
            f"Beklenen gol: {forecast['expected_home']:.2f}–{forecast['expected_away']:.2f}. "
            f"{tuning_note}"
            "Model maç oranlarından türetilmez ve gol sayılarını bağımsız Poisson değişkenleri varsayar. "
            "Kadro, sakatlık, kart ve maç içi olaylar bu hesaplamaya dahil değildir; olasılıklar istatistiksel tahmindir."
        )
        if forecast["tuning_market_metrics"]:
            market_labels = {
                "result": "Maç sonucu (1X2)",
                "over_2_5": "Toplam gol 2.5 üst",
                "btts": "Karşılıklı gol var",
            }
            evaluation_rows = []
            for market, metrics in forecast["tuning_market_metrics"].items():
                evaluation_rows.append({
                    "Market": market_labels[market],
                    "Örnek": metrics["sample_size"],
                    "Model log-loss": metrics["model_log_loss"],
                    "Sabit log-loss": metrics["baseline_log_loss"],
                    "Model Brier": metrics["model_brier_score"],
                    "Sabit Brier": metrics["baseline_brier_score"],
                    "Model kalibrasyon": metrics["model_calibration_error"],
                    "Sabit kalibrasyon": metrics["baseline_calibration_error"],
                })
            with st.expander("Geçmiş maçlarda zaman sıralı model değerlendirmesi"):
                st.caption(
                    "Ağırlıklar erken dönemden seçildi; aşağıdaki log-loss daha sonraki, "
                    "seçimde kullanılmayan maçlarda hesaplandı. Brier ve 10 dilimli kalibrasyon "
                    "hatası da gösterilir (düşük değer daha iyi). Lig bazlıdır; küçük örneklem "
                    "tek başına modelin gelecekte daha başarılı olacağını göstermez."
                )
                st.dataframe(
                    pd.DataFrame(evaluation_rows),
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "Model log-loss": st.column_config.NumberColumn(format="%.3f"),
                        "Sabit log-loss": st.column_config.NumberColumn(format="%.3f"),
                        "Model Brier": st.column_config.NumberColumn(format="%.3f"),
                        "Sabit Brier": st.column_config.NumberColumn(format="%.3f"),
                        "Model kalibrasyon": st.column_config.NumberColumn(format="%.3f"),
                        "Sabit kalibrasyon": st.column_config.NumberColumn(format="%.3f"),
                    },
                )
        st.markdown("#### Tahminler ve karşılık gelen oranlar")
        st.markdown("**Tüm tahminler**")
        total_rows = []
        for line in (0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5):
            over = sum(forecast["score_matrix"][h][a] for h in range(len(forecast["score_matrix"])) for a in range(len(forecast["score_matrix"][h])) if h + a > line)
            for outcome, probability in (("over", over), ("under", 1 - over)):
                total_rows.append({"Pazar": f"Toplam Gol Alt/Üst · {line}", "Seçim": "Üst" if outcome == "over" else "Alt", "Model tahmini": probability, **quote_columns("total_goals", line, outcome)})
        for key, label in (("1", "1 · Ev sahibi"), ("x", "X · Beraberlik"), ("2", "2 · Deplasman")):
            probability = {"1": forecast["home_win"], "x": forecast["draw"], "2": forecast["away_win"]}[key]
            total_rows.append({"Pazar": "Maç Sonucu", "Seçim": label, "Model tahmini": probability, **quote_columns("result", key, key)})
        for key, probability in (("1x", forecast["home_win"] + forecast["draw"]), ("x2", forecast["draw"] + forecast["away_win"]), ("12", forecast["home_win"] + forecast["away_win"])):
            total_rows.append({"Pazar": "Çifte Şans", "Seçim": key.upper(), "Model tahmini": probability, **quote_columns("double_chance", key, key)})
        for outcome, probability in (("yes", forecast["btts"]), ("no", 1 - forecast["btts"])):
            total_rows.append({"Pazar": "Karşılıklı Gol", "Seçim": "Var" if outcome == "yes" else "Yok", "Model tahmini": probability, **quote_columns("btts", "", outcome)})
        def render_half_table(half: int, half_label: str) -> None:
            half_forecast = (
                precomputed.get("halves", {}).get(str(half))
                if precomputed is not None
                else model_half_forecast(row, history, half=half)
            )
            if half_forecast is None:
                st.caption(f"{half_label}: en az 20 geçmiş devre skoru olmadığı için tahmin üretilemedi.")
                return
            st.caption(
                f"Yarıya özel Poisson modeli · {half_forecast['history_count']} lig maçı; "
                f"beklenen gol {half_forecast['expected_home']:.2f}–{half_forecast['expected_away']:.2f}."
            )
            def add_row(category: str, line: str, outcome: str, label: str, probability: float) -> None:
                quote = half_quote_index.get((half, category, line, outcome))
                total_rows.append({
                    "Pazar": f"{half_label} · {label}",
                    "Seçim": "Üst" if outcome == "over" else "Alt" if outcome == "under" else {"1": "1 · Ev sahibi", "x": "X · Beraberlik", "2": "2 · Deplasman"}.get(outcome, outcome),
                    "Model tahmini": probability,
                    "Oran": f"{quote['odds']:.2f}" if quote else "—",
                    "Piyasa %": f"%{quote['market_probability'] * 100:.1f}" if quote else "—",
                })

            matrix = half_forecast["score_matrix"]
            for key, label, probability in (
                ("1", "Yarı sonucu", half_forecast["home_win"]),
                ("x", "Yarı sonucu", half_forecast["draw"]),
                ("2", "Yarı sonucu", half_forecast["away_win"]),
            ):
                add_row("result", "", key, label, probability)
            for line in (0.5, 1.5, 2.5):
                over = sum(matrix[h][a] for h in range(len(matrix)) for a in range(len(matrix[h])) if h + a > line)
                for outcome, probability in (("over", over), ("under", 1 - over)):
                    add_row("total", str(line), outcome, f"Yarı toplam gol · {line}", probability)
            for line in (0.5, 1.5):
                for side, category, prefix in (("home", "home_goals", "Ev"), ("away", "away_goals", "Dep")):
                    over = sum(matrix[h][a] for h in range(len(matrix)) for a in range(len(matrix[h])) if (h if side == "home" else a) > line)
                    for outcome, probability in (("over", over), ("under", 1 - over)):
                        add_row(category, str(line), outcome, f"Takım golü · {line} · {prefix}", probability)

        render_half_table(1, "1. Yarı")
        render_half_table(2, "2. Yarı")

        matrix = forecast["score_matrix"]
        for line in (0.5, 1.5, 2.5, 3.5):
            for side, prefix in (("home", "Ev"), ("away", "Dep")):
                over = sum(matrix[h][a] for h in range(len(matrix)) for a in range(len(matrix[h])) if (h if side == "home" else a) > line)
                category = "home_goals" if side == "home" else "away_goals"
                for outcome, probability in (("over", over), ("under", 1 - over)):
                    total_rows.append({"Pazar": f"Takım golü · {line} · {prefix}", "Seçim": "Üst" if outcome == "over" else "Alt", "Model tahmini": probability, **quote_columns(category, line, outcome)})

        if corners is None:
            st.caption("Korner tahmini için bu ligde yeterli tamamlanmış korner istatistiği bulunmuyor.")
            for (category, line, outcome), quote in quote_index.items():
                if category not in {"corners", "home_corners", "away_corners"}:
                    continue
                total_rows.append({
                    "Pazar": f"{quote['market']} · {line}",
                    "Seçim": "Üst" if outcome == "over" else "Alt" if outcome == "under" else outcome,
                    "Model tahmini": None, "Oran": f"{quote['odds']:.2f}",
                    "Piyasa %": f"%{quote['market_probability'] * 100:.1f}",
                })
        else:
            for line in (6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5):
                over = sum(corners["score_matrix"][h][a] for h in range(len(corners["score_matrix"])) for a in range(len(corners["score_matrix"][h])) if h + a > line)
                for outcome, probability in (("over", over), ("under", 1 - over)):
                    total_rows.append({"Pazar": f"Toplam korner · {line}", "Seçim": "Üst" if outcome == "over" else "Alt", "Model tahmini": probability, **quote_columns("corners", line, outcome)})
            for line in (2.5, 3.5, 4.5, 5.5, 6.5):
                for side, prefix in (("home", "Ev"), ("away", "Dep")):
                    over = sum(corners["score_matrix"][h][a] for h in range(len(corners["score_matrix"])) for a in range(len(corners["score_matrix"][h])) if (h if side == "home" else a) > line)
                    for outcome, probability in (("over", over), ("under", 1 - over)):
                        total_rows.append({"Pazar": f"Takım korneri · {line} · {prefix}", "Seçim": "Üst" if outcome == "over" else "Alt", "Model tahmini": probability, **quote_columns(f"{side}_corners", line, outcome)})
            st.caption(f"Korner Poisson modeli · {corners['history_count']} lig maçı; beklenen korner: {corners['home_expected']:.2f}–{corners['away_expected']:.2f}.")

        matrix = forecast["score_matrix"]
        scores = sorted(((matrix[h][a], h, a) for h in range(len(matrix)) for a in range(len(matrix[h]))), reverse=True)[:5]
        for probability, home, away in scores:
            score_key = f"{home}-{away}"
            score_quote = quote_index.get(("score", "", score_key), {})
            total_rows.append({"Pazar": "Doğru skor", "Seçim": score_key.replace("-", "–"), "Model tahmini": probability, "Oran": f"{score_quote['odds']:.2f}" if score_quote else "—", "Piyasa %": f"%{score_quote['market_probability'] * 100:.1f}" if score_quote else "—"})

        st.dataframe(
            pd.DataFrame(total_rows),
            width="stretch",
            hide_index=True,
            column_config={
                "Model tahmini": st.column_config.NumberColumn(
                    "Model tahmini",
                    format="percent",
                )
            },
        )

        st.markdown("**Veri kapsamı**")
        st.info("Maç sonu ve yarı tahminleri geçmiş skor dağılımlarından hesaplanır. Yarı modelleri, ilgili ligde en az 20 devre skoru kaydı oluşunca devreye girer; kart istatistikleri arşivde bulunmadığı için kart tahmini üretilmez.")
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
                st.dataframe(pd.DataFrame(log), width="stretch", hide_index=True)


def build_precomputed_match_analysis(
    row: dict[str, Any],
    history: pd.DataFrame,
    recent_form_tuning: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build display-ready match analysis without Streamlit rendering calls."""
    kickoff = row["kickoff"]
    league_id = int(row["league_id"])
    home_recent = team_recent_matches(history, row["home_team_id"], kickoff, league_id)
    away_recent = team_recent_matches(history, row["away_team_id"], kickoff, league_id)
    history_by_id = {
        int(match["id"]): match
        for match in (*home_recent.to_dict("records"), *away_recent.to_dict("records"))
    }
    return {
        "history": list(history_by_id.values()),
        "forecast": model_match_forecast(row, history, recent_form_tuning),
        "corners": model_corner_forecast(row, history),
        "halves": {
            str(half): model_half_forecast(row, history, half=half)
            for half in (1, 2)
        },
    }


def _cached_analysis_for_match(
    record: dict[str, Any] | None, row: dict[str, Any]
) -> dict[str, Any] | None:
    if not record or record.get("model_version") != ANALYSIS_MODEL_VERSION:
        return None
    try:
        if any(
            int(record.get(key, -1)) != int(row[key])
            for key in ("league_id", "home_team_id", "away_team_id")
        ):
            return None
        cached_date = pd.to_datetime(record.get("match_date"), utc=True, errors="coerce")
        fixture_date = pd.to_datetime(row.get("match_date"), utc=True, errors="coerce")
        if pd.isna(cached_date) or pd.isna(fixture_date) or cached_date != fixture_date:
            return None
    except (KeyError, TypeError, ValueError):
        return None
    analysis = record.get("analysis_json")
    if (
        not isinstance(analysis, dict)
        or not isinstance(analysis.get("history"), list)
        or not isinstance(analysis.get("halves"), dict)
    ):
        return None
    return analysis


def render_tactical_match_readout(
    row: dict[str, Any], history: pd.DataFrame, forecast: dict[str, Any] | None,
    precomputed_risks: dict[str, Any] | None = None,
    ai_risks_stale: bool = False,
) -> None:
    """Summarize each fixture from its pre-match form and independent goal model."""
    st.markdown("#### Taktik eşleşme ve senaryolar")
    if forecast is None:
        st.info("Bu eşleşme için yeterli maç geçmişi bulunmadığından veriye dayalı senaryo üretilemedi.")
        return

    home_id, away_id = row["home_team_id"], row["away_team_id"]
    home_recent = team_recent_matches(history, home_id, row["kickoff"], row["league_id"])
    away_recent = team_recent_matches(history, away_id, row["kickoff"], row["league_id"])
    home_form = summarize_team_form(home_recent, home_id)
    away_form = summarize_team_form(away_recent, away_id)
    home_xg, away_xg = forecast["expected_home"], forecast["expected_away"]
    home_p, draw_p, away_p = forecast["home_win"], forecast["draw"], forecast["away_win"]
    btts_no = 1 - forecast["btts"]

    home_name, away_name = row["home_name"], row["away_name"]
    total_xg = home_xg + away_xg
    st.markdown("**Takımların son maçları**")
    form_rows = []
    for name, team_form, recent in (
        (home_name, home_form, home_recent),
        (away_name, away_form, away_recent),
    ):
        form_rows.append({
            "Takım": name,
            "Sonuçlar (yeniden eskiye)": team_form["form"],
            "Maç sayısı": len(recent),
            "Puan / maç": f"{team_form['points']:.2f}" if team_form["points"] is not None else "Veri yok",
            "Gol attı / yedi / maç": (
                f"{team_form['goals_for']:.1f} / {team_form['goals_against']:.1f}"
                if team_form["goals_for"] is not None else "Veri yok"
            ),
            "xG / maç": f"{team_form['xg_for']:.2f}" if team_form["xg_for"] is not None else "Veri yok",
        })
    st.dataframe(pd.DataFrame(form_rows), width="stretch", hide_index=True)
    st.caption("G: galibiyet · B: beraberlik · M: mağlubiyet. xG, geçmiş maçlardaki gol fırsatlarının kalitesinden hesaplanan beklenen gol değeridir.")

    st.markdown("**Modelin maç tahmini**")
    home_metric, draw_metric, away_metric, goals_metric, btts_metric = st.columns(5)
    home_metric.metric(f"{home_name} kazanır", f"{home_p:.0%}")
    draw_metric.metric("Beraberlik", f"{draw_p:.0%}")
    away_metric.metric(f"{away_name} kazanır", f"{away_p:.0%}")
    goals_metric.metric("Toplam gol beklentisi", f"{total_xg:.2f}")
    btts_metric.metric("İki takım da gol atar", f"{forecast['btts']:.0%}")
    st.caption(
        f"Takım başına beklenen gol: {home_name} {home_xg:.2f} · {away_name} {away_xg:.2f}. "
        f"En olası tek skor {forecast['most_likely_score']}; bu skorun kesin veya yüksek olasılıklı olduğu anlamına gelmez."
    )
    strongest_outcome, strongest_probability = max(
        ((home_name, home_p), ("Beraberlik", draw_p), (away_name, away_p)),
        key=lambda item: item[1],
    )
    st.info(
        f"Kısa özet: Modelde en yüksek maç sonucu olasılığı {strongest_outcome} için "
        f"{strongest_probability:.0%}. Bu, sonucun garanti olduğu anlamına gelmez."
    )
    with st.expander("Bu tahminleri nasıl okumalıyım?"):
        st.markdown(
            "- **Maç sonucu yüzdeleri** ev sahibi galibiyeti, beraberlik ve deplasman galibiyetinin model olasılıklarıdır; toplamları yaklaşık %100 eder.\n"
            "- **Toplam gol beklentisi** kesin skor değildir; geçmiş verilerden hesaplanan ortalama gol sayısıdır.\n"
            "- **İki takım da gol atar** yüzdesi, iki takımın da en az bir gol bulma olasılığıdır.\n"
            "- **En olası tek skor**, olası skorlar içindeki en yüksek değerdir; tek başına güçlü bir kesinlik göstermez.\n"
            "- Son maç sonuçları geçmiş performanstır. Bunlar model tahminiyle aynı şey değildir."
        )
    matrix = forecast["score_matrix"]
    def probability(predicate: Any) -> float:
        return sum(
            matrix[h][a]
            for h in range(len(matrix))
            for a in range(len(matrix[h]))
            if predicate(h, a)
        )

    candidates = [
        ("1X", home_p + draw_p, "Ev sahibi kazanır veya maç berabere biter."),
        ("X2", away_p + draw_p, "Deplasman kazanır veya maç berabere biter."),
        ("12", home_p + away_p, "Maçta kazanan çıkar; beraberlik olmaz."),
        (f"Maç sonucu 1 · {home_name}", home_p, f"Model ev sahibine {home_p:.0%} galibiyet olasılığı veriyor."),
        ("Beraberlik", draw_p, f"Beraberlik olasılığı {draw_p:.0%}; iki takımın beklenen golleri {home_xg:.2f}–{away_xg:.2f}."),
        (f"Maç sonucu 2 · {away_name}", away_p, f"Model deplasman galibiyetini {away_p:.0%} hesaplıyor."),
        ("0,5 gol üst", probability(lambda h, a: h + a >= 1), "Maçta en az bir gol çıkar."),
        ("0,5 gol alt", probability(lambda h, a: h + a == 0), "Maç golsüz berabere biter."),
        ("1,5 gol üst", probability(lambda h, a: h + a >= 2), f"İki veya daha fazla gol gerekir; beklenen toplam gol {total_xg:.2f}."),
        ("1,5 gol alt", probability(lambda h, a: h + a <= 1), "Maçta en fazla bir gol çıkar."),
        ("2,5 gol üst", probability(lambda h, a: h + a >= 3), f"En az üç gol gerekir; beklenen toplam gol {total_xg:.2f}."),
        ("2,5 gol alt", probability(lambda h, a: h + a <= 2), "En fazla iki gol çıkması gerekir."),
        ("3,5 gol üst", probability(lambda h, a: h + a >= 4), "Maçta en az dört gol çıkar."),
        ("3,5 gol alt", probability(lambda h, a: h + a <= 3), "Dört veya daha fazla gol çıkmaması gerekir."),
        ("4,5 gol üst", probability(lambda h, a: h + a >= 5), "Maçta en az beş gol çıkar."),
        ("4,5 gol alt", probability(lambda h, a: h + a <= 4), "Maçta en fazla dört gol çıkar."),
        ("Karşılıklı gol · Var", forecast["btts"], f"İki takımın da gol bulma olasılığı {forecast['btts']:.0%}."),
        ("Karşılıklı gol · Yok", btts_no, f"En az bir takımın gol bulamama olasılığı {btts_no:.0%}."),
        (f"{home_name} 0,5 gol üst", probability(lambda h, a: h >= 1), f"Ev sahibinin en az bir gol bulma ihtimali; beklenen gol {home_xg:.2f}."),
        (f"{home_name} 0,5 gol alt", probability(lambda h, a: h == 0), f"Ev sahibinin gol bulamama olasılığı; beklenen gol {home_xg:.2f}."),
        (f"{home_name} 1,5 gol üst", probability(lambda h, a: h >= 2), f"Ev sahibinin en az iki gol bulması gerekir; beklenen gol {home_xg:.2f}."),
        (f"{home_name} 1,5 gol alt", probability(lambda h, a: h <= 1), f"Ev sahibi en fazla bir gol atar; beklenen gol {home_xg:.2f}."),
        (f"{home_name} 2,5 gol üst", probability(lambda h, a: h >= 3), f"Ev sahibinin en az üç gol bulması gerekir; beklenen gol {home_xg:.2f}."),
        (f"{away_name} 0,5 gol üst", probability(lambda h, a: a >= 1), f"Deplasmanın en az bir gol bulma ihtimali; beklenen gol {away_xg:.2f}."),
        (f"{away_name} 0,5 gol alt", probability(lambda h, a: a == 0), f"Deplasmanın gol bulamama olasılığı; beklenen gol {away_xg:.2f}."),
        (f"{away_name} 1,5 gol üst", probability(lambda h, a: a >= 2), f"Deplasmanın en az iki gol bulması gerekir; beklenen gol {away_xg:.2f}."),
        (f"{away_name} 1,5 gol alt", probability(lambda h, a: a <= 1), f"Deplasman en fazla bir gol atar; beklenen gol {away_xg:.2f}."),
        (f"{away_name} 2,5 gol üst", probability(lambda h, a: a >= 3), f"Deplasmanın en az üç gol bulması gerekir; beklenen gol {away_xg:.2f}."),
        (f"{home_name} kazanır ve KG Yok", probability(lambda h, a: h > a and a == 0), "Ev sahibinin galibiyetiyle birlikte deplasmanın gol atamaması gerekir."),
        (f"{away_name} kazanır ve KG Yok", probability(lambda h, a: a > h and h == 0), "Deplasman galibiyetiyle birlikte ev sahibinin gol atamaması gerekir."),
        (f"{home_name} kazanır ve 2,5 üst", probability(lambda h, a: h > a and h + a >= 3), "Ev sahibi kazanmalı ve maçta en az üç gol çıkmalı."),
        (f"{away_name} kazanır ve 2,5 üst", probability(lambda h, a: a > h and h + a >= 3), "Deplasman kazanmalı ve maçta en az üç gol çıkmalı."),
        ("1X ve 1,5 gol üst", probability(lambda h, a: h >= a and h + a >= 2), "Ev sahibi yenilmez ve maçta en az iki gol çıkar."),
        ("X2 ve 1,5 gol üst", probability(lambda h, a: a >= h and h + a >= 2), "Deplasman yenilmez ve maçta en az iki gol çıkar."),
        ("1X ve 3,5 gol alt", probability(lambda h, a: h >= a and h + a <= 3), "Ev sahibi yenilmez ve maçta en fazla üç gol çıkar."),
        ("X2 ve 3,5 gol alt", probability(lambda h, a: a >= h and h + a <= 3), "Deplasman yenilmez ve maçta en fazla üç gol çıkar."),
    ]
    candidates.extend([
        (f"Tam skor {forecast['most_likely_score']}", probability(
            lambda h, a: f"{h}-{a}" == str(forecast["most_likely_score"]).replace("–", "-").replace(":", "-")
        ), "Model matrisinde en yüksek olasılığa sahip tek skor."),
    ])
    scorelines = sorted(
        (
            (float(matrix[h][a]), h, a)
            for h in range(len(matrix))
            for a in range(len(matrix[h]))
        ),
        reverse=True,
    )[:5]
    candidates.extend(
        (f"Tam skor {h}-{a}", probability(lambda home_goals, away_goals, h=h, a=a: home_goals == h and away_goals == a), "Model skor dağılımında öne çıkan sonuç.")
        for _, h, a in scorelines
    )
    candidate_probabilities = {label: chance for label, chance, _ in candidates}
    ai_risks = precomputed_risks or {}
    low, medium = st.columns(2)
    for column, title, profile, explanation in (
        (low, "Görece daha olası", "low", "Yapay zekânın temkinli bulduğu, model olasılığı yüksek seçimler:"),
        (medium, "Dengeli risk", "medium", "Yapay zekânın olasılık ve maç senaryosu bakımından dengeli bulduğu seçimler:"),
    ):
        with column:
            st.markdown(f"**{title}**")
            st.caption(explanation)
            selections = ai_risks.get(profile, [])
            if ai_risks_stale and selections:
                st.caption(
                    "Bu AI seçimleri yorum hazırlanırken kayıtlı olan oranlara dayanır; oranlar değişmiş olabilir."
                )
            if not selections:
                if ai_risks_stale:
                    st.caption(
                        "Oranlar AI yorumundan sonra değiştiği için önceki seçimler güncellenene kadar gösterilmiyor."
                    )
                elif precomputed_risks is not None:
                    st.caption("Bu profil için doğrulanmış bir AI seçimi üretilemedi.")
                else:
                    st.caption("Yapay zekâ seçimi gece iş akışında hazırlanacak; henüz hazır değil.")
            for selection in selections:
                label = selection["market"]
                chance = selection.get(
                    "model_probability", candidate_probabilities.get(label)
                )
                if chance is not None:
                    st.markdown(f"**{label} · %{chance * 100:.1f}**  \n{selection['reason']}")
    st.caption("Senaryolar tahmin garantisi veya bahis tavsiyesi değildir. Birleşik koşulların gerçekleşme olasılığı daha düşüktür; maç önü kadro ve sakatlık bilgileri bu model özetinde yer almaz.")


def render_analysis(
    matches_data: list[dict[str, Any]],
    quotes_data: list[dict[str, Any]],
    analyses_data: list[dict[str, Any]],
    teams: pd.DataFrame,
    leagues: pd.DataFrame,
    match_day: date,
) -> None:
    matches = pd.DataFrame(matches_data)
    if matches.empty:
        st.info(f"{match_day:%d.%m.%Y} için kayıtlı maç bulunamadı.")
        return
    team_names = dict(zip(teams.get("id", []), teams.get("name", []))) if not teams.empty else {}
    league_names = dict(zip(leagues.get("id", []), leagues.get("name", []))) if not leagues.empty else {}
    league_countries = dict(zip(leagues.get("id", []), leagues.get("country", []))) if not leagues.empty else {}
    latest: dict[int, dict[str, Any]] = {}
    for quote in sorted(quotes_data, key=lambda q: str(q.get("captured_at", ""))):
        try:
            latest[int(quote["match_id"])] = quote
        except (KeyError, TypeError, ValueError):
            continue
    analyses_by_id = {
        int(record["match_id"]): record
        for record in analyses_data
        if record.get("match_id") is not None
    }

    matches["home_name"] = matches["home_team_id"].map(team_names).fillna("Ev sahibi")
    matches["away_name"] = matches["away_team_id"].map(team_names).fillna("Deplasman")
    matches["league_name"] = matches["league_id"].map(league_names).fillna("Lig bilgisi yok")
    matches["league_name"] = matches["league_name"].map(lambda name: LEAGUE_NAMES_TR.get(name, name))
    matches["country_name"] = matches["league_id"].map(league_countries).fillna("").map(lambda country: COUNTRY_NAMES_TR.get(country, country))
    matches["league_label"] = matches.apply(lambda row: f"{row['country_name']} · {row['league_name']}" if row["country_name"] else row["league_name"], axis=1)
    matches["kickoff"] = pd.to_datetime(matches["match_date"], utc=True, errors="coerce").dt.tz_convert(ISTANBUL)
    now = pd.Timestamp.now(tz=ISTANBUL)
    live_or_finished = matches["status"].fillna("").astype(str).str.casefold().isin(
        {"live", "finished", *NON_PLAYABLE_MATCH_STATUSES}
    )
    kickoff_reached = matches["kickoff"].notna() & (matches["kickoff"] <= now)
    matches = matches[~(live_or_finished | kickoff_reached)].copy()
    left, right, third = st.columns([1, 1, 2])
    with left:
        league_filter = st.selectbox("Lig", ["Tüm ligler", *sorted(matches["league_label"].unique())])
    with right:
        coverage = st.selectbox("Oran durumu", ["Tüm maçlar", "Oranı olanlar", "Oranı olmayanlar"])
    with third:
        search = st.text_input("Takım ara", placeholder="Takım adı yazın")
    filtered = matches.copy()
    if league_filter != "Tüm ligler":
        filtered = filtered[filtered["league_label"] == league_filter]
    if search.strip():
        q = search.strip().casefold()
        filtered = filtered[filtered["home_name"].str.casefold().str.contains(q) | filtered["away_name"].str.casefold().str.contains(q)]
    if coverage == "Oranı olanlar":
        filtered = filtered[filtered["id"].map(lambda x: int(x) in latest)]
    elif coverage == "Oranı olmayanlar":
        filtered = filtered[~filtered["id"].map(lambda x: int(x) in latest)]

    daily_match_ids = {int(match_id) for match_id in matches["id"]}
    quoted_ids = {int(row["match_id"]) for row in quotes_data if row.get("match_id") is not None} & daily_match_ids
    market_count = sum(
        len(unpack_payload(quote.get("odds")).get("markets", []))
        for match_id, quote in latest.items()
        if match_id in daily_match_ids
    )
    a, b, c, d = st.columns(4)
    a.metric("Filtreye uyan maç", f"{len(filtered)}")
    b.metric("Gün toplamı · oranlı", f"{len(quoted_ids)} / {len(matches)}", help="Oran kapsamı, seçili filtreye değil günün tüm maçlarına göredir.")
    c.metric("Gün toplamı · oransız", f"{len(matches) - len(quoted_ids)}", help="Oran kapsamı, seçili filtreye değil günün tüm maçlarına göredir.")
    d.metric("Mevcut market satırı", f"{market_count}")
    capture_times = pd.to_datetime(
        [quote.get("captured_at") for quote in quotes_data], utc=True, errors="coerce"
    ).dropna()
    if not capture_times.empty:
        last_capture = capture_times.max().tz_convert(ISTANBUL).strftime("%d.%m.%Y %H:%M")
        st.caption(
            f"Seçili gün için son oran kaydı: {last_capture} (İstanbul). "
            "Oran kaydı olmayan maç kaynakta bulunmuyor veya henüz eşleşmemiş olabilir."
        )
    else:
        st.caption("Seçili gün için henüz oran kaydı yok; maçlar kaynakta bulunmuyor veya eşleşmemiş olabilir.")
    st.caption("Piyasa yüzdeleri, market içindeki IdDaa oranlarının marjdan arındırılmış karşılığıdır; bağımsız model tahmini değildir.")
    st.divider()

    if filtered.empty:
        st.info("Bu filtrelerle eşleşen maç yok.")
        return
    for row in filtered.sort_values("kickoff", na_position="last").to_dict("records"):
        match_id = int(row["id"])
        row["iddaa_odds_quote"] = latest.get(match_id)
        kickoff = row["kickoff"]
        time_text = kickoff.strftime("%H:%M") if pd.notna(kickoff) else "Saat bilgisi yok"
        match_expander = st.expander(
            f"{time_text}  ·  {row['home_name']} — {row['away_name']}  ·  {row['league_label']}",
            expanded=False,
            key=f"match_{match_id}",
            on_change="rerun",
        )
        if not match_expander.open:
            continue
        with match_expander:
            cached_record = analyses_by_id.get(match_id)
            cached_analysis = _cached_analysis_for_match(cached_record, row)
            raw_odds_quote = row.get("iddaa_odds_quote")
            odds_quote = raw_odds_quote if isinstance(raw_odds_quote, dict) else {}
            odds_revision_payload = {
                "captured_at": odds_quote.get("captured_at"),
                "source_updated_at": odds_quote.get("source_updated_at"),
                "odds": odds_quote.get("odds"),
            }
            odds_revision = hashlib.sha256(
                json.dumps(
                    odds_revision_payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            commentary_has_current_odds = (
                cached_analysis is not None
                and cached_analysis.get("commentary_odds_revision") == odds_revision
            )
            if cached_analysis is not None and not commentary_has_current_odds:
                st.caption(
                    "Oranlar AI yorumu oluşturulduktan sonra güncellenmiş; yorumdaki oran bilgisi güncel olmayabilir."
                )
            history = pd.DataFrame()
            if cached_analysis is not None:
                history = pd.DataFrame(cached_analysis["history"])
                if not history.empty:
                    history["match_date"] = pd.to_datetime(history["match_date"], utc=True, errors="coerce")
                    for column in ("id", "league_id", "home_team_id", "away_team_id", "home_score", "away_score", "home_halftime_score", "away_halftime_score", "home_corners", "away_corners", "home_shots", "away_shots", "home_shots_on_target", "away_shots_on_target", "home_possession", "away_possession"):
                        if column in history:
                            history[column] = pd.to_numeric(history[column], errors="coerce")
                st.caption(f"Analiz gece önceden hesaplandı · {cached_record.get('calculated_at', '')}")
            else:
                try:
                    history_data = load_league_match_history(int(row["league_id"]))
                    team_history = load_recent_team_matches(
                        int(row["league_id"]),
                        (int(row["home_team_id"]), int(row["away_team_id"])),
                        row["kickoff"].isoformat(),
                    )
                    history_by_id = {int(match["id"]): match for match in history_data}
                    history_by_id.update(
                        (int(match["id"]), match) for match in team_history
                    )
                    history = pd.DataFrame(history_by_id.values())
                    if not history.empty:
                        history["match_date"] = pd.to_datetime(history["match_date"], utc=True, errors="coerce")
                        history = history[history["match_date"] < row["kickoff"]].copy()
                        for column in ("id", "league_id", "home_team_id", "away_team_id", "home_score", "away_score", "home_halftime_score", "away_halftime_score", "home_corners", "away_corners", "home_shots", "away_shots", "home_shots_on_target", "away_shots_on_target", "home_possession", "away_possession"):
                            if column in history:
                                history[column] = pd.to_numeric(history[column], errors="coerce")
                except Exception as exc:
                    st.warning(f"Geçmiş maçlar yüklenemedi; model tahmini ve form analizi gösterilemiyor: {exc}")
                st.caption("Bu maçın gece analiz önbelleği yok; geçmiş ve tahmin şimdi hazırlanıyor.")
            try:
                prior_matches = load_previous_team_match_history(
                    (int(row["home_team_id"]), int(row["away_team_id"])),
                    row["kickoff"].isoformat(),
                )
                prior_match_ids = previous_match_ids_for_fixture(
                    row, pd.DataFrame(prior_matches)
                )
                prior_lineups = load_fixture_lineups_for_matches(
                    tuple(prior_match_ids.values())
                )
                prior_dates = {
                    int(item["id"]): pd.to_datetime(item["match_date"], utc=True).isoformat()
                    for item in prior_matches
                    if item.get("id") is not None and item.get("match_date") is not None
                }
                prior_competitions = {
                    int(item["id"]): league_names.get(int(item["league_id"]))
                    for item in prior_matches
                    if item.get("id") is not None and item.get("league_id") is not None
                }
                row["previous_fixture_lineups"] = [
                    lineup | {
                        "previous_match_date": prior_dates.get(int(lineup["match_id"])),
                        "previous_competition": prior_competitions.get(int(lineup["match_id"])),
                    }
                    for lineup in prior_lineups
                ]
            except Exception as exc:
                row["previous_fixture_lineups"] = []
                st.caption(f"Önceki maç kadro verisi alınamadı: {exc}")
            meta, status_col = st.columns([4, 1])
            meta.caption(row["league_name"])
            status_col.caption(str(row.get("status", "")).replace("_", " ").title())
            render_confirmed_lineups(row)
            model_forecast, _ = render_model_forecast(
                row, history, [], precomputed=cached_analysis, show_details=False
            )
            render_match_data_coverage(
                row, history, cached_analysis, model_forecast
            )
            st.divider()
            render_tactical_match_readout(
                row,
                history,
                model_forecast,
                precomputed_risks=(
                    (cached_analysis or {}).get("ai_risk_selections")
                ),
                ai_risks_stale=(
                    cached_analysis is not None
                    and bool((cached_analysis or {}).get("ai_risk_selections"))
                    and not commentary_has_current_odds
                ),
            )
            if model_forecast is not None:
                render_ai_match_analysis(
                    row,
                    history,
                    model_forecast,
                    match_id,
                    precomputed_analysis=(cached_analysis or {}).get("ai_analysis"),
                    precomputed_halves=(cached_analysis or {}).get("halves"),
                )
            st.divider()
            render_recent_form(row, history, team_names)
            st.divider()


def render_match_data_coverage(
    row: dict[str, Any],
    history: pd.DataFrame,
    precomputed: dict[str, Any] | None,
    forecast: dict[str, Any] | None,
) -> None:
    """Show observed sample sizes and data recency for each match card."""
    kickoff = pd.to_datetime(
        row.get("kickoff") or row.get("match_date"), utc=True, errors="coerce"
    )
    if pd.isna(kickoff):
        return
    teams = (
        (int(row["home_team_id"]), str(row.get("home_name") or "Ev sahibi")),
        (int(row["away_team_id"]), str(row.get("away_name") or "Deplasman")),
    )
    metric_columns = {
        "Devre skoru": ("home_halftime_score", "away_halftime_score"),
        "xG": ("home_xg", "away_xg"),
        "xA": ("home_xa", "away_xa"),
        "Korner": ("home_corners", "away_corners"),
        "Şut": ("home_shots", "away_shots"),
        "İsabetli şut": ("home_shots_on_target", "away_shots_on_target"),
        "Topa sahip olma": ("home_possession", "away_possession"),
    }
    if history.empty:
        prior = pd.DataFrame(
            columns=[
                "id", "league_id", "match_date", "home_team_id", "away_team_id",
                *[column for pair in metric_columns.values() for column in pair],
            ]
        )
    else:
        prior = history[
            (pd.to_numeric(history.get("league_id"), errors="coerce") == int(row["league_id"]))
            & (pd.to_datetime(history.get("match_date"), utc=True, errors="coerce") < kickoff)
        ]
    rows = []
    latest_dates: list[tuple[str, pd.Timestamp]] = []
    for team_id, team_name in teams:
        team_history = prior[
            (pd.to_numeric(prior.get("home_team_id"), errors="coerce") == team_id)
            | (pd.to_numeric(prior.get("away_team_id"), errors="coerce") == team_id)
        ].sort_values(["match_date", "id"], ascending=False).head(5)
        recent_count = len(team_history)
        if recent_count:
            if "expected_metrics_checked_at" in team_history.columns:
                check_time_by_match = {
                    int(item["id"]): item.get("expected_metrics_checked_at")
                    for item in team_history.to_dict("records")
                    if item.get("id") is not None
                }
            else:
                check_time_by_match = load_match_statistics_check_times(
                    tuple(
                        int(match_id)
                        for match_id in team_history["id"]
                        if pd.notna(match_id)
                    )
                )
            checked_count = sum(
                pd.notna(check_time_by_match.get(int(match_id)))
                for match_id in team_history["id"]
                if pd.notna(match_id)
            )
        else:
            checked_count = 0
        latest_match = (
            pd.to_datetime(team_history.iloc[0].get("match_date"), utc=True, errors="coerce")
            if recent_count else pd.NaT
        )
        if pd.notna(latest_match):
            latest_dates.append((team_name, latest_match))
        coverage = {
            "Skor": recent_count,
            "Devre skoru": 0,
            "xG": 0,
            "xA": 0,
            "Korner": 0,
            "Şut": 0,
            "İsabetli şut": 0,
            "Topa sahip olma": 0,
        }
        for match in team_history.to_dict("records"):
            is_home = int(match.get("home_team_id", -1)) == team_id
            if (
                _optional_data_number(match.get("home_halftime_score")) is not None
                and _optional_data_number(match.get("away_halftime_score")) is not None
            ):
                coverage["Devre skoru"] += 1
            for label, (home_column, away_column) in metric_columns.items():
                if label == "Devre skoru":
                    continue
                column = home_column if is_home else away_column
                if _optional_data_number(match.get(column)) is not None:
                    coverage[label] += 1
        rows.append(
            {
                "Takım": team_name,
                "Son 5 tamamlanmış maç": f"{recent_count}/5",
                "API istatistik kontrolü": (
                    f"{checked_count}/{recent_count}" if recent_count else "0/0"
                ),
                "Skor": f"{coverage['Skor']}/{recent_count}" if recent_count else "0/0",
                "Devre": f"{coverage['Devre skoru']}/{recent_count}" if recent_count else "0/0",
                "xG": f"{coverage['xG']}/{recent_count}" if recent_count else "0/0",
                "xA": f"{coverage['xA']}/{recent_count}" if recent_count else "0/0",
                "Korner": f"{coverage['Korner']}/{recent_count}" if recent_count else "0/0",
                "Şut": f"{coverage['Şut']}/{recent_count}" if recent_count else "0/0",
                "İsabetli şut": f"{coverage['İsabetli şut']}/{recent_count}" if recent_count else "0/0",
                "Topa sahip olma": f"{coverage['Topa sahip olma']}/{recent_count}" if recent_count else "0/0",
            }
        )
    with st.expander("Veri kapsamı ve güncelliği", expanded=False):
        st.caption(
            "Her ölçüm, son 5 tamamlanmış maçtaki gerçek kayıt sayısını gösterir. "
            "API istatistik kontrolü, istatistik uç noktasının bu maç için sorgulanıp sorgulanmadığını belirtir; "
            "kontrol edilmiş maçta boş alan, son API yanıtında o ölçümün gelmediği anlamına gelir. "
            "xA şu ana kadar API-Football arşivindeki maçlarda dönmedi."
        )
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        model_history = int((forecast or {}).get("history_count") or 0)
        st.caption(f"Maç sonu Poisson modeli: {model_history} geçmiş lig maçı.")
        halves = (precomputed or {}).get("halves") or {}
        for half, label in ((1, "İlk yarı"), (2, "İkinci yarı")):
            half_forecast = halves.get(str(half))
            if precomputed is None:
                half_forecast = model_half_forecast(row, history, half=half)
            if half_forecast:
                sample = int(half_forecast.get("history_count") or 0)
                st.caption(
                    f"{label} tahmini: {sample} geçmiş devre skoru · 20 kayıt eşiği sağlandı."
                )
            else:
                st.caption(
                    f"{label} tahmini üretilemedi · modelin 20 geçmiş devre skoru eşiği karşılanmadı."
                )
        recent_history_dates = [match_date for _, match_date in latest_dates]
        if recent_history_dates:
            latest = max(recent_history_dates).tz_convert(ISTANBUL)
            age_days = max(0, int((kickoff - latest.tz_convert("UTC")).total_seconds() // 86400))
            st.caption(
                f"Takımların son tamamlanmış maçlarından en günceli: "
                f"{latest:%d.%m.%Y %H:%M} TSI · maç gününe göre {age_days} gün önce."
            )
        else:
            st.caption("Bu eşleşme için maç öncesi tamamlanmış takım geçmişi bulunamadı.")
        lineup_context = confirmed_lineup_context(row)
        projected_context = previous_lineup_context(row)
        confirmed_count = len((lineup_context or {}).get("teams", []))
        projected_count = len((projected_context or {}).get("teams", []))
        st.caption(
            f"Kadro kapsamı: {confirmed_count}/2 bu maç için doğrulanmış · "
            f"{projected_count}/2 önceki maçtan tahmin edilebilen takım."
        )


def _optional_data_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def render_confirmed_lineups(row: dict[str, Any]) -> None:
    """Show official lineups or clearly labeled previous-match projections."""
    confirmed = confirmed_lineup_context(row)
    projected = previous_lineup_context(row)
    previous_reference = previous_lineup_context(row, include_confirmed=True)
    previous_by_team = {
        lineup["team"]: lineup
        for lineup in (previous_reference or {}).get("teams", [])
    }
    teams = [
        (lineup, "confirmed")
        for lineup in (confirmed or {}).get("teams", [])
    ] + [
        (lineup, "previous")
        for lineup in (projected or {}).get("teams", [])
    ]
    if not teams:
        return
    source_label = (
        "Kesinleşen kadrolar ve önceki maçtan tahmin"
        if confirmed and projected
        else "Kesinleşen kadrolar"
        if confirmed
        else "Önceki maçtan tahmini kadro"
    )
    with st.expander(source_label, expanded=False):
        if projected:
            st.caption(
                "Önceki maçtan alınan liste bugünkü kesin kadro değildir; sakatlık, ceza "
                "ve rotasyon nedeniyle değişebilir."
            )
        for lineup, source in teams:
            st.markdown(f"**{lineup['team']}** · {lineup.get('formation') or 'Diziliş bilgisi yok'}")
            starters = lineup.get("starters") or []
            substitutes = lineup.get("substitutes") or []
            starter_names = [str(player.get("name") or "") for player in starters if isinstance(player, dict)]
            substitute_names = [str(player.get("name") or "") for player in substitutes if isinstance(player, dict)]
            st.write("İlk 11: " + (", ".join(name for name in starter_names if name) or "Veri yok"))
            if substitute_names:
                st.write("Yedekler: " + ", ".join(name for name in substitute_names if name))
            confirmed_at = pd.to_datetime(lineup.get("confirmed_at"), utc=True, errors="coerce")
            if source == "confirmed" and pd.notna(confirmed_at):
                local_time = confirmed_at.tz_convert(ISTANBUL).strftime("%d.%m.%Y %H:%M")
                st.caption(f"API-Football · {local_time} TSI itibarıyla doğrulandı")
            if source == "confirmed" and lineup["team"] in previous_by_team:
                previous_starters = previous_by_team[lineup["team"]].get("starters") or []

                def player_identity(player: dict[str, Any]) -> str:
                    player_id = player.get("id")
                    return f"id:{player_id}" if player_id is not None else str(
                        player.get("name") or ""
                    ).strip().casefold()

                previous_by_identity = {
                    player_identity(player): str(player.get("name") or "")
                    for player in previous_starters
                    if isinstance(player, dict) and player_identity(player)
                }
                current_by_identity = {
                    player_identity(player): str(player.get("name") or "")
                    for player in starters
                    if isinstance(player, dict) and player_identity(player)
                }
                no_longer_starting = [
                    name
                    for identity, name in previous_by_identity.items()
                    if identity not in current_by_identity
                ]
                newly_starting = [
                    name
                    for identity, name in current_by_identity.items()
                    if identity not in previous_by_identity
                ]
                if no_longer_starting or newly_starting:
                    st.caption("İlk 11 değişikliği, bir önceki maça göre:")
                    if no_longer_starting:
                        st.write("Önceki ilk 11'den bu kez başlamayan: " + ", ".join(no_longer_starting))
                    if newly_starting:
                        st.write("Bu maçta başlayan, önceki ilk 11'de olmayan: " + ", ".join(newly_starting))
            previous_date = pd.to_datetime(
                lineup.get("previous_match_date"), utc=True, errors="coerce"
            )
            if pd.notna(previous_date):
                st.caption(
                    "Kaynak: bir önceki maç"
                    + (f" · {lineup['previous_competition']}" if lineup.get("previous_competition") else "")
                    + f" · {previous_date.tz_convert(ISTANBUL).strftime('%d.%m.%Y')}"
                )


def render_ai_match_analysis(
    row: dict[str, Any],
    history: pd.DataFrame,
    forecast: dict[str, Any],
    match_id: int,
    precomputed_analysis: str | None = None,
    precomputed_halves: dict[str, Any] | None = None,
) -> None:
    """Generate an optional, session-cached AI interpretation for one match."""
    st.markdown("#### Yapay zekâ destekli detaylı yorum")
    st.caption(
        "Yapay zekâ, uygulamanın maç geçmişi ve Poisson olasılıklarını yorumlar; "
        "hesaplanmış model tahminlerini değiştirmez. Kadro bilgisi yoksa oyuncu veya sakatlık iddiası üretmez."
    )
    validation_matches = int(forecast.get("tuning_matches") or 0)
    result_log_loss = forecast.get("tuning_log_loss")
    baseline_log_loss = forecast.get("tuning_baseline_log_loss")
    if (
        validation_matches >= 30
        and result_log_loss is not None
        and baseline_log_loss is not None
    ):
        st.caption(
            f"Geçmiş ayrılmış değerlendirme: {validation_matches} maç · "
            f"log-loss {float(result_log_loss):.3f} (sabit model: "
            f"{float(baseline_log_loss):.3f}; düşük değer daha iyi). "
            "Bu ölçüm mevcut maçın sonucunu garanti etmez."
        )
    else:
        st.caption("Modelin geçmiş doğrulama örneği bu maç için güvenilir performans özeti vermeye yetmiyor.")
    if precomputed_analysis:
        st.caption("Yapay zekâ destekli maç yorumu önceden oluşturuldu.")
        st.markdown(precomputed_analysis)
        analysis_text = precomputed_analysis
    else:
        st.caption("Yapay zekâ yorumu gece iş akışında hazırlanacak; henüz hazır değil.")
        analysis_text = None
    if analysis_text:
        if not precomputed_analysis:
            st.markdown(analysis_text)
        render_ai_analysis_feedback(match_id, analysis_text)


def render_ai_analysis_feedback(match_id: int, analysis: str) -> None:
    """Collect a minimal anonymous usefulness rating for one exact AI comment."""
    analysis_hash = hashlib.sha256(analysis.encode("utf-8")).hexdigest()
    state_key = f"ai_feedback_submitted_{match_id}_{analysis_hash[:12]}"
    if st.session_state.get(state_key):
        st.caption("Bu yorum için geri bildirimin alındı. Teşekkürler.")
        return
    st.caption("Yalnızca oy ve seçtiğin sorun etiketleri kaydedilir; yorum metni ve kişisel bilgi gönderilmez.")
    with st.form(f"ai_feedback_form_{match_id}_{analysis_hash[:12]}"):
        rating = st.radio(
            "Bu yorum bu maçı anlamana yardımcı oldu mu?",
            options=["Evet", "Geliştirilmeli"],
            index=None,
            horizontal=True,
        )
        issue_labels = {
            "generic": "Yorum fazla genel veya tekrarlı",
            "missing_data": "Önemli veri eksik",
            "unclear_reasoning": "Tahmin gerekçesi belirsiz",
            "data_error": "Bir istatistik hatalı görünüyor",
        }
        selected_labels = st.multiselect(
            "İstersen sorun başlığını seç",
            options=list(issue_labels.values()),
        )
        submitted = st.form_submit_button("Geri bildirimi gönder")
    if not submitted:
        return
    if rating is None:
        st.warning("Göndermeden önce yorumun yararlı olup olmadığını seç.")
        return
    session_id = st.session_state.setdefault(
        "ai_feedback_session_id", str(uuid4())
    )
    issue_codes = [
        code for code, label in issue_labels.items() if label in selected_labels
    ]
    try:
        submit_ai_analysis_feedback(
            match_id=match_id,
            model_version=ANALYSIS_MODEL_VERSION,
            analysis_hash=analysis_hash,
            session_id=session_id,
            helpful=rating == "Evet",
            issue_codes=issue_codes,
        )
    except Exception as exc:
        st.error(f"Geri bildirim kaydedilemedi: {exc}")
        return
    st.session_state[state_key] = True
    st.success("Geri bildirimin alındı. Teşekkürler.")


def coupon_market_candidates(row: dict[str, Any], forecast: dict[str, Any]) -> list[dict[str, Any]]:
    """Build model-derived single-match selections for the automatic coupon."""
    matrix = forecast["score_matrix"]
    home, away = row["home_name"], row["away_name"]

    def chance(predicate: Any) -> float:
        return sum(
            matrix[h][a]
            for h in range(len(matrix))
            for a in range(len(matrix[h]))
            if predicate(h, a)
        )

    candidates = [
        ("1X", "Çifte şans", chance(lambda h, a: h >= a), "Ev sahibi yenilmez."),
        ("X2", "Çifte şans", chance(lambda h, a: a >= h), "Deplasman yenilmez."),
        ("12", "Çifte şans", chance(lambda h, a: h != a), "Maç berabere bitmez."),
        (f"1 · {home}", "Maç sonucu", forecast["home_win"], f"{home} galibiyeti."),
        ("X · Beraberlik", "Maç sonucu", forecast["draw"], "Maç berabere biter."),
        (f"2 · {away}", "Maç sonucu", forecast["away_win"], f"{away} galibiyeti."),
        ("0,5 gol üst", "Toplam gol", chance(lambda h, a: h + a >= 1), "Maçta en az bir gol çıkar."),
        ("0,5 gol alt", "Toplam gol", chance(lambda h, a: h + a == 0), "Maç golsüz berabere biter."),
        ("1,5 gol üst", "Toplam gol", chance(lambda h, a: h + a >= 2), "Maçta en az iki gol çıkar."),
        ("1,5 gol alt", "Toplam gol", chance(lambda h, a: h + a <= 1), "Maçta en fazla bir gol çıkar."),
        ("2,5 gol üst", "Toplam gol", chance(lambda h, a: h + a >= 3), "Maçta en az üç gol çıkar."),
        ("2,5 gol alt", "Toplam gol", chance(lambda h, a: h + a <= 2), "Maçta en fazla iki gol çıkar."),
        ("3,5 gol üst", "Toplam gol", chance(lambda h, a: h + a >= 4), "Maçta en az dört gol çıkar."),
        ("3,5 gol alt", "Toplam gol", chance(lambda h, a: h + a <= 3), "Maçta en fazla üç gol çıkar."),
        ("4,5 gol üst", "Toplam gol", chance(lambda h, a: h + a >= 5), "Maçta en az beş gol çıkar."),
        ("4,5 gol alt", "Toplam gol", chance(lambda h, a: h + a <= 4), "Maçta en fazla dört gol çıkar."),
        ("Karşılıklı gol var", "Karşılıklı gol", forecast["btts"], "İki takım da gol atar."),
        ("Karşılıklı gol yok", "Karşılıklı gol", 1 - forecast["btts"], "Takımlardan en az biri gol atamaz."),
        (f"{home} 0,5 gol üst", "Takım golü", chance(lambda h, a: h >= 1), f"{home} en az bir gol atar."),
        (f"{home} 0,5 gol alt", "Takım golü", chance(lambda h, a: h == 0), f"{home} gol atamaz."),
        (f"{away} 0,5 gol üst", "Takım golü", chance(lambda h, a: a >= 1), f"{away} en az bir gol atar."),
        (f"{away} 0,5 gol alt", "Takım golü", chance(lambda h, a: a == 0), f"{away} gol atamaz."),
        (f"{home} 1,5 gol üst", "Takım golü", chance(lambda h, a: h >= 2), f"{home} en az iki gol atar."),
        (f"{home} 1,5 gol alt", "Takım golü", chance(lambda h, a: h <= 1), f"{home} en fazla bir gol atar."),
        (f"{away} 1,5 gol üst", "Takım golü", chance(lambda h, a: a >= 2), f"{away} en az iki gol atar."),
        (f"{away} 1,5 gol alt", "Takım golü", chance(lambda h, a: a <= 1), f"{away} en fazla bir gol atar."),
        (f"{home} 2,5 gol üst", "Takım golü", chance(lambda h, a: h >= 3), f"{home} en az üç gol atar."),
        (f"{away} 2,5 gol üst", "Takım golü", chance(lambda h, a: a >= 3), f"{away} en az üç gol atar."),
        (f"{home} kazanır · KG yok", "Birleşik seçim", chance(lambda h, a: h > a and a == 0), f"{home} kazanır ve {away} gol atamaz."),
        (f"{away} kazanır · KG yok", "Birleşik seçim", chance(lambda h, a: a > h and h == 0), f"{away} kazanır ve {home} gol atamaz."),
        (f"{home} kazanır · 2,5 üst", "Birleşik seçim", chance(lambda h, a: h > a and h + a >= 3), f"{home} kazanır ve toplam gol en az üç olur."),
        (f"{away} kazanır · 2,5 üst", "Birleşik seçim", chance(lambda h, a: a > h and h + a >= 3), f"{away} kazanır ve toplam gol en az üç olur."),
        ("1X ve 1,5 gol üst", "Birleşik seçim", chance(lambda h, a: h >= a and h + a >= 2), "Ev sahibi yenilmez ve en az iki gol çıkar."),
        ("X2 ve 1,5 gol üst", "Birleşik seçim", chance(lambda h, a: a >= h and h + a >= 2), "Deplasman yenilmez ve en az iki gol çıkar."),
        ("1X ve 3,5 gol alt", "Birleşik seçim", chance(lambda h, a: h >= a and h + a <= 3), "Ev sahibi yenilmez ve en fazla üç gol çıkar."),
        ("X2 ve 3,5 gol alt", "Birleşik seçim", chance(lambda h, a: a >= h and h + a <= 3), "Deplasman yenilmez ve en fazla üç gol çıkar."),
    ]
    for score_h in range(len(matrix)):
        for score_a in range(len(matrix[score_h])):
            candidates.append((
                f"Tam skor {score_h}–{score_a}", "Tam skor", matrix[score_h][score_a],
                f"Modelin bu kesin skora verdiği olasılık {matrix[score_h][score_a]:.1%}.",
            ))
    return [
        {"selection": label, "market": market, "probability": float(probability), "reason": reason}
        for label, market, probability, reason in candidates
        if math.isfinite(float(probability)) and 0 < float(probability) < 1
    ]


def render_auto_coupon(
    matches_data: list[dict[str, Any]],
    analyses_data: list[dict[str, Any]],
    teams: pd.DataFrame,
    selected_day: date,
) -> None:
    """Generate a sample coupon from the selected day's pre-match model forecasts."""
    st.markdown(
        "<section class='hero'><div class='hero-kicker'>Model olasılıklarına göre otomatik seçim</div>"
        "<h1>Otomatik Kupon Oluşturucu</h1><p>Tahmin profilini ve maç sayısını seç; sistem farklı market türlerinden uygun seçimleri sıralasın.</p></section>",
        unsafe_allow_html=True,
    )
    st.info("Bu bölüm model olasılıklarına göre örnek seçim listesi üretir; bahis sitesine bağlanmaz, gerçek oran veya kazanç vaadi içermez.")
    matches = pd.DataFrame(matches_data)
    if matches.empty:
        st.info(f"{selected_day:%d.%m.%Y} için fikstür bulunamadı.")
        return

    team_names = dict(zip(teams.get("id", []), teams.get("name", []))) if not teams.empty else {}
    matches["home_name"] = matches["home_team_id"].map(team_names).fillna("Ev sahibi")
    matches["away_name"] = matches["away_team_id"].map(team_names).fillna("Deplasman")
    matches["kickoff"] = pd.to_datetime(matches["match_date"], utc=True, errors="coerce").dt.tz_convert(ISTANBUL)
    now = pd.Timestamp.now(tz=ISTANBUL)
    status_done = matches["status"].fillna("").astype(str).str.casefold().isin(
        {"live", "finished", *NON_PLAYABLE_MATCH_STATUSES}
    )
    upcoming = matches[
        ~status_done
        & matches["kickoff"].notna()
        & (matches["kickoff"] > now)
    ].copy()
    if upcoming.empty:
        st.info("Seçili tarihte henüz başlamamış maç bulunmuyor.")
        return

    c1, c2 = st.columns(2)
    with c1:
        risk = st.selectbox("Tahmin profili", ["Daha temkinli", "Dengeli"], index=1)
    with c2:
        legs = st.number_input("Kupondaki maç sayısı", min_value=2, max_value=10, value=5, step=1)

    analyses_by_id = {
        int(record["match_id"]): record
        for record in analyses_data
        if record.get("match_id") is not None
    }
    league_history: dict[int, pd.DataFrame] = {}
    fixture_candidates: list[dict[str, Any]] = []
    for row in upcoming.sort_values("kickoff").to_dict("records"):
        match_id = int(row["id"])
        cached = _cached_analysis_for_match(analyses_by_id.get(match_id), row)
        forecast = cached.get("forecast") if cached else None
        if forecast is None:
            league_id = int(row["league_id"])
            if league_id not in league_history:
                try:
                    history = pd.DataFrame(load_league_match_history(league_id))
                except Exception:
                    history = pd.DataFrame()
                if not history.empty:
                    history["match_date"] = pd.to_datetime(history["match_date"], utc=True, errors="coerce")
                    for column in ("id", "league_id", "home_team_id", "away_team_id", "home_score", "away_score"):
                        if column in history:
                            history[column] = pd.to_numeric(history[column], errors="coerce")
                league_history[league_id] = history
            forecast = model_match_forecast(row, league_history[league_id])
        if forecast is None:
            continue
        for candidate in coupon_market_candidates(row, forecast):
            p = candidate["probability"]
            market = candidate["market"]
            if risk == "Daha temkinli":
                eligible = p >= 0.65 and market not in {"Birleşik seçim", "Tam skor"}
                score = p
            elif risk == "Dengeli":
                eligible = 0.30 <= p <= 0.72 and market != "Tam skor"
                score = 1 - abs(p - 0.56)
            if eligible:
                fixture_candidates.append({
                    "match_id": match_id,
                    "match": f"{row['home_name']} – {row['away_name']}",
                    "kickoff": row["kickoff"],
                    "score": score,
                    **candidate,
                })

    if not fixture_candidates:
        st.warning(f"{risk} profilinde uygun tahmin üretilemedi. Başka bir risk profili seçmeyi deneyin.")
        return

    # Choose greedily across all outcomes, allowing at most one pick per match
    # and mildly favoring market diversity when alternatives are comparable.
    choices = fixture_candidates.copy()
    selected: list[dict[str, Any]] = []
    category_counts: dict[str, int] = {}
    used_matches: set[int] = set()
    while choices and len(selected) < int(legs):
        eligible_choices = [item for item in choices if item["match_id"] not in used_matches]
        if not eligible_choices:
            break
        best = max(
            eligible_choices,
            key=lambda item: item["score"] - 0.08 * category_counts.get(item["market"], 0),
        )
        choices.remove(best)
        selected.append(best)
        used_matches.add(best["match_id"])
        category_counts[best["market"]] = category_counts.get(best["market"], 0) + 1

    if len(selected) < int(legs):
        st.warning(f"Bu risk profiline uygun {len(selected)} maç bulundu; kupon bu sayıyla oluşturuldu.")
    combined_probability = math.prod(item["probability"] for item in selected)
    a, b, c = st.columns(3)
    a.metric("Seçilen maç", f"{len(selected)}")
    b.metric("Ortalama seçim olasılığı", f"%{sum(item['probability'] for item in selected) / len(selected) * 100:.1f}" if selected else "—")
    c.metric("Birleşik model olasılığı", f"%{combined_probability * 100:.3f}" if selected else "—")
    st.dataframe(
        pd.DataFrame([
            {
                "Başlama": item["kickoff"].strftime("%H:%M"),
                "Maç": item["match"],
                "Seçim": item["selection"],
                "Model %": item["probability"] * 100,
                "Gerekçe": item["reason"],
            }
            for item in selected
        ]),
        width="stretch",
        hide_index=True,
        column_config={"Model %": st.column_config.NumberColumn(format="%.1f%%")},
    )
    st.caption("Birleşik olasılık, maç sonuçlarının birbirinden bağımsız olduğunu varsayan yaklaşık model çarpımıdır; gerçek oran veya kupon getirisi değildir. Her maçtan en fazla bir seçim alınır.")






if __name__ == '__main__':
    st.markdown("<section class='hero'><div class='hero-kicker'>Günlük karşılaşmalar · iddaa marketleri</div><h1>Maç Merkezi</h1><p>Maçları inceleyin, marketleri karşılaştırın ve maç istatistikleri arşivine ulaşın.</p></section>", unsafe_allow_html=True)
    today = datetime.now(ISTANBUL).date()
    # Keep a user's manually selected day during normal reruns, but advance it
    # automatically after midnight so yesterday's finished fixtures do not
    # make today's page appear empty.
    if st.session_state.get("_match_day_session_date") != today:
        st.session_state["match_day_picker"] = today
        st.session_state["_match_day_session_date"] = today
    col, refresh_col, _ = st.columns([1, 1, 2])
    with col:
        selected_day = st.date_input(
            "Maç günü", format="DD.MM.YYYY", key="match_day_picker"
        )
    with refresh_col:
        st.write("")
        if st.button("Verileri şimdi yenile", width="stretch"):
            load_reference_catalog.clear()
            load_match_day.clear()
            count_completed_match_history.clear()
            load_completed_match_history.clear()
            load_evaluated_prediction_results.clear()
            load_league_match_history.clear()
            load_recent_team_matches.clear()
            load_fixture_lineups_for_matches.clear()
            load_previous_team_match_history.clear()
            load_match_statistics_check_times.clear()
            load_iddaa_market_labels.clear()
            st.rerun()
    try:
        teams, leagues = load_reference_catalog()
    except Exception as exc:
        st.error(f"Takım ve lig bilgileri yüklenemedi: {exc}")
        teams, leagues = pd.DataFrame(), pd.DataFrame()
    section = st.radio(
        "Bölüm",
        ["⚽  Maç Analizleri", "🧠  Günlük Taktik Analiz", "🎟️  Kupon Oluşturucu", "📊  İstatistik Arşivi", "📈  Tahmin Performansı"],
        horizontal=True,
        label_visibility="collapsed",
    )
    if section in {"⚽  Maç Analizleri", "🧠  Günlük Taktik Analiz"}:
        try:
            match_records, quote_records, analyses = load_match_day(selected_day)
            render_analysis(
                match_records, quote_records, analyses, teams, leagues, selected_day
            )
        except Exception as exc:
            st.error(f"Maç ve oran verileri yüklenemedi: {exc}")
    elif section == "🎟️  Kupon Oluşturucu":
        try:
            match_records, _, analyses = load_match_day(selected_day)
            render_auto_coupon(match_records, analyses, teams, selected_day)
        except Exception as exc:
            st.error(f"Otomatik kupon verileri hazırlanamadı: {exc}")
    elif section == "📊  İstatistik Arşivi":
        render_archive()
    else:
        render_performance()
