"""Prediction performance data, metrics and Streamlit screen."""

from __future__ import annotations

import math
from datetime import date, datetime, time, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from app.components.data import get_db
from app.metrics import _expected_calibration_error

ISTANBUL = ZoneInfo("Europe/Istanbul")


@st.cache_data(ttl=900, show_spinner="Tahmin performansı yükleniyor…")
def load_evaluated_prediction_filter_options() -> dict[str, Any]:
    """Load compact date, model, and league options for the performance filters."""
    rows = get_db().rpc("evaluated_prediction_filter_options")
    return rows[0] if rows else {}


@st.cache_data(ttl=900, show_spinner="Tahmin performansı yükleniyor…")
def load_evaluated_prediction_results(
    start_day: date,
    end_day: date,
    model_version: str | None = None,
    league_ids: tuple[int, ...] | None = None,
) -> list[dict[str, Any]]:
    """Read only evaluated forecasts matching the selected filters."""
    if start_day > end_day:
        raise ValueError("Performance start date must not be after end date")
    start = datetime.combine(start_day, time.min, tzinfo=ISTANBUL).astimezone(
        timezone.utc
    )
    end = datetime.combine(
        end_day + timedelta(days=1), time.min, tzinfo=ISTANBUL
    ).astimezone(timezone.utc)
    filters: dict[str, str | int] = {
        "match_date": f"gte.{start.isoformat()}",
        "and": f"(match_date.lt.{end.isoformat()})",
    }
    if model_version is not None:
        filters["model_version"] = f"eq.{model_version}"
    if league_ids is not None:
        normalized_ids = tuple(sorted({int(value) for value in league_ids}))
        if not normalized_ids:
            return []
        filters["league_id"] = (
            f"eq.{normalized_ids[0]}"
            if len(normalized_ids) == 1
            else f"in.({','.join(map(str, normalized_ids))})"
        )
    return get_db().select_all(
        "evaluated_prediction_results",
        columns=(
            "prediction_id,match_id,match_date,predicted_at,model_version,actual_result,was_correct,brier_score,"
            "league_id,league_name,"
            "prob_home_win,prob_draw,prob_away_win,prob_over_2_5,over_2_5_actual,"
            "over_2_5_was_correct,over_2_5_brier_score,prob_btts,btts_actual,"
            "btts_was_correct,btts_brier_score,market_probabilities,market_performance"
        ),
        filters=filters,
        order="match_date.asc,match_id.asc,predicted_at.asc,prediction_id.asc",
    )

@st.cache_data(ttl=900, max_entries=16, show_spinner=False)
def market_performance_rows(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Summarize accuracy and probability quality for each evaluated market."""
    observations: dict[str, list[tuple[float, bool, float | None, float | None]]] = {}

    for record in records:
        result_probs = [record.get("prob_home_win"), record.get("prob_draw"), record.get("prob_away_win")]
        result_keys = ("home_win", "draw", "away_win")
        try:
            result_probs = [float(probability) for probability in result_probs]
        except (TypeError, ValueError):
            result_probs = []
        actual_result = str(record.get("actual_result") or "").casefold()
        result_aliases = {"home": "home_win", "1": "home_win", "away": "away_win", "2": "away_win", "x": "draw"}
        actual_result = result_aliases.get(actual_result, actual_result)
        if len(result_probs) == 3 and all(math.isfinite(p) and 0 <= p <= 1 for p in result_probs) and actual_result in result_keys:
            actual_index = result_keys.index(actual_result)
            confidence = max(result_probs)
            correct = result_keys[result_probs.index(confidence)] == actual_result
            brier = sum((p - float(index == actual_index)) ** 2 for index, p in enumerate(result_probs))
            loss = -math.log(max(1e-15, result_probs[actual_index]))
            observations.setdefault("Maç sonucu (1X2)", []).append((confidence, correct, brier, loss))

        for prefix, label in (("over_2_5", "Toplam gol 2,5 üst"), ("btts", "Karşılıklı gol (KG Var)")):
            p, actual, correct, stored_brier = (
                record.get(f"prob_{prefix}"), record.get(f"{prefix}_actual"),
                record.get(f"{prefix}_was_correct"), record.get(f"{prefix}_brier_score"),
            )
            try:
                p = float(p)
                brier = float(stored_brier) if stored_brier is not None else None
            except (TypeError, ValueError):
                continue
            if isinstance(actual, bool) and isinstance(correct, bool) and math.isfinite(p) and 0 <= p <= 1:
                loss = -math.log(max(1e-15, p if actual else 1 - p))
                observations.setdefault(label, []).append((max(p, 1 - p), correct, brier if brier is not None else (p - float(actual)) ** 2, loss))

        performance = record.get("market_performance") or {}
        if not isinstance(performance, dict):
            continue
        total_goals = performance.get("total_goals") or {}
        if isinstance(total_goals, dict):
            by_line: dict[str, dict[str, dict[str, Any]]] = {}
            for key, value in total_goals.items():
                parts = key.split("_", 1)
                if len(parts) == 2 and parts[0] in {"over", "under"} and isinstance(value, dict):
                    by_line.setdefault(parts[1], {})[parts[0]] = value
            for line, sides in by_line.items():
                over, under = sides.get("over"), sides.get("under")
                try:
                    p_over = float(over["probability"]) if over else 1 - float(under["probability"])
                    actual_over = over.get("actual") if over else not under.get("actual")
                except (TypeError, ValueError, KeyError):
                    continue
                if line == "2_5" or not isinstance(actual_over, bool) or not math.isfinite(p_over) or not 0 <= p_over <= 1:
                    continue
                correct = (p_over >= 0.5) == actual_over
                p_actual = p_over if actual_over else 1 - p_over
                brier = (p_over - float(actual_over)) ** 2
                observations.setdefault(f"Toplam gol {line.replace('_', ',')} çizgisi", []).append((max(p_over, 1 - p_over), correct, brier, -math.log(max(1e-15, p_actual))))

        team_goals = performance.get("team_goals") or {}
        if isinstance(team_goals, dict):
            for key, value in team_goals.items():
                if not isinstance(value, dict):
                    continue
                parts = key.split("_over_", 1)
                if len(parts) != 2 or parts[0] not in {"home", "away"}:
                    continue
                try:
                    p = float(value["probability"])
                    actual = value["actual"]
                except (TypeError, ValueError, KeyError):
                    continue
                if not isinstance(actual, bool) or not math.isfinite(p) or not 0 <= p <= 1:
                    continue
                correct = (p >= 0.5) == actual
                label = "Ev sahibi" if parts[0] == "home" else "Deplasman"
                line = parts[1].replace("_", ",")
                observations.setdefault(f"{label} gol {line} üst", []).append((max(p, 1 - p), correct, (p - float(actual)) ** 2, -math.log(max(1e-15, p if actual else 1 - p))))

        double_chance = performance.get("double_chance") or {}
        if isinstance(double_chance, dict):
            choices = [value for value in double_chance.values() if isinstance(value, dict) and isinstance(value.get("actual"), bool)]
            try:
                choice = max(choices, key=lambda value: float(value["probability"]))
                p, actual = float(choice["probability"]), choice["actual"]
            except (ValueError, TypeError, KeyError):
                p = None
            if p is not None and 0 <= p <= 1:
                observations.setdefault("Çifte şans", []).append((p, actual, (p - float(actual)) ** 2, -math.log(max(1e-15, p if actual else 1 - p))))

        correct_score = performance.get("correct_score") or {}
        scores = (record.get("market_probabilities") or {}).get("correct_scores", [])
        if isinstance(correct_score, dict) and isinstance(correct_score.get("top_3_hit"), bool) and isinstance(scores, list):
            confidence = sum(float(score.get("probability", 0)) for score in scores if isinstance(score, dict))
            hit = correct_score["top_3_hit"]
            if math.isfinite(confidence) and 0 <= confidence <= 1:
                observations.setdefault("Doğru skor (ilk 3)", []).append((confidence, hit, (confidence - float(hit)) ** 2, -math.log(max(1e-15, confidence if hit else 1 - confidence))))

    rows = []
    for market, values in observations.items():
        if not values:
            continue
        count = len(values)
        hits = sum(hit for _, hit, _, _ in values)
        accuracy = hits / count
        z = 1.96
        denominator = 1 + z * z / count
        center = (accuracy + z * z / (2 * count)) / denominator
        half_width = z * math.sqrt(
            accuracy * (1 - accuracy) / count + z * z / (4 * count * count)
        ) / denominator
        pairs = [(confidence, float(hit)) for confidence, hit, _, _ in values]
        rows.append({
            "Pazar": market,
            "Örnek": count,
            "Doğru": hits,
            "Başarı %": 100 * accuracy,
            "%95 alt sınır": 100 * max(0.0, center - half_width),
            "%95 üst sınır": 100 * min(1.0, center + half_width),
            "Brier": sum(brier for _, _, brier, _ in values if brier is not None) / count,
            "Log-loss": sum(loss for _, _, _, loss in values if loss is not None) / count,
            "Sabit baz log-loss": math.log(3) if market == "Maç sonucu (1X2)" else math.log(2),
            "Kalibrasyon hatası": _expected_calibration_error(pairs),
        })
    return pd.DataFrame(rows).sort_values("Pazar").reset_index(drop=True) if rows else pd.DataFrame()

def render_performance() -> None:
    try:
        filter_options = load_evaluated_prediction_filter_options()
    except Exception as exc:
        st.error(f"Tahmin performansı filtreleri yüklenemedi: {exc}")
        return
    minimum = pd.to_datetime(filter_options.get("min_match_date"), utc=True, errors="coerce")
    maximum = pd.to_datetime(filter_options.get("max_match_date"), utc=True, errors="coerce")
    if pd.isna(minimum) or pd.isna(maximum):
        st.info("Henüz değerlendirilmiş tahmin bulunmuyor.")
        return
    minimum_day = minimum.tz_convert(ISTANBUL).date()
    maximum_day = maximum.tz_convert(ISTANBUL).date()
    versions = sorted(
        {str(version) for version in filter_options.get("model_versions", []) if version},
        reverse=True,
    )
    raw_leagues = filter_options.get("league_options") or []
    league_ids_by_name: dict[str, set[int]] = {}
    if isinstance(raw_leagues, list):
        for league in raw_leagues:
            if not isinstance(league, dict) or not league.get("name"):
                continue
            try:
                league_ids_by_name.setdefault(str(league["name"]), set()).add(
                    int(league["id"])
                )
            except (KeyError, TypeError, ValueError):
                continue
    league_options = sorted(league_ids_by_name)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        start_day = st.date_input(
            "Başlangıç",
            value=minimum_day,
            format="DD.MM.YYYY",
            key="performance_start",
        )
    with c2:
        end_day = st.date_input(
            "Bitiş",
            value=maximum_day,
            format="DD.MM.YYYY",
            key="performance_end",
        )
    with c3:
        version_filter = st.selectbox("Model sürümü", ["Tüm sürümler", *versions])
    with c4:
        league_filter = st.selectbox("Lig", ["Tüm ligler", *league_options])
    if start_day > end_day:
        st.warning("Başlangıç tarihi bitiş tarihinden sonra olamaz.")
        return
    selected_league_ids = (
        tuple(sorted(league_ids_by_name[league_filter]))
        if league_filter != "Tüm ligler"
        else None
    )
    try:
        selected = load_evaluated_prediction_results(
            start_day,
            end_day,
            model_version=(
                version_filter if version_filter != "Tüm sürümler" else None
            ),
            league_ids=selected_league_ids,
        )
    except Exception as exc:
        st.error(f"Tahmin performansı yüklenemedi: {exc}")
        return
    if not selected:
        st.info("Seçilen dönemde ölçülebilir pazar tahmini bulunamadı.")
        return
    table = market_performance_rows(selected)
    if table.empty:
        st.info("Seçilen dönemde ölçülebilir pazar tahmini bulunamadı.")
        return
    total = int(table["Örnek"].sum())
    correct = int(table["Doğru"].sum())
    a, b, c = st.columns(3)
    a.metric("Değerlendirilen maç", f"{len(selected):,}")
    b.metric("Tüm pazar kararları", f"{correct:,} / {total:,} · %{100 * correct / total:.1f}")
    c.metric("Pazar sayısı", f"{len(table)}")
    st.caption(
        f"Dönem: {start_day:%d.%m.%Y}–{end_day:%d.%m.%Y}. Her pazarın örnek sayısı farklıdır; "
        "toplu oran karar başına eşit ağırlıklıdır ve bahis getirisi değildir. "
        "Başarı oranının %95 aralığı Wilson güven aralığıdır; küçük örneklemde belirsizlik artar. "
        "Sabit baz log-loss, 1X2 için üç eşit olasılık, ikili marketler için %50 olasılık kullanır. "
        "Kârlılık hesabı için bu maçlarla eşleştirilmiş tarihsel oran kaydı bulunmadığından ROI gösterilmiyor."
    )
    st.dataframe(
        table,
        width="stretch",
        hide_index=True,
        column_config={
            "Başarı %": st.column_config.NumberColumn(format="%.1f%%"),
            "%95 alt sınır": st.column_config.NumberColumn(format="%.1f%%"),
            "%95 üst sınır": st.column_config.NumberColumn(format="%.1f%%"),
            "Brier": st.column_config.NumberColumn(format="%.3f"),
            "Log-loss": st.column_config.NumberColumn(format="%.3f"),
            "Sabit baz log-loss": st.column_config.NumberColumn(format="%.3f"),
            "Kalibrasyon hatası": st.column_config.NumberColumn(format="%.3f"),
        },
    )
    st.caption("Brier, log-loss ve 10 dilimli kalibrasyon hatasında düşük değer daha iyidir. Kalibrasyon, seçilen sonucun güven olasılığını gerçekleşen isabetle karşılaştırır.")
