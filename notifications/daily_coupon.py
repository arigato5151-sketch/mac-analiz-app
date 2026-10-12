"""Build and send the current day's model-and-odds coupon summary."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from config.settings import get_settings
from db.db_client import SupabaseRestClient
from app.components.data import _has_suspicious_result_quote, _nesine_market_key
from app.components.freshness import odds_are_current
from models.coupon_policy import diversified_coupon_rows
from models.value_analysis import (
    MAX_PUBLISHABLE_EXPECTED_VALUE,
    ValueAssessment,
    assess_market_value,
    derive_combo_probabilities,
)
from notifications.telegram import send_telegram_message
from app.components.nesine_labels import nesine_market_name, nesine_selection_name

TZ = ZoneInfo("Europe/Istanbul")


def _label(item: ValueAssessment) -> str:
    names = {
        "home_win": "Maç Sonucu · 1",
        "draw": "Maç Sonucu · X",
        "away_win": "Maç Sonucu · 2",
        "over_2_5": "2,5 Gol Alt/Üst · Üst",
        "btts_yes": "Karşılıklı Gol · Var",
    }
    market = names.get(
        item.key,
        f"{nesine_market_name(item.key)} · {nesine_selection_name(item.key)}",
    )
    return f"{market} · %{item.model_probability * 100:.0f} · oran {item.odds:.2f}"


def _rows(db: SupabaseRestClient, now: datetime) -> list[dict[str, Any]]:
    local = now.astimezone(TZ)
    end = now + timedelta(days=7)
    matches = db.select_all(
        "matches",
        columns="id,league_id,home_team_id,away_team_id,match_date,status",
        filters={
            "status": "eq.scheduled",
            "and": f"(match_date.gte.{now.isoformat()},match_date.lte.{end.isoformat()})",
        },
        order="match_date.asc,id.asc",
    )
    if not matches:
        return []
    match_dates = {
        int(match["id"]): datetime.fromisoformat(
            str(match["match_date"]).replace("Z", "+00:00")
        ).astimezone(TZ).date()
        for match in matches
    }
    today = local.date()
    selected_date = today if today in match_dates.values() else min(match_dates.values())
    matches = [match for match in matches if match_dates[int(match["id"])] == selected_date]
    ids = ",".join(str(int(m["id"])) for m in matches)
    team_ids = ",".join(str(int(m[k])) for m in matches for k in ("home_team_id", "away_team_id"))
    league_ids = ",".join(str(int(m["league_id"])) for m in matches)
    teams = {int(r["id"]): r["name"] for r in db.select_all("teams", columns="id,name", filters={"id": f"in.({team_ids})"})}
    leagues = {int(r["id"]): r["name"] for r in db.select_all("leagues", columns="id,name", filters={"id": f"in.({league_ids})"})}
    predictions = db.select_all(
        "predictions",
        columns="match_id,prob_home_win,prob_draw,prob_away_win,prob_over_2_5,prob_btts,market_probabilities,predicted_at",
        filters={"match_id": f"in.({ids})"},
        order="predicted_at.desc",
    )
    odds = db.select_all(
        "odds_quote_history",
        columns="match_id,bookmaker,odds,captured_at",
        filters={"match_id": f"in.({ids})"},
        order="captured_at.desc",
    )
    nesine_quotes = db.select_all(
        "nesine_market_quotes",
        columns="match_id,market_code,market_name,selection_code,selection_name,odd,captured_at",
        filters={"match_id": f"in.({ids})"},
        order="captured_at.desc",
    )
    latest_predictions: dict[int, dict[str, Any]] = {}
    latest_odds: dict[int, dict[str, Any]] = {}
    for row in predictions:
        latest_predictions.setdefault(int(row["match_id"]), row)
    for row in odds:
        if isinstance(row.get("odds"), dict) and not _has_suspicious_result_quote(row["odds"]):
            latest_odds.setdefault(int(row["match_id"]), row)
    latest_nesine_time: dict[int, datetime] = {}
    nesine_odds: dict[int, dict[str, float]] = {}
    for quote in nesine_quotes:
        match_id = int(quote["match_id"])
        try:
            captured_at = datetime.fromisoformat(
                str(quote["captured_at"]).replace("Z", "+00:00")
            )
            odd = float(quote["odd"])
        except (KeyError, TypeError, ValueError):
            continue
        if odd <= 1:
            continue
        latest = latest_nesine_time.get(match_id)
        if latest is None or captured_at > latest:
            latest_nesine_time[match_id] = captured_at
            nesine_odds[match_id] = {}
        if captured_at == latest_nesine_time[match_id]:
            key = _nesine_market_key(quote)
            if key:
                nesine_odds[match_id][key] = odd
    result = []
    for match in matches:
        match_id = int(match["id"])
        prediction = latest_predictions.get(match_id)
        quote = latest_odds.get(match_id)
        if not prediction:
            continue

        probabilities = {
            "home_win": prediction["prob_home_win"],
            "draw": prediction["prob_draw"],
            "away_win": prediction["prob_away_win"],
            "over_2_5": prediction["prob_over_2_5"],
            "btts_yes": prediction["prob_btts"],
        }
        persisted_markets = prediction.get("market_probabilities") or {}
        if isinstance(persisted_markets, str):
            try:
                persisted_markets = json.loads(persisted_markets)
            except json.JSONDecodeError:
                persisted_markets = {}
        if isinstance(persisted_markets, dict):
            probabilities.update(persisted_markets)

        match_odds = dict(quote["odds"]) if quote else {}
        nesine_time = latest_nesine_time.get(match_id)
        base_quote_time = quote.get("captured_at") if quote else nesine_time
        if not match_odds and nesine_time and odds_are_current(nesine_time, now=now):
            match_odds.update(nesine_odds.get(match_id, {}))
        if not match_odds or not odds_are_current(base_quote_time, now=now):
            continue
        if nesine_time and odds_are_current(nesine_time, now=now):
            match_odds.update(nesine_odds.get(match_id, {}))
        if _fails_prediction_sanity_check(probabilities):
            continue
        probabilities.update(
            derive_combo_probabilities(
                probabilities,
                score_matrix=(persisted_markets or {}).get("score_matrix")
                if isinstance(persisted_markets, dict)
                else None,
            )
        )
        assessments = [
            item
            for item in assess_market_value(probabilities, match_odds)
            if item.model_probability >= (0.55 if item.key in {"btts_yes", "btts_no"} else 0.50)
            and item.odds <= 20.0
            and item.expected_value <= MAX_PUBLISHABLE_EXPECTED_VALUE
        ]
        if not assessments:
            continue
        best = max(assessments, key=lambda item: item.expected_value)
        kickoff = datetime.fromisoformat(
            str(match["match_date"]).replace("Z", "+00:00")
        ).astimezone(TZ)
        result.append({
            "match_id": match_id,
            "match": f"{teams.get(int(match['home_team_id']), 'Ev sahibi')} — {teams.get(int(match['away_team_id']), 'Deplasman')}",
            "time": kickoff.strftime("%H:%M"),
            "league": leagues.get(int(match["league_id"]), "Lig"),
            "best": best,
            "assessments": assessments,
        })
    return sorted(result, key=lambda row: row["best"].expected_value, reverse=True)


def _fails_prediction_sanity_check(probabilities: dict[str, Any]) -> bool:
    market_probabilities = probabilities.get("market_probabilities")
    if not isinstance(market_probabilities, dict):
        return False
    expected_goals = market_probabilities.get("expected_goals")
    if not isinstance(expected_goals, dict):
        return False
    try:
        home_goals = float(expected_goals.get("home"))
        away_goals = float(expected_goals.get("away"))
        home_win = float(probabilities["home_win"])
        away_win = float(probabilities["away_win"])
    except (KeyError, TypeError, ValueError):
        return False
    total_goals = home_goals + away_goals
    return (
        (away_goals >= 3.5 and away_win < 0.55)
        or (home_goals >= 3.5 and home_win < 0.55)
        or (total_goals >= 4.5 and float(probabilities.get("over_2_5", 0)) < 0.65)
    )


def build_daily_coupon_message(rows: list[dict[str, Any]], *, market_warning: str | None = None) -> str:
    if not rows:
        return "🎟️ Günlük kupon\nBugünün kalan maçlarında eşleşen oran ve model tahmini bulunamadı."
    groups = _coupon_groups(rows)
    low, balanced, high = groups["low_risk"], groups["balanced"], groups["high_odds"]
    sections = [f"🎟️ Günlük kupon · {len(rows)} oranlı maç"]
    if market_warning:
        sections.append(f"⚠️ {market_warning}")
    for title, selected in (("🟢 Düşük risk", low), ("🟡 Dengeli", balanced), ("🔴 Yüksek oran", high)):
        sections.append(f"\n{title}")
        if not selected:
            sections.append("Seçim yok.")
            continue
        total = 1.0
        for row in selected:
            item = row["best"]
            total *= item.odds
            sections.append(f"{row['time']} {row['match']}\n{_label(item)} · EV %{item.expected_value * 100:+.1f}")
        sections.append(f"Toplam yaklaşık oran: {total:.2f}")
    sections.append("\nTahminler istatistiksel olasılıktır; garanti veya kesin sonuç değildir.")
    return "\n".join(sections)


def _coupon_groups(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    low = diversified_coupon_rows(
        rows, max_items=None, min_probability=0.55, min_ev=1e-9
    )
    balanced = diversified_coupon_rows(rows, max_items=None, min_ev=0.03)
    high = diversified_coupon_rows(
        rows, max_items=None, min_ev=0.03, high_odds=True
    )
    return {"low_risk": low, "balanced": balanced, "high_odds": high}


def _persist_coupon_runs(db: SupabaseRestClient, rows: list[dict[str, Any]], coupon_date: str) -> None:
    for coupon_type, selected in _coupon_groups(rows).items():
        if not selected:
            continue
        total_odds = 1.0
        selections = []
        for row in selected:
            item = row["best"]
            total_odds *= item.odds
            selections.append({"match_id": int(row["match_id"]), "key": item.key, "odds": item.odds, "label": _label(item)})
        db.upsert("daily_coupon_runs", [{"coupon_date": coupon_date, "coupon_type": coupon_type, "selections": selections, "total_odds": total_odds}], on_conflict="coupon_date,coupon_type")


def _market_gate_warning(db: SupabaseRestClient) -> str | None:
    promoted = db.select_all(
        "model_candidates",
        columns="status,offline_metrics,promoted_at",
        filters={"status": "eq.promoted"},
        order="promoted_at.desc",
    )
    if not promoted:
        return "Model-piyasa karşılaştırması bulunamadı; kupon yalnızca istatistiksel sinyaldir."
    comparison = (promoted[0].get("offline_metrics") or {}).get("market_comparison") or {}
    if comparison.get("status") != "passed":
        return "Model piyasayı anlamlı biçimde geçemedi; EV sinyalleri temkinli değerlendirilmelidir."
    return None


def run_daily_coupon() -> dict[str, int | str]:
    settings = get_settings()
    db = SupabaseRestClient(settings.supabase_url, settings.supabase_service_role_key)
    rows = _rows(db, datetime.now(timezone.utc))
    _persist_coupon_runs(db, rows, datetime.now(TZ).date().isoformat())
    market_warning = _market_gate_warning(db)
    token, chat_id = os.getenv("TELEGRAM_BOT_TOKEN", "").strip(), os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        return {"matches": len(rows), "sent": 0, "skipped": "telegram_not_configured"}
    send_telegram_message(build_daily_coupon_message(rows, market_warning=market_warning), bot_token=token, chat_id=chat_id)
    return {"matches": len(rows), "sent": 1}


if __name__ == "__main__":
    print(json.dumps(run_daily_coupon(), ensure_ascii=False))
