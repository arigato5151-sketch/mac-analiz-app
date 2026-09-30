"""Build and send the current day's model-and-odds coupon summary."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

from config.settings import get_settings
from db.db_client import SupabaseRestClient
from models.coupon_policy import _market_family, diversified_coupon_rows
from models.value_analysis import ValueAssessment, assess_market_value
from notifications.telegram import send_telegram_message

TZ = ZoneInfo("Europe/Istanbul")


def _label(item: ValueAssessment) -> str:
    names = {"home_win": "MS 1", "draw": "MS X", "away_win": "MS 2", "over_2_5": "Üst 2.5", "btts_yes": "KG Var"}
    return f"{names.get(item.key, item.key)} %{item.model_probability * 100:.0f} · oran {item.odds:.2f}"


def _rows(db: SupabaseRestClient, now: datetime) -> list[dict[str, Any]]:
    local = now.astimezone(TZ)
    start = local.replace(hour=0, minute=0, second=0, microsecond=0).astimezone(timezone.utc)
    end = local.replace(hour=23, minute=59, second=59, microsecond=999999).astimezone(timezone.utc)
    matches = db.select_all("matches", columns="id,league_id,home_team_id,away_team_id,match_date,status", filters={"status": "eq.scheduled", "and": f"(match_date.gte.{now.isoformat()},match_date.lte.{end.isoformat()})"}, order="match_date.asc")
    matches = [m for m in matches if str(m["match_date"]) >= start.isoformat()]
    if not matches:
        return []
    ids = ",".join(str(int(m["id"])) for m in matches)
    team_ids = ",".join(str(int(m[k])) for m in matches for k in ("home_team_id", "away_team_id"))
    league_ids = ",".join(str(int(m["league_id"])) for m in matches)
    teams = {int(r["id"]): r["name"] for r in db.select_all("teams", columns="id,name", filters={"id": f"in.({team_ids})"})}
    leagues = {int(r["id"]): r["name"] for r in db.select_all("leagues", columns="id,name", filters={"id": f"in.({league_ids})"})}
    predictions = db.select_all("predictions", columns="match_id,prob_home_win,prob_draw,prob_away_win,prob_over_2_5,prob_btts,market_probabilities,predicted_at", filters={"match_id": f"in.({ids})"}, order="predicted_at.desc")
    odds = db.select_all("odds_quote_history", columns="match_id,odds,captured_at", filters={"match_id": f"in.({ids})"}, order="captured_at.desc")
    latest_predictions: dict[int, dict[str, Any]] = {}
    latest_odds: dict[int, dict[str, Any]] = {}
    for row in predictions:
        latest_predictions.setdefault(int(row["match_id"]), row)
    for row in odds:
        latest_odds.setdefault(int(row["match_id"]), row)
    result = []
    for match in matches:
        prediction = latest_predictions.get(int(match["id"]))
        quote = latest_odds.get(int(match["id"]))
        if not prediction or not isinstance(quote.get("odds") if quote else None, dict):
            continue
        probabilities = {"home_win": prediction["prob_home_win"], "draw": prediction["prob_draw"], "away_win": prediction["prob_away_win"], "over_2_5": prediction["prob_over_2_5"], "btts_yes": prediction["prob_btts"]}
        if isinstance(prediction.get("market_probabilities"), dict):
            probabilities.update(prediction["market_probabilities"])
        assessments = assess_market_value(probabilities, quote["odds"])
        if assessments:
            best = max(assessments, key=lambda item: item.expected_value)
            kickoff = datetime.fromisoformat(str(match["match_date"]).replace("Z", "+00:00")).astimezone(TZ)
            result.append({"match_id": int(match["id"]), "match": f"{teams.get(int(match['home_team_id']), 'Ev sahibi')} — {teams.get(int(match['away_team_id']), 'Deplasman')}", "time": kickoff.strftime("%H:%M"), "league": leagues.get(int(match["league_id"]), "Lig"), "best": best, "assessments": assessments})
    return sorted(result, key=lambda row: row["best"].expected_value, reverse=True)


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
    low = diversified_coupon_rows(rows, max_items=None, min_probability=0.55, min_ev=0.0)
    blocked = {(int(row["match_id"]), _market_family(str(row["best"].key))) for row in low}
    balanced = diversified_coupon_rows(
        rows, max_items=None, min_ev=0.03, excluded_match_families=blocked
    )
    blocked.update(
        (int(row["match_id"]), _market_family(str(row["best"].key)))
        for row in balanced
    )
    high = diversified_coupon_rows(
        rows, max_items=None, min_ev=0.03, high_odds=True,
        excluded_match_families=blocked,
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
