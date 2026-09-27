"""Settle persisted daily coupons and notify Telegram once per coupon."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from config.settings import get_settings
from db.db_client import SupabaseRestClient
from notifications.telegram import send_telegram_message


def settle_daily_coupons() -> dict[str, int]:
    settings = get_settings()
    db = SupabaseRestClient(settings.supabase_url, settings.supabase_service_role_key)
    pending = db.select_all("daily_coupon_runs", columns="id,coupon_date,coupon_type,selections,total_odds,status,result_message_sent_at", filters={"status": "eq.pending", "result_message_sent_at": "is.null"})
    matches_checked = 0
    sent = 0
    token, chat_id = os.getenv("TELEGRAM_BOT_TOKEN", "").strip(), os.getenv("TELEGRAM_CHAT_ID", "").strip()
    for coupon in pending:
        selections = coupon.get("selections") or []
        ids = ",".join(str(int(item["match_id"])) for item in selections)
        if not ids:
            continue
        matches = db.select_all("matches", columns="id,status,home_score,away_score", filters={"id": f"in.({ids})"})
        matches_checked += len(matches)
        if len(matches) != len(selections) or any(row.get("status") != "finished" or row.get("home_score") is None or row.get("away_score") is None for row in matches):
            continue
        by_id = {int(row["id"]): row for row in matches}
        outcomes: list[str] = []
        won = True
        for item in selections:
            match = by_id[int(item["match_id"])]
            home, away = int(match["home_score"]), int(match["away_score"])
            key = item["key"]
            hit = ((key == "home_win" and home > away) or (key == "draw" and home == away) or (key == "away_win" and away > home) or (key == "over_2_5" and home + away >= 3) or (key == "btts_yes" and home > 0 and away > 0))
            won = won and hit
            outcomes.append(f"{'✅' if hit else '❌'} {item.get('label', key)} ({home}-{away})")
        status = "won" if won else "lost"
        now = datetime.now(timezone.utc).isoformat()
        db.update("daily_coupon_runs", {"status": status, "settled_at": now}, filters={"id": f"eq.{coupon['id']}"})
        if token and chat_id:
            title = "tuttu ✅" if won else "yattı ❌"
            message = "🎟️ Günlük kupon sonucu\n" + f"{coupon['coupon_type']} · {title}\nToplam oran: {float(coupon['total_odds']):.2f}\n" + "\n".join(outcomes)
            send_telegram_message(message, bot_token=token, chat_id=chat_id)
            db.update("daily_coupon_runs", {"result_message_sent_at": now}, filters={"id": f"eq.{coupon['id']}"})
            sent += 1
    return {"coupons": len(pending), "matches_checked": matches_checked, "messages_sent": sent}


if __name__ == "__main__":
    print(json.dumps(settle_daily_coupons(), ensure_ascii=False))
