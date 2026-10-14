"""Send a completion status for a locally scheduled pipeline run."""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from zoneinfo import ZoneInfo

from config.settings import _load_env_file, PROJECT_ROOT
from notifications.telegram import send_telegram_message


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", choices=("success", "failure"), required=True)
    parser.add_argument("--mode", choices=("daily", "odds-refresh"), required=True)
    parser.add_argument("--message", default="")
    args = parser.parse_args()

    _load_env_file(PROJECT_ROOT / ".env")
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        raise RuntimeError("Telegram bot token/chat ID missing from .env")

    mode = "Günlük ana akış" if args.mode == "daily" else "Nesine oran yenileme"
    now = datetime.now(ZoneInfo("Europe/Istanbul")).strftime("%d.%m.%Y %H:%M:%S")
    if args.status == "success":
        message = f"✅ Maç Analiz · Yerel iş tamamlandı\nİş: {mode}\nBitiş: {now} (İstanbul)"
    else:
        detail = " ".join(args.message.split())[:500] or "Ayrıntı log dosyasında."
        message = f"❌ Maç Analiz · Yerel iş başarısız\nİş: {mode}\nZaman: {now} (İstanbul)\nHata: {detail}"
    send_telegram_message(message, bot_token=token, chat_id=chat_id)
    print(f"Telegram status sent: {args.status}, {args.mode}")


if __name__ == "__main__":
    main()
