"""Send a pipeline failure alert through Telegram when configured."""

from __future__ import annotations

import argparse
import logging
import os

import requests

from config.settings import PROJECT_ROOT, _load_env_file

LOG = logging.getLogger(__name__)


def send_telegram_alert(message: str) -> bool:
    """Send a best-effort alert; missing credentials never fail the pipeline."""
    _load_env_file(PROJECT_ROOT / ".env")
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        LOG.info("Telegram failure alert skipped; credentials are not configured")
        return False

    try:
        response = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": message[:4000]},
            timeout=10,
        )
        response.raise_for_status()
        if not response.json().get("ok"):
            LOG.error("Telegram rejected the pipeline failure alert")
            return False
    except requests.RequestException:
        LOG.exception("Could not deliver the pipeline failure alert")
        return False
    LOG.info("Pipeline failure alert sent")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("message", nargs="+", help="Alert message without secrets")
    args = parser.parse_args()
    send_telegram_alert(" ".join(args.message))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
