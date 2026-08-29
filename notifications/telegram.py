"""Small, failure-safe Telegram Bot API client."""

from __future__ import annotations

import os

import requests


class TelegramError(RuntimeError):
    """Raised when Telegram rejects a notification request."""


def send_telegram_message(
    text: str,
    *,
    bot_token: str,
    chat_id: str,
    session: requests.Session | None = None,
) -> None:
    """Send a plain-text message without including credentials in errors."""
    if not bot_token or not chat_id:
        raise ValueError("Telegram bot token and chat ID are required")
    if not text.strip():
        raise ValueError("Telegram message must not be empty")

    response = (session or requests.Session()).post(
        f"https://api.telegram.org/bot{bot_token}/sendMessage",
        json={"chat_id": chat_id, "text": text, "disable_web_page_preview": True},
        timeout=20,
    )
    if not response.ok:
        raise TelegramError(f"Telegram request failed ({response.status_code})")
    payload = response.json()
    if not payload.get("ok"):
        raise TelegramError("Telegram rejected the notification request")


def send_from_environment(text: str) -> bool:
    """Send when configured; return False when notifications are intentionally disabled."""
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not bot_token or not chat_id:
        return False
    send_telegram_message(text, bot_token=bot_token, chat_id=chat_id)
    return True
