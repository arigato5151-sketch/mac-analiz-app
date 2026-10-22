"""Report the size of the retained IdDaa odds history without downloading it."""

from __future__ import annotations

import logging
import os
import re

import requests

from config.settings import get_supabase_admin_settings
from data_pipeline.notify_failure import send_telegram_alert

LOG = logging.getLogger(__name__)
CONTENT_RANGE = re.compile(r"(?:\d+-\d+|\*)/(\d+|\*)")


def count_odds_history() -> int:
    settings = get_supabase_admin_settings()
    headers = {
        "apikey": settings.supabase_service_role_key,
        "Prefer": "count=exact",
        "Range": "0-0",
    }
    if settings.supabase_service_role_key.count(".") == 2:
        headers["Authorization"] = f"Bearer {settings.supabase_service_role_key}"
    response = requests.get(
        f"{settings.supabase_url}/rest/v1/odds_quote_history",
        params={"select": "id"},
        headers=headers,
        timeout=settings.request_timeout_seconds,
    )
    response.raise_for_status()
    content_range = response.headers.get("Content-Range", "")
    match = CONTENT_RANGE.fullmatch(content_range)
    if not match or match.group(1) == "*":
        raise RuntimeError(f"Supabase returned an unexpected Content-Range: {content_range!r}")
    return int(match.group(1))


def main() -> None:
    row_count = count_odds_history()
    print({"table": "odds_quote_history", "rows": row_count})
    summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as summary_file:
            summary_file.write(
                f"\n## Oran geçmişi boyutu\n- `odds_quote_history`: {row_count:,} satır\n"
            )

    raw_threshold = os.getenv("ODDS_HISTORY_ALERT_ROWS", "100000").strip()
    if raw_threshold:
        threshold = int(raw_threshold)
        if threshold > 0 and row_count >= threshold:
            message = (
                f"odds_quote_history {row_count:,} satıra ulaştı; "
                f"tanımlı uyarı eşiği {threshold:,}. Geçmiş kayıtları gözden geçirin."
            )
            LOG.warning(message)
            send_telegram_alert(message)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
