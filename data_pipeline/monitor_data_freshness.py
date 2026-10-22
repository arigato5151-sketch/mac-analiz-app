"""Report upcoming fixture and bookmaker-quote coverage without modifying data."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from config.settings import get_settings
from db.db_client import SupabaseRestClient

ISTANBUL = ZoneInfo("Europe/Istanbul")


def collect_data_health(
    db: SupabaseRestClient, *, now: datetime | None = None, days: int = 7
) -> dict[str, Any]:
    """Summarize upcoming fixtures, quote coverage, and quote freshness."""
    reference = now or datetime.now(timezone.utc)
    if reference.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    start = reference.astimezone(timezone.utc)
    end = start + timedelta(days=days)
    matches = db.select_all(
        "matches",
        columns="id,match_date,status",
        filters={
            "match_date": f"gte.{start.isoformat()}",
            "and": f"(match_date.lt.{end.isoformat()},status.eq.scheduled)",
        },
        order="match_date.asc,id.asc",
    )
    match_ids = [int(row["id"]) for row in matches]
    quotes: list[dict[str, Any]] = []
    for offset in range(0, len(match_ids), 100):
        chunk = match_ids[offset : offset + 100]
        quotes.extend(
            db.select_all(
                "odds_quote_history",
                columns="id,match_id,captured_at",
                filters={
                    "bookmaker": "eq.iddaa.com",
                    "match_id": f"in.({','.join(map(str, chunk))})",
                },
                order="captured_at.desc,id.desc",
            )
        )

    covered_ids = {int(row["match_id"]) for row in quotes}
    latest_values = [
        datetime.fromisoformat(str(row["captured_at"]).replace("Z", "+00:00"))
        for row in quotes
        if row.get("captured_at")
    ]
    latest = max(latest_values) if latest_values else None
    return {
        "window_start": start,
        "window_end": end,
        "scheduled_matches": len(matches),
        "matches_with_iddaa_odds": len(covered_ids),
        "matches_without_iddaa_odds": len(match_ids) - len(covered_ids),
        "latest_iddaa_quote_at": latest,
        "quote_age_hours": (
            round((reference.astimezone(timezone.utc) - latest).total_seconds() / 3600, 1)
            if latest
            else None
        ),
    }


def main() -> None:
    settings = get_settings()
    db = SupabaseRestClient(settings.supabase_url, settings.supabase_service_role_key)
    health = collect_data_health(db)
    start = health["window_start"].astimezone(ISTANBUL).strftime("%d.%m.%Y")
    end = (health["window_end"] - timedelta(seconds=1)).astimezone(ISTANBUL).strftime("%d.%m.%Y")
    latest = health["latest_iddaa_quote_at"]
    latest_text = latest.astimezone(ISTANBUL).strftime("%d.%m.%Y %H:%M") if latest else "yok"
    print(
        {
            **health,
            "window_start": health["window_start"].isoformat(),
            "window_end": health["window_end"].isoformat(),
            "latest_iddaa_quote_at": latest.isoformat() if latest else None,
        }
    )

    summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as summary_file:
            summary_file.write(
                "\n## Veri güncelliği ve oran kapsamı\n"
                f"- Kontrol aralığı: {start}–{end} (İstanbul)\n"
                f"- Planlanmış maç: {health['scheduled_matches']}\n"
                f"- IdDaa oranı bulunan maç: {health['matches_with_iddaa_odds']}\n"
                f"- IdDaa oranı bulunmayan maç: {health['matches_without_iddaa_odds']}\n"
                f"- Son oran kaydı: {latest_text} (İstanbul)\n"
                f"- Son oran kaydının yaşı: {health['quote_age_hours']} saat\n"
            )


if __name__ == "__main__":
    main()
