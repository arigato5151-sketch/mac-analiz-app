"""Print aggregate operational-event diagnostics without exposing event payloads."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timedelta, timezone

from config.settings import get_settings
from db.db_client import SupabaseRestClient


def build_report(rows: list[dict[str, object]]) -> dict[str, object]:
    """Create a compact, non-sensitive summary suitable for CI logs."""
    by_severity = Counter(str(row.get("severity", "unknown")) for row in rows)
    by_component = Counter(str(row.get("component", "unknown")) for row in rows)
    latest_warning = next(
        (
            {
                "occurred_at": row.get("occurred_at"),
                "severity": row.get("severity"),
                "component": row.get("component"),
                "event_type": row.get("event_type"),
            }
            for row in rows
            if row.get("severity") in {"warning", "error"}
        ),
        None,
    )
    return {
        "event_count": len(rows),
        "by_severity": dict(by_severity),
        "by_component": dict(by_component),
        "latest_warning_or_error": latest_warning,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=int, default=24)
    args = parser.parse_args()
    if args.hours < 1 or args.hours > 24 * 31:
        parser.error("--hours must be between 1 and 744")

    settings = get_settings()
    db = SupabaseRestClient(settings.supabase_url, settings.supabase_service_role_key)
    cutoff = datetime.now(timezone.utc) - timedelta(hours=args.hours)
    rows = db.select_all(
        "operational_events",
        columns="occurred_at,severity,component,event_type",
        filters={"occurred_at": f"gte.{cutoff.isoformat()}"},
        order="occurred_at.desc",
    )
    print(json.dumps(build_report(rows), ensure_ascii=False))


if __name__ == "__main__":
    main()
