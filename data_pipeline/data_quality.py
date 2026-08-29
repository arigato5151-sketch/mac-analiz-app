"""Validate morning fixture, prediction, form, and availability freshness."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from typing import Any

from config.settings import get_settings
from db.db_client import SupabaseRestClient


FRESHNESS_LIMIT = timedelta(hours=36)


def assess_morning_quality(
    scheduled_matches: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    team_forms: list[dict[str, Any]],
    availability_snapshots: list[dict[str, Any]],
    *,
    now: datetime,
) -> dict[str, Any]:
    """Return an auditable quality report and fail only on actionable gaps."""
    target_team_ids = {
        int(team_id)
        for match in scheduled_matches
        for team_id in (match["home_team_id"], match["away_team_id"])
    }
    predicted_match_ids = {int(row["match_id"]) for row in predictions}
    missing_predictions = sorted(
        int(match["id"]) for match in scheduled_matches if int(match["id"]) not in predicted_match_ids
    )

    def fresh_team_ids(rows: list[dict[str, Any]], timestamp_key: str) -> set[int]:
        fresh: set[int] = set()
        for row in rows:
            timestamp = row.get(timestamp_key)
            if not timestamp:
                continue
            refreshed_at = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
            if now - refreshed_at <= FRESHNESS_LIMIT:
                fresh.add(int(row["team_id"]))
        return fresh

    fresh_forms = fresh_team_ids(team_forms, "calculated_at")
    fresh_availability = fresh_team_ids(availability_snapshots, "refreshed_at")
    missing_forms = sorted(target_team_ids - fresh_forms)
    missing_availability = sorted(target_team_ids - fresh_availability)
    return {
        "scheduled_matches": len(scheduled_matches),
        "target_teams": len(target_team_ids),
        "predictions": len(predicted_match_ids),
        "missing_prediction_ids": missing_predictions,
        "stale_form_team_ids": missing_forms,
        "stale_availability_team_ids": missing_availability,
        "healthy": not (missing_predictions or missing_forms or missing_availability),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=3)
    args = parser.parse_args()
    if args.days < 1 or args.days > 14:
        parser.error("--days must be between 1 and 14")

    settings = get_settings()
    db = SupabaseRestClient(settings.supabase_url, settings.supabase_service_role_key)
    now = datetime.now(timezone.utc)
    end = now + timedelta(days=args.days)
    scheduled = db.select_all(
        "matches",
        columns="id,home_team_id,away_team_id,match_date",
        filters={
            "status": "eq.scheduled",
            "and": f"(match_date.gte.{now.isoformat()},match_date.lte.{end.isoformat()})",
        },
    )
    predictions = db.select_all("predictions", columns="match_id,predicted_at")
    forms = db.select_all("team_form", columns="team_id,calculated_at")
    availability = db.select_all(
        "team_availability_status", columns="team_id,refreshed_at"
    )
    report = assess_morning_quality(scheduled, predictions, forms, availability, now=now)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["healthy"]:
        raise RuntimeError("Morning data-quality validation failed")


if __name__ == "__main__":
    main()
