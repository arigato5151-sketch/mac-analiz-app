"""Fetch missing post-match statistics for archived fixtures from API-Football.

The job is resumable: successfully checked fixtures receive
``expected_metrics_checked_at`` even when the provider has no optional metrics.
It respects the provider's per-minute and daily request limits.
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from typing import Any

from config.settings import get_settings
from data_pipeline.api_client import ApiFootballClient, ApiFootballError
from data_pipeline.fetch_results import extract_corner_counts, extract_expected_metrics
from db.db_client import SupabaseRestClient

STAT_COLUMNS = (
    "home_xg", "away_xg", "home_corners", "away_corners", "home_shots",
    "away_shots", "home_shots_on_target", "away_shots_on_target",
    "home_possession", "away_possession",
)


def backfill_match_statistics(
    api: ApiFootballClient,
    db: SupabaseRestClient,
    *,
    max_requests: int = 7_200,
    daily_reserve: int = 200,
    request_interval: float = 0.26,
) -> dict[str, Any]:
    if max_requests < 1 or daily_reserve < 0 or request_interval < 0.2:
        raise ValueError("Invalid quota or rate-limit settings")

    matches = db.select_all(
        "matches",
        columns=(
            "id,home_team_id,away_team_id,home_xg,away_xg,home_corners,away_corners,"
            "home_shots,away_shots,home_shots_on_target,away_shots_on_target,"
            "home_possession,away_possession,expected_metrics_checked_at"
        ),
        filters={"status": "eq.finished"},
        order="id.asc",
    )
    pending = [
        row for row in matches
        if row.get("expected_metrics_checked_at") is None
        and any(row.get(column) is None for column in STAT_COLUMNS)
    ]

    checked = updated = unavailable = errors = 0
    started = datetime.now(timezone.utc).isoformat()
    for row in pending:
        if checked >= max_requests:
            break
        remaining = api.diagnostics().get("rate_limit", {})
        remaining = next(
            (value for key, value in remaining.items() if "requests-remaining" in key.lower()),
            None,
        )
        if remaining is not None and int(remaining) <= daily_reserve:
            break
        if checked:
            time.sleep(request_interval)
        fixture_id = int(row["id"])
        try:
            payload = api.get("fixtures/statistics", {"fixture": fixture_id})
        except ApiFootballError as exc:
            errors += 1
            print(f"API error for fixture {fixture_id}: {exc}", flush=True)
            # Leave the timestamp empty so a later run can retry this fixture.
            continue

        metrics = extract_expected_metrics(
            payload,
            home_team_id=int(row["home_team_id"]),
            away_team_id=int(row["away_team_id"]),
        )
        corners = extract_corner_counts(
            payload,
            home_team_id=int(row["home_team_id"]),
            away_team_id=int(row["away_team_id"]),
        )
        if corners is not None:
            metrics["home_corners"], metrics["away_corners"] = corners

        values = {
            key: value for key, value in metrics.items()
            if value is not None and row.get(key) is None
        }
        values["expected_metrics_checked_at"] = datetime.now(timezone.utc).isoformat()
        db.update("matches", values, filters={"id": f"eq.{fixture_id}"})
        checked += 1
        if values.keys() - {"expected_metrics_checked_at"}:
            updated += 1
        else:
            unavailable += 1
        if checked % 100 == 0:
            print(
                f"Progress: checked={checked}, updated={updated}, "
                f"no_supported_stats={unavailable}, pending={len(pending)-checked}, "
                f"api={api.diagnostics()}",
                flush=True,
            )

    return {
        "started_at": started,
        "pending_at_start": len(pending),
        "checked": checked,
        "rows_updated": updated,
        "no_supported_stats": unavailable,
        "errors": errors,
        "remaining_pending_estimate": max(0, len(pending) - checked),
        "api": api.diagnostics(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-requests", type=int, default=7_200)
    parser.add_argument("--daily-reserve", type=int, default=200)
    args = parser.parse_args()
    settings = get_settings()
    summary = backfill_match_statistics(
        ApiFootballClient(settings.api_football_key),
        SupabaseRestClient(settings.supabase_url, settings.supabase_service_role_key),
        max_requests=args.max_requests,
        daily_reserve=args.daily_reserve,
    )
    print(summary, flush=True)


if __name__ == "__main__":
    main()
