"""Fetch a date's fixtures and persist completed results."""

from __future__ import annotations

import argparse
import re
import time
from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from config.leagues import TRACKED_LEAGUE_IDS
from config.settings import get_settings
from data_pipeline.evaluate_predictions import evaluate_completed_predictions
from data_pipeline.api_client import ApiFootballClient, ApiFootballError
from data_pipeline.fetch_fixtures import SyncSummary, transform_fixtures
from db.db_client import SupabaseRestClient

EXPECTED_GOALS_STAT_TYPES = frozenset({"expected_goals", "xg"})
EXPECTED_ASSISTS_STAT_TYPES = frozenset(
    {"expected_assists", "expected_assist", "xa", "x_a"}
)
TACTICAL_STAT_TYPES = {
    "shots": frozenset({"total_shots", "shots"}),
    "shots_on_target": frozenset({"shots_on_goal", "shots_on_target"}),
    "possession": frozenset({"ball_possession", "possession"}),
}
STALE_ACTIVE_GRACE = timedelta(hours=4)
MAX_STALE_ACTIVE_RECONCILIATIONS = 4
# The scheduled results job runs twice a day. Keep this retry budget bounded,
# while allowing the last week's late-arriving fixture stats to be revisited.
MAX_EXPECTED_METRIC_REQUESTS_PER_RUN = 50
EXPECTED_METRICS_RECHECK_HOURS = 12
API_REQUEST_DAILY_RESERVE = 200
API_REQUEST_INTERVAL_SECONDS = 0.26


def _statistic_key(value: object) -> str:
    """Normalize provider labels without relying on an exact spelling."""
    return "_".join(str(value).strip().lower().replace("-", " ").split())


def _non_negative_float(value: object) -> float | None:
    """Return a valid provider metric or ignore unavailable/non-numeric values."""
    try:
        parsed = float(str(value).replace("%", "").replace(",", ".").strip())
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def sync_results(
    api: ApiFootballClient, db: SupabaseRestClient, *, match_date: date
) -> SyncSummary:
    """Apply completed scores to existing fixtures without changing the bulletin."""
    api_fixtures = api.get(
        "fixtures",
        {"date": match_date.isoformat(), "timezone": "Europe/Istanbul"},
    )
    _, transformed = transform_fixtures(api_fixtures)
    tracked = [
        row
        for row in transformed
        if int(row["league_id"]) in TRACKED_LEAGUE_IDS
    ]
    finished = [row for row in tracked if row["status"] == "finished"]
    fixture_ids = sorted({int(row["id"]) for row in finished})
    existing_ids: set[int] = set()
    for offset in range(0, len(fixture_ids), 100):
        chunk = fixture_ids[offset : offset + 100]
        existing_ids.update(
            int(row["id"])
            for row in db.select_all(
                "matches",
                columns="id",
                filters={"id": f"in.({','.join(map(str, chunk))})"},
            )
        )
    updates = [
        {
            "id": int(row["id"]),
            "status": "finished",
            "home_score": row["home_score"],
            "away_score": row["away_score"],
            "home_halftime_score": row["home_halftime_score"],
            "away_halftime_score": row["away_halftime_score"],
        }
        for row in finished
        if int(row["id"]) in existing_ids
    ]
    for row in updates:
        match_id = row.pop("id")
        db.update("matches", row, filters={"id": f"eq.{match_id}"})
    day = match_date.isoformat()
    return SyncSummary(
        requested_start=day,
        requested_end=day,
        completed_start=day,
        completed_end=day,
        skipped_start=None,
        skipped_end=None,
        is_complete=True,
        api_fixtures=len(api_fixtures),
        tracked_fixtures=len(tracked),
        leagues_upserted=0,
        teams_upserted=0,
        matches_upserted=len(updates),
    )


def sync_recent_results(
    api: ApiFootballClient,
    db: SupabaseRestClient,
    *,
    today: date,
    lookback_days: int,
) -> SyncSummary:
    """Refresh completed results while leaving fixture dates to the bulletin."""
    if lookback_days < 0 or lookback_days > 14:
        raise ValueError("lookback_days must be between 0 and 14")
    requested_start = today - timedelta(days=lookback_days)
    daily: list[SyncSummary] = []
    skipped_days: list[date] = []
    cursor = requested_start
    while cursor <= today:
        try:
            daily.append(sync_results(api, db, match_date=cursor))
            cursor += timedelta(days=1)
        except ApiFootballError as exc:
            plan_window = re.search(
                r"Free plans do not have access to this date, try from "
                r"(\d{4}-\d{2}-\d{2}) to (\d{4}-\d{2}-\d{2})",
                str(exc),
            )
            if not plan_window:
                raise
            allowed_start = date.fromisoformat(plan_window.group(1))
            allowed_end = date.fromisoformat(plan_window.group(2))
            if cursor < allowed_start:
                last_skipped = min(today, allowed_start - timedelta(days=1))
                skipped_days.extend(
                    cursor + timedelta(days=offset)
                    for offset in range((last_skipped - cursor).days + 1)
                )
                cursor = allowed_start
                continue
            if cursor > allowed_end:
                skipped_days.extend(
                    cursor + timedelta(days=offset)
                    for offset in range((today - cursor).days + 1)
                )
                break
            raise
    summary = SyncSummary(
        requested_start=requested_start.isoformat(),
        requested_end=today.isoformat(),
        completed_start=daily[0].completed_start if daily else None,
        completed_end=daily[-1].completed_end if daily else None,
        skipped_start=min(skipped_days).isoformat() if skipped_days else None,
        skipped_end=max(skipped_days).isoformat() if skipped_days else None,
        is_complete=not skipped_days,
        api_fixtures=sum(item.api_fixtures for item in daily),
        tracked_fixtures=sum(item.tracked_fixtures for item in daily),
        leagues_upserted=0,
        teams_upserted=0,
        matches_upserted=sum(item.matches_upserted for item in daily),
    )
    sync_recent_expected_metrics(api, db, today=today, lookback_days=lookback_days)
    return summary


def reconcile_stale_active_fixtures(
    api: ApiFootballClient,
    db: SupabaseRestClient,
    *,
    now: datetime | None = None,
) -> int:
    """Reconcile stale active fixtures by ID to prevent date changes from stranding them.

    Problem: Some matches remain stuck in `scheduled` or `live` status long after
    their scheduled kickoff time (e.g., postponed, cancelled, or API-Football
    date corrections). The standard date-based sweep only looks forward, so
    these stale fixtures are never re-fetched and their status/scores never
    update.

    This function:
    1. Finds matches with status `scheduled` or `live` whose `match_date`
       is older than `now - STALE_ACTIVE_GRACE` (default 4 hours).
    2. Re-fetches each fixture by ID from API-Football.
    3. Updates only the mutable match state fields (`match_date`, `status`,
       `home_score`, `away_score`) — teams and league are immutable once created.
    4. Limits reconciliation to `MAX_STALE_ACTIVE_RECONCILIATIONS` fixtures per
       run (default 20) to bound API quota usage.

    Args:
        api: API-Football client.
        db: Supabase REST client with service-role write access.
        now: Reference time (timezone-aware). Defaults to `datetime.now(timezone.utc)`.

    Returns:
        Number of fixtures successfully reconciled.

    Notes:
        - Single provider anomalies are logged but do not halt the batch.
        - Fixtures that 404 or disappear from the API are skipped silently.
        - The 4-hour grace window (`STALE_ACTIVE_GRACE`) balances between
          catching genuine date changes and avoiding premature re-fetching of
          matches that are merely delayed.
        - The 20-fixture cap (`MAX_STALE_ACTIVE_RECONCILIATIONS`) bounds
          API quota consumption per workflow run.
    """
    reference_time = now or datetime.now(timezone.utc)
    if reference_time.tzinfo is None:
        raise ValueError("now must be timezone-aware")

    stale_matches = db.select_all(
        "matches",
        columns="id",
        filters={
            "status": "in.(scheduled,live)",
            "match_date": f"lte.{(reference_time - STALE_ACTIVE_GRACE).isoformat()}",
        },
        order="match_date.asc",
    )
    reconciled = 0
    for stale_match in stale_matches[:MAX_STALE_ACTIVE_RECONCILIATIONS]:
        fixture_id = int(stale_match["id"])
        try:
            payload = api.get("fixtures", {"id": fixture_id})
            _, matches = transform_fixtures(payload)
            current = next(
                (match for match in matches if int(match["id"]) == fixture_id), None
            )
            if current is None:
                continue

            # Patch only mutable match state; teams and league already exist.
            db.update(
                "matches",
                {
                    "match_date": current["match_date"],
                    "status": current["status"],
                    "home_score": current["home_score"],
                    "away_score": current["away_score"],
                    "home_halftime_score": current.get("home_halftime_score"),
                    "away_halftime_score": current.get("away_halftime_score"),
                },
                filters={"id": f"eq.{fixture_id}"},
            )
            reconciled += 1
        except Exception as error:
            # A single provider anomaly must not block the normal date-based sweep.
            print(f"Stale fixture {fixture_id} skipped: {type(error).__name__}")
    return reconciled


def extract_expected_goals(
    payload: list[dict[str, Any]],
    *,
    home_team_id: int,
    away_team_id: int,
) -> tuple[float, float] | None:
    """Read API-Football xG when the plan/provider exposes the statistic."""
    metrics = extract_expected_metrics(
        payload, home_team_id=home_team_id, away_team_id=away_team_id
    )
    home_xg = metrics["home_xg"]
    away_xg = metrics["away_xg"]
    if home_xg is None or away_xg is None:
        return None
    return home_xg, away_xg


def extract_expected_metrics(
    payload: list[dict[str, Any]],
    *,
    home_team_id: int,
    away_team_id: int,
) -> dict[str, float | int | None]:
    """Read expected and basic tactical statistics without filling missing values.

    API-Football availability differs by competition and plan. Missing expected-assist
    statistics are intentionally preserved as ``None`` rather than inferred from
    actual assists, which would leak post-match outcomes into a predictive feature.
    """
    values: dict[int, dict[str, float]] = {}
    for team_stats in payload:
        team_id = (team_stats.get("team") or {}).get("id")
        if team_id not in {home_team_id, away_team_id}:
            continue
        for stat in team_stats.get("statistics", []):
            kind = _statistic_key(stat.get("type", ""))
            metric = (
                "xg"
                if kind in EXPECTED_GOALS_STAT_TYPES
                else "xa" if kind in EXPECTED_ASSISTS_STAT_TYPES else None
            )
            if metric is None:
                metric = next(
                    (name for name, labels in TACTICAL_STAT_TYPES.items() if kind in labels),
                    None,
                )
            if metric is None:
                continue
            value = _non_negative_float(stat.get("value"))
            if metric == "possession" and value is not None and value > 100:
                value = None
            if value is not None:
                values.setdefault(int(team_id), {})[metric] = (
                    int(value) if metric in {"shots", "shots_on_target"} else value
                )

    return {
        f"{side}_{metric}": values.get(team_id, {}).get(metric)
        for side, team_id in (("home", home_team_id), ("away", away_team_id))
        for metric in ("xg", "xa", "shots", "shots_on_target", "possession")
    }


def extract_corner_counts(
    payload: list[dict[str, Any]], *, home_team_id: int, away_team_id: int
) -> tuple[int, int] | None:
    """Extract full-time corners for both fixture teams, preserving missing data."""
    values: dict[int, int] = {}
    for team_stats in payload:
        team_id = (team_stats.get("team") or {}).get("id")
        if team_id not in {home_team_id, away_team_id}:
            continue
        for stat in team_stats.get("statistics", []):
            if _statistic_key(stat.get("type", "")) not in {"corner_kicks", "corner_kick", "corners"}:
                continue
            value = _non_negative_float(stat.get("value"))
            if value is not None and value.is_integer():
                values[int(team_id)] = int(value)
    if home_team_id not in values or away_team_id not in values:
        return None
    return values[home_team_id], values[away_team_id]


def sync_recent_expected_metrics(
    api: ApiFootballClient,
    db: SupabaseRestClient,
    *,
    today: date,
    lookback_days: int,
) -> int:
    """Persist available xG/xA for recent finished fixtures; missing data is harmless."""
    istanbul = ZoneInfo("Europe/Istanbul")
    start = datetime.combine(
        today - timedelta(days=lookback_days), datetime.min.time(), tzinfo=istanbul
    ).astimezone(timezone.utc).isoformat()
    end = datetime.combine(
        today + timedelta(days=1), datetime.min.time(), tzinfo=istanbul
    ).astimezone(timezone.utc).isoformat()
    matches = db.select_all(
        "matches",
        columns=(
            "id,home_team_id,away_team_id,home_xg,away_xg,home_xa,away_xa,"
            "expected_metrics_checked_at,home_corners,away_corners,home_shots,away_shots,"
            "home_shots_on_target,away_shots_on_target,home_possession,away_possession"
        ),
        filters={
            "status": "eq.finished",
            "and": f"(match_date.gte.{start},match_date.lt.{end})",
        },
        order="match_date.desc,id.desc",
    )
    api_requests = 0
    rows_updated = 0
    no_supported_stats = 0
    for match in matches:
        checked_at = match.get("expected_metrics_checked_at")
        already_complete = all(
            match.get(column) is not None
            for column in (
                # API-Football does not consistently return xA. Do not let that
                # unsupported optional metric trigger repeated lookups by itself.
                "home_xg", "away_xg", "home_corners", "away_corners",
                "home_shots", "away_shots", "home_shots_on_target", "away_shots_on_target",
                "home_possession", "away_possession",
            )
        )
        # Recent finished fixtures can gain stats after the first fetch. Retry
        # incomplete rows after a cooldown, but only while inside this function's
        # bounded lookback window.
        recently_checked = checked_at is not None and datetime.fromisoformat(
            str(checked_at).replace("Z", "+00:00")
        ) >= datetime.now(ZoneInfo("UTC")) - timedelta(
            hours=EXPECTED_METRICS_RECHECK_HOURS
        )
        if already_complete or recently_checked:
            continue
        if api_requests >= MAX_EXPECTED_METRIC_REQUESTS_PER_RUN:
            print(
                "Deferring remaining expected-metric lookups to a later run "
                f"after {MAX_EXPECTED_METRIC_REQUESTS_PER_RUN} API requests."
            )
            break
        rate_limit = api.diagnostics().get("rate_limit", {})
        daily_remaining = next(
            (
                value
                for key, value in rate_limit.items()
                if "requests-remaining" in key.lower()
            ),
            None,
        )
        if daily_remaining is not None and int(daily_remaining) <= API_REQUEST_DAILY_RESERVE:
            print(
                "Deferring expected-metric lookups to preserve the daily API reserve "
                f"of {API_REQUEST_DAILY_RESERVE} requests."
            )
            break
        if api_requests:
            time.sleep(API_REQUEST_INTERVAL_SECONDS)
        try:
            api_requests += 1
            statistics = api.get("fixtures/statistics", {"fixture": int(match["id"])})
        except Exception as error:
            # Keep final scores even when optional xG/xA is unavailable.
            print(
                f"Expected-metric fetch skipped for fixture {int(match['id'])}: {type(error).__name__}"
            )
            continue
        metrics = extract_expected_metrics(
            statistics,
            home_team_id=int(match["home_team_id"]),
            away_team_id=int(match["away_team_id"]),
        )
        corners = extract_corner_counts(
            statistics,
            home_team_id=int(match["home_team_id"]),
            away_team_id=int(match["away_team_id"]),
        )
        if corners is not None:
            metrics["home_corners"] = corners[0]
            metrics["away_corners"] = corners[1]
        values = {
            column: value
            for column, value in metrics.items()
            if value is not None and match.get(column) is None
        }
        for column in tuple(values):
            if column.endswith(("_xg", "_xa")):
                values[f"{column}_source"] = "api-football"
        values["expected_metrics_checked_at"] = datetime.now(
            ZoneInfo("UTC")
        ).isoformat()
        # These rows already exist; PATCH avoids re-validating required insert fields.
        db.update("matches", values, filters={"id": f"eq.{int(match['id'])}"})
        if len(values) > 1:
            rows_updated += 1
        else:
            no_supported_stats += 1
    print(
        "Recent match-stat retry summary: "
        f"requests={api_requests}, updated={rows_updated}, "
        f"no_new_supported_stats={no_supported_stats}"
    )
    return api_requests


def sync_recent_xg(
    api: ApiFootballClient,
    db: SupabaseRestClient,
    *,
    today: date,
    lookback_days: int,
) -> int:
    """Backward-compatible name for callers that only knew about xG."""
    return sync_recent_expected_metrics(
        api, db, today=today, lookback_days=lookback_days
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--date",
        type=date.fromisoformat,
        help="Reconcile one date instead of the rolling lookback window.",
    )
    parser.add_argument("--lookback-days", type=int, default=7)
    args = parser.parse_args()
    settings = get_settings()
    api = ApiFootballClient(settings.api_football_key)
    db = SupabaseRestClient(settings.supabase_url, settings.supabase_service_role_key)
    if args.date:
        summary = sync_results(api, db, match_date=args.date)
        evaluation = evaluate_completed_predictions(
            db, today=args.date, lookback_days=0
        )
    else:
        today = datetime.now(ZoneInfo("Europe/Istanbul")).date()
        summary = sync_recent_results(
            api,
            db,
            today=today,
            lookback_days=args.lookback_days,
        )
        # Scores can arrive late or an earlier evaluator run can be missed.
        # Revisit two weeks of already stored finished fixtures on every run.
        evaluation = evaluate_completed_predictions(
            db, today=today, lookback_days=max(args.lookback_days, 14)
        )
    diagnostics = api.diagnostics()
    print({"sync": asdict(summary), "evaluation": evaluation, "api": diagnostics})


if __name__ == "__main__":
    main()
