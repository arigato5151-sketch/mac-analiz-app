"""Fetch a date's fixtures and persist completed results."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from config.settings import get_settings
from data_pipeline.api_client import ApiFootballClient
from data_pipeline.fetch_fixtures import SyncSummary, sync_fixtures
from db.db_client import SupabaseRestClient
from monitoring.operational_events import record_api_diagnostics, record_exception


EXPECTED_GOALS_STAT_TYPES = frozenset({"expected_goals", "xg"})
EXPECTED_ASSISTS_STAT_TYPES = frozenset({"expected_assists", "expected_assist", "xa", "x_a"})


def _statistic_key(value: object) -> str:
    """Normalize provider labels without relying on an exact spelling."""
    return "_".join(str(value).strip().lower().replace("-", " ").split())


def _non_negative_float(value: object) -> float | None:
    """Return a valid provider metric or ignore unavailable/non-numeric values."""
    try:
        parsed = float(str(value).replace(",", "."))
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def sync_results(
    api: ApiFootballClient, db: SupabaseRestClient, *, match_date: date
) -> SyncSummary:
    # The fixture upsert is idempotent and updates scores/status atomically.
    return sync_fixtures(
        api, db, start_date=match_date, end_date=match_date
    )


def sync_recent_results(
    api: ApiFootballClient,
    db: SupabaseRestClient,
    *,
    today: date,
    lookback_days: int,
) -> SyncSummary:
    """Reconcile recent days so a missed run cannot strand stale fixtures."""
    if lookback_days < 0 or lookback_days > 14:
        raise ValueError("lookback_days must be between 0 and 14")
    summary = sync_fixtures(
        api,
        db,
        start_date=today - timedelta(days=lookback_days),
        end_date=today,
    )
    sync_recent_expected_metrics(api, db, today=today, lookback_days=lookback_days)
    return summary


def extract_expected_goals(
    payload: list[dict[str, Any]], *, home_team_id: int, away_team_id: int,
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
    payload: list[dict[str, Any]], *, home_team_id: int, away_team_id: int,
) -> dict[str, float | None]:
    """Read provider xG/xA statistics for both teams without inventing missing xA.

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
                "xg" if kind in EXPECTED_GOALS_STAT_TYPES
                else "xa" if kind in EXPECTED_ASSISTS_STAT_TYPES
                else None
            )
            if metric is None:
                continue
            value = _non_negative_float(stat.get("value"))
            if value is not None:
                values.setdefault(int(team_id), {})[metric] = value

    return {
        "home_xg": values.get(home_team_id, {}).get("xg"),
        "away_xg": values.get(away_team_id, {}).get("xg"),
        "home_xa": values.get(home_team_id, {}).get("xa"),
        "away_xa": values.get(away_team_id, {}).get("xa"),
    }


def sync_recent_expected_metrics(
    api: ApiFootballClient, db: SupabaseRestClient, *, today: date, lookback_days: int,
) -> int:
    """Persist available xG/xA for recent finished fixtures; missing data is harmless."""
    start = datetime.combine(today - timedelta(days=lookback_days), datetime.min.time()).isoformat()
    end = datetime.combine(today + timedelta(days=1), datetime.min.time()).isoformat()
    matches = db.select_all(
        "matches",
        columns=(
            "id,home_team_id,away_team_id,home_xg,away_xg,home_xa,away_xa,"
            "expected_metrics_checked_at"
        ),
        filters={"status": "eq.finished", "and": f"(match_date.gte.{start},match_date.lt.{end})"},
    )
    updates: list[dict[str, Any]] = []
    for match in matches:
        checked_at = match.get("expected_metrics_checked_at")
        already_complete = all(
            match.get(column) is not None
            for column in ("home_xg", "away_xg", "home_xa", "away_xa")
        )
        # Avoid paying an API request every run when a competition does not expose xA.
        recently_checked = (
            checked_at is not None
            and datetime.fromisoformat(str(checked_at).replace("Z", "+00:00"))
            >= datetime.now(ZoneInfo("UTC")) - timedelta(hours=12)
        )
        if already_complete or recently_checked:
            continue
        try:
            statistics = api.get("fixtures/statistics", {"fixture": int(match["id"])})
        except Exception as error:
            # Final scores remain critical; optional xG/xA must not stop evaluation.
            record_exception(
                db,
                component="fetch_results",
                operation="expected-metric fetch",
                error=error,
                context={"fixture_id": int(match["id"])},
            )
            print(f"Expected-metric fetch skipped for fixture {int(match['id'])}: {type(error).__name__}")
            continue
        metrics = extract_expected_metrics(
            statistics,
            home_team_id=int(match["home_team_id"]), away_team_id=int(match["away_team_id"]),
        )
        updates.append({
            "id": int(match["id"]),
            **{column: value for column, value in metrics.items() if value is not None},
            "expected_metrics_checked_at": datetime.now(ZoneInfo("UTC")).isoformat(),
        })
    if updates:
        db.upsert("matches", updates, on_conflict="id")
    return len(updates)


def sync_recent_xg(
    api: ApiFootballClient, db: SupabaseRestClient, *, today: date, lookback_days: int,
) -> int:
    """Backward-compatible name for callers that only knew about xG."""
    return sync_recent_expected_metrics(api, db, today=today, lookback_days=lookback_days)


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
    try:
        if args.date:
            summary = sync_results(api, db, match_date=args.date)
        else:
            summary = sync_recent_results(
                api,
                db,
                today=datetime.now(ZoneInfo("Europe/Istanbul")).date(),
                lookback_days=args.lookback_days,
            )
    except Exception as error:
        record_exception(db, component="fetch_results", operation="result sync", error=error)
        raise
    diagnostics = api.diagnostics()
    record_api_diagnostics(db, component="fetch_results", diagnostics=diagnostics)
    print({"sync": asdict(summary), "api": diagnostics})


if __name__ == "__main__":
    main()
