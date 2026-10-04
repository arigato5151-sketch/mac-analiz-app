"""Validate morning fixture, prediction, form, and availability freshness."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from math import isfinite
from typing import Any

from config.settings import UPCOMING_HORIZON_DAYS, get_settings
from db.db_client import SupabaseRestClient

FRESHNESS_LIMIT = timedelta(hours=36)
PREDICTION_FRESHNESS_LIMIT = timedelta(hours=36)
FIXTURE_SYNC_FRESHNESS_LIMIT = timedelta(hours=36)
STALE_ACTIVE_GRACE = timedelta(hours=4)
FUTURE_TIMESTAMP_TOLERANCE = timedelta(minutes=5)
RESULT_PROBABILITY_FIELDS = (
    "prob_home_win",
    "prob_draw",
    "prob_away_win",
)
ALL_PROBABILITY_FIELDS = (
    *RESULT_PROBABILITY_FIELDS,
    "prob_over_2_5",
    "prob_btts",
)


def _parse_timestamp(value: object) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _probability_is_valid(value: object) -> bool:
    if isinstance(value, bool):
        return False
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return False
    return isfinite(parsed) and 0.0 <= parsed <= 1.0


def _market_probabilities_are_valid(value: object) -> bool:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return False
    if not isinstance(value, Mapping):
        return False

    def valid_tree(node: object, key: str | None = None) -> bool:
        if key == "expected_goals":
            return (
                isinstance(node, Mapping)
                and {"home", "away"}.issubset(node)
                and all(_is_finite_nonnegative(item) for item in node.values())
            )
        if key == "score_matrix":
            if not isinstance(node, list) or not node or not all(
                isinstance(row, list) and row for row in node
            ):
                return False
            row_width = len(node[0])
            if any(len(row) != row_width for row in node):
                return False
            values = [item for row in node for item in row]
            if not all(_probability_is_valid(item) for item in values):
                return False
            return abs(sum(float(item) for item in values) - 1.0) <= 0.001
        if isinstance(node, Mapping):
            return all(valid_tree(item, str(child_key)) for child_key, item in node.items())
        if isinstance(node, list):
            return all(valid_tree(item, key) for item in node)
        if key == "score":
            return isinstance(node, str) and bool(node.strip())
        return _probability_is_valid(node)

    return bool(value) and valid_tree(value)


def _is_finite_nonnegative(value: object) -> bool:
    if isinstance(value, bool):
        return False
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return False
    return isfinite(parsed) and parsed >= 0.0


def assess_morning_quality(
    scheduled_matches: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    team_forms: list[dict[str, Any]],
    availability_snapshots: list[dict[str, Any]],
    active_matches: list[dict[str, Any]] | None = None,
    *,
    now: datetime,
    latest_fixture_sync_at: object = None,
) -> dict[str, Any]:
    """Return an auditable quality report and fail only on actionable gaps."""
    now = _parse_timestamp(now) or datetime.now(timezone.utc)
    fixture_sync_at = _parse_timestamp(latest_fixture_sync_at)
    fixture_sync_age_minutes = (
        round((now - fixture_sync_at).total_seconds() / 60, 1)
        if fixture_sync_at is not None
        else None
    )
    stale_fixture_sync = fixture_sync_at is not None and (
        fixture_sync_at > now + FUTURE_TIMESTAMP_TOLERANCE
        or now - fixture_sync_at > FIXTURE_SYNC_FRESHNESS_LIMIT
    )
    target_team_ids = {
        int(team_id)
        for match in scheduled_matches
        for team_id in (match["home_team_id"], match["away_team_id"])
    }
    latest_predictions: dict[int, dict[str, Any]] = {}
    invalid_prediction_timestamp_ids: set[int] = set()
    for row in predictions:
        match_id = int(row["match_id"])
        current = latest_predictions.get(match_id)
        timestamp = _parse_timestamp(row.get("predicted_at"))
        current_timestamp = _parse_timestamp(current.get("predicted_at")) if current else None
        if current is None or (
            timestamp is not None
            and (current_timestamp is None or timestamp > current_timestamp)
        ):
            latest_predictions[match_id] = row

    predicted_match_ids = set(latest_predictions)
    missing_predictions = sorted(
        int(match["id"])
        for match in scheduled_matches
        if int(match["id"]) not in predicted_match_ids
    )
    stale_predictions: set[int] = set()
    future_predictions: set[int] = set()
    predictions_after_kickoff: set[int] = set()
    invalid_probability_ids: set[int] = set()
    invalid_market_probability_ids: set[int] = set()
    invalid_fixture_ids: set[int] = set()

    for match in scheduled_matches:
        match_id = int(match["id"])
        if "home_team_id" in match and "away_team_id" in match:
            try:
                home_id = int(match["home_team_id"])
                away_id = int(match["away_team_id"])
                if home_id <= 0 or away_id <= 0 or home_id == away_id:
                    invalid_fixture_ids.add(match_id)
            except (TypeError, ValueError):
                invalid_fixture_ids.add(match_id)

        kickoff = _parse_timestamp(match.get("match_date"))
        if "match_date" in match and kickoff is None:
            invalid_fixture_ids.add(match_id)

        prediction = latest_predictions.get(match_id)
        if prediction is None:
            continue
        predicted_at = _parse_timestamp(prediction.get("predicted_at"))
        if "predicted_at" in prediction:
            if predicted_at is None:
                invalid_prediction_timestamp_ids.add(match_id)
            elif predicted_at > now + FUTURE_TIMESTAMP_TOLERANCE:
                future_predictions.add(match_id)
            elif now - predicted_at > PREDICTION_FRESHNESS_LIMIT:
                stale_predictions.add(match_id)
            if kickoff is not None and predicted_at is not None and predicted_at > kickoff:
                predictions_after_kickoff.add(match_id)

        if any(field in prediction for field in ALL_PROBABILITY_FIELDS):
            probabilities = [prediction.get(field) for field in ALL_PROBABILITY_FIELDS]
            if not all(_probability_is_valid(value) for value in probabilities):
                invalid_probability_ids.add(match_id)
            else:
                result_total = sum(float(prediction[field]) for field in RESULT_PROBABILITY_FIELDS)
                if abs(result_total - 1.0) > 0.001:
                    invalid_probability_ids.add(match_id)

        if "market_probabilities" in prediction and not _market_probabilities_are_valid(
            prediction.get("market_probabilities")
        ):
            invalid_market_probability_ids.add(match_id)

    def fresh_team_ids(rows: list[dict[str, Any]], timestamp_key: str) -> set[int]:
        fresh: set[int] = set()
        for row in rows:
            refreshed_at = _parse_timestamp(row.get(timestamp_key))
            if refreshed_at is None or refreshed_at > now + FUTURE_TIMESTAMP_TOLERANCE:
                continue
            if now - refreshed_at <= FRESHNESS_LIMIT:
                fresh.add(int(row["team_id"]))
        return fresh

    fresh_forms = fresh_team_ids(team_forms, "calculated_at")
    fresh_availability = fresh_team_ids(availability_snapshots, "refreshed_at")
    missing_forms = sorted(target_team_ids - fresh_forms)
    missing_availability = sorted(target_team_ids - fresh_availability)
    stale_active_matches = []
    for match in active_matches or []:
        kickoff = _parse_timestamp(match.get("match_date"))
        if kickoff is not None and kickoff < now - STALE_ACTIVE_GRACE:
            stale_active_matches.append(int(match["id"]))
    return {
        "scheduled_matches": len(scheduled_matches),
        "latest_fixture_sync_at": fixture_sync_at.isoformat() if fixture_sync_at else None,
        "fixture_sync_age_minutes": fixture_sync_age_minutes,
        "stale_fixture_sync": stale_fixture_sync,
        "target_teams": len(target_team_ids),
        "predictions": len(predicted_match_ids),
        "missing_prediction_ids": missing_predictions,
        "stale_prediction_ids": sorted(stale_predictions),
        "future_prediction_ids": sorted(future_predictions),
        "prediction_after_kickoff_ids": sorted(predictions_after_kickoff),
        "invalid_prediction_timestamp_ids": sorted(invalid_prediction_timestamp_ids),
        "invalid_probability_ids": sorted(invalid_probability_ids),
        "invalid_market_probability_ids": sorted(invalid_market_probability_ids),
        "invalid_fixture_ids": sorted(invalid_fixture_ids),
        "stale_form_team_ids": missing_forms,
        "stale_availability_team_ids": missing_availability,
        "stale_active_match_ids": stale_active_matches,
        "healthy": not (
            stale_fixture_sync
            or missing_predictions
            or stale_predictions
            or future_predictions
            or predictions_after_kickoff
            or invalid_prediction_timestamp_ids
            or invalid_probability_ids
            or invalid_market_probability_ids
            or invalid_fixture_ids
            or missing_forms
            or missing_availability
            or stale_active_matches
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=UPCOMING_HORIZON_DAYS)
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
        order="id.asc",
    )
    scheduled_ids = ",".join(str(int(row["id"])) for row in scheduled)
    predictions = (
        db.select_all(
            "predictions",
            columns=(
                "match_id,model_version,predicted_at,prob_home_win,prob_draw,"
                "prob_away_win,prob_over_2_5,prob_btts,market_probabilities"
            ),
            filters={"match_id": f"in.({scheduled_ids})"},
            order="predicted_at.desc",
        )
        if scheduled_ids
        else []
    )
    forms = db.select_all(
        "team_form", columns="team_id,calculated_at", order="team_id.asc"
    )
    availability = db.select_all(
        "team_availability_status",
        columns="team_id,refreshed_at",
        order="team_id.asc",
    )
    active_matches = db.select_all(
        "matches",
        columns="id,match_date",
        filters={
            "status": "in.(scheduled,live)",
            "match_date": f"lt.{(now - STALE_ACTIVE_GRACE).isoformat()}",
        },
        order="id.asc",
    )
    latest_fixture_sync = db.select(
        "operational_events",
        columns="occurred_at",
        filters={
            "component": "eq.fetch_fixtures",
            "event_type": "eq.sync_succeeded",
        },
        limit=1,
        order="occurred_at.desc",
    )
    # Legacy deployments may not yet have sync_succeeded events. The current
    # workflow runs this quality check immediately after fetch_fixtures, so use
    # that run's start time as a safe fallback while reporting the absence.
    latest_fixture_sync_at = (
        latest_fixture_sync[0].get("occurred_at")
        if latest_fixture_sync
        else now.isoformat()
    )
    report = assess_morning_quality(
        scheduled,
        predictions,
        forms,
        availability,
        active_matches=active_matches,
        now=now,
        latest_fixture_sync_at=latest_fixture_sync_at,
    )
    report["fixture_sync_event_missing"] = not bool(latest_fixture_sync)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["healthy"]:
        raise RuntimeError("Morning data-quality validation failed")


if __name__ == "__main__":
    main()
