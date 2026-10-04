"""Capture and report predictions available at fixed pre-match horizons."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

from config.settings import get_settings
from data_pipeline.isotime import parse_iso_datetime
from db.db_client import DatabaseError, SupabaseRestClient
from evaluation.track_performance import build_performance_row

# Capture windows tolerate cron delay while keeping the horizon groups distinct.
HORIZON_WINDOWS: dict[str, tuple[timedelta, timedelta]] = {
    "24h": (timedelta(hours=23, minutes=30), timedelta(hours=24, minutes=30)),
    "6h": (timedelta(hours=5, minutes=45), timedelta(hours=6, minutes=15)),
    "20m": (timedelta(minutes=15), timedelta(minutes=25)),
}
MATCH_ID_QUERY_BATCH_SIZE = 100


def _client() -> SupabaseRestClient:
    settings = get_settings()
    return SupabaseRestClient(settings.supabase_url, settings.supabase_service_role_key)


def capture_horizon_snapshots(
    db: SupabaseRestClient, *, now: datetime | None = None
) -> dict[str, Any]:
    """Save the newest forecast available during each target horizon window."""
    captured_at = now or datetime.now(timezone.utc)
    if captured_at.tzinfo is None:
        raise ValueError("now must include a timezone")
    captured_at = captured_at.astimezone(timezone.utc)
    matches = db.select_all(
        "matches",
        columns="id,match_date,status",
        filters={
            "status": "eq.scheduled",
            "and": (
                f"(match_date.gte.{captured_at.isoformat()},"
                f"match_date.lte.{(captured_at + timedelta(hours=25)).isoformat()})"
            ),
        },
        order="match_date.asc,id.asc",
    )
    targets: dict[int, tuple[dict[str, Any], str, float]] = {}
    for match in matches:
        match_id = int(match["id"])
        kickoff = parse_iso_datetime(str(match["match_date"]))
        lead = kickoff - captured_at
        for horizon, (minimum, maximum) in HORIZON_WINDOWS.items():
            if minimum <= lead <= maximum:
                targets[match_id] = (match, horizon, lead.total_seconds() / 60.0)
                break
    if not targets:
        return {
            "eligible_matches": 0,
            "captured": 0,
            "already_captured": 0,
            "missing_prediction": 0,
        }

    match_ids = sorted(targets)
    predictions: list[dict[str, Any]] = []
    for start in range(0, len(match_ids), MATCH_ID_QUERY_BATCH_SIZE):
        batch = match_ids[start : start + MATCH_ID_QUERY_BATCH_SIZE]
        predictions.extend(
            db.select_all(
                "predictions",
                columns=(
                    "id,match_id,model_version,prob_home_win,prob_draw,prob_away_win,"
                    "prob_over_2_5,prob_btts,market_probabilities,predicted_at"
                ),
                filters={"match_id": f"in.({','.join(map(str, batch))})"},
                order="predicted_at.desc,id.desc",
            )
        )
    latest_by_match: dict[int, dict[str, Any]] = {}
    for prediction in predictions:
        match_id = int(prediction["match_id"])
        if match_id in latest_by_match:
            continue
        predicted_at = parse_iso_datetime(str(prediction["predicted_at"]))
        if predicted_at <= captured_at and predicted_at <= parse_iso_datetime(
            str(targets[match_id][0]["match_date"])
        ):
            latest_by_match[match_id] = prediction

    existing_snapshots: set[tuple[int, str]] = set()
    for start in range(0, len(match_ids), MATCH_ID_QUERY_BATCH_SIZE):
        batch = match_ids[start : start + MATCH_ID_QUERY_BATCH_SIZE]
        existing_snapshots.update(
            (int(row["match_id"]), str(row["horizon_key"]))
            for row in db.select_all(
                "prediction_horizon_snapshots",
                columns="match_id,horizon_key",
                filters={"match_id": f"in.({','.join(map(str, batch))})"},
            )
        )

    written = 0
    missing_prediction = 0
    already_captured = 0
    for match_id, (match, horizon, lead_minutes) in targets.items():
        if (match_id, horizon) in existing_snapshots:
            already_captured += 1
            continue
        prediction = latest_by_match.get(match_id)
        if prediction is None:
            missing_prediction += 1
            continue
        predicted_at = parse_iso_datetime(str(prediction["predicted_at"]))
        age_minutes = (captured_at - predicted_at).total_seconds() / 60.0
        snapshot = {
            "source_prediction_id": int(prediction["id"]),
            "match_id": match_id,
            "horizon_key": horizon,
            "model_version": str(prediction["model_version"]),
            "prob_home_win": float(prediction["prob_home_win"]),
            "prob_draw": float(prediction["prob_draw"]),
            "prob_away_win": float(prediction["prob_away_win"]),
            "prob_over_2_5": prediction.get("prob_over_2_5"),
            "prob_btts": prediction.get("prob_btts"),
            "market_probabilities": prediction.get("market_probabilities") or {},
            "source_predicted_at": predicted_at.isoformat(),
            "captured_at": captured_at.isoformat(),
            "lead_minutes": round(lead_minutes, 2),
            "prediction_age_minutes": round(age_minutes, 2),
        }
        try:
            db.insert("prediction_horizon_snapshots", [snapshot])
            written += 1
        except DatabaseError as error:
            if "(409)" in str(error):
                already_captured += 1
                continue
            raise
    return {
        "eligible_matches": len(targets),
        "captured": written,
        "already_captured": already_captured,
        "missing_prediction": missing_prediction,
    }


def horizon_performance_report(db: SupabaseRestClient) -> dict[str, Any]:
    """Compare captured forecast horizons on their completed fixtures."""
    snapshots = db.select_all(
        "prediction_horizon_snapshots",
        columns=(
            "id,source_prediction_id,match_id,horizon_key,model_version,prob_home_win,"
            "prob_draw,prob_away_win,prob_over_2_5,prob_btts,market_probabilities,"
            "captured_at,lead_minutes,prediction_age_minutes"
        ),
        order="captured_at.asc,id.asc",
    )
    match_ids = sorted({int(row["match_id"]) for row in snapshots})
    matches: list[dict[str, Any]] = []
    for start in range(0, len(match_ids), MATCH_ID_QUERY_BATCH_SIZE):
        batch = match_ids[start : start + MATCH_ID_QUERY_BATCH_SIZE]
        matches.extend(
            db.select_all(
                "matches",
                columns="id,league_id,match_date,status,home_score,away_score",
                filters={
                    "id": f"in.({','.join(map(str, batch))})",
                    "status": "eq.finished",
                    "home_score": "not.is.null",
                    "away_score": "not.is.null",
                },
            )
        )
    completed_by_id = {int(row["id"]): row for row in matches}
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    unscored = 0
    for snapshot in snapshots:
        match = completed_by_id.get(int(snapshot["match_id"]))
        if match is None:
            unscored += 1
            continue
        prediction = {
            "id": int(snapshot["source_prediction_id"]),
            "match_id": int(snapshot["match_id"]),
            "prob_home_win": snapshot["prob_home_win"],
            "prob_draw": snapshot["prob_draw"],
            "prob_away_win": snapshot["prob_away_win"],
            "prob_over_2_5": snapshot.get("prob_over_2_5"),
            "prob_btts": snapshot.get("prob_btts"),
            "market_probabilities": snapshot.get("market_probabilities") or {},
        }
        metrics = build_performance_row(
            prediction,
            match,
            evaluated_at=datetime.now(timezone.utc).isoformat(),
        )
        groups[str(snapshot["horizon_key"])].append({**snapshot, **metrics})

    report: dict[str, Any] = {"unscored_snapshots": unscored, "horizons": {}}
    for horizon in HORIZON_WINDOWS:
        rows = groups.get(horizon, [])
        count = len(rows)
        report["horizons"][horizon] = {
            "sample_size": count,
            "accuracy": (
                sum(bool(row["was_correct"]) for row in rows) / count if count else None
            ),
            "brier_score": (
                sum(float(row["brier_score"]) for row in rows) / count if count else None
            ),
            "log_loss": (
                sum(float(row["log_loss"]) for row in rows) / count if count else None
            ),
            "mean_actual_lead_minutes": (
                sum(float(row["lead_minutes"]) for row in rows) / count if count else None
            ),
            "mean_prediction_age_minutes": (
                sum(float(row["prediction_age_minutes"]) for row in rows) / count
                if count
                else None
            ),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--capture", action="store_true")
    action.add_argument("--report", action="store_true")
    args = parser.parse_args()
    db = _client()
    result = (
        capture_horizon_snapshots(db)
        if args.capture
        else horizon_performance_report(db)
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
