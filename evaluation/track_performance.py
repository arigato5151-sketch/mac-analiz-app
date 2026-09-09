"""Compare completed fixtures with stored predictions and persist metrics."""

from __future__ import annotations

import json
from math import log
from datetime import datetime, timedelta, timezone
from typing import Any

from config.settings import get_settings
from data_pipeline.isotime import parse_iso_datetime
from db.db_client import SupabaseRestClient


RESULT_LABELS = ("home_win", "draw", "away_win")
PROBABILITY_COLUMNS = ("prob_home_win", "prob_draw", "prob_away_win")
RESULT_NOTIFICATION_MAX_AGE = timedelta(hours=24)
# Evaluations only ever target recently finished fixtures; bounding the finished
# scan keeps the `match_id=in.(...)` filter small even after many seasons.
EVALUATION_LOOKBACK_DAYS = 14
# Keep PostgREST `in.(...)` query strings comfortably below proxy URL limits.
MATCH_ID_QUERY_BATCH_SIZE = 100
LOG_LOSS_EPSILON = 1e-15


def actual_result(home_score: int, away_score: int) -> str:
    """Return the canonical 1X2 label for a completed scoreline."""
    if home_score > away_score:
        return "home_win"
    if home_score < away_score:
        return "away_win"
    return "draw"


def binary_market_performance(
    probability: float | None, *, actual_positive: bool
) -> tuple[bool | None, float | None]:
    """Evaluate one binary market without fabricating a missing prediction."""
    if probability is None:
        return None, None
    if not 0 <= probability <= 1:
        raise ValueError("Binary-market probability must be between 0 and 1")
    target = float(actual_positive)
    return (probability >= 0.5) == actual_positive, float((probability - target) ** 2)


def multiclass_log_loss(
    probabilities: tuple[float, float, float], outcome_index: int
) -> float:
    """Return stable per-fixture negative log likelihood for the realised 1-X-2 class."""
    if not 0 <= outcome_index < len(probabilities):
        raise ValueError("outcome_index must reference a 1-X-2 probability")
    return float(-log(max(probabilities[outcome_index], LOG_LOSS_EPSILON)))


def build_performance_row(
    prediction: dict[str, Any], match: dict[str, Any], *, evaluated_at: str
) -> dict[str, Any]:
    """Build one deterministic multiclass evaluation row."""
    if match.get("home_score") is None or match.get("away_score") is None:
        raise ValueError("Completed match must contain both scores")

    probabilities = tuple(float(prediction[column]) for column in PROBABILITY_COLUMNS)
    if any(probability < 0 or probability > 1 for probability in probabilities):
        raise ValueError("Prediction probabilities must be between 0 and 1")
    if abs(sum(probabilities) - 1.0) > 0.001:
        raise ValueError("Prediction probabilities must sum to one")

    outcome = actual_result(int(match["home_score"]), int(match["away_score"]))
    outcome_index = RESULT_LABELS.index(outcome)
    predicted_index = max(range(len(probabilities)), key=probabilities.__getitem__)
    one_hot = tuple(1.0 if index == outcome_index else 0.0 for index in range(3))
    brier_score = sum(
        (probability - target) ** 2
        for probability, target in zip(probabilities, one_hot, strict=True)
    )
    log_loss = multiclass_log_loss(probabilities, outcome_index)
    home_score = int(match["home_score"])
    away_score = int(match["away_score"])
    over_actual = home_score + away_score >= 3
    btts_actual = home_score > 0 and away_score > 0
    over_correct, over_brier = binary_market_performance(
        (
            float(prediction["prob_over_2_5"])
            if prediction.get("prob_over_2_5") is not None
            else None
        ),
        actual_positive=over_actual,
    )
    btts_correct, btts_brier = binary_market_performance(
        (
            float(prediction["prob_btts"])
            if prediction.get("prob_btts") is not None
            else None
        ),
        actual_positive=btts_actual,
    )

    row = {
        "prediction_id": int(prediction["id"]),
        "match_id": int(match["id"]),
        "actual_result": outcome,
        "was_correct": predicted_index == outcome_index,
        "brier_score": float(brier_score),
        "log_loss": log_loss,
        "over_2_5_actual": over_actual,
        "over_2_5_was_correct": over_correct,
        "over_2_5_brier_score": over_brier,
        "btts_actual": btts_actual,
        "btts_was_correct": btts_correct,
        "btts_brier_score": btts_brier,
        "evaluated_at": evaluated_at,
        # PostgREST bulk writes require every object to expose the same keys.
        "snapshot_id": (
            int(prediction["snapshot_id"])
            if prediction.get("snapshot_id") is not None
            else None
        ),
    }
    return row


def select_official_predictions(
    predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep only the latest valid pre-kickoff prediction for each fixture.

    Retraining may leave several model versions for one fixture. Production
    performance must score the one that was actually available latest before
    kickoff, not every historical model snapshot.
    """
    latest: dict[int, dict[str, Any]] = {}
    for prediction in predictions:
        match_id = int(prediction["match_id"])
        current = latest.get(match_id)
        candidate_key = (parse_iso_datetime(prediction["predicted_at"]), int(prediction["id"]))
        current_key = (
            (parse_iso_datetime(current["predicted_at"]), int(current["id"]))
            if current is not None
            else None
        )
        if current_key is None or candidate_key > current_key:
            latest[match_id] = prediction
    return list(latest.values())


def _select_rows_for_match_ids(
    db: SupabaseRestClient,
    table: str,
    *,
    columns: str,
    match_ids: list[int],
    filters: dict[str, str] | None = None,
    batch_size: int = MATCH_ID_QUERY_BATCH_SIZE,
) -> list[dict[str, Any]]:
    """Fetch rows in bounded `match_id` batches to avoid HTTP 414 errors."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    rows: list[dict[str, Any]] = []
    unique_match_ids = sorted(set(int(match_id) for match_id in match_ids))
    for start in range(0, len(unique_match_ids), batch_size):
        batch = unique_match_ids[start : start + batch_size]
        batch_filter = f"in.({','.join(str(match_id) for match_id in batch)})"
        rows.extend(
            db.select_all(
                table,
                columns=columns,
                filters={**(filters or {}), "match_id": batch_filter},
            )
        )
    return rows


def evaluate_pending_predictions(db: SupabaseRestClient) -> list[dict[str, Any]]:
    """Evaluate predictions once, using idempotent upserts for safe retries."""
    evaluated_match_ids = {
        int(row["match_id"])
        for row in db.select_all("prediction_performance", columns="match_id")
    }
    lookback_cutoff = (
        datetime.now(timezone.utc) - timedelta(days=EVALUATION_LOOKBACK_DAYS)
    ).isoformat()
    finished_matches = db.select_all(
        "matches",
        columns="id,match_date,home_score,away_score",
        filters={
            "status": "eq.finished",
            "home_score": "not.is.null",
            "away_score": "not.is.null",
            "match_date": f"gte.{lookback_cutoff}",
        },
    )
    matches_by_id = {int(row["id"]): row for row in finished_matches}
    pending_match_ids = [
        int(match_id)
        for match_id in matches_by_id
        if int(match_id) not in evaluated_match_ids
    ]
    if not pending_match_ids:
        return []
    predictions = _select_rows_for_match_ids(
        db,
        "predictions",
        columns="id,match_id,prob_home_win,prob_draw,prob_away_win,predicted_at",
        match_ids=pending_match_ids,
    )
    snapshots = _select_rows_for_match_ids(
        db,
        "prediction_snapshots",
        columns=(
            "id,source_prediction_id,match_id,model_version,prob_home_win,prob_draw,"
            "prob_away_win,prob_over_2_5,prob_btts,captured_at"
        ),
        match_ids=pending_match_ids,
        filters={"snapshot_type": "eq.pre_match_60m"},
    )
    snapshots_by_match = {
        int(snapshot["match_id"]): {
            "id": int(snapshot["source_prediction_id"]),
            "snapshot_id": int(snapshot["id"]),
            "match_id": int(snapshot["match_id"]),
            "model_version": snapshot["model_version"],
            "prob_home_win": snapshot["prob_home_win"],
            "prob_draw": snapshot["prob_draw"],
            "prob_away_win": snapshot["prob_away_win"],
            "prob_over_2_5": snapshot["prob_over_2_5"],
            "prob_btts": snapshot["prob_btts"],
            "predicted_at": snapshot["captured_at"],
        }
        for snapshot in snapshots
    }

    evaluated_at = datetime.now(timezone.utc).isoformat()
    fallback_predictions = {
        int(prediction["match_id"]): prediction
        for prediction in select_official_predictions(predictions)
    }
    official_predictions = [
        snapshots_by_match.get(match_id, prediction)
        for match_id, prediction in fallback_predictions.items()
    ]
    rows = [
        build_performance_row(prediction, match, evaluated_at=evaluated_at)
        for prediction in official_predictions
        if (match := matches_by_id.get(int(prediction["match_id"]))) is not None
        and parse_iso_datetime(prediction["predicted_at"]) <= parse_iso_datetime(match["match_date"])
    ]
    persisted = db.upsert("prediction_performance", rows, on_conflict="prediction_id")
    queue_rows = [
        {
            "prediction_id": int(row["prediction_id"]),
            "match_id": int(row["match_id"]),
            "available_at": evaluated_at,
            "next_attempt_at": evaluated_at,
        }
        for row in persisted
        if (match := matches_by_id.get(int(row["match_id"]))) is not None
        and parse_iso_datetime(match["match_date"])
        >= datetime.now(timezone.utc) - RESULT_NOTIFICATION_MAX_AGE
    ]
    db.upsert("result_notification_queue", queue_rows, on_conflict="prediction_id")
    return persisted


def main() -> None:
    from models.shadow import evaluate_shadow_predictions

    settings = get_settings()
    db = SupabaseRestClient(settings.supabase_url, settings.supabase_service_role_key)
    rows = evaluate_pending_predictions(db)
    shadow_rows = evaluate_shadow_predictions(db)
    print(
        json.dumps(
            {
                "evaluated_predictions": len(rows),
                "correct_predictions": sum(bool(row["was_correct"]) for row in rows),
                "shadow_predictions_evaluated": len(shadow_rows),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
