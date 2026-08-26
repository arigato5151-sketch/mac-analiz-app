"""Compare completed fixtures with stored predictions and persist metrics."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from config.settings import get_settings
from db.db_client import SupabaseRestClient


RESULT_LABELS = ("home_win", "draw", "away_win")
PROBABILITY_COLUMNS = ("prob_home_win", "prob_draw", "prob_away_win")


def actual_result(home_score: int, away_score: int) -> str:
    """Return the canonical 1X2 label for a completed scoreline."""
    if home_score > away_score:
        return "home_win"
    if home_score < away_score:
        return "away_win"
    return "draw"


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

    return {
        "prediction_id": int(prediction["id"]),
        "match_id": int(match["id"]),
        "actual_result": outcome,
        "was_correct": predicted_index == outcome_index,
        "brier_score": float(brier_score),
        "evaluated_at": evaluated_at,
    }


def evaluate_pending_predictions(db: SupabaseRestClient) -> list[dict[str, Any]]:
    """Evaluate predictions once, using idempotent upserts for safe retries."""
    predictions = db.select_all(
        "predictions",
        columns=(
            "id,match_id,prob_home_win,prob_draw,prob_away_win,predicted_at"
        ),
    )
    evaluated_ids = {
        int(row["prediction_id"])
        for row in db.select_all("prediction_performance", columns="prediction_id")
    }
    pending = [row for row in predictions if int(row["id"]) not in evaluated_ids]
    if not pending:
        return []

    finished_matches = db.select_all(
        "matches",
        columns="id,status,home_score,away_score",
        filters={
            "status": "eq.finished",
            "home_score": "not.is.null",
            "away_score": "not.is.null",
        },
    )
    matches_by_id = {int(row["id"]): row for row in finished_matches}
    evaluated_at = datetime.now(timezone.utc).isoformat()
    rows = [
        build_performance_row(prediction, match, evaluated_at=evaluated_at)
        for prediction in pending
        if (match := matches_by_id.get(int(prediction["match_id"]))) is not None
    ]
    return db.upsert(
        "prediction_performance", rows, on_conflict="prediction_id"
    )


def main() -> None:
    settings = get_settings()
    db = SupabaseRestClient(
        settings.supabase_url, settings.supabase_service_role_key
    )
    rows = evaluate_pending_predictions(db)
    print(
        json.dumps(
            {
                "evaluated_predictions": len(rows),
                "correct_predictions": sum(bool(row["was_correct"]) for row in rows),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
