"""Evaluate stored match predictions against completed fixture scores."""

from __future__ import annotations

import argparse
import math
from datetime import date, datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from db.db_client import SupabaseRestClient

ISTANBUL = ZoneInfo("Europe/Istanbul")
MAX_LOOKBACK_DAYS = 3650


def _number(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) and 0 <= parsed <= 1 else None


def _binary_market(probabilities: dict[str, Any], actual: bool) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, raw_probability in probabilities.items():
        probability = _number(raw_probability)
        if probability is None:
            continue
        result[key] = {
            "probability": probability,
            "actual": actual,
            "brier_score": (probability - float(actual)) ** 2,
        }
    return result


def evaluate_prediction(prediction: dict[str, Any], match: dict[str, Any]) -> dict[str, Any] | None:
    """Build one immutable performance row from a saved forecast and final score."""
    try:
        home_score, away_score = int(match["home_score"]), int(match["away_score"])
        prediction_id, match_id = int(prediction["id"]), int(prediction["match_id"])
    except (KeyError, TypeError, ValueError):
        return None
    if home_score < 0 or away_score < 0:
        return None
    try:
        kickoff = datetime.fromisoformat(str(match["match_date"]).replace("Z", "+00:00"))
        predicted_at = datetime.fromisoformat(str(prediction["predicted_at"]).replace("Z", "+00:00"))
    except (KeyError, TypeError, ValueError):
        return None
    if predicted_at >= kickoff:
        return None

    result_probabilities = [
        _number(prediction.get("prob_home_win")),
        _number(prediction.get("prob_draw")),
        _number(prediction.get("prob_away_win")),
    ]
    if any(value is None for value in result_probabilities):
        return None
    if abs(sum(value for value in result_probabilities if value is not None) - 1) > 0.01:
        return None

    actual_index = 0 if home_score > away_score else 1 if home_score == away_score else 2
    actual_result = ("home_win", "draw", "away_win")[actual_index]
    winner_index = max(range(3), key=lambda index: result_probabilities[index])
    brier_score = sum(
        (float(probability) - float(index == actual_index)) ** 2
        for index, probability in enumerate(result_probabilities)
    )
    actual_probability = float(result_probabilities[actual_index])
    row: dict[str, Any] = {
        "prediction_id": prediction_id,
        "match_id": match_id,
        "actual_result": actual_result,
        "was_correct": winner_index == actual_index,
        "brier_score": brier_score,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "log_loss": -math.log(max(1e-15, actual_probability)),
        "market_performance": {},
    }

    market_performance: dict[str, Any] = {}
    total_goals = home_score + away_score
    for key, threshold in (("over_2_5", 2.5),):
        probability = _number(prediction.get(f"prob_{key}"))
        if probability is not None:
            actual = total_goals > threshold
            row[f"{key}_actual"] = actual
            row[f"{key}_was_correct"] = (probability >= 0.5) == actual
            row[f"{key}_brier_score"] = (probability - float(actual)) ** 2

    btts_probability = _number(prediction.get("prob_btts"))
    btts_actual = home_score > 0 and away_score > 0
    if btts_probability is not None:
        row.update(
            {
                "btts_actual": btts_actual,
                "btts_was_correct": (btts_probability >= 0.5) == btts_actual,
                "btts_brier_score": (btts_probability - float(btts_actual)) ** 2,
            }
        )

    markets = prediction.get("market_probabilities") or {}
    if isinstance(markets, str):
        import json

        try:
            markets = json.loads(markets)
        except json.JSONDecodeError:
            markets = {}
    if isinstance(markets, dict):
        total_market_results: dict[str, Any] = {}
        raw_total_goals = markets.get("total_goals") or {}
        if isinstance(raw_total_goals, dict):
            for key, probability in raw_total_goals.items():
                if "_" not in key:
                    continue
                side, line_text = key.split("_", 1)
                try:
                    line = float(line_text.replace("_", "."))
                except ValueError:
                    continue
                if side == "over":
                    actual = total_goals > line
                elif side == "under":
                    actual = total_goals < line
                else:
                    continue
                total_market_results.update(_binary_market({key: probability}, actual))
        if total_market_results:
            market_performance["total_goals"] = total_market_results

        team_market_results: dict[str, Any] = {}
        raw_team_goals = markets.get("team_goals") or {}
        if isinstance(raw_team_goals, dict):
            for key, probability in raw_team_goals.items():
                if "_over_" not in key:
                    continue
                team, line_text = key.split("_over_", 1)
                try:
                    line = float(line_text.replace("_", "."))
                except ValueError:
                    continue
                score = home_score if team == "home" else away_score if team == "away" else None
                if score is None:
                    continue
                team_market_results.update(_binary_market({key: probability}, score > line))
        if team_market_results:
            market_performance["team_goals"] = team_market_results

        double_chance_results: dict[str, Any] = {}
        raw_double_chance = markets.get("double_chance") or {}
        if isinstance(raw_double_chance, dict):
            outcome = {"1": home_score > away_score, "X": home_score == away_score, "2": away_score > home_score}
            for key, probability in raw_double_chance.items():
                normalized = str(key).upper().replace(" ", "")
                if normalized not in {"1X", "X2", "12"}:
                    continue
                actual = any(outcome[item] for item in normalized)
                double_chance_results.update(_binary_market({normalized: probability}, actual))
        if double_chance_results:
            market_performance["double_chance"] = double_chance_results

        correct_scores = markets.get("correct_scores") or []
        if isinstance(correct_scores, list) and correct_scores:
            actual_score = f"{home_score}-{away_score}"
            market_performance["correct_score"] = {
                "actual": actual_score,
                "top_3_hit": any(
                    isinstance(item, dict) and str(item.get("score")) == actual_score
                    for item in correct_scores[:3]
                ),
            }

    row["market_performance"] = market_performance
    return row


def evaluate_completed_predictions(
    db: SupabaseRestClient,
    *,
    today: date | None = None,
    lookback_days: int = 14,
) -> dict[str, int]:
    """Insert evaluations for finished matches, skipping already evaluated predictions."""
    if not 0 <= lookback_days <= MAX_LOOKBACK_DAYS:
        raise ValueError(f"lookback_days must be between 0 and {MAX_LOOKBACK_DAYS}")
    day = today or datetime.now(ISTANBUL).date()
    start = datetime.combine(day - timedelta(days=lookback_days), datetime.min.time(), tzinfo=ISTANBUL).astimezone(timezone.utc)
    end = datetime.combine(day + timedelta(days=1), datetime.min.time(), tzinfo=ISTANBUL).astimezone(timezone.utc)
    matches = db.select_all(
        "matches",
        columns="id,match_date,home_score,away_score,status",
        filters={
            "status": "eq.finished",
            "home_score": "not.is.null",
            "away_score": "not.is.null",
            "match_date": f"gte.{start.isoformat()}",
            "and": f"(match_date.lt.{end.isoformat()})",
        },
        order="match_date.asc,id.asc",
    )
    predictions: list[dict[str, Any]] = []
    match_ids = [int(match["id"]) for match in matches]
    for offset in range(0, len(match_ids), 100):
        chunk = match_ids[offset : offset + 100]
        predictions.extend(
            db.select_all(
                "predictions",
                columns=(
                    "id,match_id,model_version,predicted_at,prob_home_win,prob_draw,"
                    "prob_away_win,prob_over_2_5,prob_btts,market_probabilities"
                ),
                filters={"match_id": f"in.({','.join(map(str, chunk))})"},
                order="match_id.asc,predicted_at.asc,id.asc",
            )
        )

    existing_ids: set[int] = set()
    prediction_ids = [int(row["id"]) for row in predictions]
    for offset in range(0, len(prediction_ids), 100):
        chunk = prediction_ids[offset : offset + 100]
        existing_ids.update(
            int(row["prediction_id"])
            for row in db.select_all(
                "prediction_performance",
                columns="prediction_id",
                filters={"prediction_id": f"in.({','.join(map(str, chunk))})"},
                order="prediction_id.asc",
            )
        )

    matches_by_id = {int(match["id"]): match for match in matches}
    records = []
    skipped = 0
    for prediction in predictions:
        if int(prediction["id"]) in existing_ids:
            continue
        result = evaluate_prediction(prediction, matches_by_id[int(prediction["match_id"])])
        if result is None:
            skipped += 1
        else:
            records.append(result)

    for offset in range(0, len(records), 100):
        db.insert("prediction_performance", records[offset : offset + 100])
    return {
        "finished_matches": len(matches),
        "predictions_found": len(predictions),
        "evaluations_inserted": len(records),
        "already_evaluated": sum(int(row["id"]) in existing_ids for row in predictions),
        "skipped_invalid": skipped,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lookback-days", type=int, default=14)
    args = parser.parse_args()
    from config.settings import get_supabase_admin_settings

    settings = get_supabase_admin_settings()
    db = SupabaseRestClient(settings.supabase_url, settings.supabase_service_role_key)
    print(evaluate_completed_predictions(db, lookback_days=args.lookback_days))


if __name__ == "__main__":
    main()
