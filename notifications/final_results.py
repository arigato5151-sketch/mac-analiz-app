"""Send one audited Telegram result card for each newly evaluated fixture."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any

from config.settings import get_settings
from db.db_client import SupabaseRestClient
from notifications.telegram import TelegramError, send_telegram_message


MAX_DELIVERIES_PER_RUN = 20


def _outcome_label(home_score: int, away_score: int) -> str:
    if home_score > away_score:
        return "Ev kazanır"
    if home_score == away_score:
        return "Beraberlik"
    return "Deplasman kazanır"


def _result_line(
    probability: float,
    *,
    actual_positive: bool,
    positive_label: str,
    negative_label: str,
) -> str:
    predicted_positive = probability >= 0.5
    predicted_label = positive_label if predicted_positive else negative_label
    predicted_probability = probability if predicted_positive else 1 - probability
    correct = predicted_positive == actual_positive
    return f"{predicted_label} (%{predicted_probability * 100:.0f}) {'✓' if correct else '✗'}"


def final_result_message(
    match: dict[str, Any],
    prediction: dict[str, Any],
    *,
    home_team: str,
    away_team: str,
    league_name: str,
) -> str:
    """Build one concise audited result card with all model markets."""
    home_score = int(match["home_score"])
    away_score = int(match["away_score"])
    outcome_probabilities = {
        "Ev kazanır": float(prediction["prob_home_win"]),
        "Beraberlik": float(prediction["prob_draw"]),
        "Deplasman kazanır": float(prediction["prob_away_win"]),
    }
    predicted_outcome, outcome_probability = max(
        outcome_probabilities.items(), key=lambda item: item[1]
    )
    outcome_correct = predicted_outcome == _outcome_label(home_score, away_score)
    return "\n".join(
        (
            f"🏁 {home_team} {home_score} — {away_score} {away_team}",
            "Tahmin sonuçları",
            f"1-X-2: {predicted_outcome} %{outcome_probability * 100:.0f} {'✓' if outcome_correct else '✗'}",
            "Üst 2.5: "
            + _result_line(
                float(prediction["prob_over_2_5"]),
                actual_positive=home_score + away_score >= 3,
                positive_label="Üst",
                negative_label="Alt",
            ),
            "KG Var: "
            + _result_line(
                float(prediction["prob_btts"]),
                actual_positive=home_score > 0 and away_score > 0,
                positive_label="KG Var",
                negative_label="KG Yok",
            ),
        )
    )


def _pending_results(
    db: SupabaseRestClient,
    *,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Return queued deliveries that have not yet succeeded."""
    current_time = now or datetime.now(timezone.utc)
    evaluated = db.select(
        "result_notification_queue",
        columns="match_id,prediction_id,attempts",
        filters={
            "sent_at": "is.null",
            "next_attempt_at": f"lte.{current_time.isoformat()}",
        },
        limit=MAX_DELIVERIES_PER_RUN,
        order="available_at.asc",
    )
    return evaluated


def run_final_result_notifications() -> dict[str, int | str]:
    """Deliver each new evaluation once and persist the delivery marker."""
    settings = get_settings()
    db = SupabaseRestClient(settings.supabase_url, settings.supabase_service_role_key)
    pending = _pending_results(db)
    if not pending:
        return {"pending": 0, "sent": 0}
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not bot_token or not chat_id:
        return {"pending": len(pending), "sent": 0, "skipped": "telegram_not_configured"}

    match_ids = ",".join(str(int(row["match_id"])) for row in pending)
    prediction_ids = ",".join(str(int(row["prediction_id"])) for row in pending)
    matches = {
        int(row["id"]): row
        for row in db.select_all(
            "matches",
            columns="id,league_id,home_team_id,away_team_id,home_score,away_score",
            filters={"id": f"in.({match_ids})"},
        )
    }
    predictions = {
        int(row["id"]): row
        for row in db.select_all(
            "predictions",
            columns="id,model_version,prob_home_win,prob_draw,prob_away_win,prob_over_2_5,prob_btts",
            filters={"id": f"in.({prediction_ids})"},
        )
    }
    snapshots = {
        int(row["match_id"]): row
        for row in db.select_all(
            "prediction_snapshots",
            columns=(
                "match_id,model_version,prob_home_win,prob_draw,prob_away_win,"
                "prob_over_2_5,prob_btts,captured_at"
            ),
            filters={"match_id": f"in.({match_ids})", "snapshot_type": "eq.pre_match_60m"},
        )
    }
    team_ids = {int(team_id) for match in matches.values() for team_id in (match["home_team_id"], match["away_team_id"])}
    league_ids = {int(match["league_id"]) for match in matches.values()}
    teams = {int(row["id"]): str(row["name"]) for row in db.select_all("teams", columns="id,name", filters={"id": f"in.({','.join(map(str, team_ids))})"})}
    leagues = {int(row["id"]): str(row["name"]) for row in db.select_all("leagues", columns="id,name", filters={"id": f"in.({','.join(map(str, league_ids))})"})}

    sent = 0
    failed = 0
    delivered_at = datetime.now(timezone.utc)
    for evaluation in pending:
        match = matches.get(int(evaluation["match_id"]))
        prediction = snapshots.get(int(evaluation["match_id"])) or predictions.get(
            int(evaluation["prediction_id"])
        )
        if match is None or prediction is None:
            continue
        try:
            send_telegram_message(
                final_result_message(
                    match,
                    prediction,
                    home_team=teams.get(int(match["home_team_id"]), "Ev sahibi"),
                    away_team=teams.get(int(match["away_team_id"]), "Deplasman"),
                    league_name=leagues.get(int(match["league_id"]), "Lig"),
                ),
                bot_token=bot_token,
                chat_id=chat_id,
            )
        except TelegramError as error:
            attempts = int(evaluation.get("attempts", 0)) + 1
            retry_at = delivered_at + timedelta(minutes=min(60, 5 * (2 ** min(attempts, 4))))
            db.upsert(
                "result_notification_queue",
                [{
                    "prediction_id": int(evaluation["prediction_id"]),
                    "match_id": int(evaluation["match_id"]),
                    "attempts": attempts,
                    "next_attempt_at": retry_at.isoformat(),
                    "last_error": type(error).__name__,
                }],
                on_conflict="prediction_id",
            )
            failed += 1
            continue
        db.upsert(
            "result_notification_queue",
            [{
                "prediction_id": int(evaluation["prediction_id"]),
                "match_id": int(evaluation["match_id"]),
                "sent_at": delivered_at.isoformat(),
                "attempts": int(evaluation.get("attempts", 0)) + 1,
                "next_attempt_at": delivered_at.isoformat(),
                "last_error": None,
            }],
            on_conflict="prediction_id",
        )
        sent += 1
    return {"pending": len(pending), "sent": sent, "failed": failed}


def main() -> None:
    print(json.dumps(run_final_result_notifications(), ensure_ascii=False))


if __name__ == "__main__":
    main()
