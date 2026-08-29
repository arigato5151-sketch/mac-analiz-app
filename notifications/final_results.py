"""Send one audited Telegram result card for each newly evaluated fixture."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any

from config.settings import get_settings
from db.db_client import SupabaseRestClient
from notifications.telegram import send_telegram_message


NOTIFICATION_TYPE = "final_result"


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
    """Build a compact 1-X-2, totals, and BTTS result audit message."""
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
            league_name,
            f"1-X-2: {predicted_outcome} (%{outcome_probability * 100:.0f}) {'✓' if outcome_correct else '✗'}",
            "Üst/Alt 2.5: "
            + _result_line(
                float(prediction["prob_over_2_5"]),
                actual_positive=home_score + away_score >= 3,
                positive_label="Üst",
                negative_label="Alt",
            ),
            "KG Var/Yok: "
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
    """Return only evaluations made by the current nightly processing run.

    Restricting the window prevents the first deployment from replaying an
    archive of historical fixtures to Telegram.
    """
    current_time = now or datetime.now(timezone.utc)
    recent_cutoff = (current_time - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
    evaluated = db.select(
        "prediction_performance",
        columns="match_id,prediction_id",
        filters={"evaluated_at": f"gte.{recent_cutoff}"},
        limit=100,
        order="evaluated_at.desc",
    )
    if not evaluated:
        return []
    match_ids = ",".join(str(int(row["match_id"])) for row in evaluated)
    sent = db.select_all(
        "notification_log",
        columns="match_id",
        filters={
            "match_id": f"in.({match_ids})",
            "notification_type": f"eq.{NOTIFICATION_TYPE}",
        },
    )
    sent_ids = {int(row["match_id"]) for row in sent}
    return [row for row in evaluated if int(row["match_id"]) not in sent_ids]


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
    team_ids = {int(team_id) for match in matches.values() for team_id in (match["home_team_id"], match["away_team_id"])}
    league_ids = {int(match["league_id"]) for match in matches.values()}
    teams = {int(row["id"]): str(row["name"]) for row in db.select_all("teams", columns="id,name", filters={"id": f"in.({','.join(map(str, team_ids))})"})}
    leagues = {int(row["id"]): str(row["name"]) for row in db.select_all("leagues", columns="id,name", filters={"id": f"in.({','.join(map(str, league_ids))})"})}

    sent = 0
    for evaluation in pending:
        match = matches.get(int(evaluation["match_id"]))
        prediction = predictions.get(int(evaluation["prediction_id"]))
        if match is None or prediction is None:
            continue
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
        db.upsert(
            "notification_log",
            [{"match_id": int(match["id"]), "notification_type": NOTIFICATION_TYPE, "model_version": prediction["model_version"]}],
            on_conflict="match_id,notification_type",
        )
        sent += 1
    return {"pending": len(pending), "sent": sent}


def main() -> None:
    print(json.dumps(run_final_result_notifications(), ensure_ascii=False))


if __name__ == "__main__":
    main()
