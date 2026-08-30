"""Send concise morning prediction and nightly performance Telegram summaries."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from config.settings import get_settings
from db.db_client import SupabaseRestClient
from notifications.telegram import send_from_environment, send_many_from_environment


def build_morning_messages(
    matches: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    teams: list[dict[str, Any]],
    leagues: list[dict[str, Any]],
) -> list[str]:
    """Build one compact, complete probability card per upcoming fixture."""
    teams_by_id = {int(row["id"]): str(row["name"]) for row in teams}
    leagues_by_id = {int(row["id"]): str(row["name"]) for row in leagues}
    latest_predictions: dict[int, dict[str, Any]] = {}
    for row in sorted(predictions, key=lambda item: str(item["predicted_at"]), reverse=True):
        latest_predictions.setdefault(int(row["match_id"]), row)

    messages: list[str] = []
    for match in sorted(matches, key=lambda item: str(item["match_date"])):
        prediction = latest_predictions.get(int(match["id"]))
        if prediction is None:
            continue
        home = teams_by_id.get(int(match["home_team_id"]), "Ev sahibi")
        away = teams_by_id.get(int(match["away_team_id"]), "Deplasman")
        league = leagues_by_id.get(int(match["league_id"]), "Lig")
        kickoff = datetime.fromisoformat(str(match["match_date"]).replace("Z", "+00:00"))
        messages.append(
            "\n".join(
                (
                    f"⚽ {home} — {away}",
                    f"{league} · {kickoff.astimezone(ZoneInfo('Europe/Istanbul')).strftime('%d.%m %H:%M')}",
                    "1-X-2: "
                    f"1 %{float(prediction['prob_home_win']) * 100:.0f} · "
                    f"X %{float(prediction['prob_draw']) * 100:.0f} · "
                    f"2 %{float(prediction['prob_away_win']) * 100:.0f}",
                    "Üst/Alt 2.5: "
                    f"Üst %{float(prediction['prob_over_2_5']) * 100:.0f} · "
                    f"Alt %{(1 - float(prediction['prob_over_2_5'])) * 100:.0f}",
                    "KG Var/Yok: "
                    f"Var %{float(prediction['prob_btts']) * 100:.0f} · "
                    f"Yok %{(1 - float(prediction['prob_btts'])) * 100:.0f}",
                    "İstatistiksel olasılıktır; kesin sonuç değildir.",
                )
            )
        )
    return messages


def build_morning_message(
    matches: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    teams: list[dict[str, Any]],
    leagues: list[dict[str, Any]],
) -> str:
    """Compatibility helper for text previews and the no-fixture case."""
    messages = build_morning_messages(matches, predictions, teams, leagues)
    return "\n\n".join(messages) if messages else "Önümüzdeki 3 gün için tahmin bulunamadı."


def build_night_message(performance: list[dict[str, Any]]) -> str:
    """Build a recent evaluated-performance summary."""
    if not performance:
        return "⚽ Maç Analiz · Gece özeti\nHenüz değerlendirilmiş tahmin yok."
    recent = performance[:30]
    accuracy = sum(bool(row["was_correct"]) for row in recent) / len(recent)
    brier = sum(float(row["brier_score"]) for row in recent) / len(recent)
    def binary_accuracy(column: str) -> str:
        available = [bool(row[column]) for row in recent if row.get(column) is not None]
        return f"%{sum(available) / len(available) * 100:.1f}" if available else "—"
    return (
        "⚽ Maç Analiz · Gece performans özeti\n"
        f"Son {len(recent)} değerlendirme: %{accuracy * 100:.1f} 1-X-2 isabet\n"
        f"Üst/Alt 2.5: {binary_accuracy('over_2_5_was_correct')} · "
        f"KG Var/Yok: {binary_accuracy('btts_was_correct')}\n"
        f"Ortalama Brier: {brier:.3f}\n"
        "Tahminler istatistiksel olasılıktır; kesin sonuç değildir."
    )


def _official_performance_rows(
    performance: list[dict[str, Any]], predictions: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Choose one latest model snapshot per match for the nightly summary."""
    predicted_at_by_id = {
        int(row["id"]): str(row["predicted_at"]) for row in predictions
    }
    selected: dict[int, dict[str, Any]] = {}
    for row in performance:
        match_id = int(row["match_id"])
        current = selected.get(match_id)
        key = (predicted_at_by_id.get(int(row["prediction_id"]), ""), int(row["prediction_id"]))
        current_key = (
            (
                predicted_at_by_id.get(int(current["prediction_id"]), ""),
                int(current["prediction_id"]),
            )
            if current is not None
            else None
        )
        if current_key is None or key > current_key:
            selected[match_id] = row
    return sorted(selected.values(), key=lambda row: str(row["evaluated_at"]), reverse=True)


def _morning_data(db: SupabaseRestClient) -> tuple[list[dict[str, Any]], ...]:
    now = datetime.now(timezone.utc)
    end = now + timedelta(days=3)
    matches = db.select_all(
        "matches",
        columns="id,league_id,home_team_id,away_team_id,match_date",
        filters={
            "status": "eq.scheduled",
            "and": f"(match_date.gte.{now.isoformat()},match_date.lte.{end.isoformat()})",
        },
        order="match_date.asc",
    )
    return (
        matches,
        db.select_all(
            "predictions",
            columns=(
                "match_id,prob_home_win,prob_draw,prob_away_win,prob_over_2_5,"
                "prob_btts,predicted_at"
            ),
            order="predicted_at.desc",
        ),
        db.select_all("teams", columns="id,name"),
        db.select_all("leagues", columns="id,name"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("morning", "night"), required=True)
    args = parser.parse_args()
    settings = get_settings()
    db = SupabaseRestClient(settings.supabase_url, settings.supabase_service_role_key)
    if args.mode == "morning":
        messages = build_morning_messages(*_morning_data(db))
        sent = send_many_from_environment(messages)
        if messages and sent:
            print(f"Telegram morning messages sent: {sent}")
        elif messages:
            print("Telegram notification skipped: credentials are not configured")
        else:
            print("Telegram morning notification skipped: no predictions available")
        return
    else:
        performance_rows = db.select_all(
            "prediction_performance",
            columns=(
                "prediction_id,match_id,was_correct,brier_score,over_2_5_was_correct,"
                "btts_was_correct,evaluated_at"
            ),
        )
        message = build_night_message(
            _official_performance_rows(
                performance_rows,
                db.select_all("predictions", columns="id,predicted_at"),
            )[:30]
        )

    if send_from_environment(message):
        print(f"Telegram {args.mode} summary sent")
    else:
        print("Telegram notification skipped: credentials are not configured")


if __name__ == "__main__":
    main()
