"""Send concise morning prediction and nightly performance Telegram summaries."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from config.settings import get_settings
from db.db_client import SupabaseRestClient
from notifications.telegram import send_from_environment


def _prediction_signal(row: dict[str, Any]) -> tuple[str, float]:
    candidates = (
        ("Ev kazanır", float(row["prob_home_win"])),
        ("Beraberlik", float(row["prob_draw"])),
        ("Deplasman kazanır", float(row["prob_away_win"])),
        ("Üst 2.5", float(row["prob_over_2_5"])),
        ("KG Var", float(row["prob_btts"])),
    )
    return max(candidates, key=lambda item: item[1])


def build_morning_message(
    matches: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    teams: list[dict[str, Any]],
    leagues: list[dict[str, Any]],
) -> str:
    """Build a bounded, readable upcoming-fixture summary without betting language."""
    teams_by_id = {int(row["id"]): str(row["name"]) for row in teams}
    leagues_by_id = {int(row["id"]): str(row["name"]) for row in leagues}
    latest_predictions: dict[int, dict[str, Any]] = {}
    for row in sorted(predictions, key=lambda item: str(item["predicted_at"]), reverse=True):
        latest_predictions.setdefault(int(row["match_id"]), row)

    lines = ["⚽ Maç Analiz · Günlük tahmin özeti"]
    for match in sorted(matches, key=lambda item: str(item["match_date"])):
        prediction = latest_predictions.get(int(match["id"]))
        if prediction is None:
            continue
        signal, probability = _prediction_signal(prediction)
        home = teams_by_id.get(int(match["home_team_id"]), "Ev sahibi")
        away = teams_by_id.get(int(match["away_team_id"]), "Deplasman")
        league = leagues_by_id.get(int(match["league_id"]), "Lig")
        kickoff = datetime.fromisoformat(str(match["match_date"]).replace("Z", "+00:00"))
        lines.append(
            f"• {kickoff.astimezone(ZoneInfo('Europe/Istanbul')).strftime('%d.%m %H:%M')} · {league}\n"
            f"  {home} — {away}: {signal} (%{probability * 100:.0f})"
        )
        if len(lines) == 6:
            break
    if len(lines) == 1:
        lines.append("Önümüzdeki 3 gün için gönderilebilir tahmin bulunamadı.")
    lines.append("Tahminler istatistiksel olasılıktır; kesin sonuç değildir.")
    return "\n".join(lines)


def build_night_message(performance: list[dict[str, Any]]) -> str:
    """Build a recent evaluated-performance summary."""
    if not performance:
        return "⚽ Maç Analiz · Gece özeti\nHenüz değerlendirilmiş tahmin yok."
    recent = performance[:30]
    accuracy = sum(bool(row["was_correct"]) for row in recent) / len(recent)
    brier = sum(float(row["brier_score"]) for row in recent) / len(recent)
    return (
        "⚽ Maç Analiz · Gece performans özeti\n"
        f"Son {len(recent)} değerlendirme: %{accuracy * 100:.1f} 1-X-2 isabet\n"
        f"Ortalama Brier: {brier:.3f}\n"
        "Tahminler istatistiksel olasılıktır; kesin sonuç değildir."
    )


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
        message = build_morning_message(*_morning_data(db))
    else:
        message = build_night_message(
            db.select("prediction_performance", columns="was_correct,brier_score", limit=30, order="evaluated_at.desc")
        )

    if send_from_environment(message):
        print(f"Telegram {args.mode} summary sent")
    else:
        print("Telegram notification skipped: credentials are not configured")


if __name__ == "__main__":
    main()
