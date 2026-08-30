"""Refresh context and send one Telegram message around 60 minutes before kickoff."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import joblib

from config.leagues import LEAGUES_BY_ID
from config.settings import PROJECT_ROOT, get_settings
from data_pipeline.api_client import ApiFootballClient
from data_pipeline.fetch_injuries import sync_injuries
from data_pipeline.fetch_lineups import sync_fixture_lineups
from data_pipeline.fetch_team_stats import sync_team_form
from data_pipeline.odds import MatchOdds, fetch_match_odds, record_odds_quote
from data_pipeline.refresh_context import current_elo_ratings
from db.db_client import DatabaseError, SupabaseRestClient
from models.predict import generate_prediction_rows, load_latest_team_forms, persist_predictions, resolve_model_path
from models.shadow import run_shadow_predictions
from models.train_model import load_completed_matches
from notifications.telegram import send_telegram_message


NOTIFICATION_TYPE = "pre_match_60m"
WINDOW_START_MINUTES = 45
WINDOW_END_MINUTES = 75
LINEUP_LOOKAHEAD_MINUTES = 90


def persist_production_snapshot(
    db: SupabaseRestClient,
    prediction: dict[str, Any],
    *,
    match_id: int,
    model_version: str,
    captured_at: str,
) -> dict[str, Any]:
    """Store the user-facing pre-match output once; never rewrite it on retries."""
    existing = db.select(
        "prediction_snapshots",
        columns="id,source_prediction_id,match_id,snapshot_type,model_version,prob_home_win,prob_draw,prob_away_win,prob_over_2_5,prob_btts,source_predicted_at,context,captured_at",
        filters={"match_id": f"eq.{match_id}", "snapshot_type": f"eq.{NOTIFICATION_TYPE}"},
        limit=1,
    )
    if existing:
        return existing[0]

    snapshot = {
        "source_prediction_id": int(prediction["id"]),
        "match_id": match_id,
        "snapshot_type": NOTIFICATION_TYPE,
        "model_version": model_version,
        "prob_home_win": float(prediction["prob_home_win"]),
        "prob_draw": float(prediction["prob_draw"]),
        "prob_away_win": float(prediction["prob_away_win"]),
        "prob_over_2_5": float(prediction["prob_over_2_5"]),
        "prob_btts": float(prediction["prob_btts"]),
        "source_predicted_at": str(prediction["predicted_at"]),
        "context": {
            "notification_type": NOTIFICATION_TYPE,
            "notification_window_minutes": [WINDOW_START_MINUTES, WINDOW_END_MINUTES],
        },
        "captured_at": captured_at,
    }
    try:
        return db.insert("prediction_snapshots", [snapshot])[0]
    except DatabaseError:
        # A concurrent retry can win the unique constraint race. Read its
        # immutable winner instead of overwriting the historical record.
        existing = db.select(
            "prediction_snapshots",
            columns="id,source_prediction_id,match_id,snapshot_type,model_version,prob_home_win,prob_draw,prob_away_win,prob_over_2_5,prob_btts,source_predicted_at,context,captured_at",
            filters={"match_id": f"eq.{match_id}", "snapshot_type": f"eq.{NOTIFICATION_TYPE}"},
            limit=1,
        )
        if existing:
            return existing[0]
        raise


def due_matches(matches: list[dict[str, Any]], *, now: datetime) -> list[dict[str, Any]]:
    """Return scheduled fixtures within the 60-minute notification window."""
    start = now + timedelta(minutes=WINDOW_START_MINUTES)
    end = now + timedelta(minutes=WINDOW_END_MINUTES)
    eligible: list[dict[str, Any]] = []
    for match in matches:
        kickoff = datetime.fromisoformat(str(match["match_date"]).replace("Z", "+00:00"))
        if start <= kickoff <= end:
            eligible.append(match)
    return sorted(eligible, key=lambda row: str(row["match_date"]))


def pre_match_message(
    match: dict[str, Any],
    prediction: dict[str, Any],
    *,
    home_team: str,
    away_team: str,
    league_name: str,
    odds: MatchOdds | None = None,
) -> str:
    kickoff = datetime.fromisoformat(str(match["match_date"]).replace("Z", "+00:00"))
    lines = [
        f"⚽ {home_team} — {away_team}",
        f"{league_name} · {kickoff.astimezone(ZoneInfo('Europe/Istanbul')).strftime('%d.%m %H:%M')}",
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
    ]
    if odds is not None:
        if all((odds.home_win, odds.draw, odds.away_win)):
            lines.append(
                f"{odds.bookmaker} 1-X-2: 1 @ {odds.home_win} · "
                f"X @ {odds.draw} · 2 @ {odds.away_win}"
            )
        if all((odds.over_2_5, odds.under_2_5)):
            lines.append(
                f"{odds.bookmaker} Üst/Alt 2.5: Üst @ {odds.over_2_5} · "
                f"Alt @ {odds.under_2_5}"
            )
        if all((odds.btts_yes, odds.btts_no)):
            lines.append(
                f"{odds.bookmaker} KG Var/Yok: Var @ {odds.btts_yes} · "
                f"Yok @ {odds.btts_no}"
            )
    lines.append("İstatistiksel olasılıktır; kesin sonuç değildir.")
    return "\n".join(lines)


def _pending_matches(db: SupabaseRestClient, *, now: datetime) -> list[dict[str, Any]]:
    end = now + timedelta(minutes=WINDOW_END_MINUTES)
    scheduled = db.select_all(
        "matches",
        columns="id,league_id,home_team_id,away_team_id,match_date,status",
        filters={
            "status": "eq.scheduled",
            "and": f"(match_date.gte.{now.isoformat()},match_date.lte.{end.isoformat()})",
        },
        order="match_date.asc",
    )
    candidates = due_matches(scheduled, now=now)
    if not candidates:
        return []
    match_ids = ",".join(str(int(match["id"])) for match in candidates)
    sent = db.select_all(
        "notification_log",
        columns="match_id",
        filters={
            "match_id": f"in.({match_ids})",
            "notification_type": f"eq.{NOTIFICATION_TYPE}",
        },
    )
    sent_ids = {int(row["match_id"]) for row in sent}
    return [match for match in candidates if int(match["id"]) not in sent_ids]


def sync_soon_lineups(api: ApiFootballClient, db: SupabaseRestClient, *, now: datetime) -> int:
    """Poll only the next 90 minutes until both official XIs are available."""
    soon = db.select_all(
        "matches",
        columns="id",
        filters={
            "status": "eq.scheduled",
            "and": f"(match_date.gte.{now.isoformat()},match_date.lte.{(now + timedelta(minutes=LINEUP_LOOKAHEAD_MINUTES)).isoformat()})",
        },
    )
    if not soon:
        return 0
    match_ids = ",".join(str(int(row["id"])) for row in soon)
    existing = db.select_all(
        "fixture_lineups", columns="match_id", filters={"match_id": f"in.({match_ids})"}
    )
    confirmed_counts: dict[int, int] = {}
    for row in existing:
        match_id = int(row["match_id"])
        confirmed_counts[match_id] = confirmed_counts.get(match_id, 0) + 1
    written = 0
    for match in soon:
        match_id = int(match["id"])
        if confirmed_counts.get(match_id, 0) >= 2:
            continue
        written += sync_fixture_lineups(api, db, match_id=match_id)
    return written


def sync_soon_odds(api: ApiFootballClient, db: SupabaseRestClient, *, now: datetime) -> int:
    """Capture meaningful market moves in the same bounded pre-kickoff window."""
    soon = db.select_all(
        "matches", columns="id",
        filters={"status": "eq.scheduled", "and": f"(match_date.gte.{now.isoformat()},match_date.lte.{(now + timedelta(minutes=LINEUP_LOOKAHEAD_MINUTES)).isoformat()})"},
    )
    written = 0
    for match in soon:
        match_id = int(match["id"])
        try:
            odds = fetch_match_odds(api, fixture_id=match_id)
            if odds and record_odds_quote(db, match_id=match_id, odds=odds, captured_at=now.isoformat()):
                written += 1
        except Exception as error:
            print(f"Odds history unavailable for fixture {match_id}: {type(error).__name__}")
    return written


def _refresh_and_predict(
    api: ApiFootballClient,
    db: SupabaseRestClient,
    matches: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    """Refresh only the affected teams, then regenerate their probabilities."""
    historical = load_completed_matches(db)
    team_ids = {
        int(team_id)
        for match in matches
        for team_id in (match["home_team_id"], match["away_team_id"])
    }
    elo_by_team = current_elo_ratings(historical, team_ids)
    teams_by_league: dict[int, set[int]] = {}
    for match in matches:
        league_id = int(match["league_id"])
        if league_id not in LEAGUES_BY_ID:
            raise ValueError(f"Untracked league in pre-match queue: {league_id}")
        teams_by_league.setdefault(league_id, set()).update(
            (int(match["home_team_id"]), int(match["away_team_id"]))
        )
    for league_id, affected_teams in teams_by_league.items():
        season = LEAGUES_BY_ID[league_id].season
        for team_id in sorted(affected_teams):
            sync_team_form(api, db, team_id=team_id, season=season, elo_rating=elo_by_team[team_id])
        sync_injuries(api, db, league_id=league_id, team_ids=affected_teams)

    model_path = resolve_model_path(PROJECT_ROOT / "models" / "saved_models" / "latest.joblib")
    bundle = joblib.load(model_path)
    model_version = str(bundle.get("model_version", model_path.stem))
    rows = generate_prediction_rows(
        bundle,
        historical,
        matches,
        model_version=model_version,
        team_form_by_id=load_latest_team_forms(db, team_ids),
    )
    persisted = persist_predictions(db, rows)
    try:
        run_shadow_predictions(
            db,
            matches=matches,
            historical_matches=historical,
            team_form_by_id=load_latest_team_forms(db, team_ids),
        )
    except Exception as error:
        # Shadow evaluation must never delay or suppress a production alert.
        print(f"Shadow forecast skipped: {type(error).__name__}")
    return persisted, model_version


def run_pre_match_notifications(now: datetime | None = None) -> dict[str, Any]:
    """Run one idempotent notification cycle for fixtures near kickoff."""
    settings = get_settings()
    db = SupabaseRestClient(settings.supabase_url, settings.supabase_service_role_key)
    now = now or datetime.now(timezone.utc)
    matches = _pending_matches(db, now=now)
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not bot_token or not chat_id:
        return {"due_matches": len(matches), "sent": 0, "skipped": "telegram_not_configured"}

    api = ApiFootballClient(settings.api_football_key)
    lineup_rows = sync_soon_lineups(api, db, now=now)
    odds_history_rows = sync_soon_odds(api, db, now=now)
    if not matches:
        return {"due_matches": 0, "sent": 0, "lineup_rows": lineup_rows, "odds_history_rows": odds_history_rows, "api": api.diagnostics()}
    predictions, model_version = _refresh_and_predict(api, db, matches)
    predictions_by_match = {int(row["match_id"]): row for row in predictions}
    team_ids = {int(team_id) for match in matches for team_id in (match["home_team_id"], match["away_team_id"])}
    league_ids = {int(match["league_id"]) for match in matches}
    teams = {
        int(row["id"]): str(row["name"])
        for row in db.select_all("teams", columns="id,name", filters={"id": f"in.({','.join(map(str, team_ids))})"})
    }
    leagues = {
        int(row["id"]): str(row["name"])
        for row in db.select_all("leagues", columns="id,name", filters={"id": f"in.({','.join(map(str, league_ids))})"})
    }
    sent = 0
    for match in matches:
        match_id = int(match["id"])
        prediction = predictions_by_match[match_id]
        snapshot = persist_production_snapshot(
            db,
            prediction,
            match_id=match_id,
            model_version=model_version,
            captured_at=now.isoformat(),
        )
        try:
            odds = fetch_match_odds(api, fixture_id=match_id)
        except Exception as error:  # Odds are optional; never suppress a prediction alert.
            print(f"Odds unavailable for fixture {match_id}: {type(error).__name__}")
            odds = None
        if odds is not None:
            record_odds_quote(
                db, match_id=match_id, odds=odds, captured_at=now.isoformat(),
                notification_reference=True,
            )
            db.upsert(
                "bookmaker_odds_snapshots",
                [{
                    "prediction_id": int(snapshot["source_prediction_id"]),
                    "match_id": match_id,
                    "bookmaker": odds.bookmaker,
                    "source_updated_at": odds.source_updated_at,
                    "odds": odds.as_snapshot(),
                    "captured_at": now.isoformat(),
                }],
                on_conflict="prediction_id",
            )
        send_telegram_message(
            pre_match_message(
                match,
                prediction,
                home_team=teams.get(int(match["home_team_id"]), "Ev sahibi"),
                away_team=teams.get(int(match["away_team_id"]), "Deplasman"),
                league_name=leagues.get(int(match["league_id"]), "Lig"),
                odds=odds,
            ),
            bot_token=bot_token,
            chat_id=chat_id,
        )
        db.upsert(
            "notification_log",
            [{"match_id": match_id, "notification_type": NOTIFICATION_TYPE, "model_version": model_version, "sent_at": now.isoformat()}],
            on_conflict="match_id,notification_type",
        )
        sent += 1
    return {"due_matches": len(matches), "sent": sent, "lineup_rows": lineup_rows, "odds_history_rows": odds_history_rows, "api": api.diagnostics()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(json.dumps(run_pre_match_notifications(), ensure_ascii=False))


if __name__ == "__main__":
    main()
