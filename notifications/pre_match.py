"""Refresh context and send one Telegram message about 20 minutes before kickoff."""

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
from data_pipeline.match_commentary import MatchCommentaryError, generate_match_commentary
from data_pipeline.odds import MatchOdds, fetch_match_odds, record_odds_quote
from data_pipeline.refresh_context import current_elo_ratings
from db.db_client import DatabaseError, SupabaseRestClient
from models.feature_engineering import CausalFeatureState
from models.predict import generate_prediction_rows, load_latest_team_forms, persist_predictions, resolve_model_path
from models.shadow import run_shadow_predictions
from models.train_model import load_completed_matches
from notifications.telegram import send_telegram_message


NOTIFICATION_TYPE = "pre_match_60m"
WINDOW_START_MINUTES = 15
WINDOW_END_MINUTES = 25
LINEUP_LOOKAHEAD_MINUTES = 90
MAX_TELEGRAM_COMMENTARY_CHARS = 1_400


def _form_summary(form: dict[str, Any] | None) -> str:
    """Format refreshed team-form data without claiming unavailable match history."""
    if not form:
        return "Güncel form verisi yok"
    return (
        f"Son 5 galibiyet oranı %{float(form.get('win_rate_last5') or 0) * 100:.0f}; "
        f"atılan gol {float(form.get('avg_goals_scored_last5') or 0):.2f}; "
        f"yenilen gol {float(form.get('avg_goals_conceded_last5') or 0):.2f}"
    )


def _absence_summary(rows: list[dict[str, Any]], *, team_id: int) -> list[str]:
    labels = {"injured": "sakat", "suspended": "cezalı", "doubtful": "şüpheli"}
    return [
        f"{str(row['player_name'])[:100]} ({labels[str(row['status'])]})"
        for row in rows
        if int(row.get("team_id", -1)) == team_id
        and str(row.get("status")) in labels
        and row.get("player_name")
    ][:12]


def _telegram_commentary(text: str | None) -> str | None:
    """Keep the optional AI section safely below Telegram's message size limit."""
    if not text or not text.strip():
        return None
    normalized = text.strip()
    if len(normalized) > MAX_TELEGRAM_COMMENTARY_CHARS:
        normalized = normalized[:MAX_TELEGRAM_COMMENTARY_CHARS].rsplit(" ", 1)[0] + "…"
    return normalized


def generate_notification_commentary(
    *,
    match: dict[str, Any],
    prediction: dict[str, Any],
    historical: list[dict[str, Any]],
    team_forms: dict[int, dict[str, Any]],
    availability_rows: list[dict[str, Any]],
    home_team: str,
    away_team: str,
) -> str:
    """Generate one pre-kickoff commentary using the same refreshed model context."""
    state = CausalFeatureState()
    for completed_match in historical:
        state.update(completed_match)
    baseline = state.poisson_baseline(match)
    home_id = int(match["home_team_id"])
    away_id = int(match["away_team_id"])
    return generate_match_commentary(
        home_team=home_team,
        away_team=away_team,
        home_xg=baseline.home_expected_goals,
        away_xg=baseline.away_expected_goals,
        home_absences=_absence_summary(availability_rows, team_id=home_id),
        away_absences=_absence_summary(availability_rows, team_id=away_id),
        home_form=_form_summary(team_forms.get(home_id)),
        away_form=_form_summary(team_forms.get(away_id)),
        home_win_probability=float(prediction["prob_home_win"]),
        draw_probability=float(prediction["prob_draw"]),
        away_win_probability=float(prediction["prob_away_win"]),
    )


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
    """Return fixtures inside the five-minute tolerance around the 20-minute target."""
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
    commentary: str | None = None,
) -> str:
    kickoff = datetime.fromisoformat(str(match["match_date"]).replace("Z", "+00:00"))
    outcomes = {
        "Ev kazanır": float(prediction["prob_home_win"]),
        "Beraberlik": float(prediction["prob_draw"]),
        "Deplasman kazanır": float(prediction["prob_away_win"]),
    }
    outcome, probability = max(outcomes.items(), key=lambda item: item[1])
    lines = [
        f"⚽ {home_team} — {away_team}",
        f"⏰ {kickoff.astimezone(ZoneInfo('Europe/Istanbul')).strftime('%d.%m · %H:%M')}",
        "",
        f"Tahmin: {outcome} %{probability * 100:.0f}",
        f"Üst 2.5: %{float(prediction['prob_over_2_5']) * 100:.0f} · "
        f"KG Var: %{float(prediction['prob_btts']) * 100:.0f}",
    ]
    safe_commentary = _telegram_commentary(commentary)
    if safe_commentary:
        lines.extend(("", "🧠 Maç yorumu", safe_commentary))
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
) -> tuple[
    list[dict[str, Any]], str, list[dict[str, Any]], dict[int, dict[str, Any]]
]:
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
    team_forms = load_latest_team_forms(db, team_ids)
    rows = generate_prediction_rows(
        bundle,
        historical,
        matches,
        model_version=model_version,
        team_form_by_id=team_forms,
    )
    persisted = persist_predictions(db, rows)
    try:
        run_shadow_predictions(
            db,
            matches=matches,
            historical_matches=historical,
            team_form_by_id=team_forms,
        )
    except Exception as error:
        # Shadow evaluation must never delay or suppress a production alert.
        print(f"Shadow forecast skipped: {type(error).__name__}")
    return persisted, model_version, historical, team_forms


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
    predictions, model_version, historical, team_forms = _refresh_and_predict(api, db, matches)
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
    availability_rows = db.select_all(
        "player_availability",
        columns="team_id,player_name,status",
        filters={"team_id": f"in.({','.join(map(str, team_ids))})"},
    )
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
        try:
            commentary = generate_notification_commentary(
                match=match,
                prediction=prediction,
                historical=historical,
                team_forms=team_forms,
                availability_rows=availability_rows,
                home_team=teams.get(int(match["home_team_id"]), "Ev sahibi"),
                away_team=teams.get(int(match["away_team_id"]), "Deplasman"),
            )
        except MatchCommentaryError as error:
            # An optional LLM must never suppress the time-sensitive model alert.
            print(f"Gemini commentary skipped for fixture {match_id}: {type(error).__name__}")
            commentary = None
        send_telegram_message(
            pre_match_message(
                match,
                prediction,
                home_team=teams.get(int(match["home_team_id"]), "Ev sahibi"),
                away_team=teams.get(int(match["away_team_id"]), "Deplasman"),
                league_name=leagues.get(int(match["league_id"]), "Lig"),
                odds=odds,
                commentary=commentary,
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
