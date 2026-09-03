"""Generate and persist model probabilities for upcoming fixtures."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from config.settings import PROJECT_ROOT, get_settings
from db.db_client import SupabaseRestClient
from data_pipeline.odds import attach_pre_match_odds
from models.calibration import apply_binary_temperature, apply_multiclass_temperature
from models.feature_engineering import FEATURE_COLUMNS, build_upcoming_features
from models.train_model import load_completed_matches


def resolve_model_path(model_path: Path | None = None) -> Path:
    if model_path is not None:
        if not model_path.is_file():
            raise FileNotFoundError(f"Model not found: {model_path}")
        return model_path

    model_dir = PROJECT_ROOT / "models" / "saved_models"
    latest_path = model_dir / "latest.joblib"
    if latest_path.is_file():
        return latest_path
    candidates = sorted(model_dir.glob("model_v*.joblib"))
    if not candidates:
        raise FileNotFoundError("No versioned trained model was found")
    return candidates[-1]


def load_upcoming_matches(
    db: SupabaseRestClient,
    *,
    now: datetime,
    horizon_days: int,
) -> list[dict[str, Any]]:
    if horizon_days < 1 or horizon_days > 14:
        raise ValueError("horizon_days must be between 1 and 14")
    end = now + timedelta(days=horizon_days)
    matches = db.select_all(
        "matches",
        columns=(
            "id,league_id,home_team_id,away_team_id,match_date,status,"
            "home_score,away_score"
        ),
        filters={
            "status": "eq.scheduled",
            "and": f"(match_date.gte.{now.isoformat()},match_date.lte.{end.isoformat()})",
        },
        order="match_date.asc,id.asc",
    )
    quotes = db.select_all("odds_quote_history", columns="match_id,odds,captured_at", order="captured_at.asc")
    availability = db.select_all(
        "team_availability_status",
        columns="team_id,available_count,unavailable_count",
    )
    available_counts = {
        int(row["team_id"]): int(row["available_count"]) for row in availability
    }
    unavailable_counts = {
        int(row["team_id"]): int(row["unavailable_count"]) for row in availability
    }
    unavailable_players = db.select_all("player_availability", columns="team_id,status")
    unavailable_by_team: dict[int, list[dict[str, Any]]] = {}
    for player in unavailable_players:
        unavailable_by_team.setdefault(int(player["team_id"]), []).append(player)
    lineups = db.select_all("fixture_lineups", columns="match_id,team_id")
    confirmed = {(int(row["match_id"]), int(row["team_id"])) for row in lineups}
    enriched = attach_pre_match_odds(matches, quotes, observed_at=now)
    for match in enriched:
        home_id, away_id = int(match["home_team_id"]), int(match["away_team_id"])
        match["home_available_count"] = available_counts.get(home_id, 22)
        match["away_available_count"] = available_counts.get(away_id, 22)
        match["home_unavailable_count"] = unavailable_counts.get(home_id, 0)
        match["away_unavailable_count"] = unavailable_counts.get(away_id, 0)
        match["home_unavailable_players"] = unavailable_by_team.get(home_id, [])
        match["away_unavailable_players"] = unavailable_by_team.get(away_id, [])
        match["home_lineup_confirmed"] = (int(match["id"]), home_id) in confirmed
        match["away_lineup_confirmed"] = (int(match["id"]), away_id) in confirmed
    return enriched


def generate_prediction_rows(
    bundle: dict[str, Any],
    historical_matches: list[dict[str, Any]],
    upcoming_matches: list[dict[str, Any]],
    *,
    model_version: str,
    team_form_by_id: dict[int, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    expected_columns = list(FEATURE_COLUMNS)
    if bundle.get("feature_columns") != expected_columns:
        raise ValueError("Saved model feature contract is incompatible")
    if not upcoming_matches:
        return []

    ordered_upcoming = sorted(
        upcoming_matches, key=lambda row: (row["match_date"], int(row["id"]))
    )
    features = build_upcoming_features(
        historical_matches,
        ordered_upcoming,
        team_form_by_id=team_form_by_id,
    )
    calibration = bundle.get("calibration", {})
    result_probabilities = apply_multiclass_temperature(
        bundle["result_model"].predict_proba(features),
        float(calibration.get("result_temperature", 1.0)),
    )
    over_probabilities = apply_binary_temperature(
        bundle["over_2_5_model"].predict_proba(features)[:, 1],
        float(calibration.get("over_2_5_temperature", 1.0)),
    )
    btts_probabilities = apply_binary_temperature(
        bundle["btts_model"].predict_proba(features)[:, 1],
        float(calibration.get("btts_temperature", 1.0)),
    )

    if result_probabilities.shape != (len(ordered_upcoming), 3):
        raise RuntimeError("Result model returned an unexpected probability shape")
    if not np.allclose(result_probabilities.sum(axis=1), 1.0, atol=1e-6):
        raise RuntimeError("Result probabilities are not normalized")

    predicted_at = datetime.now(timezone.utc).isoformat()
    rows: list[dict[str, Any]] = []
    for index, match in enumerate(ordered_upcoming):
        rows.append(
            {
                "match_id": int(match["id"]),
                "model_version": model_version,
                "prob_home_win": float(result_probabilities[index, 0]),
                "prob_draw": float(result_probabilities[index, 1]),
                "prob_away_win": float(result_probabilities[index, 2]),
                "prob_over_2_5": float(over_probabilities[index]),
                "prob_btts": float(btts_probabilities[index]),
                "predicted_at": predicted_at,
            }
        )
    return rows


def load_latest_team_forms(
    db: SupabaseRestClient, team_ids: set[int]
) -> dict[int, dict[str, Any]]:
    """Return the latest stored context row for each upcoming team."""
    if not team_ids:
        return {}
    rows = db.select_all(
        "team_form",
        columns=(
            "team_id,calculated_at,elo_rating,avg_goals_scored_last5,"
            "avg_goals_conceded_last5,win_rate_last5,home_away_split"
        ),
        filters={"team_id": f"in.({','.join(str(team_id) for team_id in sorted(team_ids))})"},
        order="calculated_at.desc",
    )
    latest: dict[int, dict[str, Any]] = {}
    for row in rows:
        team_id = int(row["team_id"])
        if team_id in team_ids and team_id not in latest:
            latest[team_id] = row
    return latest


def persist_predictions(
    db: SupabaseRestClient, rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    return db.upsert(
        "predictions", rows, on_conflict="match_id,model_version"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--days", type=int, default=3)
    args = parser.parse_args()

    settings = get_settings()
    db = SupabaseRestClient(settings.supabase_url, settings.supabase_service_role_key)
    model_path = resolve_model_path(args.model)
    bundle = joblib.load(model_path)
    now = datetime.now(timezone.utc)
    historical = load_completed_matches(db)
    upcoming = load_upcoming_matches(db, now=now, horizon_days=args.days)
    upcoming_team_ids = {
        int(team_id)
        for match in upcoming
        for team_id in (match["home_team_id"], match["away_team_id"])
    }
    team_forms = load_latest_team_forms(db, upcoming_team_ids)
    rows = generate_prediction_rows(
        bundle,
        historical,
        upcoming,
        model_version=str(bundle.get("model_version", model_path.stem)),
        team_form_by_id=team_forms,
    )
    persisted = persist_predictions(db, rows)
    print(
        json.dumps(
            {
                "model_version": str(bundle.get("model_version", model_path.stem)),
                "historical_matches": len(historical),
                "upcoming_matches": len(upcoming),
                "predictions_written": len(persisted),
                "teams_with_live_form": len(team_forms),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
