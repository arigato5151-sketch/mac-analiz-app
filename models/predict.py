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
from models.feature_engineering import FEATURE_COLUMNS, build_upcoming_features
from models.train_model import load_completed_matches


def resolve_model_path(model_path: Path | None = None) -> Path:
    if model_path is not None:
        if not model_path.is_file():
            raise FileNotFoundError(f"Model not found: {model_path}")
        return model_path

    model_dir = PROJECT_ROOT / "models" / "saved_models"
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
    return db.select_all(
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


def generate_prediction_rows(
    bundle: dict[str, Any],
    historical_matches: list[dict[str, Any]],
    upcoming_matches: list[dict[str, Any]],
    *,
    model_version: str,
) -> list[dict[str, Any]]:
    expected_columns = list(FEATURE_COLUMNS)
    if bundle.get("feature_columns") != expected_columns:
        raise ValueError("Saved model feature contract is incompatible")
    if not upcoming_matches:
        return []

    ordered_upcoming = sorted(
        upcoming_matches, key=lambda row: (row["match_date"], int(row["id"]))
    )
    features = build_upcoming_features(historical_matches, ordered_upcoming)
    result_probabilities = bundle["result_model"].predict_proba(features)
    over_probabilities = bundle["over_2_5_model"].predict_proba(features)[:, 1]
    btts_probabilities = bundle["btts_model"].predict_proba(features)[:, 1]

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
    rows = generate_prediction_rows(
        bundle,
        historical,
        upcoming,
        model_version=model_path.stem,
    )
    persisted = persist_predictions(db, rows)
    print(
        json.dumps(
            {
                "model_version": model_path.stem,
                "historical_matches": len(historical),
                "upcoming_matches": len(upcoming),
                "predictions_written": len(persisted),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
