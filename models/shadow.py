"""Run candidate models in shadow mode and require evidence before promotion."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import joblib

from config.settings import PROJECT_ROOT, get_settings
from data_pipeline.isotime import parse_iso_datetime
from db.db_client import DatabaseError, SupabaseRestClient
from evaluation.track_performance import EVALUATION_LOOKBACK_DAYS
from models.artifact_store import ArtifactStoreError, download_model
from models.predict import (
    generate_prediction_rows,
    load_latest_team_forms,
    load_upcoming_matches,
    newest_versioned_model,
)
from models.train_model import load_historical_matches


MINIMUM_PROMOTION_SAMPLE = 100
MINIMUM_BRIER_IMPROVEMENT = 0.005


def candidate_path(model_version: str) -> Path:
    path = PROJECT_ROOT / "models" / "saved_models" / f"{model_version}.joblib"
    if not path.is_file():
        # A fresh clone or CI checkout may not carry the binary. Re-hydrate it
        # from Supabase Storage rather than failing shadow evaluation.
        try:
            return download_model(f"{model_version}.joblib", dest_dir=path.parent)
        except ArtifactStoreError as error:
            raise FileNotFoundError(
                f"Candidate model artifact is missing: {model_version}"
            ) from error
    return path


def register_newest_candidate(db: SupabaseRestClient) -> dict[str, Any]:
    """Register the newest versioned artifact; `latest.joblib` is never a candidate."""
    model_dir = PROJECT_ROOT / "models" / "saved_models"
    path = newest_versioned_model(model_dir)
    if path is None:
        raise FileNotFoundError("No versioned model artifact was found")
    bundle = joblib.load(path)
    version = str(bundle.get("model_version", path.stem))
    existing = db.select(
        "model_candidates",
        columns="model_version,status",
        filters={"model_version": f"eq.{version}"},
        limit=1,
    )
    if existing and existing[0]["status"] == "promoted":
        # Re-registering a promoted version would flip it back to shadow and let
        # promote_candidate "re-promote" the live model for no reason.
        return existing[0]
    row = {
        "model_version": version,
        "status": "shadow",
        "offline_metrics": bundle.get("metrics", {}),
        "registered_at": datetime.now(timezone.utc).isoformat(),
    }
    return db.upsert("model_candidates", [row], on_conflict="model_version")[0]


def run_shadow_predictions(
    db: SupabaseRestClient,
    *,
    matches: list[dict[str, Any]],
    historical_matches: list[dict[str, Any]],
    team_form_by_id: dict[int, dict[str, Any]],
) -> int:
    """Persist the first candidate forecast for each fixture without changing it later."""
    candidates = db.select_all(
        "model_candidates",
        columns="model_version",
        filters={"status": "eq.shadow"},
    )
    written = 0
    for candidate in candidates:
        model_version = str(candidate["model_version"])
        try:
            bundle = joblib.load(candidate_path(model_version))
        except FileNotFoundError:
            # Never allow a stale candidate artifact to block a user-facing alert.
            print(f"Shadow candidate artifact unavailable: {model_version}")
            continue
        rows = generate_prediction_rows(
            bundle,
            historical_matches,
            matches,
            model_version=model_version,
            team_form_by_id=team_form_by_id,
        )
        for row in rows:
            shadow_row = {key: value for key, value in row.items() if key != "model_version"}
            shadow_row["model_version"] = model_version
            try:
                db.insert("shadow_predictions", [shadow_row])
                written += 1
            except DatabaseError as error:
                # A unique conflict means an earlier run already captured the
                # immutable candidate output; never overwrite it on refresh.
                if "(409)" not in str(error):
                    raise
    return written


def evaluate_shadow_predictions(db: SupabaseRestClient) -> list[dict[str, Any]]:
    """Score shadow forecasts when their fixtures become final."""
    from evaluation.track_performance import build_performance_row

    already_evaluated = {
        int(row["shadow_prediction_id"])
        for row in db.select_all(
            "shadow_prediction_performance", columns="shadow_prediction_id"
        )
    }
    lookback_cutoff = (
        datetime.now(timezone.utc) - timedelta(days=EVALUATION_LOOKBACK_DAYS)
    ).isoformat()
    finished = {
        int(row["id"]): row
        for row in db.select_all(
            "matches",
            columns="id,status,match_date,home_score,away_score",
            filters={
                "status": "eq.finished",
                "home_score": "not.is.null",
                "away_score": "not.is.null",
                "match_date": f"gte.{lookback_cutoff}",
            },
        )
    }
    if not finished:
        return []
    finished_filter = f"in.({','.join(str(match_id) for match_id in sorted(finished))})"
    predictions = db.select_all(
        "shadow_predictions",
        columns=(
            "id,match_id,prob_home_win,prob_draw,prob_away_win,prob_over_2_5,"
            "prob_btts,predicted_at"
        ),
        filters={"match_id": finished_filter},
    )
    evaluated_at = datetime.now(timezone.utc).isoformat()
    rows: list[dict[str, Any]] = []
    for prediction in predictions:
        prediction_id = int(prediction["id"])
        if prediction_id in already_evaluated:
            continue
        match = finished.get(int(prediction["match_id"]))
        if match is None:
            continue
        if parse_iso_datetime(prediction["predicted_at"]) > parse_iso_datetime(match["match_date"]):
            continue
        # Reuse production metric calculations, then map to shadow schema.
        source = {**prediction, "id": prediction_id}
        performance = build_performance_row(source, match, evaluated_at=evaluated_at)
        rows.append(
            {
                "shadow_prediction_id": prediction_id,
                "match_id": int(match["id"]),
                "was_correct": performance["was_correct"],
                "brier_score": performance["brier_score"],
                "over_2_5_was_correct": performance["over_2_5_was_correct"],
                "over_2_5_brier_score": performance["over_2_5_brier_score"],
                "btts_was_correct": performance["btts_was_correct"],
                "btts_brier_score": performance["btts_brier_score"],
                "evaluated_at": evaluated_at,
            }
        )
    return db.upsert(
        "shadow_prediction_performance", rows, on_conflict="shadow_prediction_id"
    )


def promotion_decision(
    *,
    candidate_brier: float,
    candidate_accuracy: float,
    production_brier: float,
    production_accuracy: float,
    sample_size: int,
) -> tuple[bool, str]:
    """Use a conservative two-metric gate; no sample means no promotion."""
    if sample_size < MINIMUM_PROMOTION_SAMPLE:
        return False, f"Gölge örneklemi yetersiz: {sample_size}/{MINIMUM_PROMOTION_SAMPLE}"
    if candidate_brier > production_brier - MINIMUM_BRIER_IMPROVEMENT:
        return False, (
            "Aday modelin Brier iyileşmesi promotion eşiğini karşılamıyor"
        )
    if candidate_accuracy < production_accuracy:
        return False, "Aday modelin 1-X-2 isabeti canlı modelden düşük"
    return True, "Aday model gölge karşılaştırmasını geçti"


def promote_candidate(db: SupabaseRestClient, model_version: str) -> str:
    """Promote only a candidate that beat production on the same completed matches."""
    candidate = db.select(
        "model_candidates",
        columns="model_version,status,offline_metrics,registered_at,promoted_at",
        filters={"model_version": f"eq.{model_version}"},
        limit=1,
    )
    if not candidate or candidate[0]["status"] != "shadow":
        raise ValueError("Candidate must exist and be in shadow status")
    shadows = db.select_all(
        "shadow_predictions", columns="id,match_id", filters={"model_version": f"eq.{model_version}"}
    )
    if not shadows:
        raise RuntimeError("Candidate has no shadow predictions")
    shadow_ids = ",".join(str(int(row["id"])) for row in shadows)
    candidate_performance = db.select_all(
        "shadow_prediction_performance", columns="shadow_prediction_id,match_id,was_correct,brier_score",
        filters={"shadow_prediction_id": f"in.({shadow_ids})"},
    )
    match_ids = {int(row["match_id"]) for row in candidate_performance}
    if not match_ids:
        raise RuntimeError("Candidate has no completed shadow predictions")
    production_rows = db.select_all(
        "prediction_performance", columns="match_id,was_correct,brier_score,evaluated_at",
        filters={"match_id": f"in.({','.join(map(str, sorted(match_ids)))})"},
    )
    production_by_match: dict[int, dict[str, Any]] = {}
    for row in production_rows:
        match_id = int(row["match_id"])
        current = production_by_match.get(match_id)
        if current is None or parse_iso_datetime(row["evaluated_at"]) > parse_iso_datetime(current["evaluated_at"]):
            production_by_match[match_id] = row
    paired = [
        (candidate_row, production_by_match[int(candidate_row["match_id"])])
        for candidate_row in candidate_performance
        if int(candidate_row["match_id"]) in production_by_match
    ]
    if not paired:
        raise RuntimeError("No same-match production baseline is available")
    accepted, reason = promotion_decision(
        candidate_brier=sum(float(row[0]["brier_score"]) for row in paired) / len(paired),
        candidate_accuracy=sum(bool(row[0]["was_correct"]) for row in paired) / len(paired),
        production_brier=sum(float(row[1]["brier_score"]) for row in paired) / len(paired),
        production_accuracy=sum(bool(row[1]["was_correct"]) for row in paired) / len(paired),
        sample_size=len(paired),
    )
    if not accepted:
        raise RuntimeError(reason)
    shutil.copyfile(candidate_path(model_version), PROJECT_ROOT / "models" / "saved_models" / "latest.joblib")
    db.upsert(
        "model_candidates",
        [{
            **candidate[0],
            "status": "promoted",
            "promoted_at": datetime.now(timezone.utc).isoformat(),
        }],
        on_conflict="model_version",
    )
    return reason


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register-newest", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--promote")
    parser.add_argument("--days", type=int, default=3)
    args = parser.parse_args()
    settings = get_settings()
    db = SupabaseRestClient(settings.supabase_url, settings.supabase_service_role_key)
    result: dict[str, Any] = {}
    if args.register_newest:
        result["candidate"] = register_newest_candidate(db)["model_version"]
    if args.evaluate:
        result["evaluated"] = len(evaluate_shadow_predictions(db))
    if args.promote:
        result["promotion"] = promote_candidate(db, args.promote)
    if not args.register_newest and not args.evaluate and not args.promote:
        now = datetime.now(timezone.utc)
        matches = load_upcoming_matches(db, now=now, horizon_days=args.days)
        team_ids = {
            int(team_id)
            for match in matches
            for team_id in (match["home_team_id"], match["away_team_id"])
        }
        result["shadow_predictions"] = run_shadow_predictions(
            db,
            matches=matches,
            historical_matches=load_historical_matches(db),
            team_form_by_id=load_latest_team_forms(db, team_ids),
        )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
