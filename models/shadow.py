"""Run candidate models in shadow mode and require evidence before promotion."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timedelta, timezone
from math import log
from pathlib import Path
from typing import Any

import numpy as np

from config.settings import PROJECT_ROOT, UPCOMING_HORIZON_DAYS, get_settings
from data_pipeline.isotime import parse_iso_datetime
from data_pipeline.odds import vig_free_market_probabilities
from db.db_client import DatabaseError, SupabaseRestClient
from evaluation.track_performance import EVALUATION_LOOKBACK_DAYS
from models.artifact_store import (
    ArtifactStoreError,
    download_model,
    load_verified_joblib,
)
from models.predict import (
    generate_prediction_rows,
    load_latest_team_forms,
    load_upcoming_matches,
    newest_versioned_model,
)
from models.train_model import load_historical_matches

MINIMUM_PROMOTION_SAMPLE = 100
MINIMUM_BRIER_IMPROVEMENT = 0.005
MINIMUM_PROMOTION_ACCURACY = 0.50
MAXIMUM_PROMOTION_BRIER = 0.62
# A candidate whose calibrated loss is worse than its raw loss is not
# production-ready, and its calibration must not regress beyond production's.
MINIMUM_BASELINE_ADVANTAGE = 0.01
MAXIMUM_ECE_REGRESSION = 1.25
MINIMUM_MARKET_SAMPLE = 200
MARKET_BOOTSTRAP_SAMPLES = 10_000
MARKET_REQUIRED_MEAN_ADVANTAGE = -0.010
MARKET_REQUIRED_CI_UPPER = -0.005


def candidate_path(model_version: str) -> Path:
    path = PROJECT_ROOT / "models" / "saved_models" / f"{model_version}.joblib"
    if path.is_file():
        load_verified_joblib(path)
        return path
    # Normal weekly candidates share the production bundle manifest.
    try:
        result = download_model(f"{model_version}.joblib", dest_dir=path.parent)
        download_model("artifact_manifest.json", dest_dir=path.parent)
        load_verified_joblib(result)
        return result
    except ArtifactStoreError:
        pass
    # Recovery candidates have their own manifest. Keep it isolated so it
    # cannot replace the production manifest in a scheduled runner.
    recovery_dir = path.parent / "recovery" / model_version
    prefix = f"recovery/{model_version}/"
    try:
        result = download_model(prefix + path.name, dest_dir=recovery_dir)
        download_model(prefix + "artifact_manifest.json", dest_dir=recovery_dir)
        load_verified_joblib(result)
        return result
    except ArtifactStoreError as error:
        raise FileNotFoundError(
            f"Verified candidate artifact is missing: {model_version}"
        ) from error


def register_candidate_artifact(db: SupabaseRestClient, path: Path) -> dict[str, Any]:
    """Register an explicitly selected, locally verified shadow candidate."""
    bundle = load_verified_joblib(path)
    version = str(bundle.get("model_version", "")).strip()
    if not version or version != path.stem or not version.startswith("model_v"):
        raise ValueError("Candidate artifact version does not match its filename")
    existing = db.select(
        "model_candidates",
        columns="model_version,status",
        filters={"model_version": f"eq.{version}"},
        limit=1,
    )
    if existing and existing[0]["status"] == "promoted":
        return existing[0]
    row = {
        "model_version": version,
        "status": "shadow",
        "offline_metrics": bundle.get("metrics", {}),
        "registered_at": datetime.now(timezone.utc).isoformat(),
    }
    return db.upsert("model_candidates", [row], on_conflict="model_version")[0]


def register_newest_candidate(db: SupabaseRestClient) -> dict[str, Any]:
    """Register the newest versioned artifact; `latest.joblib` is never a candidate."""
    model_dir = PROJECT_ROOT / "models" / "saved_models"
    path = newest_versioned_model(model_dir)
    if path is None:
        raise FileNotFoundError("No versioned model artifact was found")
    return register_candidate_artifact(db, path)


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
        order="registered_at.asc",
    )
    written = 0
    for candidate in candidates:
        model_version = str(candidate["model_version"])
        try:
            bundle = load_verified_joblib(candidate_path(model_version))
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
            "shadow_prediction_performance",
            columns="shadow_prediction_id",
            order="shadow_prediction_id.asc",
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
            "prob_btts,market_probabilities,predicted_at"
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
                "market_performance": performance["market_performance"],
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
    candidate_calibrated_log_loss: float | None = None,
    candidate_raw_log_loss: float | None = None,
    candidate_ece: float | None = None,
    production_ece: float | None = None,
    candidate_log_loss: float | None = None,
    candidate_baseline_log_loss: float | None = None,
) -> tuple[bool, str]:
    """Use a conservative multi-metric gate; no sample means no promotion."""
    if sample_size < MINIMUM_PROMOTION_SAMPLE:
        return False, f"Gölge örneklemi yetersiz: {sample_size}/{MINIMUM_PROMOTION_SAMPLE}"
    if candidate_accuracy < MINIMUM_PROMOTION_ACCURACY:
        return False, (
            "Aday model mutlak isabet tabanını karşılamıyor: "
            f"{candidate_accuracy:.1%} < {MINIMUM_PROMOTION_ACCURACY:.1%}"
        )
    if candidate_brier > MAXIMUM_PROMOTION_BRIER:
        return False, (
            "Aday model mutlak Brier kalite tabanını karşılamıyor: "
            f"{candidate_brier:.3f} > {MAXIMUM_PROMOTION_BRIER:.3f}"
        )
    if candidate_brier > production_brier - MINIMUM_BRIER_IMPROVEMENT:
        return False, (
            "Aday modelin Brier iyileşmesi promotion eşiğini karşılamıyor"
        )
    if candidate_accuracy < production_accuracy:
        return False, "Aday modelin 1-X-2 isabeti canlı modelden düşük"
    if (
        candidate_calibrated_log_loss is not None
        and candidate_raw_log_loss is not None
        and candidate_calibrated_log_loss > candidate_raw_log_loss
    ):
        return False, (
            "Aday modelin kalibrasyonu ham modelden kötü; production'a uygun değil"
        )
    if (
        candidate_ece is not None
        and production_ece is not None
        and candidate_ece > production_ece * MAXIMUM_ECE_REGRESSION
    ):
        return False, (
            "Aday modelin kalibrasyonu (ECE) üretim modelinden kabul edilemez "
            f"derecede kötü: {candidate_ece:.4f} > {production_ece * MAXIMUM_ECE_REGRESSION:.4f}"
        )
    if (
        candidate_log_loss is not None
        and candidate_baseline_log_loss is not None
        and candidate_log_loss > candidate_baseline_log_loss - MINIMUM_BASELINE_ADVANTAGE
    ):
        return False, (
            "Aday model, kronolojik frekans baseline'ına karşı anlamlı üstünlük "
            "sağlamıyor"
        )
    return True, "Aday model gölge karşılaştırmasını geçti"


def paired_market_comparison(
    samples: list[tuple[float, float]],
    *,
    minimum_sample: int = MINIMUM_MARKET_SAMPLE,
    bootstrap_samples: int = MARKET_BOOTSTRAP_SAMPLES,
    seed: int = 42,
) -> dict[str, Any]:
    """Compare candidate and vig-free market log-loss on identical matches."""
    if len(samples) < minimum_sample:
        return {"status": "insufficient_evidence", "sample_size": len(samples)}
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    differences = np.asarray([candidate - market for candidate, market in samples], dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(differences), size=(bootstrap_samples, len(differences)))
    bootstrap_means = differences[draws].mean(axis=1)
    mean_difference = float(differences.mean())
    lower, upper = np.quantile(bootstrap_means, [0.025, 0.975])
    passed = mean_difference <= MARKET_REQUIRED_MEAN_ADVANTAGE and float(upper) <= MARKET_REQUIRED_CI_UPPER
    return {
        "status": "passed" if passed else "failed",
        "sample_size": len(samples),
        "candidate_log_loss": float(np.mean([item[0] for item in samples])),
        "market_log_loss": float(np.mean([item[1] for item in samples])),
        "mean_difference": mean_difference,
        "bootstrap_ci_lower": float(lower),
        "bootstrap_ci_upper": float(upper),
        "bootstrap_samples": bootstrap_samples,
    }


def _market_log_loss_sample(
    shadow: dict[str, Any], match: dict[str, Any], quote: dict[str, Any]
) -> tuple[float, float] | None:
    market = vig_free_market_probabilities(quote.get("odds") or {})
    if not market or match.get("home_score") is None or match.get("away_score") is None:
        return None
    actual = (
        "market_implied_home_win"
        if int(match["home_score"]) > int(match["away_score"])
        else "market_implied_away_win"
        if int(match["home_score"]) < int(match["away_score"])
        else "market_implied_draw"
    )
    try:
        candidate_key = {
            "market_implied_home_win": "prob_home_win",
            "market_implied_draw": "prob_draw",
            "market_implied_away_win": "prob_away_win",
        }[actual]
        return (
            -log(max(float(shadow[candidate_key]), 1e-15)),
            -log(max(float(market[actual]), 1e-15)),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _production_offline_metrics() -> dict[str, Any]:
    """Offline metrics of the current production artifact, when available."""
    latest = PROJECT_ROOT / "models" / "saved_models" / "latest.joblib"
    if not latest.is_file():
        return {}
    bundle = load_verified_joblib(latest)
    return dict(bundle.get("metrics", {}))


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
        "shadow_predictions", columns="id,match_id,predicted_at", filters={"model_version": f"eq.{model_version}"}
    )
    if not shadows:
        raise RuntimeError("Candidate has no shadow predictions")
    shadow_ids = ",".join(str(int(row["id"])) for row in shadows)
    candidate_performance = db.select_all(
        "shadow_prediction_performance", columns="shadow_prediction_id,match_id,was_correct,brier_score",
        filters={"shadow_prediction_id": f"in.({shadow_ids})"},
        order="shadow_prediction_id.asc",
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
    completed_matches = db.select_all(
        "matches", columns="id,home_score,away_score", filters={"id": f"in.({','.join(map(str, sorted(match_ids)))})"}
    )
    matches_by_id = {int(row["id"]): row for row in completed_matches}
    odds_rows = db.select_all(
        "odds_quote_history", columns="match_id,odds,captured_at",
        filters={"match_id": f"in.({','.join(map(str, sorted(match_ids)))})"},
        order="captured_at.desc",
    )
    shadows_by_match = {int(row["match_id"]): row for row in shadows}
    market_samples: list[tuple[float, float]] = []
    market_matches_seen: set[int] = set()
    for quote in odds_rows:
        match_id = int(quote["match_id"])
        if match_id in market_matches_seen:
            continue
        shadow = shadows_by_match.get(match_id)
        match = matches_by_id.get(match_id)
        if not shadow or not match:
            continue
        if parse_iso_datetime(str(quote["captured_at"])) > parse_iso_datetime(str(shadow["predicted_at"])):
            continue
        market_matches_seen.add(match_id)
        sample = _market_log_loss_sample(shadow, match, quote)
        if sample is not None:
            market_samples.append(sample)
    market_comparison = paired_market_comparison(market_samples)
    candidate_offline = dict(candidate[0].get("offline_metrics") or {})
    candidate_offline["market_comparison"] = market_comparison
    db.upsert("model_candidates", [{**candidate[0], "offline_metrics": candidate_offline}], on_conflict="model_version")
    if market_comparison["status"] != "passed":
        raise RuntimeError(
            "Piyasa karşılaştırması terfi için yeterli değil: "
            f"{market_comparison['status']} ({market_comparison['sample_size']}/{MINIMUM_MARKET_SAMPLE})"
        )
    production_offline = _production_offline_metrics()
    accepted, reason = promotion_decision(
        candidate_brier=sum(float(row[0]["brier_score"]) for row in paired) / len(paired),
        candidate_accuracy=sum(bool(row[0]["was_correct"]) for row in paired) / len(paired),
        production_brier=sum(float(row[1]["brier_score"]) for row in paired) / len(paired),
        production_accuracy=sum(bool(row[1]["was_correct"]) for row in paired) / len(paired),
        sample_size=len(paired),
        candidate_calibrated_log_loss=candidate_offline.get("log_loss"),
        candidate_raw_log_loss=candidate_offline.get("raw_log_loss"),
        candidate_ece=candidate_offline.get("expected_calibration_error"),
        production_ece=production_offline.get("expected_calibration_error"),
        candidate_log_loss=candidate_offline.get("log_loss"),
        candidate_baseline_log_loss=candidate_offline.get("baseline_log_loss"),
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


def shadow_report(db: SupabaseRestClient) -> dict[str, Any]:
    """Production live metrics and per-candidate shadow metrics, kept separate."""
    production = db.select_all("prediction_performance", columns="was_correct,brier_score")
    production_summary: dict[str, Any] = {
        "n": len(production),
        "accuracy": (
            sum(bool(row["was_correct"]) for row in production) / len(production)
            if production
            else None
        ),
        "brier": (
            sum(float(row["brier_score"]) for row in production) / len(production)
            if production
            else None
        ),
    }
    candidates = db.select_all(
        "model_candidates",
        columns="model_version,status,offline_metrics",
        order="registered_at.asc",
    )
    version_by_shadow_id = {
        int(row["id"]): str(row["model_version"])
        for row in db.select_all("shadow_predictions", columns="id,model_version")
    }
    shadow_rows = db.select_all(
        "shadow_prediction_performance",
        columns="shadow_prediction_id,was_correct,brier_score",
        order="shadow_prediction_id.asc",
    )
    by_version: dict[str, list[dict[str, Any]]] = {}
    for row in shadow_rows:
        version = version_by_shadow_id.get(int(row["shadow_prediction_id"]))
        if version:
            by_version.setdefault(version, []).append(row)
    candidate_summary = []
    for candidate in candidates:
        version = str(candidate["model_version"])
        rows = by_version.get(version, [])
        candidate_summary.append(
            {
                "model_version": version,
                "status": str(candidate["status"]),
                "n": len(rows),
                "accuracy": (
                    sum(bool(r["was_correct"]) for r in rows) / len(rows)
                    if rows
                    else None
                ),
                "brier": (
                    sum(float(r["brier_score"]) for r in rows) / len(rows)
                    if rows
                    else None
                ),
                "offline_log_loss": (candidate.get("offline_metrics") or {}).get("log_loss"),
            }
        )
    return {"production": production_summary, "candidates": candidate_summary}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register-newest", action="store_true")
    parser.add_argument("--register-artifact", type=Path)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--promote")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--days", type=int, default=UPCOMING_HORIZON_DAYS)
    args = parser.parse_args()
    settings = get_settings()
    db = SupabaseRestClient(settings.supabase_url, settings.supabase_service_role_key)
    result: dict[str, Any] = {}
    if args.register_newest:
        result["candidate"] = register_newest_candidate(db)["model_version"]
    if args.register_artifact:
        result["candidate"] = register_candidate_artifact(
            db, args.register_artifact
        )["model_version"]
    if args.evaluate:
        result["evaluated"] = len(evaluate_shadow_predictions(db))
    if args.promote:
        result["promotion"] = promote_candidate(db, args.promote)
    if args.report:
        result["report"] = shadow_report(db)
    if (
        not args.register_newest
        and not args.register_artifact
        and not args.evaluate
        and not args.promote
        and not args.report
    ):
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
