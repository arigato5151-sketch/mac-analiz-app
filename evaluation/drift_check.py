"""Automated data and model drift check for CI/CD quality gates."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from config.settings import PROJECT_ROOT, get_settings
from db.db_client import SupabaseRestClient
from monitoring.drift_service import DriftMonitoringService, DriftReportSummary
from monitoring.feature_snapshot import (
    CURRENT_SNAPSHOT_NAME,
    REFERENCE_SNAPSHOT_NAME,
    extract_snapshot_metadata,
    load_feature_snapshot,
    resolve_snapshot_path,
    snapshot_feature_columns,
)

LOGGER = logging.getLogger(__name__)


def _format_period(meta: dict[str, Any]) -> str:
    """Format human-readable period range from snapshot metadata."""
    p_start = meta.get("period_start", "")
    p_end = meta.get("period_end", "")
    if p_start and p_end:
        return f"{p_start[:10]} .. {p_end[:10]}"
    if p_start:
        return f"from {p_start[:10]}"
    return meta.get("saved_at", "")[:19] if meta.get("saved_at") else ""


def run_drift_check(
    *,
    fail_on_critical: bool = False,
    output_path: Path | None = None,
    reference_snapshot_path: Path | None = None,
    current_snapshot_path: Path | None = None,
    allow_skip: bool | None = None,
    model_version: str | None = None,
) -> int:
    """Run drift check and return exit code (0 for success or skipped, 1 for critical failure).

    If genuine production snapshots or DB data are not available:
    - Yields SKIPPED status instead of an artificial 'OK'.
    - Returns 0 when allow_skip is True, or 1 if fail_on_critical is set and skip is disallowed.

    Controls:
    - allow_skip: If None, resolved from env var ALLOW_DRIFT_SKIP (default True for dev).
      In production/CI with --strict, allow_skip is False (fail-closed).
    - Checks schema compatibility between reference and current feature sets.
    """
    # Resolve allow_skip from env if not explicitly provided
    if allow_skip is None:
        env_val = os.getenv("ALLOW_DRIFT_SKIP")
        if env_val is not None:
            allow_skip = env_val.strip().lower() not in ("false", "0", "no")
        else:
            allow_skip = True

    models_dir = PROJECT_ROOT / "models" / "saved_models"
    ref_path = reference_snapshot_path or resolve_snapshot_path(
        models_dir, kind="ref", model_version=model_version
    )
    cur_path = current_snapshot_path or resolve_snapshot_path(
        models_dir, kind="current", model_version=model_version
    )

    ref_df = load_feature_snapshot(ref_path)
    cur_df = load_feature_snapshot(cur_path)

    ref_meta = extract_snapshot_metadata(ref_df, path=ref_path)
    cur_meta = extract_snapshot_metadata(cur_df, path=cur_path)

    active_model_version = (
        model_version
        or ref_meta.get("model_version")
        or cur_meta.get("model_version")
        or "unknown"
    )

    # If active model version is still unknown, inspect metadata files
    if active_model_version == "unknown":
        metadata_files = sorted(
            models_dir.glob("model_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if metadata_files:
            active_model_version = metadata_files[0].stem

    ref_period = _format_period(ref_meta)
    cur_period = _format_period(cur_meta)

    # Check live DB evaluations if available
    perf_df = pd.DataFrame()
    try:
        settings = get_settings()
        if settings.supabase_url and settings.supabase_service_role_key:
            db = SupabaseRestClient(
                settings.supabase_url, settings.supabase_service_role_key
            )
            perf_rows = db.select_all("prediction_performance", limit=200)
            if perf_rows:
                perf_df = pd.DataFrame(perf_rows)
    except Exception as exc:
        LOGGER.debug("Live DB connection unavailable for drift check: %s", exc)

    has_features = (
        ref_df is not None
        and cur_df is not None
        and not ref_df.empty
        and not cur_df.empty
    )

    if not has_features:
        status_label = "FAILED" if (fail_on_critical and not allow_skip) else "SKIPPED"
        print(f"Drift Check Status: {status_label} (Real feature snapshots not found)")
        print(f"  Reference snapshot ({ref_path.name}): {'Found' if ref_df is not None else 'Missing'}")
        print(f"  Current snapshot ({cur_path.name}): {'Found' if cur_df is not None else 'Missing'}")
        print(f"  Database performance rows: {len(perf_df)} rows")

        skipped_report = {
            "status": status_label,
            "reason": "Feature snapshots not available; synthetic data is disallowed in production/CI gates",
            "data_source": "missing_snapshots",
            "model_version": active_model_version,
            "reference_rows": ref_meta.get("rows", 0),
            "current_rows": cur_meta.get("rows", 0),
            "reference_period": ref_period,
            "current_period": cur_period,
            "critical_alerts": (
                [
                    f"Feature snapshots missing: ref={'found' if ref_df is not None else 'missing'}, "
                    f"cur={'found' if cur_df is not None else 'missing'}"
                ]
                if status_label == "FAILED"
                else []
            ),
        }

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(skipped_report, f, indent=2, ensure_ascii=False)
            print(f"Drift report written to {output_path}")

        if fail_on_critical and not allow_skip:
            print(
                "FAILED: Feature snapshots required but missing under strict/fail-closed mode!",
                file=sys.stderr,
            )
            return 1
        return 0

    # Genuine feature snapshots exist — validate schema compatibility
    ref_cols = set(snapshot_feature_columns(ref_df))
    cur_cols = set(snapshot_feature_columns(cur_df))
    common_cols = ref_cols.intersection(cur_cols)

    schema_mismatch = False
    schema_alerts: list[str] = []
    if len(common_cols) == 0:
        schema_mismatch = True
        schema_alerts.append(
            "Kritik: Referans ve güncel özellik kümeleri tamamen ayrık (0 ortak özellik)!"
        )
    elif len(common_cols) < len(ref_cols) * 0.7:
        schema_mismatch = True
        missing_count = len(ref_cols) - len(common_cols)
        schema_alerts.append(
            f"Kritik: Özellik şeması uyumsuzluğu: {missing_count} referans özelliği "
            "güncel veri kümesinde bulunamadı!"
        )

    feature_columns = sorted(common_cols) if common_cols else sorted(ref_cols)

    reference_metrics: dict[str, float] = {}
    metadata_files = sorted(
        models_dir.glob("model_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if metadata_files:
        try:
            with open(metadata_files[0], encoding="utf-8") as f:
                meta = json.load(f)
                reference_metrics = meta.get("metrics", {})
        except Exception as exc:
            LOGGER.warning("Could not read model metadata: %s", exc)

    service = DriftMonitoringService()
    report: DriftReportSummary = service.generate_report(
        reference_features=ref_df,
        current_features=cur_df,
        feature_columns=feature_columns,
        performance_evaluations=perf_df if not perf_df.empty else None,
        reference_metrics=reference_metrics,
        data_source="parquet_snapshots",
        reference_period=ref_period,
        current_period=cur_period,
    )

    report_dict = report.to_dict()
    report_dict["model_version"] = active_model_version

    if schema_mismatch:
        report_dict["status"] = "CRITICAL"
        report_dict["critical_alerts"].extend(schema_alerts)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False)
        print(f"Drift report written to {output_path}")

    print(f"Drift Status: {report_dict['status']}")
    print(f"Model version: {active_model_version}")
    print(f"Data source: {report_dict['data_source']}")
    print(f"Reference: {report_dict['reference_rows']} rows ({ref_period})")
    print(f"Current: {report_dict['current_rows']} rows ({cur_period})")
    print(
        f"Drifted features share: %{report_dict['drifted_features_share'] * 100:.1f} "
        f"({report_dict['drifted_features_count']}/{report_dict['total_features_count']})"
    )

    if report_dict.get("critical_alerts"):
        print("CRITICAL DRIFT ALERTS:")
        for alert in report_dict["critical_alerts"]:
            print(f"  - {alert}")

    if fail_on_critical and report_dict["status"] in ("CRITICAL", "FAILED"):
        print(
            "FAILED: Critical drift thresholds or schema incompatibility detected!",
            file=sys.stderr,
        )
        return 1

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fail-on-critical",
        action="store_true",
        help="Exit with non-zero code if critical drift threshold is exceeded",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Disallow skipping when snapshots are missing (fail-closed)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write JSON drift report",
    )
    parser.add_argument(
        "--reference-snapshot",
        type=Path,
        default=None,
        help="Custom path to reference feature snapshot parquet",
    )
    parser.add_argument(
        "--current-snapshot",
        type=Path,
        default=None,
        help="Custom path to current feature snapshot parquet",
    )
    parser.add_argument(
        "--model-version",
        type=str,
        default=None,
        help="Optional specific model version identifier to resolve snapshots",
    )
    args = parser.parse_args()

    allow_skip = False if args.strict else None
    code = run_drift_check(
        fail_on_critical=args.fail_on_critical,
        output_path=args.output,
        reference_snapshot_path=args.reference_snapshot,
        current_snapshot_path=args.current_snapshot,
        allow_skip=allow_skip,
        model_version=args.model_version,
    )
    sys.exit(code)


if __name__ == "__main__":
    main()
