"""Automated data and model drift check for CI/CD quality gates."""

from __future__ import annotations

import argparse
import json
import logging
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
    snapshot_feature_columns,
)

LOGGER = logging.getLogger(__name__)


def run_drift_check(
    *,
    fail_on_critical: bool = False,
    output_path: Path | None = None,
    reference_snapshot_path: Path | None = None,
    current_snapshot_path: Path | None = None,
    allow_skip: bool = True,
) -> int:
    """Run drift check and return exit code (0 for success or skipped, 1 for critical failure).

    If genuine production snapshots or DB data are not available:
    - Yields SKIPPED status instead of an artificial 'OK'.
    - Returns 0 when allow_skip is True, or 1 if fail_on_critical is set and skip is disallowed.
    """
    models_dir = PROJECT_ROOT / "models" / "saved_models"
    ref_path = reference_snapshot_path or (models_dir / REFERENCE_SNAPSHOT_NAME)
    cur_path = current_snapshot_path or (models_dir / CURRENT_SNAPSHOT_NAME)

    ref_df = load_feature_snapshot(ref_path)
    cur_df = load_feature_snapshot(cur_path)

    ref_meta = extract_snapshot_metadata(ref_df)
    cur_meta = extract_snapshot_metadata(cur_df)

    # If snapshots are not present on disk, check whether live DB evaluations are available
    perf_df = pd.DataFrame()
    db_connected = False
    try:
        settings = get_settings()
        if settings.supabase_url and settings.supabase_service_role_key:
            db = SupabaseRestClient(settings.supabase_url, settings.supabase_service_role_key)
            perf_rows = db.select_all("prediction_performance", limit=200)
            if perf_rows:
                perf_df = pd.DataFrame(perf_rows)
                db_connected = True
    except Exception as exc:
        LOGGER.debug("Live DB connection unavailable for drift check: %s", exc)

    has_features = (
        ref_df is not None
        and cur_df is not None
        and not ref_df.empty
        and not cur_df.empty
    )

    if not has_features:
        # Genuine data is missing — do not invent synthetic numbers or claim "OK"
        print("Drift Check Status: SKIPPED (Real feature snapshots not found)")
        print(f"  Reference snapshot ({ref_path.name}): {'Found' if ref_df is not None else 'Missing'}")
        print(f"  Current snapshot ({cur_path.name}): {'Found' if cur_df is not None else 'Missing'}")
        print(f"  Database performance rows: {len(perf_df)} rows")

        skipped_report = {
            "status": "SKIPPED",
            "reason": "Feature snapshots not available; synthetic data is disallowed in production/CI gates",
            "data_source": "missing_snapshots",
            "reference_rows": ref_meta["rows"],
            "current_rows": cur_meta["rows"],
            "reference_period": ref_meta.get("saved_at") or "",
            "current_period": cur_meta.get("saved_at") or "",
            "critical_alerts": [],
        }

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(skipped_report, f, indent=2, ensure_ascii=False)
            print(f"Drift report written to {output_path}")

        if fail_on_critical and not allow_skip:
            print("FAILED: Feature snapshots required but missing under strict mode!", file=sys.stderr)
            return 1
        return 0

    # Genuine feature snapshots exist
    feature_columns = snapshot_feature_columns(ref_df)
    reference_metrics: dict[str, float] = {}
    metadata_files = sorted(models_dir.glob("model_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
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
        reference_period=ref_meta.get("saved_at") or "",
        current_period=cur_meta.get("saved_at") or "",
    )

    report_dict = report.to_dict()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False)
        print(f"Drift report written to {output_path}")

    print(f"Drift Status: {report.status}")
    print(f"Data source: {report.data_source}")
    print(f"Reference rows: {report.reference_rows} · Current rows: {report.current_rows}")
    print(f"Drifted features share: %{report.drifted_features_share * 100:.1f} ({report.drifted_features_count}/{report.total_features_count})")

    if report.critical_alerts:
        print("CRITICAL DRIFT ALERTS:")
        for alert in report.critical_alerts:
            print(f"  - {alert}")

    if fail_on_critical and report.status == "CRITICAL":
        print("FAILED: Critical drift thresholds exceeded on genuine data!", file=sys.stderr)
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
        help="Disallow skipping when snapshots are missing",
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
    args = parser.parse_args()
    code = run_drift_check(
        fail_on_critical=args.fail_on_critical,
        output_path=args.output,
        reference_snapshot_path=args.reference_snapshot,
        current_snapshot_path=args.current_snapshot,
        allow_skip=not args.strict,
    )
    sys.exit(code)


if __name__ == "__main__":
    main()
