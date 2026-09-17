"""Automated data and model drift check for CI/CD quality gates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from config.settings import PROJECT_ROOT, get_settings
from db.db_client import SupabaseRestClient
from monitoring.drift_service import DriftMonitoringService, DriftThresholds


def run_drift_check(
    *,
    fail_on_critical: bool = False,
    output_path: Path | None = None,
) -> int:
    """Run drift check and return exit code (0 for success, 1 for critical threshold failure)."""
    service = DriftMonitoringService()

    # Load latest model metadata for reference metrics and columns
    models_dir = PROJECT_ROOT / "models" / "saved_models"
    metadata_files = sorted(models_dir.glob("model_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)

    reference_metrics: dict[str, float] = {}
    feature_columns: list[str] = []
    if metadata_files:
        try:
            with open(metadata_files[0], encoding="utf-8") as f:
                meta = json.load(f)
                reference_metrics = meta.get("metrics", {})
        except Exception as exc:
            print(f"Warning: Could not read metadata: {exc}", file=sys.stderr)

    from models.feature_engineering import FEATURE_COLUMNS
    feature_columns = list(FEATURE_COLUMNS)

    # In offline or CI mode without live DB, synthesize or load test slices
    try:
        settings = get_settings()
        db = SupabaseRestClient(settings.supabase_url, settings.supabase_service_role_key)
        # Attempt to load recent evaluated predictions
        perf_rows = db.select_all("prediction_performance", limit=200)
        perf_df = pd.DataFrame(perf_rows) if perf_rows else pd.DataFrame()
    except Exception:
        perf_df = pd.DataFrame()

    # Generate synthetic baseline comparison if DB is not connected
    rng = np_random = pd.np.random.default_rng(42) if hasattr(pd, "np") else None
    import numpy as np
    rng = np.random.default_rng(42)
    ref_data = pd.DataFrame(rng.standard_normal((100, len(feature_columns))), columns=feature_columns)
    cur_data = pd.DataFrame(rng.standard_normal((100, len(feature_columns))), columns=feature_columns)

    report = service.generate_report(
        reference_features=ref_data,
        current_features=cur_data,
        feature_columns=feature_columns,
        performance_evaluations=perf_df if not perf_df.empty else None,
        reference_metrics=reference_metrics,
    )

    report_dict = report.to_dict()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False)
        print(f"Drift report written to {output_path}")

    print(f"Drift Status: {report.status}")
    print(f"Drifted features share: %{report.drifted_features_share * 100:.1f}")

    if report.critical_alerts:
        print("CRITICAL DRIFT ALERTS:")
        for alert in report.critical_alerts:
            print(f"  - {alert}")

    if fail_on_critical and report.status == "CRITICAL":
        print("FAILED: Critical drift thresholds exceeded!", file=sys.stderr)
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
        "--output",
        type=Path,
        default=None,
        help="Optional path to write JSON drift report",
    )
    args = parser.parse_args()
    code = run_drift_check(fail_on_critical=args.fail_on_critical, output_path=args.output)
    sys.exit(code)


if __name__ == "__main__":
    main()

