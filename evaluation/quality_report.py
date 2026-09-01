"""Report production prediction quality from persisted Brier and Log Loss metrics."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import date, timedelta
from typing import Any

from config.settings import get_settings
from db.db_client import SupabaseRestClient


def summarize_quality(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate daily view rows with sample-size weighting across model versions."""
    grouped: defaultdict[str, dict[str, float]] = defaultdict(
        lambda: {"sample_size": 0.0, "correct": 0.0, "brier_total": 0.0, "log_loss_total": 0.0}
    )
    for row in rows:
        sample_size = int(row["sample_size"])
        if sample_size < 1:
            continue
        bucket = grouped[str(row["model_version"])]
        bucket["sample_size"] += sample_size
        bucket["correct"] += float(row["accuracy"]) * sample_size
        bucket["brier_total"] += float(row["avg_brier_score"]) * sample_size
        bucket["log_loss_total"] += float(row["avg_log_loss"]) * sample_size

    report = []
    for model_version, values in sorted(grouped.items()):
        sample_size = int(values["sample_size"])
        report.append({
            "model_version": model_version,
            "sample_size": sample_size,
            "accuracy": values["correct"] / sample_size,
            "avg_brier_score": values["brier_total"] / sample_size,
            "avg_log_loss": values["log_loss_total"] / sample_size,
        })
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=30)
    args = parser.parse_args()
    if args.days < 1 or args.days > 3650:
        raise ValueError("days must be between 1 and 3650")

    settings = get_settings()
    db = SupabaseRestClient(settings.supabase_url, settings.supabase_service_role_key)
    start = date.today() - timedelta(days=args.days - 1)
    rows = db.select_all(
        "prediction_quality_daily",
        columns="evaluated_day,model_version,sample_size,accuracy,avg_brier_score,avg_log_loss",
        filters={"evaluated_day": f"gte.{start.isoformat()}"},
        order="evaluated_day.asc",
    )
    print(json.dumps({"days": args.days, "models": summarize_quality(rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
