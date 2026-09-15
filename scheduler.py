"""Scheduler configuration using APScheduler."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from config.settings import get_settings


def create_scheduler() -> BackgroundScheduler:
    """Create and configure the background scheduler."""
    scheduler = BackgroundScheduler(timezone="UTC")
    
    settings = get_settings()
    
    # Add jobs
    # Morning data update - 03:00 UTC (06:00 Turkey time)
    scheduler.add_job(
        run_morning_update,
        CronTrigger(hour=3, minute=0, timezone="UTC"),
        id="morning_update",
        max_instances=1,
        replace_existing=True,
    )
    
    # Evening data update - 20:30 UTC (23:30 Turkey time)
    scheduler.add_job(
        run_evening_update,
        CronTrigger(hour=20, minute=30, timezone="UTC"),
        id="evening_update",
        max_instances=1,
        replace_existing=True,
    )
    
    # Night data update - 23:30 UTC (02:30 Turkey time)
    scheduler.add_job(
        run_night_update,
        CronTrigger(hour=23, minute=30, timezone="UTC"),
        id="night_update",
        max_instances=1,
        replace_existing=True,
    )
    
    # Pre-match notifications - every 5 minutes
    scheduler.add_job(
        run_pre_match_notifications,
        IntervalTrigger(minutes=5),
        id="pre_match_notifications",
        max_instances=1,
        replace_existing=True,
    )
    
    # Result notifications - every 15 minutes
    scheduler.add_job(
        run_result_notifications,
        IntervalTrigger(minutes=15),
        id="result_notifications",
        max_instances=1,
        replace_existing=True,
    )
    
    # Daily summary - 06:00 UTC (09:00 Turkey time)
    scheduler.add_job(
        run_daily_summary,
        CronTrigger(hour=6, minute=0, timezone="UTC"),
        id="daily_summary",
        max_instances=1,
        replace_existing=True,
    )
    
    # Weekly retrain - Sunday 03:00 UTC
    scheduler.add_job(
        run_weekly_retrain,
        CronTrigger(day_of_week="sun", hour=3, minute=0, timezone="UTC"),
        id="weekly_retrain",
        max_instances=1,
        replace_existing=True,
    )
    
    # Operational alerts - every 10 minutes
    scheduler.add_job(
        run_operational_alerts,
        IntervalTrigger(minutes=10),
        id="operational_alerts",
        max_instances=1,
        replace_existing=True,
    )
    
    return scheduler


# Job functions (imported lazily to avoid circular imports)
def run_morning_update() -> dict[str, Any]:
    """Morning data update - fetch fixtures, refresh context, generate predictions."""
    from data_pipeline.fetch_fixtures import main as fetch_fixtures_main
    from data_pipeline.refresh_context import main as refresh_context_main
    from models.predict import main as predict_main
    from data_pipeline.data_quality import main as data_quality_main
    
    results = {}
    results["fixtures"] = fetch_fixtures_main(["--days", "3"])
    results["refresh_context"] = refresh_context_main(["--days", "3"])
    results["predictions"] = predict_main(["--days", "3"])
    results["quality"] = data_quality_main(["--days", "3"])
    return results


def run_evening_update() -> dict[str, Any]:
    """Evening data update - fetch results, evaluate predictions."""
    from data_pipeline.fetch_results import main as fetch_results_main
    from evaluation.track_performance import main as track_performance_main
    
    results = {}
    results["results"] = fetch_results_main(["--lookback-days", "7"])
    results["performance"] = track_performance_main()
    return results


def run_night_update() -> dict[str, Any]:
    """Night update - send daily summary, evaluate shadow models."""
    from notifications.daily_summary import main as daily_summary_main
    from evaluation.track_performance import main as track_performance_main
    from models.shadow import main as shadow_main
    
    results = {}
    results["summary"] = daily_summary_main(["--mode", "night"])
    results["performance"] = track_performance_main()
    results["shadow"] = shadow_main()
    return results


def run_pre_match_notifications() -> dict[str, Any]:
    """Send pre-match Telegram notifications."""
    from notifications.pre_match import run_pre_match_notifications
    return run_pre_match_notifications()


def run_result_notifications() -> dict[str, Any]:
    """Send result notifications."""
    from notifications.final_results import run_result_notifications
    return run_result_notifications()


def run_daily_summary() -> dict[str, Any]:
    """Send daily summary to Telegram."""
    from notifications.daily_summary import main as daily_summary_main
    return daily_summary_main(["--mode", "night"])


def run_weekly_retrain() -> dict[str, Any]:
    """Weekly model retraining."""
    from models.train_model import main as train_model_main
    from models.artifact_store import main as artifact_store_main
    from models.shadow import main as shadow_main
    
    results = {}
    results["training"] = train_model_main(["--publish-latest"])
    results["artifact_store"] = artifact_store_main(["--push", "models/saved_models"])
    results["shadow"] = shadow_main()
    return results


def run_operational_alerts() -> dict[str, Any]:
    """Check operational health and send alerts if needed."""
    from notifications.operational_alerts import run_operational_alerts
    return run_operational_alerts()


def start_scheduler() -> None:
    """Start the background scheduler."""
    scheduler = create_scheduler()
    scheduler.start()
    print(f"Scheduler started with {len(scheduler.get_jobs())} jobs")
    return scheduler


def shutdown_scheduler(scheduler: Any) -> None:
    """Shutdown the scheduler gracefully."""
    scheduler.shutdown(wait=True)
    print("Scheduler shut down")


if __name__ == "__main__":
    import signal
    import sys
    
    scheduler = start_scheduler()
    
    def signal_handler(sig, frame):
        print("Shutting down scheduler...")
        shutdown_scheduler(scheduler)
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("Scheduler running. Press Ctrl+C to stop.")
    try:
        while True:
            pass
    except KeyboardInterrupt:
        shutdown_scheduler(scheduler)