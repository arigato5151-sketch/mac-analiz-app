"""Scheduler configuration using APScheduler."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from threading import Event
from typing import Any

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger


PROJECT_ROOT = Path(__file__).resolve().parent


def create_scheduler() -> BackgroundScheduler:
    """Create and configure the background scheduler."""
    scheduler = BackgroundScheduler(timezone="UTC")
    
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


def _run_module(module: str, *arguments: str) -> dict[str, Any]:
    """Run a CLI module in an isolated process and return its captured output."""
    completed = subprocess.run(
        [sys.executable, "-m", module, *arguments],
        check=True,
        capture_output=True,
        cwd=PROJECT_ROOT,
        text=True,
    )
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


# Job functions use subprocesses for CLI tasks so concurrent jobs never mutate sys.argv.
def run_morning_update() -> dict[str, Any]:
    """Morning data update - fetch fixtures, refresh context, generate predictions."""
    return {
        "fixtures": _run_module("data_pipeline.fetch_fixtures", "--days", "3"),
        "refresh_context": _run_module("data_pipeline.refresh_context", "--days", "3"),
        "predictions": _run_module("models.predict", "--days", "3"),
        "quality": _run_module("data_pipeline.data_quality", "--days", "3"),
    }


def run_evening_update() -> dict[str, Any]:
    """Evening data update - fetch results, evaluate predictions."""
    return {
        "results": _run_module("data_pipeline.fetch_results", "--lookback-days", "7"),
        "performance": _run_module("evaluation.track_performance"),
    }


def run_night_update() -> dict[str, Any]:
    """Night update - send daily summary, evaluate shadow models."""
    return {
        "summary": _run_module("notifications.daily_summary", "--mode", "night"),
        "performance": _run_module("evaluation.track_performance"),
        "shadow": _run_module("models.shadow"),
    }


def run_pre_match_notifications() -> dict[str, Any]:
    """Send pre-match Telegram notifications."""
    from notifications.pre_match import run_pre_match_notifications
    return run_pre_match_notifications()


def run_result_notifications() -> dict[str, Any]:
    """Send result notifications."""
    from notifications.final_results import run_final_result_notifications

    return run_final_result_notifications()


def run_daily_summary() -> dict[str, Any]:
    """Send daily summary to Telegram."""
    return _run_module("notifications.daily_summary", "--mode", "morning")


def run_weekly_retrain() -> dict[str, Any]:
    """Weekly model retraining."""
    return {
        "training": _run_module("models.train_model", "--publish-latest"),
        "artifact_store": _run_module(
            "models.artifact_store", "--push", "models/saved_models"
        ),
        "shadow": _run_module("models.shadow", "--register-newest"),
    }


def run_operational_alerts() -> dict[str, Any]:
    """Check operational health and send alerts if needed."""
    return _run_module("notifications.operational_alerts", "--notify")


def start_scheduler() -> BackgroundScheduler:
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

    scheduler = start_scheduler()
    stop_event = Event()

    def signal_handler(_signal_number: int, _frame: Any) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("Scheduler running. Press Ctrl+C to stop.")
    try:
        stop_event.wait()
    finally:
        shutdown_scheduler(scheduler)
