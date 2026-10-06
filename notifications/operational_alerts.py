"""Detect delivery and evaluation failures, then raise a concise Telegram alert."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any

from config.settings import get_settings
from db.db_client import SupabaseRestClient
from monitoring.operational_events import record_event
from notifications.telegram import TelegramError, send_telegram_message

EVALUATION_GRACE = timedelta(hours=6)
QUEUE_GRACE = timedelta(hours=3)
STALE_ACTIVE_GRACE = timedelta(hours=4)
SAMPLE_LIMIT = 10
EVENT_LOOKBACK = timedelta(hours=2)


def _parse_timestamp(value: object) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def build_health_report(
    *,
    finished_matches: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    evaluated_match_ids: set[int],
    pending_result_notifications: list[dict[str, Any]],
    active_matches: list[dict[str, Any]],
    now: datetime,
) -> dict[str, Any]:
    """Return only actionable, bounded diagnostics suitable for a public log."""
    predicted_before_kickoff = {
        int(row["match_id"])
        for row in predictions
        if _parse_timestamp(row["predicted_at"]) <= _parse_timestamp(row["match_date"])
    }
    overdue_evaluations = sorted(
        int(match["id"])
        for match in finished_matches
        if int(match["id"]) in predicted_before_kickoff
        and int(match["id"]) not in evaluated_match_ids
        and _parse_timestamp(match["match_date"]) <= now - EVALUATION_GRACE
    )
    stuck_notifications = sorted(
        int(row["match_id"])
        for row in pending_result_notifications
        if _parse_timestamp(row["available_at"]) <= now - QUEUE_GRACE
    )
    stale_active = sorted(
        int(match["id"])
        for match in active_matches
        if _parse_timestamp(match["match_date"]) <= now - STALE_ACTIVE_GRACE
    )
    return {
        "healthy": not (overdue_evaluations or stuck_notifications or stale_active),
        "overdue_evaluation_count": len(overdue_evaluations),
        "stuck_result_notification_count": len(stuck_notifications),
        "stale_active_match_count": len(stale_active),
        "overdue_evaluation_samples": overdue_evaluations[:SAMPLE_LIMIT],
        "stuck_result_notification_samples": stuck_notifications[:SAMPLE_LIMIT],
        "stale_active_match_samples": stale_active[:SAMPLE_LIMIT],
    }


def collect_health_report(db: SupabaseRestClient, *, now: datetime) -> dict[str, Any]:
    """Read durable pipeline state; no external side effects occur here."""
    finished_matches = db.select_all(
        "matches",
        columns="id,match_date",
        filters={
            "status": "eq.finished",
            "match_date": f"lte.{(now - EVALUATION_GRACE).isoformat()}",
        },
    )
    predictions = db.select_all("predictions", columns="match_id,predicted_at")
    match_dates = {
        int(row["id"]): row["match_date"]
        for row in db.select_all("matches", columns="id,match_date")
    }
    enriched_predictions = [
        {**row, "match_date": match_dates[int(row["match_id"])]}
        for row in predictions
        if int(row["match_id"]) in match_dates
    ]
    evaluated_match_ids = {
        int(row["match_id"])
        for row in db.select_all("prediction_performance", columns="match_id")
    }
    pending = db.select_all(
        "result_notification_queue",
        columns="match_id,available_at",
        filters={"sent_at": "is.null"},
        order="prediction_id.asc",
    )
    active = db.select_all(
        "matches",
        columns="id,match_date",
        filters={"status": "in.(scheduled,live)"},
    )
    return build_health_report(
        finished_matches=finished_matches,
        predictions=enriched_predictions,
        evaluated_match_ids=evaluated_match_ids,
        pending_result_notifications=pending,
        active_matches=active,
        now=now,
    )


def format_alert(report: dict[str, Any], *, reason: str | None = None) -> str:
    """Avoid IDs and secrets in Telegram; they add no operational value there."""
    lines = ["⚠️ Maç Analiz · Operasyon uyarısı"]
    if reason:
        lines.append(reason)
    if report["overdue_evaluation_count"]:
        lines.append(f"Gecikmiş değerlendirme: {report['overdue_evaluation_count']}")
    if report["stuck_result_notification_count"]:
        lines.append(f"Takılı sonuç bildirimi: {report['stuck_result_notification_count']}")
    if report["stale_active_match_count"]:
        lines.append(f"Eski aktif maç kaydı: {report['stale_active_match_count']}")
    if report.get("workflow"):
        lines.append(f"Workflow: {report['workflow']}")
    if report.get("failed_steps"):
        lines.append("Başarısız adımlar: " + ", ".join(report["failed_steps"]))
    if report.get("health_check_error"):
        lines.append(f"Sağlık kontrolü de başarısız: {report['health_check_error']}")
    if report.get("run_url"):
        lines.append(f"Çalıştırma: {report['run_url']}")
    events = report.get("recent_errors") or []
    if events:
        lines.append("Son teknik hata kayıtları:")
        for event in events[:5]:
            occurred_at = str(event.get("occurred_at", ""))[:19].replace("T", " ")
            component = str(event.get("component", "uygulama"))[:50]
            message = str(event.get("message", event.get("event_type", "Hata")))[:220]
            context = event.get("context") or {}
            location = context.get("error_location") if isinstance(context, dict) else None
            suffix = f" · {location}" if location else ""
            lines.append(f"• {occurred_at} · {component}: {message}{suffix}")
    if len(lines) == 1:
        lines.append("İş akışı başarısız oldu; GitHub Actions günlüklerini kontrol edin.")
    return "\n".join(lines)[:3800]


def collect_recent_errors(
    db: SupabaseRestClient, *, now: datetime, since: datetime | None = None
) -> list[dict[str, Any]]:
    """Read a small sanitized sample of recent runtime errors for failure alerts."""
    cutoff = (since or now - EVENT_LOOKBACK).isoformat()
    return db.select(
        "operational_events",
        columns="occurred_at,severity,component,event_type,message,context",
        filters={"severity": "eq.error", "occurred_at": f"gte.{cutoff}"},
        limit=5,
        order="occurred_at.desc",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notify", action="store_true")
    parser.add_argument("--fail-on-critical", action="store_true")
    parser.add_argument("--reason")
    parser.add_argument("--workflow", default=os.getenv("GITHUB_WORKFLOW"))
    parser.add_argument("--run-url", default=os.getenv("APP_RUN_URL"))
    parser.add_argument("--failed-step", action="append", default=[])
    parser.add_argument("--events-only", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    db = SupabaseRestClient(settings.supabase_url, settings.supabase_service_role_key)
    now = datetime.now(timezone.utc)
    if args.events_only:
        report = {
            "healthy": True,
            "overdue_evaluation_count": 0,
            "stuck_result_notification_count": 0,
            "stale_active_match_count": 0,
        }
    else:
        try:
            report = collect_health_report(db, now=now)
        except Exception as error:
            if not args.reason:
                raise
            report = {
                "healthy": False,
                "overdue_evaluation_count": 0,
                "stuck_result_notification_count": 0,
                "stale_active_match_count": 0,
                "health_check_error": type(error).__name__,
            }
    report["workflow"] = args.workflow
    report["run_url"] = args.run_url
    report["failed_steps"] = [
        step for step in [*args.failed_step, *os.getenv("FAILED_STEPS", "").split(";")]
        if step.strip()
    ]
    try:
        since = os.getenv("APP_RUN_STARTED_AT", "").strip()
        run_started_at = datetime.fromisoformat(since.replace("Z", "+00:00")) if since else None
        report["recent_errors"] = collect_recent_errors(db, now=now, since=run_started_at)
    except Exception:
        report["recent_errors"] = []
    if not args.events_only:
        record_event(
            db,
            severity="info" if report["healthy"] else "warning",
            component="operational_alerts",
            event_type="health_check",
            message="Operational health check completed",
            context={
                "healthy": report["healthy"],
                "overdue_evaluation_count": report["overdue_evaluation_count"],
                "stuck_result_notification_count": report["stuck_result_notification_count"],
                "stale_active_match_count": report["stale_active_match_count"],
                "workflow_failure_reason_provided": bool(args.reason),
                "recent_error_count": len(report.get("recent_errors", [])),
            },
        )
    print(json.dumps(report, ensure_ascii=False))
    should_notify = args.notify and (
        not report["healthy"] or args.reason or report.get("recent_errors")
    )
    if should_notify:
        token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
        if not token or not chat_id:
            raise RuntimeError("Telegram alert credentials are not configured")
        try:
            send_telegram_message(format_alert(report, reason=args.reason), bot_token=token, chat_id=chat_id)
        except TelegramError as error:
            raise RuntimeError("Operational Telegram alert failed") from error
    if args.fail_on_critical and not report["healthy"]:
        raise RuntimeError("Operational health check found critical pipeline gaps")


if __name__ == "__main__":
    main()
