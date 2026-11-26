"""Send a detailed GitHub Actions run report to Telegram."""

from __future__ import annotations

import json
import logging
import os
import re
import zipfile
from io import BytesIO

import requests

from data_pipeline.notify_failure import send_telegram_alert

LOG = logging.getLogger(__name__)
API_ROOT = "https://api.github.com"
ERROR_LINE = re.compile(
    r"(error|exception|traceback|failed|failure|timed? ?out|quota|429|500|503|denied)",
    re.IGNORECASE,
)


def _github_headers(token: str) -> dict[str, str]:
    return {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _failure_details(repo: str, run_id: str, token: str) -> list[str]:
    """Read failed job logs and return a short, redacted error excerpt."""
    if not (repo and run_id and token):
        return []
    try:
        response = requests.get(
            f"{API_ROOT}/repos/{repo}/actions/runs/{run_id}/jobs",
            params={"per_page": 100},
            headers=_github_headers(token),
            timeout=15,
        )
        response.raise_for_status()
        jobs = response.json().get("jobs", [])
    except (requests.RequestException, ValueError) as exc:
        LOG.warning("Could not retrieve failed job details: %s", exc)
        return []

    excerpts: list[str] = []
    for job in jobs:
        if job.get("conclusion") not in {"failure", "timed_out", "action_required"}:
            continue
        failed_steps = [
            step.get("name", "Bilinmeyen adım")
            for step in job.get("steps", [])
            if step.get("conclusion") in {"failure", "timed_out", "action_required"}
        ]
        excerpt_lines: list[str] = []
        try:
            log_response = requests.get(
                f"{API_ROOT}/repos/{repo}/actions/jobs/{job['id']}/logs",
                headers=_github_headers(token),
                timeout=25,
            )
            log_response.raise_for_status()
            content = log_response.content
            if zipfile.is_zipfile(BytesIO(content)):
                with zipfile.ZipFile(BytesIO(content)) as archive:
                    log_text = "\n".join(
                        archive.read(name).decode("utf-8", errors="replace")
                        for name in archive.namelist()
                        if not name.endswith("/")
                    )
            else:
                log_text = content.decode("utf-8", errors="replace")
            excerpt_lines = [
                line.strip()[-350:]
                for line in log_text.splitlines()
                if ERROR_LINE.search(line)
            ][-8:]
        except (requests.RequestException, KeyError, OSError, zipfile.BadZipFile) as exc:
            LOG.warning("Could not retrieve logs for failed job: %s", exc)

        heading = f"{job.get('name', 'İş')} — başarısız adım: "
        heading += ", ".join(failed_steps) if failed_steps else str(job.get("conclusion"))
        excerpts.append(heading)
        excerpts.extend(f"• {line}" for line in excerpt_lines)
    return excerpts


def _api_step_statuses(repo: str, run_id: str, token: str) -> list[str]:
    if not (repo and run_id and token):
        return []
    try:
        response = requests.get(
            f"{API_ROOT}/repos/{repo}/actions/runs/{run_id}/jobs",
            params={"per_page": 100},
            headers=_github_headers(token),
            timeout=15,
        )
        response.raise_for_status()
        jobs = response.json().get("jobs", [])
    except (requests.RequestException, ValueError) as exc:
        LOG.warning("Could not retrieve workflow step statuses: %s", exc)
        return []

    result: list[str] = []
    for job in jobs:
        result.append(f"{job.get('name', 'İş')}: {job.get('conclusion') or 'çalışıyor'}")
        for step in job.get("steps", []):
            conclusion = step.get("conclusion") or step.get("status") or "bekleniyor"
            icon = "✅" if conclusion == "success" else "❌" if conclusion in {
                "failure", "timed_out", "action_required"
            } else "⏭️" if conclusion == "skipped" else "⏳"
            result.append(f"{icon} {step.get('name', 'Adım')}: {conclusion}")
    return result


def build_report() -> str:
    status = os.getenv("WORKFLOW_STATUS", "unknown").strip().lower()
    successful = status == "success"
    title = "✅ İş akışı başarılı" if successful else "❌ İş akışı başarısız"
    workflow = os.getenv("GITHUB_WORKFLOW", "GitHub Actions")
    repo = os.getenv("GITHUB_REPOSITORY", "")
    run_id = os.getenv("GITHUB_RUN_ID", "")
    server = os.getenv("GITHUB_SERVER_URL", "https://github.com").rstrip("/")
    run_url = f"{server}/{repo}/actions/runs/{run_id}" if repo and run_id else "Bağlantı yok"
    lines = [
        f"{title}: {workflow}",
        f"Durum: {status}",
        f"Tetikleyici: {os.getenv('GITHUB_EVENT_NAME', 'bilinmiyor')}",
        f"Dal/etiket: {os.getenv('GITHUB_REF_NAME', 'bilinmiyor')}",
        f"Başlatan: {os.getenv('GITHUB_ACTOR', 'bilinmiyor')}",
        f"Çalıştırma: {run_url}",
    ]

    step_lines = _api_step_statuses(
        repo,
        run_id,
        os.getenv("GITHUB_TOKEN", ""),
    )
    if not step_lines:
        try:
            steps = json.loads(os.getenv("STEP_RESULTS_JSON", "{}"))
        except ValueError:
            steps = {}
        if isinstance(steps, dict):
            for name, details in steps.items():
                if isinstance(details, dict):
                    outcome = details.get("outcome") or details.get("conclusion") or "çalışmadı"
                    step_lines.append(f"{name}: {outcome}")
    if step_lines:
        lines.extend(["\nİş ve adım sonuçları:", *step_lines])

    summary_path = os.getenv("GITHUB_STEP_SUMMARY", "")
    if summary_path and os.path.isfile(summary_path):
        try:
            summary = open(summary_path, encoding="utf-8").read().strip()
            if summary:
                lines.extend(["\nİş akışı özeti:", summary[-1200:]])
        except OSError as exc:
            LOG.warning("Could not read workflow summary: %s", exc)

    if not successful:
        details = _failure_details(
            repo,
            run_id,
            os.getenv("GITHUB_TOKEN", ""),
        )
        lines.append("\nHata nedeni ve günlük ayrıntıları:")
        lines.extend(details or [
            "Başarısız adımlar yukarıda listelendi. Ayrıntılı günlük için çalıştırma bağlantısını açın."
        ])
    report = "\n".join(lines)
    if len(report) > 3800:
        report = report[:3650].rstrip() + "\n… Devamı: Actions günlük bağlantısı."
    return report


def main() -> None:
    message = build_report()
    # GitHub masks secrets in its own logs; apply the same protection before
    # sending excerpts retrieved through the Actions API to Telegram.
    for key, value in os.environ.items():
        if key.endswith("_KEY") or "TOKEN" in key or "SECRET" in key:
            if value and len(value) >= 6:
                message = message.replace(value, "[gizlendi]")
    if not send_telegram_alert(message):
        LOG.warning("Workflow report was not delivered to Telegram")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
