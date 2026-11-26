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
STATUS_TR = {
    "success": "başarılı",
    "failure": "başarısız",
    "timed_out": "zaman aşımına uğradı",
    "action_required": "işlem gerekiyor",
    "skipped": "atlandı",
    "cancelled": "iptal edildi",
    "in_progress": "sürüyor",
    "queued": "sırada",
    "unknown": "bilinmiyor",
}
EVENT_TR = {
    "schedule": "zamanlanmış çalışma",
    "workflow_dispatch": "elle başlatma",
    "push": "koda gönderim",
    "pull_request": "pull request",
}
NAME_TR = {
    "Match statistics sync": "Maç verileri eşitleme",
    "Confirmed lineups refresh": "Kesinleşen kadroları yenileme",
    "Verify AI provider keys": "Yapay zekâ sağlayıcı anahtarlarını doğrulama",
    "Continuous integration": "Sürekli bütünleştirme denetimi",
    "sync": "Senkronizasyon işi",
    "refresh": "Kadro yenileme işi",
    "verify": "Anahtar doğrulama işi",
    "checks": "Kod denetimi işi",
    "Set up job": "İşi başlat",
    "Complete job": "İşi tamamla",
    "Checkout": "Depoyu hazırla",
    "Set up Python": "Python ortamını kur",
    "Install dependencies": "Bağımlılıkları kur",
    "Install dependencies and check tools": "Bağımlılıkları ve denetim araçlarını kur",
    "Compile Python modules": "Python dosyalarını derle",
    "Ruff lint": "Ruff kod denetimi",
    "Run tests": "Testleri çalıştır",
    "Validate SQL migrations": "SQL geçişlerini doğrula",
    "Verify production migration ledger": "Canlı veritabanı geçiş kaydını doğrula",
    "Verify production public table ACLs": "Canlı tablo erişim izinlerini doğrula",
    "Refresh fixtures from IdDaa bulletin": "IdDaa fikstürlerini yenile",
    "Pull IdDaa daily fixtures and all listed markets": "Günlük IdDaa fikstür ve oranlarını al",
    "Refresh completed match scores and statistics": "Tamamlanan maçların skor ve istatistiklerini yenile",
    "Verify AI commentary key for scheduled run": "Zamanlanmış çalışma için yapay zekâ anahtarını doğrula",
    "Generate and save scheduled AI match commentary": "Yapay zekâ maç yorumlarını üret ve kaydet",
    "Report retained odds history size": "Oran geçmişi boyutunu raporla",
    "Report upcoming fixture coverage and odds freshness": "Yaklaşan maç ve oran güncelliğini raporla",
    "Fetch confirmed lineups for imminent matches": "Yaklaşan maçların kesinleşen kadrolarını al",
    "Backfill recent teams' previous-match lineups on manual runs": "Takımların önceki maç kadrolarını tamamla",
    "Refresh AI analysis on explicit manual runs": "Yapay zekâ analizlerini yenile",
    "Check AI provider credentials": "Yapay zekâ sağlayıcı anahtarlarını denetle",
    "Send detailed Telegram workflow report": "Ayrıntılı Telegram raporunu gönder",
}


def _translated_name(value: object) -> str:
    text = str(value)
    return NAME_TR.get(text, text)


def _translated_status(value: object) -> str:
    text = str(value or "unknown")
    return STATUS_TR.get(text, text)


def _explain_error_line(line: str) -> str:
    lowered = line.casefold()
    if "429" in lowered or "quota" in lowered or "rate limit" in lowered:
        return "Servis kullanım kotası veya istek sınırı aşılmış olabilir."
    if "503" in lowered or "service unavailable" in lowered:
        return "Uzak servis geçici olarak yanıt verememiş."
    if "timeout" in lowered or "timed out" in lowered:
        return "İstek belirlenen süre içinde tamamlanamamış."
    if "401" in lowered or "403" in lowered or "unauthorized" in lowered:
        return "Kimlik doğrulama veya erişim izni reddedilmiş."
    if "importerror" in lowered or "cannot import name" in lowered:
        return "Python modülünden gerekli ad içe aktarılamamış."
    if "modulenotfounderror" in lowered or "no module named" in lowered:
        return "Gerekli Python modülü ortamda bulunamamış."
    if "traceback" in lowered or "exception" in lowered:
        return "Python adımı bir istisna nedeniyle durmuş."
    if "500" in lowered or "internal server error" in lowered:
        return "Uzak servis sunucu hatası döndürmüş."
    return "Bu hata satırı başarısız adımın günlük kaydından alındı."


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

        heading = f"{_translated_name(job.get('name', 'İş'))} — başarısız adım: "
        heading += ", ".join(_translated_name(name) for name in failed_steps)
        if not failed_steps:
            heading += _translated_status(job.get("conclusion"))
        excerpts.append(heading)
        if excerpt_lines:
            excerpts.append("Olası açıklama ve hata günlüğündeki özgün kayıt:")
        for line in excerpt_lines:
            excerpts.append(f"• Açıklama: {_explain_error_line(line)}")
            excerpts.append(f"  Günlük kaydı: {line}")
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
        result.append(
            f"{_translated_name(job.get('name', 'İş'))}: "
            f"{_translated_status(job.get('conclusion') or 'in_progress')}"
        )
        for step in job.get("steps", []):
            raw_conclusion = step.get("conclusion") or step.get("status") or "queued"
            conclusion = _translated_status(raw_conclusion)
            icon = "✅" if raw_conclusion == "success" else "❌" if raw_conclusion in {
                "failure", "timed_out", "action_required"
            } else "⏭️" if raw_conclusion == "skipped" else "⏳"
            result.append(f"{icon} {_translated_name(step.get('name', 'Adım'))}: {conclusion}")
    return result


def build_report() -> str:
    status = os.getenv("WORKFLOW_STATUS", "unknown").strip().lower()
    successful = status == "success"
    title = "✅ İş akışı başarılı" if successful else "❌ İş akışı başarısız"
    workflow = _translated_name(os.getenv("GITHUB_WORKFLOW", "GitHub Actions"))
    repo = os.getenv("GITHUB_REPOSITORY", "")
    run_id = os.getenv("GITHUB_RUN_ID", "")
    server = os.getenv("GITHUB_SERVER_URL", "https://github.com").rstrip("/")
    run_url = f"{server}/{repo}/actions/runs/{run_id}" if repo and run_id else "Bağlantı yok"
    lines = [
        f"{title}: {workflow}",
        f"Durum: {_translated_status(status)}",
        f"Tetikleyici: {EVENT_TR.get(os.getenv('GITHUB_EVENT_NAME', ''), 'bilinmiyor')}",
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
                    step_lines.append(f"{_translated_name(name)}: {_translated_status(outcome)}")
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
