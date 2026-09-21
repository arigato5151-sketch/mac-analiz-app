from __future__ import annotations

import subprocess
from unittest.mock import MagicMock

import scheduler


def test_run_module_uses_an_isolated_python_process(monkeypatch) -> None:
    completed = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="done\n", stderr=""
    )
    run = MagicMock(return_value=completed)
    monkeypatch.setattr(scheduler.subprocess, "run", run)

    result = scheduler._run_module("example.module", "--days", "3")

    run.assert_called_once_with(
        [scheduler.sys.executable, "-m", "example.module", "--days", "3"],
        check=True,
        capture_output=True,
        cwd=scheduler.PROJECT_ROOT,
        text=True,
    )
    assert result == {"returncode": 0, "stdout": "done", "stderr": ""}


def test_scheduler_registers_every_job_without_loading_app_settings() -> None:
    configured = scheduler.create_scheduler()

    assert {job.id for job in configured.get_jobs()} == {
        "morning_update",
        "evening_update",
        "night_update",
        "pre_match_notifications",
        "result_notifications",
        "daily_summary",
        "weekly_retrain",
        "operational_alerts",
    }


def test_daily_summary_uses_morning_mode(monkeypatch) -> None:
    run_module = MagicMock(return_value={"returncode": 0})
    monkeypatch.setattr(scheduler, "_run_module", run_module)

    assert scheduler.run_daily_summary() == {"returncode": 0}
    run_module.assert_called_once_with(
        "notifications.daily_summary", "--mode", "morning"
    )


def test_morning_update_uses_shared_seven_day_horizon(monkeypatch) -> None:
    run_module = MagicMock(return_value={"returncode": 0})
    monkeypatch.setattr(scheduler, "_run_module", run_module)

    scheduler.run_morning_update()

    assert [call.args for call in run_module.call_args_list] == [
        ("data_pipeline.fetch_fixtures", "--days", "7"),
        ("data_pipeline.refresh_context", "--days", "7"),
        ("models.predict", "--days", "7"),
        ("data_pipeline.data_quality", "--days", "7"),
    ]


def test_result_job_calls_existing_notification_entrypoint(monkeypatch) -> None:
    from notifications import final_results

    notify = MagicMock(return_value={"pending": 1, "sent": 1})
    monkeypatch.setattr(final_results, "run_final_result_notifications", notify)

    assert scheduler.run_result_notifications() == {"pending": 1, "sent": 1}
    notify.assert_called_once_with()


def test_weekly_retrain_does_not_publish_before_shadow_promotion(monkeypatch) -> None:
    run_module = MagicMock(return_value={"returncode": 0})
    monkeypatch.setattr(scheduler, "_run_module", run_module)

    scheduler.run_weekly_retrain()

    assert run_module.call_args_list[0].args == (
        "models.train_model",
        "--optimize",
        "--optuna-trials",
        "25",
    )
    assert all("--publish-latest" not in call.args for call in run_module.call_args_list)
