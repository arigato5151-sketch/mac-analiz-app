from __future__ import annotations

import json
from datetime import date, datetime
from unittest.mock import ANY, Mock

import pytest

from data_pipeline import fetch_fixtures
from data_pipeline.api_client import ApiFootballError


def plan_error(start: str, end: str) -> ApiFootballError:
    return ApiFootballError(
        f"Free plans do not have access to this date, try from {start} to {end}"
    )


def fixture(day: str) -> dict[str, object]:
    return {
        "league": {"id": 39},
        "teams": {
            "home": {"id": 1, "name": "Home"},
            "away": {"id": 2, "name": "Away"},
        },
        "fixture": {"id": int(day[-2:]), "date": f"{day}T12:00:00Z", "status": {"short": "NS"}},
        "goals": {"home": None, "away": None},
    }


def run_sync(api: Mock, start: str, end: str) -> tuple[object, Mock]:
    db = Mock()
    summary = fetch_fixtures.sync_fixtures(
        api, db, start_date=date.fromisoformat(start), end_date=date.fromisoformat(end), league_ids=(39,)
    )
    return summary, db


def test_plan_end_shortfall_is_reported_and_accessible_data_upserted() -> None:
    api = Mock()
    api.get.side_effect = [
        [fixture("2026-10-01")],
        [fixture("2026-10-02")],
        plan_error("2026-10-01", "2026-10-02"),
    ]

    summary, db = run_sync(api, "2026-10-01", "2026-10-04")

    assert summary.is_complete is False
    assert (summary.requested_start, summary.requested_end) == ("2026-10-01", "2026-10-04")
    assert (summary.completed_start, summary.completed_end) == ("2026-10-01", "2026-10-02")
    assert (summary.skipped_start, summary.skipped_end) == ("2026-10-03", "2026-10-04")
    db.upsert.assert_any_call("matches", ANY, on_conflict="id")
    assert summary.matches_upserted == 2


def test_dates_before_plan_start_are_recorded_then_fetch_resumes() -> None:
    api = Mock()
    api.get.side_effect = [
        plan_error("2026-10-02", "2026-10-04"),
        [fixture("2026-10-02")],
        [fixture("2026-10-03")],
        [fixture("2026-10-04")],
    ]

    summary, _ = run_sync(api, "2026-10-01", "2026-10-04")

    assert summary.skipped_start == summary.skipped_end == "2026-10-01"
    assert summary.completed_start == "2026-10-02"
    assert summary.completed_end == "2026-10-04"
    assert summary.is_complete is False
    assert [call.args[1]["date"] for call in api.get.call_args_list] == [
        "2026-10-01", "2026-10-02", "2026-10-03", "2026-10-04"
    ]


def test_fully_accessible_range_is_complete() -> None:
    api = Mock()
    api.get.side_effect = [[fixture("2026-10-01")], [fixture("2026-10-02")]]

    summary, _ = run_sync(api, "2026-10-01", "2026-10-02")

    assert summary.is_complete is True
    assert summary.skipped_start is summary.skipped_end is None
    assert summary.completed_start == "2026-10-01"
    assert summary.completed_end == "2026-10-02"


def test_plan_window_can_exclude_entire_requested_range() -> None:
    api = Mock()
    api.get.side_effect = [plan_error("2026-10-05", "2026-10-07")]

    summary, _ = run_sync(api, "2026-10-01", "2026-10-03")

    assert summary.is_complete is False
    assert summary.completed_start is summary.completed_end is None
    assert (summary.skipped_start, summary.skipped_end) == ("2026-10-01", "2026-10-03")


def test_non_plan_api_error_still_raises() -> None:
    api = Mock()
    error = ApiFootballError("temporary API failure")
    api.get.side_effect = error

    with pytest.raises(ApiFootballError, match="temporary API failure"):
        run_sync(api, "2026-10-01", "2026-10-01")


def test_main_prints_machine_readable_partial_summary(monkeypatch, capsys) -> None:
    api = Mock()
    api.diagnostics.return_value = {"requests": 1}
    api.get.side_effect = [
        [fixture("2026-10-01")],
        [fixture("2026-10-02")],
        plan_error("2026-10-01", "2026-10-02"),
    ]
    db = Mock()
    monkeypatch.setattr(fetch_fixtures, "get_settings", lambda: Mock(api_football_key="fake"))
    monkeypatch.setattr(fetch_fixtures, "ApiFootballClient", lambda _: api)
    monkeypatch.setattr(fetch_fixtures, "SupabaseRestClient", lambda *_: db)
    monkeypatch.setattr(fetch_fixtures, "datetime", Mock(now=lambda _: datetime(2026, 10, 1)))
    monkeypatch.setattr("sys.argv", ["fetch_fixtures", "--days", "3"])

    fetch_fixtures.main()

    output = capsys.readouterr().out
    result = json.loads(output)
    assert result["sync"]["is_complete"] is False
    assert result["sync"]["skipped_start"] == result["sync"]["skipped_end"]
    db.upsert.assert_any_call("matches", ANY, on_conflict="id")
