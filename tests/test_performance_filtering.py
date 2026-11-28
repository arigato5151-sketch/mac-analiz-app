from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from app import performance


class FakeSupabase:
    def __init__(self) -> None:
        self.select_query: dict[str, Any] | None = None
        self.rpc_name: str | None = None

    def select_all(self, table: str, **kwargs: Any) -> list[dict[str, Any]]:
        assert table == "evaluated_prediction_results"
        self.select_query = kwargs
        return [{"match_id": 1}]

    def rpc(self, function: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        assert params is None
        self.rpc_name = function
        return [
            {
                "min_match_date": "2025-01-01T12:00:00+00:00",
                "max_match_date": "2026-11-27T12:00:00+00:00",
                "model_versions": ["model_b", "model_a"],
                "league_options": [{"id": 7, "name": "Lig A"}],
            }
        ]


def test_performance_query_applies_date_model_and_league_filters(monkeypatch) -> None:
    db = FakeSupabase()
    performance.load_evaluated_prediction_results.clear()
    monkeypatch.setattr(performance, "get_db", lambda: db)

    records = performance.load_evaluated_prediction_results(
        date(2026, 11, 27),
        date(2026, 11, 27),
        model_version="model_a",
        league_ids=(7, 9),
    )

    assert records == [{"match_id": 1}]
    assert db.select_query is not None
    filters = db.select_query["filters"]
    assert filters["match_date"].startswith("gte.2026-11-26T21:00:00")
    assert "match_date.lt.2026-11-27T21:00:00" in filters["and"]
    assert filters["model_version"] == "eq.model_a"
    assert filters["league_id"] == "in.(7,9)"
    assert db.select_query["order"] == (
        "match_date.asc,match_id.asc,predicted_at.asc,prediction_id.asc"
    )
    performance.load_evaluated_prediction_results.clear()


def test_performance_filter_metadata_uses_compact_rpc(monkeypatch) -> None:
    db = FakeSupabase()
    performance.load_evaluated_prediction_filter_options.clear()
    monkeypatch.setattr(performance, "get_db", lambda: db)

    options = performance.load_evaluated_prediction_filter_options()

    assert db.rpc_name == "evaluated_prediction_filter_options"
    assert options["model_versions"] == ["model_b", "model_a"]
    assert options["league_options"] == [{"id": 7, "name": "Lig A"}]
    performance.load_evaluated_prediction_filter_options.clear()


def test_performance_filter_rpc_is_invoker_read_only_and_role_restricted() -> None:
    migration = (
        Path(__file__).resolve().parents[1]
        / "db"
        / "migrations"
        / "048_evaluated_prediction_filter_options.sql"
    )
    sql = migration.read_text(encoding="utf-8").upper()

    assert "SECURITY INVOKER" in sql
    assert "SECURITY DEFINER" not in sql
    assert "REVOKE ALL ON FUNCTION PUBLIC.EVALUATED_PREDICTION_FILTER_OPTIONS() FROM PUBLIC" in sql
    assert "GRANT EXECUTE ON FUNCTION PUBLIC.EVALUATED_PREDICTION_FILTER_OPTIONS() TO ANON, SERVICE_ROLE" in sql
    assert "INSERT INTO" not in sql
    assert "UPDATE " not in sql
    assert "DELETE FROM" not in sql
