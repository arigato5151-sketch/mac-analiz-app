from __future__ import annotations

from datetime import date
from typing import Any

from app.components import data


class FakeSupabase:
    def __init__(self) -> None:
        self.cache_query: dict[str, Any] | None = None

    def select_all(self, table: str, **kwargs: Any) -> list[dict[str, Any]]:
        if table == "matches":
            return [
                {
                    "id": 17,
                    "league_id": 3,
                    "home_team_id": 11,
                    "away_team_id": 12,
                    "match_date": "2099-01-01T12:00:00Z",
                    "status": "scheduled",
                }
            ]
        if table == "match_analysis_cache":
            self.cache_query = kwargs
            return [{"match_id": 17, "model_version": "test"}]
        return []


def test_match_analysis_cache_query_orders_by_its_primary_key(monkeypatch) -> None:
    db = FakeSupabase()
    data.load_match_day.clear()
    monkeypatch.setattr(data, "get_db", lambda: db)

    _, _, analyses = data.load_match_day(date(2099, 1, 1))

    assert analyses == [{"match_id": 17, "model_version": "test"}]
    assert db.cache_query is not None
    assert db.cache_query["order"] == "match_id.asc"
    data.load_match_day.clear()
