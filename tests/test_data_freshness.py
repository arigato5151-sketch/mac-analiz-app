from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from data_pipeline.monitor_data_freshness import collect_data_health


class FakeSupabase:
    def __init__(self) -> None:
        self.quote_match_ids: list[int] | None = None

    def select_all(self, table: str, **kwargs: Any) -> list[dict[str, Any]]:
        assert table == "matches"
        assert kwargs["columns"] == "id,match_date,status"
        return [
            {"id": 11, "match_date": "2026-11-28T15:00:00Z", "status": "scheduled"},
            {"id": 12, "match_date": "2026-11-29T15:00:00Z", "status": "scheduled"},
        ]

    def rpc(self, function: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        assert function == "latest_iddaa_quote_times"
        self.quote_match_ids = params["p_match_ids"]
        return [{"match_id": 11, "captured_at": "2026-11-27T10:00:00Z"}]


def test_data_health_checks_only_the_latest_quote_per_fixture() -> None:
    now = datetime(2026, 11, 27, 12, tzinfo=timezone.utc)
    db = FakeSupabase()

    health = collect_data_health(db, now=now)

    assert db.quote_match_ids == [11, 12]
    assert health["scheduled_matches"] == 2
    assert health["matches_with_iddaa_odds"] == 1
    assert health["matches_without_iddaa_odds"] == 1
    assert health["quote_age_hours"] == 2.0
