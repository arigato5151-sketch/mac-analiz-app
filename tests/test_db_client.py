from db.db_client import SupabaseRestClient, _build_session
from datetime import datetime, timezone

from models.predict import load_upcoming_matches


def test_retry_policy_does_not_retry_post():
    retry = _build_session().get_adapter("https://").max_retries
    assert "POST" not in retry.allowed_methods


def test_select_all_uses_deterministic_default_order(monkeypatch):
    client = SupabaseRestClient("https://example.test", "key")
    calls = []

    def fake_select(table, **kwargs):
        calls.append(kwargs)
        return [{"id": 1}]

    monkeypatch.setattr(client, "select", fake_select)
    assert client.select_all("matches", page_size=10) == [{"id": 1}]
    assert calls[0]["order"] == "id.asc"


def test_upcoming_loader_orders_availability_by_its_real_key():
    class FakeDb:
        def __init__(self):
            self.orders = {}

        def select_all(self, table, **kwargs):
            self.orders[table] = kwargs.get("order")
            return []

    db = FakeDb()
    assert load_upcoming_matches(
        db, now=datetime(2026, 10, 1, tzinfo=timezone.utc), horizon_days=7
    ) == []
    assert db.orders["team_availability_status"] == "team_id.asc"
    assert db.orders["fixture_lineups"] == "match_id.asc,team_id.asc"
