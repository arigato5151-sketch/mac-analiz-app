from db.db_client import SupabaseRestClient, _build_session


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
