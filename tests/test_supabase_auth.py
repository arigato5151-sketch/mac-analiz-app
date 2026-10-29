from __future__ import annotations

from typing import Any

import pytest

from db.db_client import (
    AuthenticatedSupabaseRestClient,
    DatabaseError,
    PublicSupabaseRestClient,
    SupabaseRestClient,
)


class FakeResponse:
    def __init__(self, *, ok: bool = True, text: str = "", payload: Any = None) -> None:
        self.ok = ok
        self.text = text
        self.content = b"{}" if ok else text.encode()
        self._payload = payload if payload is not None else {}
        self.status_code = 200 if ok else 403

    def json(self) -> Any:
        return self._payload


class FakeSession:
    def __init__(self, response: FakeResponse | None = None) -> None:
        self.response = response or FakeResponse()
        self.calls: list[dict[str, Any]] = []

    def request(self, method: str, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append({"method": method, "url": url, **kwargs})
        return self.response


def request_headers(client: SupabaseRestClient, session: FakeSession) -> dict[str, str]:
    client.select("matches", columns="id")
    return session.calls[-1]["headers"]


def test_legacy_service_role_jwt_is_sent_as_bearer() -> None:
    fake_jwt = "header.payload.signature"
    session = FakeSession()
    headers = request_headers(
        SupabaseRestClient("https://example.supabase.co", fake_jwt, session=session),
        session,
    )
    assert headers["apikey"] == fake_jwt
    assert headers["Authorization"] == f"Bearer {fake_jwt}"


def test_new_service_secret_uses_apikey_without_authorization() -> None:
    fake_secret = "test-service-key"
    session = FakeSession()
    headers = request_headers(
        SupabaseRestClient("https://example.supabase.co", fake_secret, session=session),
        session,
    )
    assert headers["apikey"] == fake_secret
    assert "Authorization" not in headers


@pytest.mark.parametrize(
    ("method", "args", "kwargs"),
    [
        ("insert", ("matches", [{"id": 1}]), {}),
        ("upsert", ("matches", [{"id": 1}]), {}),
        ("update", ("matches", {"status": "finished"}), {"filters": {"id": "eq.1"}}),
        ("delete", ("matches",), {"filters": {"id": "eq.1"}}),
    ],
)
def test_public_client_rejects_all_mutations(
    method: str, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> None:
    client = PublicSupabaseRestClient(
        "https://example.supabase.co", "test-service-key", session=FakeSession()
    )
    with pytest.raises(PermissionError, match="read-only"):
        getattr(client, method)(*args, **kwargs)


def test_authenticated_client_sends_access_token_as_bearer() -> None:
    session = FakeSession()
    client = AuthenticatedSupabaseRestClient(
        "https://example.supabase.co",
        "test-service-key",
        "user-access-token",
        session=session,
    )
    headers = request_headers(client, session)
    assert headers["apikey"] == "test-service-key"
    assert headers["Authorization"] == "Bearer user-access-token"


def test_database_errors_do_not_expose_body_or_credentials_in_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    fake_jwt = "header.payload.signature"
    response_body = f"provider error contained {fake_jwt}"
    session = FakeSession(FakeResponse(ok=False, text=response_body))
    client = SupabaseRestClient(
        "https://example.supabase.co", fake_jwt, session=session
    )

    with pytest.raises(DatabaseError) as error:
        client.select("matches")

    # No credential or body is included in the exception that a caller may log.
    assert fake_jwt not in str(error.value)
    assert response_body not in str(error.value)
    assert fake_jwt not in caplog.text


def test_migration_ledger_comparison_is_exercised_with_source_checksums() -> None:
    from db.migrations import discover_migrations, verify_ledger

    migrations = discover_migrations()

    class SourceLedger:
        def select_all(self, *_: Any, **__: Any) -> list[dict[str, str]]:
            return [
                {"version": migration.version, "checksum": migration.checksum}
                for migration in migrations
            ]

    assert verify_ledger(migrations, SourceLedger()) == {
        "missing": [],
        "mismatched": [],
        "unknown": [],
    }
