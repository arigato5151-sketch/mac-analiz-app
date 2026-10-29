from __future__ import annotations

from pathlib import Path

import pytest

from db.db_client import DatabaseError, SupabaseRestClient
from db.verify_acl import assert_no_acl_violations


MIGRATION = (
    Path(__file__).resolve().parents[1]
    / "db"
    / "migrations"
    / "042_read_only_public_acl_check.sql"
)


def test_acl_rpc_is_read_only_and_restricted_to_service_role() -> None:
    sql = MIGRATION.read_text(encoding="utf-8").upper()

    assert "SECURITY DEFINER" in sql
    assert "SET SEARCH_PATH = PG_CATALOG, PUBLIC" in sql
    assert "REVOKE ALL ON FUNCTION PUBLIC.VERIFY_PUBLIC_TABLE_ACL()" in sql
    assert "FROM PUBLIC, ANON, AUTHENTICATED" in sql
    assert "GRANT EXECUTE ON FUNCTION PUBLIC.VERIFY_PUBLIC_TABLE_ACL() TO SERVICE_ROLE" in sql
    assert "INSERT INTO" not in sql
    assert "UPDATE " not in sql
    assert "DELETE FROM" not in sql
    assert "TRUNCATE " not in sql


def test_acl_rpc_reports_only_effective_violations() -> None:
    assert_no_acl_violations([])

    try:
        assert_no_acl_violations(
            [
                {
                    "grantee": "anon",
                    "table_name": "schema_migrations",
                    "privilege_type": "TRUNCATE",
                }
            ]
        )
    except RuntimeError as error:
        assert "anon" in str(error)
        assert "TRUNCATE" in str(error)
    else:
        raise AssertionError("ACL violation must fail the production check")


@pytest.mark.parametrize(
    "response",
    [
        {},
        None,
        "[]",
        3,
        [{"grantee": "anon", "table_name": "schema_migrations", "privilege_type": "SELECT"}],
        [{"grantee": "owner", "table_name": "schema_migrations", "privilege_type": "TRUNCATE"}],
        [{"grantee": "anon", "table_name": "unknown_table", "privilege_type": "TRUNCATE"}],
        [{"grantee": "anon", "table_name": "schema_migrations"}],
        [{"grantee": "anon", "table_name": "schema_migrations", "privilege_type": "TRUNCATE", "extra": "value"}],
        ["not a row"],
        [{"grantee": 4, "table_name": "schema_migrations", "privilege_type": "TRUNCATE"}],
    ],
)
def test_acl_rpc_rejects_invalid_response_contract(response: object) -> None:
    with pytest.raises(RuntimeError, match="invalid response"):
        assert_no_acl_violations(response)


def test_acl_rpc_accepts_object_row_normalized_by_rest_client() -> None:
    class FakeResponse:
        ok = True
        content = b"{}"

        @staticmethod
        def json() -> dict[str, str]:
            return {"grantee": "anon", "table_name": "schema_migrations", "privilege_type": "TRUNCATE"}

    class FakeSession:
        @staticmethod
        def request(*args, **kwargs) -> FakeResponse:
            return FakeResponse()

    rows = SupabaseRestClient("https://example.supabase.co", "test-service-key", session=FakeSession()).rpc(
        "verify_public_table_acl"
    )
    assert isinstance(rows, list)
    with pytest.raises(RuntimeError, match="violations"):
        assert_no_acl_violations(rows)


def test_invalid_rpc_json_is_wrapped_without_response_body() -> None:
    class FakeResponse:
        ok = True
        content = b"sensitive-response-body"

        @staticmethod
        def json() -> object:
            raise ValueError("sensitive-response-body")

    class FakeSession:
        @staticmethod
        def request(*args, **kwargs) -> FakeResponse:
            return FakeResponse()

    client = SupabaseRestClient("https://example.supabase.co", "test-service-key", session=FakeSession())
    with pytest.raises(DatabaseError, match="invalid JSON") as error:
        client.rpc("verify_public_table_acl")
    assert "sensitive-response-body" not in str(error.value)
