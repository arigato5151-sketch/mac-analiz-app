from __future__ import annotations

from pathlib import Path

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
