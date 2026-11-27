from __future__ import annotations

from pathlib import Path


MIGRATION = (
    Path(__file__).resolve().parents[1]
    / "db"
    / "migrations"
    / "047_latest_iddaa_quote_rpc.sql"
)


def test_latest_quote_rpcs_are_read_only_and_return_newest_per_match() -> None:
    sql = MIGRATION.read_text(encoding="utf-8").upper()

    assert sql.count("DISTINCT ON (QUOTE.MATCH_ID)") == 2
    assert sql.count("ORDER BY QUOTE.MATCH_ID, QUOTE.CAPTURED_AT DESC, QUOTE.ID DESC") == 2
    assert sql.count("SECURITY INVOKER") == 2
    assert "SECURITY DEFINER" not in sql
    assert "QUOTE.BOOKMAKER = 'IDDAA.COM'" in sql
    assert "REVOKE ALL ON FUNCTION PUBLIC.LATEST_IDDAA_QUOTES(INTEGER[]) FROM PUBLIC" in sql
    assert "REVOKE ALL ON FUNCTION PUBLIC.LATEST_IDDAA_QUOTE_TIMES(INTEGER[]) FROM PUBLIC" in sql
    assert sql.count("TO ANON, SERVICE_ROLE") == 2
    assert "INSERT INTO" not in sql
    assert "UPDATE " not in sql
    assert "DELETE FROM" not in sql
