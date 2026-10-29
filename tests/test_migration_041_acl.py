from __future__ import annotations

import re
from pathlib import Path

import pytest


MIGRATION = (
    Path(__file__).resolve().parents[1]
    / "db"
    / "migrations"
    / "041_revoke_dangerous_public_table_acl.sql"
)

TABLES = {
    "schema_migrations",
    "result_notification_queue",
    "notification_log",
    "model_candidates",
    "shadow_predictions",
    "shadow_prediction_performance",
    "personal_match_workspace",
    "personal_match_decisions",
    "bookmaker_odds_snapshots",
}
CREATED_IN_REPO = {
    "schema_migrations",
    "result_notification_queue",
    "model_candidates",
    "shadow_predictions",
    "shadow_prediction_performance",
    "personal_match_workspace",
    "personal_match_decisions",
}
PRODUCTION_ONLY = {"notification_log", "bookmaker_odds_snapshots"}


def migration_parts() -> tuple[str, str, str]:
    sql = MIGRATION.read_text(encoding="utf-8")
    executable_sql = "\n".join(
        line for line in sql.splitlines() if not line.lstrip().startswith("--")
    )
    do_match = re.search(r"\bDO\s+\$\$(.*?)\$\$\s*;", executable_sql, re.I | re.S)
    assert do_match is not None
    do_start = do_match.start()
    return sql, executable_sql[:do_start], do_match.group(1)


def test_migration_revokes_only_dangerous_privileges_from_public_roles() -> None:
    sql, before_do, _ = migration_parts()
    executable_sql = "\n".join(
        line for line in sql.splitlines() if not line.lstrip().startswith("--")
    ).strip()
    static_revoke = re.search(
        r"REVOKE\s+([^;]+?)\s+ON\s+TABLE\s+(.*?)\s+FROM\s+anon,\s*authenticated\s*;",
        before_do,
        flags=re.IGNORECASE | re.DOTALL,
    )

    assert static_revoke is not None
    assert {
        privilege.strip().upper()
        for privilege in static_revoke.group(1).split(",")
    } == {"TRUNCATE", "REFERENCES", "TRIGGER"}
    assert {
        table.rsplit(".", maxsplit=1)[-1].strip().lower()
        for table in static_revoke.group(2).split(",")
    } == TABLES
    assert not re.search(r"\b(?:SELECT|INSERT|UPDATE|DELETE)\b", before_do, re.I)
    assert executable_sql.upper().startswith("BEGIN;")
    assert executable_sql.upper().endswith("COMMIT;")


def test_maintain_revoke_is_conditional_on_postgresql_17() -> None:
    _, before_do, do_block = migration_parts()
    assert re.search(r"IF\s+current_setting\('server_version_num'\)::integer\s*>=\s*170000\s+THEN", do_block, re.I)
    dynamic_revoke = re.search(
        r"EXECUTE\s+'(REVOKE\s+MAINTAIN.*?FROM\s+anon,\s*authenticated)'\s*;",
        do_block,
        flags=re.IGNORECASE | re.DOTALL,
    )
    assert dynamic_revoke is not None
    dynamic_sql = dynamic_revoke.group(1)
    assert not re.search(r"\b(?:SELECT|INSERT|UPDATE|DELETE)\b", dynamic_sql, re.I)
    assert {
        table.rsplit(".", maxsplit=1)[-1].strip().lower()
        for table in re.findall(r"public\.([a-z_]+)", dynamic_sql, re.I)
    } == TABLES
    assert not re.search(r"\bMAINTAIN\b", before_do, re.I)


@pytest.mark.parametrize(
    ("server_version_num", "runs_maintain_revoke"),
    [(160000, False), (170000, True), (170006, True)],
)
def test_maintain_branch_semantics_for_postgresql_16_and_17(
    server_version_num: int, runs_maintain_revoke: bool
) -> None:
    _, _, do_block = migration_parts()
    threshold = re.search(r"server_version_num'\)::integer\s*>=\s*(\d+)", do_block)
    assert threshold is not None
    assert (server_version_num >= int(threshold.group(1))) is runs_maintain_revoke
    # MAINTAIN occurs only in the dynamic SQL literal, invisible to PG16's SQL parser.
    assert re.search(r"EXECUTE\s+'REVOKE MAINTAIN", do_block, re.I)


def test_migration_includes_seed_free_post_apply_check() -> None:
    sql, _, _ = migration_parts()
    assert "information_schema.role_table_grants" in sql
    assert "has_table_privilege" in sql
    assert "current_setting('server_version_num')::integer >= 170000" in sql
    assert "r.grantee::name" in sql
    for privilege in ("TRUNCATE", "REFERENCES", "TRIGGER", "MAINTAIN"):
        assert f"'{privilege}'" in sql
    for table in TABLES:
        assert f"'{table}'" in sql


def test_target_tables_match_repo_migrations_and_known_production_tables() -> None:
    migration_dir = MIGRATION.parent
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in migration_dir.glob("*.sql")
        if path != MIGRATION
    )
    created = set(
        re.findall(
            r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?public\.([a-z_]+)",
            source,
            flags=re.IGNORECASE,
        )
    )
    assert CREATED_IN_REPO <= created
    assert TABLES == CREATED_IN_REPO | PRODUCTION_ONLY


def test_migration_validator_requires_executable_begin_and_final_commit(
    tmp_path: Path,
) -> None:
    from db.migrations import discover_migrations

    path = tmp_path / "041_transaction_test.sql"
    path.write_text("-- BEGIN; in comment only\nSELECT 1;\nCOMMIT;\n", encoding="utf-8")
    with pytest.raises(ValueError, match="transaction-wrapped"):
        discover_migrations(tmp_path)

    path.write_text(
        "-- leading comment\nBEGIN;\nSELECT 1;\nCOMMIT;\n-- trailing comment\n",
        encoding="utf-8",
    )
    assert len(discover_migrations(tmp_path)) == 1
