"""Validate immutable SQL migrations and verify their Supabase ledger entries."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from config.settings import get_supabase_admin_settings
from db.db_client import SupabaseRestClient

MIGRATIONS_DIR = Path(__file__).with_name("migrations")
MIGRATION_NAME = re.compile(r"^(?P<version>\d{3}_[a-z0-9_]+)\.sql$")
_DOLLAR_QUOTE = re.compile(r"\$[A-Za-z_][A-Za-z0-9_]*\$|\$\$")
# 030 was intentionally reserved; migration files are immutable and are not renumbered.
ALLOWED_MIGRATION_GAPS = frozenset({30})


@dataclass(frozen=True)
class Migration:
    version: str
    path: Path
    checksum: str


def _top_level_sql_statements(sql: str) -> list[str]:
    """Split SQL at top-level semicolons, ignoring comments and quoted text."""
    statements: list[str] = []
    current: list[str] = []
    index = 0
    block_comment_depth = 0
    quote: str | None = None
    dollar_quote: str | None = None
    escape_string = False

    while index < len(sql):
        char = sql[index]
        next_char = sql[index + 1] if index + 1 < len(sql) else ""

        if block_comment_depth:
            if char == "/" and next_char == "*":
                block_comment_depth += 1
                index += 2
                continue
            if char == "*" and next_char == "/":
                block_comment_depth -= 1
                index += 2
                continue
            index += 1
            continue

        if dollar_quote is not None:
            if sql.startswith(dollar_quote, index):
                current.append(" ")
                index += len(dollar_quote)
                dollar_quote = None
            else:
                index += 1
            continue

        if quote is not None:
            if char == quote:
                if next_char == quote:
                    index += 2
                    continue
                quote = None
            elif char == "\\" and escape_string:
                index += 2
                continue
            index += 1
            continue

        if char == "-" and next_char == "-":
            index += 2
            while index < len(sql) and sql[index] not in "\r\n":
                index += 1
            current.append(" ")
            continue
        if char == "/" and next_char == "*":
            block_comment_depth = 1
            index += 2
            current.append(" ")
            continue
        if char in {"'", '"'}:
            quote = char
            escape_string = (
                char == "'"
                and index > 0
                and sql[index - 1] in "Ee"
                and (index < 2 or not (sql[index - 2].isalnum() or sql[index - 2] == "_"))
            )
            current.append(" ")
            index += 1
            continue
        if char == "$":
            match = _DOLLAR_QUOTE.match(sql, index)
            if match is not None:
                dollar_quote = match.group()
                current.append(" ")
                index = match.end()
                continue
        if char == ";":
            statement = "".join(current).strip()
            if statement:
                statements.append(statement)
            current.clear()
        else:
            current.append(char)
        index += 1

    trailing = "".join(current).strip()
    if trailing:
        statements.append(trailing)
    return statements


def discover_migrations(directory: Path = MIGRATIONS_DIR) -> list[Migration]:
    """Read numbered migrations in deterministic order and reject stray SQL files."""
    migrations: list[Migration] = []
    for path in sorted(directory.glob("*.sql")):
        match = MIGRATION_NAME.fullmatch(path.name)
        if match is None:
            raise ValueError(f"Invalid migration filename: {path.name}")
        text = path.read_text(encoding="utf-8")
        statements = _top_level_sql_statements(text)
        controls: list[str] = []
        for statement in statements:
            upper_statement = statement.upper()
            if upper_statement.startswith("START TRANSACTION"):
                controls.append("START TRANSACTION")
                continue
            if upper_statement.startswith("PREPARE TRANSACTION"):
                controls.append("PREPARE TRANSACTION")
                continue
            first_word = upper_statement.split(None, maxsplit=1)[0]
            if first_word in {"BEGIN", "COMMIT", "END", "ABORT", "ROLLBACK"}:
                controls.append(first_word)
        if (
            not statements
            or statements[0].strip().upper() != "BEGIN"
            or statements[-1].strip().upper() != "COMMIT"
            or controls != ["BEGIN", "COMMIT"]
        ):
            raise ValueError(f"Migration must be transaction-wrapped: {path.name}")
        migrations.append(
            Migration(
                version=match.group("version"),
                path=path,
                checksum=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            )
        )
    versions = [migration.version for migration in migrations]
    if not migrations:
        raise ValueError("No SQL migrations found")
    if len(versions) != len(set(versions)):
        raise ValueError("Duplicate migration versions found")
    numbers = [int(version[:3]) for version in versions]
    expected = set(range(numbers[0], numbers[-1] + 1))
    gaps = expected.difference(numbers)
    unexpected_gaps = gaps.difference(ALLOWED_MIGRATION_GAPS)
    if unexpected_gaps:
        formatted = ", ".join(f"{number:03d}" for number in sorted(unexpected_gaps))
        raise ValueError(f"Unexpected migration version gap(s): {formatted}")
    return migrations


def verify_ledger(
    migrations: list[Migration], db: SupabaseRestClient
) -> dict[str, list[str]]:
    """Compare the deployed ledger with exact repository file hashes."""
    applied = {
        str(row["version"]): str(row["checksum"])
        for row in db.select_all("schema_migrations", columns="version,checksum", order="version.asc")
    }
    expected = {migration.version: migration.checksum for migration in migrations}
    return {
        "missing": sorted(version for version in expected if version not in applied),
        "mismatched": sorted(
            version
            for version, checksum in expected.items()
            if version in applied and applied[version] != checksum
        ),
        "unknown": sorted(version for version in applied if version not in expected),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--verify-production", action="store_true")
    parser.add_argument("--manifest", action="store_true")
    args = parser.parse_args()

    migrations = discover_migrations()
    if args.manifest:
        print(json.dumps([migration.__dict__ | {"path": str(migration.path)} for migration in migrations], ensure_ascii=False, indent=2))
    if args.verify_production:
        settings = get_supabase_admin_settings()
        report = verify_ledger(
            migrations,
            SupabaseRestClient(settings.supabase_url, settings.supabase_service_role_key),
        )
        print(json.dumps(report, ensure_ascii=False))
        if any(report.values()):
            raise RuntimeError("Production migration ledger does not match the repository")
    elif not args.manifest:
        print(f"Validated {len(migrations)} SQL migrations")


if __name__ == "__main__":
    main()
