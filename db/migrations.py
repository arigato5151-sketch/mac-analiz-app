"""Validate immutable SQL migrations and verify their Supabase ledger entries."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from config.settings import get_settings
from db.db_client import SupabaseRestClient


MIGRATIONS_DIR = Path(__file__).with_name("migrations")
MIGRATION_NAME = re.compile(r"^(?P<version>\d{3}_[a-z0-9_]+)\.sql$")


@dataclass(frozen=True)
class Migration:
    version: str
    path: Path
    checksum: str


def discover_migrations(directory: Path = MIGRATIONS_DIR) -> list[Migration]:
    """Read numbered migrations in deterministic order and reject stray SQL files."""
    migrations: list[Migration] = []
    for path in sorted(directory.glob("*.sql")):
        match = MIGRATION_NAME.fullmatch(path.name)
        if match is None:
            raise ValueError(f"Invalid migration filename: {path.name}")
        text = path.read_text(encoding="utf-8")
        normalized = text.strip().upper()
        if not normalized.startswith(("--", "BEGIN;")) or "BEGIN;" not in normalized or not normalized.endswith("COMMIT;"):
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
    return migrations


def verify_ledger(
    migrations: list[Migration], db: SupabaseRestClient
) -> dict[str, list[str]]:
    """Compare the deployed ledger with exact repository file hashes."""
    applied = {
        str(row["version"]): str(row["checksum"])
        for row in db.select_all("schema_migrations", columns="version,checksum")
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
        settings = get_settings()
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
