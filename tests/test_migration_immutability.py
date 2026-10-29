from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from db.migrations import discover_migrations, verify_ledger


ROOT = Path(__file__).resolve().parents[1]
CHECKSUM_MANIFEST = ROOT / "tests" / "fixtures" / "migration_checksums.json"


def test_migration_files_match_pinned_normalized_checksums() -> None:
    expected = json.loads(CHECKSUM_MANIFEST.read_text(encoding="utf-8"))
    actual = {migration.version: migration.checksum for migration in discover_migrations()}

    assert actual == expected


def test_controller_checksum_normalizes_crlf_and_lf(tmp_path: Path) -> None:
    lf_dir = tmp_path / "lf"
    crlf_dir = tmp_path / "crlf"
    lf_dir.mkdir()
    crlf_dir.mkdir()
    sql = "BEGIN;\nSELECT 1;\nCOMMIT;\n"
    path = "001_checksum_test.sql"
    (lf_dir / path).write_bytes(sql.encode("utf-8"))
    (crlf_dir / path).write_bytes(sql.replace("\n", "\r\n").encode("utf-8"))

    lf_checksum = discover_migrations(lf_dir)[0].checksum
    crlf_checksum = discover_migrations(crlf_dir)[0].checksum

    assert lf_checksum == crlf_checksum


def test_verify_production_reports_changed_applied_migration_as_mismatch() -> None:
    migrations = discover_migrations()
    baseline = {migration.version: migration.checksum for migration in migrations}

    class ProductionLedger:
        def select_all(self, *_: Any, **__: Any) -> list[dict[str, str]]:
            return [
                {"version": version, "checksum": checksum}
                for version, checksum in baseline.items()
            ]

    target = next(
        migration
        for migration in migrations
        if migration.version == "041_revoke_dangerous_public_table_acl"
    )
    modified = [
        replace(target, checksum="0" * 64)
        if migration.version == target.version
        else migration
        for migration in migrations
    ]

    assert verify_ledger(modified, ProductionLedger()) == {
        "missing": [],
        "mismatched": [target.version],
        "unknown": [],
    }
