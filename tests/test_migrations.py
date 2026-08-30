from __future__ import annotations

import hashlib

import pytest

from db.migrations import Migration, discover_migrations, verify_ledger


def test_discover_migrations_returns_ordered_transactional_files() -> None:
    migrations = discover_migrations()

    assert [migration.version for migration in migrations] == sorted(
        migration.version for migration in migrations
    )
    assert all(len(migration.checksum) == 64 for migration in migrations)


def test_discover_migrations_rejects_unwrapped_sql(tmp_path) -> None:
    (tmp_path / "003_bad.sql").write_text("SELECT 1;", encoding="utf-8")

    with pytest.raises(ValueError, match="transaction-wrapped"):
        discover_migrations(tmp_path)


class _LedgerDb:
    def __init__(self, rows: list[dict[str, str]]) -> None:
        self.rows = rows

    def select_all(self, _table: str, **_kwargs: object) -> list[dict[str, str]]:
        return self.rows


def test_verify_ledger_reports_missing_and_changed_migrations() -> None:
    migrations = [
        Migration("001_first", __file__, hashlib.sha256(b"first").hexdigest()),
        Migration("002_second", __file__, hashlib.sha256(b"second").hexdigest()),
    ]

    report = verify_ledger(
        migrations,
        _LedgerDb([
            {"version": "001_first", "checksum": "incorrect"},
            {"version": "999_unknown", "checksum": "irrelevant"},
        ]),  # type: ignore[arg-type]
    )

    assert report == {
        "missing": ["002_second"],
        "mismatched": ["001_first"],
        "unknown": ["999_unknown"],
    }
