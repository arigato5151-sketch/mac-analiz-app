from pathlib import Path

import pytest

from db.migrations import discover_migrations


def test_repository_migrations_allow_only_the_documented_030_gap():
    migrations = discover_migrations()
    numbers = {int(item.version[:3]) for item in migrations}
    assert 30 not in numbers
    assert numbers == set(range(33)) - {30}


def test_unexpected_migration_gap_is_rejected(tmp_path: Path):
    for number in (0, 1, 3):
        (tmp_path / f"{number:03d}_migration.sql").write_text(
            "-- test\nBEGIN;\nSELECT 1;\nCOMMIT;\n", encoding="utf-8"
        )

    with pytest.raises(ValueError, match="Unexpected migration version gap"):
        discover_migrations(tmp_path)
