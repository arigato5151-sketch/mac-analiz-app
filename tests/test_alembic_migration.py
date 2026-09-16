from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path

from db.models_db import Base


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MIGRATION_PATH = PROJECT_ROOT / "alembic" / "versions" / "001_initial_schema.py"


def test_initial_migration_creates_each_table_once() -> None:
    tree = ast.parse(MIGRATION_PATH.read_text(encoding="utf-8"))
    table_names = [
        call.args[0].value
        for call in ast.walk(tree)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "create_table"
        and call.args
        and isinstance(call.args[0], ast.Constant)
        and isinstance(call.args[0].value, str)
    ]

    duplicates = {
        table_name: count
        for table_name, count in Counter(table_names).items()
        if count > 1
    }

    assert table_names
    assert duplicates == {}
    assert set(table_names) == set(Base.metadata.tables)
