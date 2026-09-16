from __future__ import annotations

import pandas as pd

from app.components.grid import build_read_only_grid_options


def test_read_only_grid_uses_only_community_supported_options() -> None:
    options = build_read_only_grid_options(
        pd.DataFrame([{"match_id": 1, "team": "Example"}])
    )

    assert options["defaultColDef"]["filter"] is True
    assert "filterable" not in options["defaultColDef"]
    assert "sideBar" not in options
    assert "rowSelection" not in options
