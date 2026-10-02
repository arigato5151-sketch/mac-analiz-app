"""Shared configuration for read-only AgGrid tables."""

from __future__ import annotations

import pandas as pd
from st_aggrid import GridOptionsBuilder


def build_read_only_grid_options(frame: pd.DataFrame) -> dict[str, object]:
    """Build community-edition options without unused selection state."""
    builder = GridOptionsBuilder.from_dataframe(frame)
    builder.configure_pagination(paginationAutoPageSize=True)
    builder.configure_default_column(
        filter=True,
        sortable=True,
        resizable=True,
        wrapHeaderText=True,
        autoHeaderHeight=True,
    )
    return builder.build()


def fit_match_dashboard_columns(options: dict[str, object]) -> dict[str, object]:
    """Fit match dashboard columns in one line, giving the signal column spare width."""
    columns = options.get("columnDefs", [])
    if isinstance(columns, list):
        widths = {
            "Tarih": 128,
            "Lig": 155,
            "Maç": 195,
            "1": 70,
            "X": 70,
            "2": 70,
            "Üst 2.5": 90,
            "KG Var": 90,
        }
        for column in columns:
            if not isinstance(column, dict):
                continue
            field = column.get("field")
            if field in widths:
                column["width"] = widths[field]
                column["flex"] = 0
            elif field == "Öne çıkan":
                column["flex"] = 1
                column["minWidth"] = 175

    default_column = options.setdefault("defaultColDef", {})
    if isinstance(default_column, dict):
        default_column.update(
            wrapText=False,
            autoHeight=False,
        )
    return options
