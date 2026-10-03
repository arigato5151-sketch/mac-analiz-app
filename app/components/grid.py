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
        detailed_market_widths = {
            "Çifte şans": 64,
            "Üst 1.5": 45,
            "Alt 3.5": 45,
            "Ev 0.5 Üst": 45,
            "Dep. 0.5 Üst": 45,
            "Ev 1.5 Üst": 45,
            "Dep. 1.5 Üst": 45,
            "Skor 1": 58,
            "Skor 2": 58,
            "Skor 3": 58,
            "1": 46,
            "X": 46,
            "2": 46,
            "Üst 2.5": 54,
            "KG Var": 54,
        }
        is_detailed = any(
            isinstance(column, dict) and column.get("field") == "Çifte şans"
            for column in columns
        )
        widths = {
            "Tarih": 80 if is_detailed else 128,
            "Lig": 95 if is_detailed else 155,
            "Maç": 130 if is_detailed else 195,
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
            if is_detailed and field in detailed_market_widths:
                column["width"] = detailed_market_widths[field]
                column["flex"] = 0
                column["filter"] = False
            elif field in widths:
                column["width"] = widths[field]
                column["flex"] = 0
                if is_detailed and field in {"1", "X", "2", "Üst 2.5", "KG Var"}:
                    column["filter"] = False
            elif field == "Öne çıkan":
                column["flex"] = 1
                column["minWidth"] = 175
            elif field == "En güçlü sinyal" and is_detailed:
                # The signal is already present in quick view; reserve width for market values.
                column["hide"] = True
                column["filter"] = False
            if is_detailed:
                column["headerStyle"] = {
                    "fontSize": "11px",
                    "paddingLeft": "2px",
                    "paddingRight": "2px",
                }

        if is_detailed:
            default_column = options.setdefault("defaultColDef", {})
            if isinstance(default_column, dict):
                default_column["cellStyle"] = {
                    "fontSize": "10px",
                    "paddingLeft": "2px",
                    "paddingRight": "2px",
                }
            for column in columns:
                if isinstance(column, dict) and column.get("field") == "Tarih":
                    column["width"] = 80
                elif isinstance(column, dict) and column.get("field") == "Lig":
                    column["width"] = 95
                elif isinstance(column, dict) and column.get("field") == "Maç":
                    column["width"] = 150
                    column["flex"] = 1
                    column["minWidth"] = 130
                    column["filter"] = False

    default_column = options.setdefault("defaultColDef", {})
    if isinstance(default_column, dict):
        default_column.update(
            wrapText=False,
            autoHeight=False,
        )
    return options


# Keep the previous helper name available while hosted instances refresh files.
fit_read_only_grid_columns = fit_match_dashboard_columns
