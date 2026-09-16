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
