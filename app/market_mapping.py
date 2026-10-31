"""Helpers for mapping individual bookmaker selections to model outcomes."""

from __future__ import annotations

import unicodedata
from typing import Any


def is_compound_selection(label: str) -> bool:
    """Prevent combo selections from being treated as a single market outcome."""
    normalized = unicodedata.normalize("NFKD", label.casefold())
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    compact = "".join(char for char in normalized if char.isalnum())
    if compact in {"1ve0", "0ve1", "0ve2", "2ve0", "1ve2", "2ve1"}:
        return False
    return " ve " in f" {normalized.strip()} " or " and " in f" {normalized.strip()} "


def total_goals_outcome(label: str) -> str | None:
    """Map only a standalone Over/Under selection, never a combo containing it."""
    normalized = unicodedata.normalize("NFKD", label.casefold())
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    normalized = " ".join(normalized.split())
    return {"ust": "over", "over": "over", "alt": "under", "under": "under"}.get(normalized)


def standard_market_price(selection: dict[str, Any]) -> float:
    """Use IdDaa's standard odd; wodd is a separate promotional price field."""
    value = selection.get("odds")
    if value is None:
        value = selection.get("display_odds")
    return float(value)
