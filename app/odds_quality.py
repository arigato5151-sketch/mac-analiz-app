"""Fail-safe checks for internally inconsistent IdDaa total-goals prices."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import defaultdict
from typing import Any


def _key_text(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or "").casefold())
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", "", text)


def total_market_key(market: dict[str, Any]) -> tuple[str, str, str]:
    try:
        line = str(float(str(market.get("line", "")).replace(",", ".")))
    except (TypeError, ValueError):
        line = str(market.get("line", ""))
    return str(market.get("sport_type", "")), str(market.get("market_id", "")), line


def inconsistent_total_market_keys(
    markets: list[dict[str, Any]],
) -> set[tuple[str, str, str]]:
    """Return (sport_type, market_id, line) keys violating total-price ordering.

    For the same total-goals market, Over prices should not fall as the line
    rises, and Under prices should not rise. Flag both sides of each violation.
    """
    grouped: dict[
        tuple[str, str], dict[float, list[tuple[dict[str, Any], dict[str, float]]]]
    ] = defaultdict(lambda: defaultdict(list))
    for market in markets:
        try:
            line = float(str(market.get("line", "")).replace(",", "."))
            sport_type, market_id, _ = total_market_key(market)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(line) or not market_id:
            continue
        market_name = _key_text(market.get("market_name"))
        market_family = (
            market_name
            if any(term in market_name for term in ("toplamgol", "altust", "overunder"))
            else market_id
        )
        prices: dict[str, float] = {}
        for selection in market.get("selections", []):
            label = _key_text(selection.get("name") or selection.get("id"))
            side = "over" if label in {"ust", "over"} else "under" if label in {"alt", "under"} else None
            try:
                price = float(selection.get("display_odds") or selection.get("odds"))
            except (TypeError, ValueError):
                continue
            if side and math.isfinite(price) and price > 1:
                prices[side] = price
        if prices:
            grouped[(sport_type, market_family)][line].append((market, prices))

    inconsistent: set[tuple[str, str, str]] = set()
    for lines in grouped.values():
        ordered = sorted(lines.items())
        for (lower_line, lower_markets), (upper_line, upper_markets) in zip(ordered, ordered[1:]):
            if upper_line <= lower_line:
                continue
            for lower_market, lower in lower_markets:
                for upper_market, upper in upper_markets:
                    over_bad = "over" in lower and "over" in upper and upper["over"] + 1e-9 < lower["over"]
                    under_bad = "under" in lower and "under" in upper and upper["under"] > lower["under"] + 1e-9
                    if over_bad or under_bad:
                        inconsistent.add(total_market_key(lower_market))
                        inconsistent.add(total_market_key(upper_market))
    return inconsistent
