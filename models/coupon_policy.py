"""Deterministic, diversified coupon selection rules."""

from __future__ import annotations

from typing import Any

RESULT_KEYS = {"home_win", "draw", "away_win"}
TOTAL_KEYS = {"over_2_5", "under_2_5"}
BTTS_KEYS = {"btts_yes", "btts_no"}
# Live evaluation shows 1-X-2 is materially weaker than binary markets.
# Keep result picks out of coupons unless the model has stronger conviction.
MIN_COUPON_PROBABILITY = 0.55
MAX_PUBLISHABLE_ODDS = 20.0
MARKET_MIN_PROBABILITY = {
    "home_win": 0.58,
    "draw": 0.58,
    "away_win": 0.58,
    "over_2_5": 0.55,
    "under_2_5": 0.55,
    "btts_yes": 0.55,
    "btts_no": 0.55,
}


def _market_family(key: str) -> str:
    if key in RESULT_KEYS:
        return "result"
    if key in TOTAL_KEYS:
        return "totals"
    if key in BTTS_KEYS:
        return "btts"
    return "other"


def diversified_coupon_rows(
    rows: list[dict[str, Any]],
    *,
    max_items: int | None = None,
    min_probability: float = 0.0,
    min_ev: float = 0.03,
    high_odds: bool = False,
    excluded_match_families: set[tuple[int, str]] | None = None,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        for assessment in row.get("assessments", [row["best"]]):
            try:
                probability = float(assessment.model_probability)
                expected_value = float(assessment.expected_value)
                odds = float(assessment.odds)
            except Exception:
                continue
            market_floor = MARKET_MIN_PROBABILITY.get(str(assessment.key), MIN_COUPON_PROBABILITY)
            if (
                probability < max(min_probability, market_floor)
                or odds > MAX_PUBLISHABLE_ODDS
                or expected_value < min_ev
            ):
                continue
            if high_odds and odds < 2.5:
                continue
            candidates.append({**row, "best": assessment})
    # Rank by value first. Market diversity is a tie-breaker only; no market
    # family is mandatory and a coupon may legitimately contain one family.
    candidates.sort(key=lambda row: (-float(row["best"].expected_value), row["best"].key))
    selected: list[dict[str, Any]] = []
    used_keys: set[str] = set()
    used_match_families: set[tuple[int, str]] = set(excluded_match_families or ())
    # A match may appear with different market families (e.g. MS + KG), but
    # the same match/family cannot be repeated in this or another coupon.
    for row in candidates:
        key = str(row["best"].key)
        match_family = (int(row["match_id"]), _market_family(key))
        if match_family in used_match_families or key in used_keys:
            continue
        selected.append(row)
        used_keys.add(key)
        used_match_families.add(match_family)
        if max_items is not None and len(selected) >= max_items:
            break
    return selected
