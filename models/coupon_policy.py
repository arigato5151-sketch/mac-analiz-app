"""Deterministic, diversified coupon selection rules."""

from __future__ import annotations

from typing import Any

RESULT_KEYS = {"home_win", "draw", "away_win"}
TOTAL_KEYS = {"over_2_5", "under_2_5"}
BTTS_KEYS = {"btts_yes", "btts_no"}


def _market_family(key: str) -> str:
    if key in RESULT_KEYS:
        return "result"
    if key in TOTAL_KEYS:
        return "totals"
    if key in BTTS_KEYS:
        return "btts"
    return "other"


def diversified_coupon_rows(rows: list[dict[str, Any]], *, max_items: int, min_probability: float = 0.0, min_ev: float = 0.03, high_odds: bool = False) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for row in rows:
        for assessment in row.get("assessments", [row["best"]]):
            if assessment.model_probability < min_probability or assessment.expected_value < min_ev:
                continue
            if high_odds and assessment.odds < 2.5:
                continue
            candidates.append({**row, "best": assessment})
    # Rank by value first. Market diversity is a tie-breaker only; no market
    # family is mandatory and a coupon may legitimately contain one family.
    candidates.sort(key=lambda row: (-row["best"].expected_value, row["best"].key))
    selected: list[dict[str, Any]] = []
    used_keys: set[str] = set()
    used_matches: set[int] = set()
    # Select the strongest eligible outcomes. Only duplicate match/key entries
    # are blocked; no requirement exists to fill result/total/BTTS families.
    for row in candidates:
        key = str(row["best"].key)
        if int(row["match_id"]) in used_matches or key in used_keys:
            continue
        selected.append(row)
        used_keys.add(key)
        used_matches.add(int(row["match_id"]))
        if len(selected) >= max_items:
            break
    return selected
