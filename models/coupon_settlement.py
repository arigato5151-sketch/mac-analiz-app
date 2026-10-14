"""Settle coupon selections only when their required result data is available."""

from __future__ import annotations

import re


def settle_coupon_selection(
    key: str,
    *,
    home_score: int,
    away_score: int,
    home_corners: int | None = None,
    away_corners: int | None = None,
) -> bool | None:
    """Return whether a full-time market won, or None when data is unavailable."""
    home, away = int(home_score), int(away_score)
    total = home + away
    result = "home_win" if home > away else "draw" if home == away else "away_win"

    result_keys = {"home_win", "draw", "away_win"}
    if key in result_keys:
        return key == result
    if key == "double_chance_1x":
        return result in {"home_win", "draw"}
    if key == "double_chance_x2":
        return result in {"draw", "away_win"}
    if key == "double_chance_12":
        return result != "draw"
    if key == "btts_yes":
        return home > 0 and away > 0
    if key == "btts_no":
        return home == 0 or away == 0

    combined = re.fullmatch(r"(home_win|draw|away_win)_(over|under)_(\d+)_(\d)", key)
    if combined:
        result_key, direction, major, minor = combined.groups()
        line = int(major) + int(minor) / 10
        result_hit = result == result_key
        total_hit = total > line if direction == "over" else total < line
        return result_hit and total_hit

    result_btts = re.fullmatch(r"(home_win|draw|away_win)_btts_(yes|no)", key)
    if result_btts:
        result_key, selection = result_btts.groups()
        btts = home > 0 and away > 0
        return result == result_key and btts == (selection == "yes")

    team_total = re.fullmatch(r"(home|away)_(over|under)_(\d+)_(\d)", key)
    if team_total:
        team, direction, major, minor = team_total.groups()
        goals = home if team == "home" else away
        line = int(major) + int(minor) / 10
        return goals > line if direction == "over" else goals < line

    total_market = re.fullmatch(r"(over|under)_(\d+)_(\d)", key)
    if total_market:
        direction, major, minor = total_market.groups()
        line = int(major) + int(minor) / 10
        return total > line if direction == "over" else total < line

    corners = re.fullmatch(r"corners_(over|under)_(\d+)_(\d)", key)
    if corners:
        if home_corners is None or away_corners is None:
            return None
        direction, major, minor = corners.groups()
        total_corners = int(home_corners) + int(away_corners)
        line = int(major) + int(minor) / 10
        return total_corners > line if direction == "over" else total_corners < line

    # Half-time, cards, and other markets cannot be evaluated from the
    # currently persisted full-time score and corner fields.
    return None
