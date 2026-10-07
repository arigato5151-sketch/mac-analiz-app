"""Canonical Turkish market and selection names used by Nesine."""

from __future__ import annotations

import re


_RESULT_SELECTIONS = {
    "home_win": "1",
    "draw": "X",
    "away_win": "2",
}


def nesine_market_name(key: str) -> str:
    """Return Nesine's market heading for an internal prediction key."""
    if key in _RESULT_SELECTIONS:
        return "Maç Sonucu"
    if key in {"double_chance_1x", "double_chance_x2", "double_chance_12"}:
        return "Çifte Şans"
    if key.startswith("first_half_") and key.removeprefix("first_half_") in _RESULT_SELECTIONS:
        return "1. Yarı Sonucu"
    if key.startswith("second_half_") and key.removeprefix("second_half_") in _RESULT_SELECTIONS:
        return "2. Yarı Sonucu"
    if key in {"btts_yes", "btts_no"}:
        return "Karşılıklı Gol"
    if key.startswith(("home_win_btts_", "draw_btts_", "away_win_btts_")):
        return "Maç Sonucu ve Karşılıklı Gol"

    combination = re.fullmatch(r"(home_win|draw|away_win)_(over|under)_(\d)_(\d)", key)
    if combination:
        _, _, major, minor = combination.groups()
        return f"Maç Sonucu ve {major},{minor} Alt/Üst"

    half_team_total = re.fullmatch(
        r"(home|away)_(first_half|second_half)_(over|under)_(\d)_(\d)", key
    )
    if half_team_total:
        team, period, _, major, minor = half_team_total.groups()
        team_name = "Ev Sahibi" if team == "home" else "Deplasman"
        half_name = "1.Y" if period == "first_half" else "2.Y"
        return f"{team_name} {half_name} {major},{minor} Gol Alt/Üst"

    team_total = re.fullmatch(r"(home|away)_(over|under)_(\d)_(\d)", key)
    if team_total:
        team, _, major, minor = team_total.groups()
        team_name = "Ev Sahibi" if team == "home" else "Deplasman"
        return f"{team_name} {major},{minor} Gol Alt/Üst"

    corners = re.fullmatch(r"corners_(over|under)_(\d)_(\d)", key)
    if corners:
        _, major, minor = corners.groups()
        return f"{major},{minor} Korner Alt/Üst"

    half_total = re.fullmatch(r"(first_half|second_half)_(over|under)_(\d)_(\d)", key)
    if half_total:
        period, _, major, minor = half_total.groups()
        prefix = "1. Yarı" if period == "first_half" else "2. Yarı"
        return f"{prefix} {major},{minor} Gol Alt/Üst"

    total = re.fullmatch(r"(over|under)_(\d)_(\d)", key)
    if total:
        _, major, minor = total.groups()
        return f"{major},{minor} Gol Alt/Üst"
    return "Maç Sonucu"


def nesine_selection_name(key: str) -> str:
    """Return the selection label convention used under Nesine market names."""
    if key in _RESULT_SELECTIONS:
        return _RESULT_SELECTIONS[key]
    if key.startswith("first_half_") and key.removeprefix("first_half_") in _RESULT_SELECTIONS:
        return f"1.Y {_RESULT_SELECTIONS[key.removeprefix('first_half_')]}"
    if key.startswith("second_half_") and key.removeprefix("second_half_") in _RESULT_SELECTIONS:
        return f"2.Y {_RESULT_SELECTIONS[key.removeprefix('second_half_')]}"
    double_chance = {
        "double_chance_1x": "ÇŞ 1-X",
        "double_chance_12": "ÇŞ 1-2",
        "double_chance_x2": "ÇŞ X-2",
    }
    if key in double_chance:
        return double_chance[key]
    result_combo = re.fullmatch(r"(home_win|draw|away_win)_(over|under)_(\d)_(\d)", key)
    if result_combo:
        result, direction, _, _ = result_combo.groups()
        return f"MS{_RESULT_SELECTIONS[result]} & {'Üst' if direction == 'over' else 'Alt'}"
    result_btts = re.fullmatch(r"(home_win|draw|away_win)_btts_(yes|no)", key)
    if result_btts:
        result, outcome = result_btts.groups()
        return f"MS{_RESULT_SELECTIONS[result]} & {'Var' if outcome == 'yes' else 'Yok'}"
    direction = "Üst" if "_over_" in f"_{key}_" else "Alt"
    line = re.search(r"(\d)_(\d)$", key)
    if line:
        major, minor = line.groups()
        if key.startswith("corners_"):
            return f"{direction} {major},{minor}"
        return direction
    if key in {"btts_yes", "btts_no"}:
        return "Var" if key == "btts_yes" else "Yok"
    return key
