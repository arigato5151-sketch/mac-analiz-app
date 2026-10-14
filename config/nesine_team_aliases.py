"""Verified Nesine team labels mapped to API-Football team IDs, scoped by league.

Only add aliases after checking the team's API-Football ID in a fixture from
the same competition. Unknown labels deliberately remain unmapped.
"""

from config.leagues import TRACKED_LEAGUE_IDS

NESINE_TEAM_IDS_BY_LEAGUE: dict[int, dict[str, int]] = {
    # Verified against the app's API-Football fixtures on 2026-10-14.
    2: {
        "feyenoord": 209,
        "como 1907": 895,
        "como": 895,
        "lask": 1026,
        "lask linz": 1026,
        "liverpool": 40,
        "shakhtar d.": 550,
        "shakhtar donetsk": 550,
        "aek athens": 575,
        "aek athens fc": 575,
        "s. bratislava": 656,
        "slovan bratislava": 656,
        "stuttgart": 172,
        "vfb stuttgart": 172,
        "bodo glimt": 327,
        "bodo/glimt": 327,
        "b. dortmund": 165,
        "borussia dortmund": 165,
        "man. city": 50,
        "manchester city": 50,
        "psg": 85,
        "paris saint germain": 85,
        "as roma": 497,
        "real madrid": 541,
        "aston villa": 66,
        "fenerbahce": 611,
        "real betis": 543,
        "fc porto": 212,
    },
    40: {
        "norwich city": 71,
        "norwich": 71,
        "southampton": 41,
        "qpr": 72,
        "derby county": 69,
        "derby": 69,
        "lincoln city": 1379,
        "lincoln": 1379,
        "west ham": 48,
        "portsmouth": 1355,
        "sheffield utd": 62,
        "sheffield united": 62,
    },
}

if set(NESINE_TEAM_IDS_BY_LEAGUE) - set(TRACKED_LEAGUE_IDS):
    raise ValueError("Nesine team aliases may only reference app-tracked leagues")
if any(
    team_id <= 0
    for aliases in NESINE_TEAM_IDS_BY_LEAGUE.values()
    for team_id in aliases.values()
):
    raise ValueError("Nesine team aliases must use positive API-Football team IDs")
