"""Verified Nesine team labels mapped to API-Football team IDs, scoped by league.

Only add aliases after checking the team's API-Football ID in a fixture from
the same competition. Unknown labels deliberately remain unmapped.
"""

from config.leagues import TRACKED_LEAGUE_IDS

NESINE_TEAM_IDS_BY_LEAGUE: dict[int, dict[str, int]] = {
    # Verified against the app's API-Football fixtures on 2026-10-18.
    71: {
        "gremio p.": 130,
        "cruzeiro mg": 135,
    },
    135: {
        "fiorentina": 502,
        "como 1907": 895,
    },
    179: {
        "celtic": 247,
        "hearts": 254,
    },
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
    # Verified against the same-day API-Football fixture IDs and kickoffs
    # reported by the local Nesine run on 2026-10-17.
    39: {
        "man. city": 50,
        "ipswich town": 57,
        "leeds utd": 63,
        "man. utd": 33,
        "nottingham f.": 65,
        "arsenal": 42,
    },
    61: {
        "psg": 85,
        "lille": 79,
        "brest": 106,
    },
    62: {
        "red star": 104,
        "st etienne": 1063,
    },
    78: {
        "elversberg": 1660,
        "augsburg": 170,
        "werder bremen": 162,
        "paderborn": 185,
        "hamburg": 175,
        "stuttgart": 172,
        "b. dortmund": 165,
        "schalke 04": 174,
        "mainz": 164,
        "b. leverkusen": 168,
        "freiburg": 160,
        "m'gladbach": 163,
        "hoffenheim": 167,
    },
    88: {
        "s. rotterdam": 426,
        "zwolle": 193,
        "go ahead eagles": 410,
        "nec nijmegen": 413,
        "f. sittard": 205,
    },
    136: {
        "carrarese c.": 1581,
        "v. entella": 527,
        "lr vicenza v.": 1584,
        "mantova 1911": 1693,
        "us palermo": 522,
        "ac pisa": 801,
        "cremonese": 520,
        "ac cesena": 509,
        "as avellino 1912": 528,
        "sampdoria": 498,
        "ss arezzo": 876,
    },
    140: {
        "atl madrid": 530,
        "ath bilbao": 531,
        "malaga": 535,
        "r. sociedad": 548,
    },
    218: {
        "aus. lustenau": 1399,
        "lask": 1026,
        "sv ried": 1028,
        "austria wien": 601,
        "salzburg": 571,
        "altach": 618,
        "wolfsberger": 1025,
        "wsg tirol": 1398,
    },
    # Verified against the same-day API-Football fixtures and kickoff times
    # reported by the local Nesine run on 2026-10-15.
    3: {
        "salzburg": 571,
        "hb sheva": 563,
        "b. leverkusen": 168,
        "sparta prag": 628,
        "u. s. gilloise": 1393,
        "r. sociedad": 548,
        "nk celje": 4360,
        "omonia": 3402,
        "d. zagreb": 620,
        "levski sofya": 646,
        "ferencvaros": 651,
        "viktoria plzen": 567,
        "hoffenheim": 167,
        "ofi crete": 1124,
        "j. bialystok": 336,
        "ararat": 3683,
    },
    848: {
        "kuopion": 1165,
        "k. almaty": 664,
        "banja luka": 3364,
        "kizil yildiz": 598,
        "egnatia r.": 3327,
        "kf egnatia r.": 3327,
        "un. craiova": 632,
        "agf aarhus": 406,
        "hajduk split": 608,
        "mjallby": 2240,
        "inter escaldes": 3342,
        "iberia 1999": 3502,
        "saint truidense": 735,
        "lincoln red": 667,
        "zalgiris kaunas": 3872,
        "hearts": 254,
        "kopenhag": 400,
        "braga": 217,
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
