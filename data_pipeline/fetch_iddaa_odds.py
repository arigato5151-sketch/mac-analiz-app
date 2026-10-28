"""Import today's IdDaa.com football bulletin and all available markets."""

from __future__ import annotations

import argparse
import logging
import os
import re
import time
import unicodedata
from difflib import SequenceMatcher
from datetime import date, datetime, time as datetime_time, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

import requests

from config.leagues import TRACKED_LEAGUES
from config.settings import get_settings
from db.db_client import SupabaseRestClient

BASE_URL = "https://sportsbookv2.iddaa.com/sportsbook"
ISTANBUL = ZoneInfo("Europe/Istanbul")
LOG = logging.getLogger("iddaa_odds")

# IdDaa exposes the corresponding API-Football ID in `cref` for most leagues.
IDDAA_COMPETITION_REFS: dict[int, int] = {
    39: 1,
    140: 36,
    135: 33,
    136: 34,  # Italian Serie B
    # IdDaa catalogue currently identifies Germany Bundesliga as cref 42.
    78: 42,
    61: 4,
    40: 2,
    94: 52,
    71: 83,
    203: 62,
    88: 39,
    253: 18,
    144: 38,
    307: 3708,
    235: 53,
    128: 170677,
    # Lower divisions and Mexico are present in TRACKED_LEAGUES and in the
    # daily match screen. Include their IdDaa competitions so eligible events
    # are not silently skipped by the odds importer.
    262: 28,  # Liga MX: IdDaa Meksika Primera Ligi
    141: 37,  # Segunda División: IdDaa İspanya La Liga 2
    62: 19,  # Ligue 2: IdDaa Fransa Ligue 2
    179: 54,  # Scottish Premiership
    218: 29,  # Austrian Bundesliga
    197: 127,  # Greek Super League
}
# IdDaa's catalogue occasionally stops exposing API-Football refs in `cref`
# and instead publishes its own competition ID. Keep these explicit mappings
# tied to the current football catalogue; never infer a league from team names.
IDDAA_COMPETITION_IDS: dict[int, int] = {
    39: 43,  # English Premier League
    141: 123,  # Spanish Segunda División
    144: 148,  # Belgian Pro League
    94: 566,  # Portuguese Primeira Liga
    128: 49038,  # Argentine Primera LPF (current season)
    62: 614,  # French Ligue 2
    78: 45,  # German Bundesliga
    262: 37,  # Mexican Liga MX (Apertura)
    235: 62,  # Russian Premier League
}
REQUIRED_IDDAA_LEAGUES = set(IDDAA_COMPETITION_REFS) | {3, 848}


def normalize(value: str) -> str:
    # NFKD does not decompose several Latin letters used in European club
    # names (ø, æ, ł), so translate them before removing combining marks.
    value = value.casefold().translate(str.maketrans({
        "æ": "ae", "ø": "o", "ł": "l", "ß": "ss", "đ": "d", "ı": "i",
        "ð": "d", "þ": "th",
    }))
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


TEAM_SUFFIXES = {
    "fc", "cf", "sc", "ac", "afc", "club", "football", "futebol",
    "fbpa", "esporte", "clube", "de", "do", "da", "the", "rb", "sv",
    "sp", "mg", "ba", "rs", "sa", "k", "fk",
}


def normalize_team(value: str) -> str:
    """Remove diacritics and common club suffixes while keeping core name."""
    tokens = normalize(value).split()
    tokens = [token for token in tokens if token not in TEAM_SUFFIXES]
    return " ".join(tokens)


TEAM_ALIASES: dict[str, tuple[str, ...]] = {
    "independiente rivadavia": (
        "independ. rivadavia",
        "ind. mendoza",
        "independiente mendoza",
        "independ. mendoza",
    ),
    "queens park rangers": ("qpr",),
    "swansea city": ("swansea",),
    "sheffield united": ("sheffield utd",),
    "derby county": ("derby",),
    "lincoln": ("lincoln city",),
    "cardiff": ("cardiff city",),
    "birmingham": ("birmingham city",),
    "rb leipzig": ("leipzig",),
    "pec zwolle": ("zwolle",),
    "fsv mainz 05": ("mainz",),
    "real sociedad ii": ("real sociedad b",),
    "racing santander": ("santander",),
    "vitoria": ("vitoria ba",),
    "atletico paranaense": ("paranaense",),
    "botafogo": ("botafogo rj",),
    "rubin": ("rubin kazan",),
    "lanus": ("ath lanus", "atletico lanus"),
    "1 fc koln": ("koln", "koln fc"),
    "saint etienne": ("st etienne",),
    "rayo vallecano": ("vallecano",),
    "baltika": ("fc baltika kaliningrad",),
    "west brom": ("west bromwich", "west bromwich albion"),
    "amed": ("amed sportif faaliyetler",),
    "hamburger sv": ("hamburg",),
    "sc paderborn 07": ("paderborn",),
    "cf montreal": ("montreal impact", "montreal"),
    "norwich": ("norwich city",),
    "blackburn": ("blackburn rovers",),
    "celtic": ("glasgow celtic",),
    "dinamo moskva": ("dinamo moskova", "dinamo moscow", "dynamo", "dynamo moscow"),
    "akron": ("akron tolyatti",),
    "oviedo": ("real oviedo",),
    "celta de vigo ii": ("celta vigo b", "celta b"),
    "levadiakos": ("levadeiakos", "apo levadeiakos"),
    "tsv hartberg": ("hartberg",),
    "wolfsberger ac": ("wolfsberg", "wolfsberger"),
    "iraklis 1908": ("iraklis", "pot iraklis"),
    "kifisia": ("ae kifisia", "ae kifisia fc"),
    "rapid vienna": ("rapid wien",),
    "austria lustenau": ("aus lustenau",),
    "wsg wattens": ("wsg tirol", "tirol"),
    "kvc westerlo": ("westerlo",),
    "lommel united": ("lommel sk",),
    "mantova": ("mantova 1911",),
    "avellino": ("as avellino 1912",),
    "arezzo": ("us arezzo", "ss arezzo"),
    "catanzaro": ("us catanzaro 1929",),
    "defensa y justicia": ("df justicia", "defensa justicia"),
    "talleres cordoba": ("talleres",),
    "rodez": ("rodez af", "rodez aveyron football"),
    "gremio": ("gremio fbpa", "gremio foot ball porto alegrense"),
    "fluminense": ("fluminense fc",),
    "santos": ("santos fc",),
    "remo": ("clube do remo", "remo pa"),
    "bragantino": ("rb bragantino", "bragantino sp", "red bull bragantino"),
    "bahia": ("bahia ba", "ec bahia"),
    "al ahli jeddah": ("ahli saudi", "al ahli saudi", "al ahli", "al-ahli jeddah"),
    "al fayha": ("al fayha fc", "al-fayha"),
    "bayern munich": ("bayern munich k", "bayern munich women", "bayern munich (k)", "bayern munchen", "bayern munih"),
    "vfb stuttgart": ("vfb stuttgart k", "vfb stuttgart (k)", "stuttgart"),
    "estudiantes plata": ("estudiantes de la plata", "estudiantes l p", "estudiantes lp"),
    "rodina moskva": ("rodina moscow",),
    "al diriyah": ("al draih", "al diraiyah", "al diriyah fc"),
    "wolves": ("wolverhampton", "wolverhampton wanderers", "wolverhampton wanderers fc"),
    "porto": ("fc porto", "porto fc"),
    "psv eindhoven": ("psv", "psv eindhoven fc"),
    "slavia prague": ("slavia praha", "slavia prag"),
    "borussia dortmund": ("b dortmund", "b dortmund u19", "borussia dortmund u19"),
    "sabah fa": ("sabah",),
    "lens": ("rc lens",),
    "brugge": ("club brugge", "club brugge kv", "brugge kv"),
    "sporting lisbon": ("sporting cp", "sporting lizbon"),
    "lask": ("lask linz",),
    "como": ("como 1907",),
    "athletic club": ("athletic bilbao", "athletic club bilbao"),
    # API-Football and IdDaa spell these Mexican clubs differently.
    "pumas unam": ("unam pumas", "u n a m pumas", "universidad nacional"),
    "toluca": ("deportivo toluca", "deportivo toluca fc"),
    "kairat almaty": ("k almaty",),
    "kuopion": ("kups", "kups kuopio"),
    "kauno zalgiris": ("zalgiris kaunas", "zalgiris kauno"),
    "crvena zvezda": ("kizil yildiz", "red star belgrade", "red star"),
    "jagiellonia": ("jagiellonia bialystok",),
    "ferencvarosi": ("ferencvaros", "ferencvarosi tc"),
    "viktoria plzen": ("plzen",),
    "omonia nicosia": ("omonia",),
    "olympiacos": ("olympiakos", "olympiakos piraeus"),
    "sparta prague": ("sparta prag", "sparta praha"),
    "ararat armenia": ("ararat", "ararat-armenia"),
    "marseille": ("marsilya",),
    "celje": ("nk celje",),
    "aarhus": ("agf aarhus",),
    "gent": ("kaa gent",),
    "union saint gilloise": ("union st gilloise", "union saint-gilloise"),
    "ofi crete": ("ofi",),
    "bayer leverkusen": ("b leverkusen",),
    "universitatea craiova": ("un craiova",),
    "mjallby": ("mjallby aif",),
    "jablonec": ("fk jablonec",),
    "heart of midlothian": ("hearts",),
    "st truiden": ("st truidense", "sint truiden", "sint truidense"),
    "hoffenheim": ("1899 hoffenheim",),
    "copenhagen": ("kobenhavn", "kopenhag", "fc copenhagen"),
    "cska sofia": ("cska sofya",),
    "levski sofia": ("levski sofya",),
    "salzburg": ("red bull salzburg",),
    "newells old boys": ("new old boys", "newell's old boys", "newells"),
    "newcastle": ("newcastle united",),
    "st louis city": ("saint louis city", "st louis city sc", "saint louis city sc"),
    "sporting gijon": ("s gijon", "sporting de gijon", "sporting gijon",),
    "sabadell": ("ce sabadell fc",),
    "nacional": ("n madeira", "nacional madeira", "nacional da madeira", "cd nacional"),
    "estac troyes": ("troyes", "est troyes", "es troyes"),
    "genk": ("krc genk",),
    "rostov": ("fc rostov", "fk rostov"),
    "lokomotiv": ("lokomotiv moskova", "lokomotiv moscow"),
    "aldosivi": ("ca aldosivi",),
    "belgrano cordoba": ("belgrano", "ca belgrano"),
    "rosario central": ("rosario c",),
    "huracan": ("ca huracan", "club atletico huracan"),
    "pumas unam": ("u.n.a.m. pumas", "pumas unam", "unam pumas"),
    "tigres uanl": ("tigres", "tigres u.a.n.l."),
    "olympiacos": ("olympiakos", "olympiakos piraeus", "olympiacos piraeus"),
    "kalamata": ("pae ps kalamata",),
    "aris thessalonikis": ("aris", "aris fc",),
    "auxerre": ("aj auxerre",),
    "toulouse": ("toulouse fc",),
    "palermo": ("us palermo",),
    "carrarese": ("carrarese calcio", "carrarese calcio 1908"),
    "leeds": ("leeds united",),
    "sporting cp": ("sporting lisbon", "sporting lizbon"),
    "academico viseu": ("a de viseu", "academico de viseu"),
    "braga": ("sc braga", "sporting de braga"),
    "estoril": ("estoril praia",),
    "standard liege": ("standard liege", "standard de liege",),
    "volos nfc": ("volos", "volos fc"),
    "cska moscow": ("cska moskova", "cska moscou"),
    "spartak moscow": ("spartak moskova", "spartak moscou"),
    "eldense": ("club deportivo eldense", "cd eldense"),
    "la galaxy": ("los angeles galaxy",),
    "barracas central": ("barracas",),
    "estudiantes de rio cuarto": ("estudiantes rio cuarto",),
    "central cordoba de santiago": ("c cordoba", "central cordoba"),
    "boulogne": ("us boulogne", "boulogne sur mer"),
}


def _team_alias_groups() -> tuple[frozenset[str], ...]:
    return tuple(
        frozenset({normalize_team(canonical), *(normalize_team(alias) for alias in aliases)})
        for canonical, aliases in TEAM_ALIASES.items()
    )


TEAM_ALIAS_GROUPS = _team_alias_groups()


def team_names_match(left: str, right: str) -> bool:
    left_norm, right_norm = normalize_team(left), normalize_team(right)
    if not left_norm or not right_norm:
        return False
    if left_norm == right_norm:
        return True
    for values in TEAM_ALIAS_GROUPS:
        if left_norm in values and right_norm in values:
            return True
    # Provider names can add a geographic/organizational suffix (e.g.
    # "Bayern Munich" vs "Bayern Munich II"). Require the shorter name to
    # have at least two meaningful tokens so common short names cannot match
    # an unrelated club by accident.
    left_tokens, right_tokens = set(left_norm.split()), set(right_norm.split())
    if (
        min(len(left_tokens), len(right_tokens)) >= 2
        and (left_tokens <= right_tokens or right_tokens <= left_tokens)
    ):
        return True
    # Accept well-known abbreviations such as "S Martin SJ" against the full
    # app name, but require a high score and avoid matching a short token alone.
    if min(len(left_norm), len(right_norm)) < 6:
        return False
    score = SequenceMatcher(None, left_norm, right_norm).ratio()
    return score >= 0.92


def _request_json(session: requests.Session, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    response = session.get(f"{BASE_URL}/{path}", params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()
    if not payload.get("isSuccess"):
        raise RuntimeError(f"IdDaa API rejected {path}")
    return payload


def market_labels(config: Any) -> dict[tuple[int, int], str]:
    """Map (sport type, market code) to IdDaa's displayed market category."""
    labels: dict[tuple[int, int], str] = {}
    groups = config.get("dmsg", {}) if isinstance(config, dict) else {}
    for group in groups.values() if isinstance(groups, dict) else []:
        name = group.get("n") if isinstance(group, dict) else None
        entries = group.get("m", []) if isinstance(group, dict) else []
        if not name:
            continue
        for entry in entries:
            try:
                sport_type, market_code = (int(part) for part in str(entry).split("_", 1))
            except (TypeError, ValueError):
                continue
            labels[(sport_type, market_code)] = str(name)
    return labels


def fetch_bulletin(
    session: requests.Session,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[tuple[int, int], str]]:
    competitions = _request_json(session, "competitions")["data"]
    try:
        labels = market_labels(_request_json(session, "get_market_config").get("data", {}))
    except (requests.RequestException, RuntimeError, ValueError):
        LOG.exception("IdDaa market-name configuration unavailable; reporting market codes")
        labels = {}
    payload = _request_json(
        session,
        "events",
        {"st": 1, "type": 0, "version": 0, "live": "true"},
    )
    events = payload.get("data", {}).get("events", [])
    return competitions, events, labels


def competition_league_ids(competitions: list[dict[str, Any]]) -> dict[int, int]:
    league_by_competition_ref = {
        competition_ref: league_id
        for league_id, competition_ref in IDDAA_COMPETITION_REFS.items()
    }
    result: dict[int, int] = {}
    for competition in competitions:
        if str(competition.get("si", "")) != "1":
            continue
        try:
            competition_ref = abs(int(competition.get("cref", 0)))
            own_competition_id = int(competition.get("i", 0))
        except (TypeError, ValueError):
            continue
        league_id = league_by_competition_ref.get(competition_ref)
        # Prefer the stable API-Football ref where available. Fall back to
        # verified IdDaa catalogue IDs for leagues whose cref is provider-local.
        if league_id is None:
            league_id = IDDAA_COMPETITION_IDS.get(own_competition_id)
        if league_id is None and own_competition_id in {1220, 1219, 1221}:
            league_id = 144
        if league_id is None and own_competition_id == 57705:
            league_id = 207  # Switzerland Super League; IdDaa publishes a provider-specific ref.
        # IdDaa's current Champions League catalogue no longer exposes the
        # API-Football league ID as cref. Match its exact football competition
        # name so the daily bulletin does not silently skip those fixtures.
        if league_id is None:
            competition_name = normalize(str(competition.get("n", "")))
            # IdDaa's current European competition catalogue uses provider
            # competition refs instead of API-Football league IDs. Match the
            # exact football competition names so these fixtures are included
            # in the odds import.
            european_competitions = {
                "uefa sampiyonlar ligi": 2,
                "uefa avrupa ligi": 3,
                "uefa avrupa konferans ligi": 848,
                "uefa konferans ligi": 848,
                "almanya bundesliga": 78,
                "arjantin primera lpf kapanis": 128,
                "arjantin primera lpf acilis": 128,
            }
            league_id = european_competitions.get(competition_name)
        if league_id is not None:
            result[int(competition["i"])] = league_id
    return result


def match_markets(
    event: dict[str, Any], labels: dict[tuple[int, int], str] | None = None
) -> dict[str, Any]:
    markets = []
    for market in event.get("m", []):
        selections = [
            {
                "id": outcome.get("no"),
                "name": outcome.get("n"),
                "odds": outcome.get("odd"),
                "display_odds": outcome.get("wodd"),
            }
            for outcome in market.get("o", [])
        ]
        if selections:
            markets.append(
                {
                    "market_id": market.get("st"),
                    "sport_type": market.get("t"),
                    "market_name": market.get("name") or market.get("n") or market.get("sn") or (labels or {}).get((int(market.get("t", -1)), int(market.get("st", -1)))),
                    "line": market.get("sov"),
                    "selections": selections,
                }
            )
    return {
        "source": "iddaa.com",
        "event_id": event.get("i"),
        "home_team": event.get("hn"),
        "away_team": event.get("an"),
        "markets": markets,
    }


def summarize_markets(
    events: list[dict[str, Any]], labels: dict[tuple[int, int], str] | None = None
) -> dict[str, Any]:
    """Summarize available market codes, labels, and event coverage."""
    counts: dict[tuple[str, str], int] = {}
    event_market_counts: list[int] = []
    for event in events:
        markets = [market for market in event.get("m", []) if market.get("o")]
        event_market_counts.append(len(markets))
        for market in markets:
            code = str(market.get("st", "unknown"))
            label = str(
                market.get("name")
                or market.get("n")
                or market.get("sn")
                or (labels or {}).get((int(market.get("t", -1)), int(market.get("st", -1))))
                or ""
            )
            key = (code, label)
            counts[key] = counts.get(key, 0) + 1
    return {
        "market_types": [
            {"market_id": code, "market_name": label or None, "events": count}
            for (code, label), count in sorted(counts.items(), key=lambda item: (item[0][0], item[0][1]))
        ],
        "event_market_count_min": min(event_market_counts, default=0),
        "event_market_count_max": max(event_market_counts, default=0),
        "event_market_count_average": round(sum(event_market_counts) / len(event_market_counts), 2)
        if event_market_counts
        else 0,
    }


def _date(value: str) -> date:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(ISTANBUL).date()


def import_daily_odds(
    db: SupabaseRestClient,
    *,
    match_day: date,
    session: requests.Session | None = None,
    only_uncovered: bool = False,
) -> dict[str, Any]:
    http = session or requests.Session()
    http.headers.update({"Accept": "application/json", "User-Agent": "mac-analiz-app/1.0"})
    competitions, events, labels = fetch_bulletin(http)
    comp_map = competition_league_ids(competitions)
    tracked = {league.id for league in TRACKED_LEAGUES}
    target_competitions = {
        competition_id: league_id
        for competition_id, league_id in comp_map.items()
        if league_id in tracked
    }
    mapped_leagues = set(target_competitions.values())
    missing_leagues = REQUIRED_IDDAA_LEAGUES - mapped_leagues
    if missing_leagues:
        LOG.warning("IdDaa catalogue currently has no exact match for app league IDs: %s", sorted(missing_leagues))
    if not target_competitions:
        raise RuntimeError("IdDaa competitions response did not match any configured target leagues")

    start = datetime.combine(match_day, datetime_time.min, tzinfo=ISTANBUL).astimezone(timezone.utc)
    end = datetime.combine(
        match_day + timedelta(days=1), datetime_time.min, tzinfo=ISTANBUL
    ).astimezone(timezone.utc)
    match_filters = {
        "league_id": f"in.({','.join(map(str, sorted(set(target_competitions.values()))))})",
        "match_date": f"gte.{start.isoformat()}",
        "and": f"(match_date.lt.{end.isoformat()})",
    }
    if not only_uncovered:
        match_filters["status"] = "in.(scheduled,live)"
    matches = db.select_all(
        "matches",
        columns="id,league_id,home_team_id,away_team_id,match_date,status",
        filters=match_filters,
        order="match_date.asc,id.asc",
    )
    team_ids = sorted(
        {
            int(team_id)
            for match in matches
            for team_id in (match.get("home_team_id"), match.get("away_team_id"))
            if team_id is not None
        }
    )
    teams: list[dict[str, Any]] = []
    for offset in range(0, len(team_ids), 100):
        team_chunk = team_ids[offset : offset + 100]
        teams.extend(
            db.select_all(
                "teams",
                columns="id,name",
                filters={"id": f"in.({','.join(map(str, team_chunk))})"},
                order="id.asc",
            )
        )
    team_names = {int(row["id"]): str(row["name"]) for row in teams}
    covered_match_ids: set[int] = set()
    if only_uncovered and matches:
        match_ids = [int(row["id"]) for row in matches]
        for offset in range(0, len(match_ids), 100):
            chunk = match_ids[offset : offset + 100]
            history = db.select_all(
                "odds_quote_history",
                columns="match_id",
                filters={"match_id": f"in.({','.join(map(str, chunk))})"},
                order="match_id.asc",
            )
            covered_match_ids.update(int(row["match_id"]) for row in history)
    index: dict[tuple[int, date], list[dict[str, Any]]] = {}
    for match in matches:
        if int(match["league_id"]) not in target_competitions.values():
            continue
        if not only_uncovered and match.get("status") not in ("scheduled", "live"):
            continue
        home = team_names.get(int(match["home_team_id"]))
        away = team_names.get(int(match["away_team_id"]))
        if not home or not away:
            continue
        day = _date(str(match["match_date"]))
        key = (int(match["league_id"]), day)
        match["_home_name"] = home
        match["_away_name"] = away
        index.setdefault(key, []).append(match)

    now = datetime.now(timezone.utc).isoformat()
    records: list[dict[str, Any]] = []
    seen_events: set[int] = set()
    matched_ids: set[int] = set()
    unmatched = 0
    unmatched_details: list[dict[str, Any]] = []
    selected_events = 0
    selected_market_events: list[dict[str, Any]] = []
    for event in events:
        competition_id = int(event.get("ci", -1))
        league_id = target_competitions.get(competition_id)
        if league_id is None:
            continue
        kickoff = datetime.fromtimestamp(int(event["d"]), timezone.utc).astimezone(ISTANBUL)
        if kickoff.date() != match_day:
            continue
        selected_events += 1
        selected_market_events.append(event)
        home, away = str(event.get("hn", "")), str(event.get("an", ""))
        day_fixtures = index.get((league_id, kickoff.date()), [])
        candidates = [
            row for row in day_fixtures
            if int(row["id"]) not in matched_ids
            and (
                team_names_match(home, str(row["_home_name"]))
                and team_names_match(away, str(row["_away_name"]))
                or team_names_match(home, str(row["_away_name"]))
                and team_names_match(away, str(row["_home_name"]))
            )
        ]
        if len(candidates) > 1:
            candidates.sort(
                key=lambda row: (
                    team_names_match(home, str(row["_home_name"]))
                    and team_names_match(away, str(row["_away_name"])),
                    -abs((datetime.fromisoformat(str(row["match_date"]).replace("Z", "+00:00")) - kickoff.astimezone(timezone.utc)).total_seconds()),
                ),
                reverse=True,
            )
        if not candidates:
            unmatched += 1
            same_day_league = index.get((league_id, kickoff.date()), [])
            team_mismatch_candidates = []
            for row in same_day_league:
                db_home = team_names.get(int(row["home_team_id"]), "")
                db_away = team_names.get(int(row["away_team_id"]), "")
                if team_names_match(db_home, home) or team_names_match(db_home, away) or team_names_match(db_away, home) or team_names_match(db_away, away):
                    team_mismatch_candidates.append(
                        {"match_id": row["id"], "home": db_home, "away": db_away}
                    )
            detail = {
                "competition_id": competition_id,
                "league_id": league_id,
                "home": home,
                "away": away,
                "kickoff": kickoff.isoformat(),
                "reason": "no_unique_team_pair_match",
                "candidate_fixtures": team_mismatch_candidates[:5],
            }
            unmatched_details.append(detail)
            LOG.warning(
                "Unmatched IdDaa event: competition_id=%s league_id=%s home=%r away=%r kickoff=%s reason=%s same_day_league_fixtures=%s partial_team_candidates=%s",
                competition_id,
                league_id,
                event.get("hn"),
                event.get("an"),
                kickoff.isoformat(),
                "team_names_or_home_away_order_do_not_match_or_fixture_already_matched",
                len(same_day_league),
                team_mismatch_candidates[:5],
            )
            continue
        target = min(
            candidates,
            key=lambda row: abs(
                (datetime.fromisoformat(str(row["match_date"]).replace("Z", "+00:00")) - kickoff.astimezone(timezone.utc)).total_seconds()
            ),
        )
        match_kickoff = datetime.fromisoformat(str(target["match_date"]).replace("Z", "+00:00"))
        if abs((match_kickoff - kickoff.astimezone(timezone.utc)).total_seconds()) > 6 * 3600:
            unmatched += 1
            detail = {
                "competition_id": competition_id,
                "league_id": league_id,
                "home": home,
                "away": away,
                "kickoff": kickoff.isoformat(),
                "reason": "kickoff_diff_over_6h",
                "candidate_match_id": target["id"],
                "candidate_home": team_names.get(int(target["home_team_id"])),
                "candidate_away": team_names.get(int(target["away_team_id"])),
                "candidate_kickoff": match_kickoff.isoformat(),
            }
            unmatched_details.append(detail)
            LOG.warning(
                "Unmatched IdDaa event: competition_id=%s league_id=%s home=%r away=%r kickoff=%s reason=kickoff_diff_over_6h candidate_match_id=%s candidate_home=%r candidate_away=%r candidate_kickoff=%s",
                competition_id,
                league_id,
                event.get("hn"),
                event.get("an"),
                kickoff.isoformat(),
                target["id"],
                team_names.get(int(target["home_team_id"])),
                team_names.get(int(target["away_team_id"])),
                match_kickoff.isoformat(),
            )
            continue
        event_id = int(event["i"])
        if event_id in seen_events:
            continue
        seen_events.add(event_id)
        matched_ids.add(int(target["id"]))
        if int(target["id"]) in covered_match_ids:
            continue
        records.append(
            {
                "match_id": int(target["id"]),
                "bookmaker": "iddaa.com",
                "odds": match_markets(event, labels),
                "source_updated_at": now,
                "captured_at": now,
            }
        )

    if selected_events and not matched_ids and not unmatched:
        raise RuntimeError(
            f"IdDaa returned {selected_events} target-league events for {match_day}, "
            "but produced no unique fixture matches"
        )
    if records:
        db.insert("odds_quote_history", records)
    return {
        "competitions_matched": len(target_competitions),
        "leagues_missing_from_iddaa_catalogue": sorted(missing_leagues),
        "events_for_day": selected_events,
        "events_matched": len(records),
        "quotes_saved": len(records),
        "events_unmatched": unmatched,
        "unmatched_details": unmatched_details,
        "markets": summarize_markets(selected_market_events, labels),
    }


def write_github_summary(match_day: date, summary: dict[str, Any]) -> None:
    """Add odds coverage and catalogue gaps to the Actions run summary."""
    summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return
    league_labels = {
        league.id: f"{league.name} ({league.country})"
        for league in TRACKED_LEAGUES
    }
    missing_leagues = ", ".join(
        f"{league_id}: {league_labels.get(league_id, 'Bilinmeyen lig')}"
        for league_id in summary["leagues_missing_from_iddaa_catalogue"]
    ) or "Yok"
    lines = [
        "## IdDaa oran kapsamı",
        f"- Tarih: {match_day.isoformat()}",
        f"- Eşleşen: {summary['events_matched']} / {summary['events_for_day']}",
        f"- Kaydedilen oran anlıkları: {summary['quotes_saved']}",
        f"- Eşleşmeyen: {summary['events_unmatched']}",
        f"- Katalogda eşleşmeyen lig ID'leri: {missing_leagues}",
    ]
    if summary["unmatched_details"]:
        lines.extend([
            "",
            "| Lig ID | IdDaa maçı | Başlama | Olası fikstür | Neden |",
            "|---:|---|---|---|---|",
        ])
        for item in summary["unmatched_details"]:
            matchup = f"{item['home']} — {item['away']}".replace("|", "\\|")
            candidates = item.get("candidate_fixtures", [])
            if candidates:
                candidate_text = "; ".join(
                    f"{candidate['home']} — {candidate['away']} (ID {candidate['match_id']})"
                    for candidate in candidates
                )
            elif item.get("candidate_match_id"):
                candidate_text = (
                    f"{item.get('candidate_home')} — {item.get('candidate_away')} "
                    f"(ID {item['candidate_match_id']})"
                )
            else:
                candidate_text = "—"
            candidate_text = candidate_text.replace("|", "\\|")
            lines.append(
                f"| {item['league_id']} | {matchup} | {item['kickoff']} | "
                f"{candidate_text} | {item['reason']} |"
            )
    try:
        with open(summary_path, "a", encoding="utf-8") as summary_file:
            summary_file.write("\n".join(lines) + "\n")
    except OSError:
        LOG.exception("Could not write IdDaa coverage to the GitHub Actions summary")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", help="Istanbul calendar date (YYYY-MM-DD); defaults to today")
    parser.add_argument("--retries", type=int, default=3, help="Attempts before reporting failure")
    parser.add_argument(
        "--allow-partial-retry",
        action="store_true",
        help="Retry a partial import after matching logic is fixed; existing quotes are retained",
    )
    args = parser.parse_args()
    if args.retries < 1 or args.retries > 5:
        parser.error("--retries must be between 1 and 5")
    match_day = date.fromisoformat(args.date) if args.date else datetime.now(ISTANBUL).date()
    settings = get_settings()
    db = SupabaseRestClient(settings.supabase_url, settings.supabase_service_role_key)

    for attempt in range(1, args.retries + 1):
        try:
            summary = import_daily_odds(
                db,
                match_day=match_day,
                only_uncovered=args.allow_partial_retry,
            )
            print({"date": match_day.isoformat(), "iddaa_odds": summary})
            write_github_summary(match_day, summary)
            if summary["events_unmatched"]:
                LOG.error(
                    "IdDaa odds import completed with partial coverage: "
                    "%s of %s target events could not be matched; saved %s quotes",
                    summary["events_unmatched"],
                    summary["events_for_day"],
                    summary["quotes_saved"],
                )
                if not args.allow_partial_retry:
                    # Records have already been saved. Avoid accidental
                    # duplicate history snapshots on routine scheduled runs.
                    raise SystemExit(2)
                LOG.warning(
                    "Partial retry explicitly enabled; unmatched events may be retried"
                )
                return
            return
        except Exception:
            LOG.exception("IdDaa odds import failed (attempt %s/%s)", attempt, args.retries)
            if attempt == args.retries:
                raise
            time.sleep(30 * attempt)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
