"""Read visible Nesine pre-match odds and persist safe, matched quotes.

The collector deliberately parses rendered rows instead of relying on private
network endpoints. Nesine can change its DOM, so parsing failures are reported
and never create fake odds.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime, time, timedelta, timezone
from difflib import SequenceMatcher
from typing import Any
from zoneinfo import ZoneInfo

from config.settings import get_settings
from config.leagues import TRACKED_LEAGUE_IDS
from config.nesine_team_aliases import NESINE_TEAM_IDS_BY_LEAGUE
from data_pipeline.api_client import ApiFootballClient
from data_pipeline.fetch_fixtures import transform_fixtures
from db.db_client import SupabaseRestClient

NESINE_BOOKMAKER = "Nesine"
DEFAULT_NESINE_URL = "https://www.nesine.com/iddaa?et=1&le=2"
_ODD = re.compile(r"^(?:[1-9]\d?)(?:[.,]\d{1,3})$")
LOGGER = logging.getLogger(__name__)
ISTANBUL = ZoneInfo("Europe/Istanbul")

# Names as displayed in Nesine's football bulletin. These names are used to
# filter the Nesine fixture feed before we read detailed markets. Keep this
# keyed by API-Football league id so only the app's configured competitions
# are considered.
NESINE_LEAGUE_NAMES: dict[int, tuple[str, ...]] = {
    39: ("ingiltere premier lig", "ingiltere premier league", "premier league"),
    140: ("ispanya la liga", "ispanya laliga", "la liga", "laliga"),
    135: ("italya serie a", "serie a italya"),
    78: ("almanya bundesliga", "almanya 1 bundesliga", "bundesliga almanya"),
    61: ("fransa ligue 1", "ligue 1 fransa"),
    2: ("uefa sampiyonlar ligi", "uefa champions league"),
    3: ("uefa avrupa ligi", "uefa europa league"),
    848: ("uefa konferans ligi", "uefa avrupa konferans ligi", "uefa europa conference league"),
    203: ("turkiye super lig", "super lig"),
    94: ("portekiz premier ligi", "portekiz 1 ligi", "portekiz ligi", "primeira liga"),
    88: ("hollanda eredivisie", "eredivisie"),
    144: ("belcika jupiler pro lig", "belcika jupiler league", "belcika 1 lig", "jupiler pro league"),
    40: ("ingiltere championship", "championship"),
    307: ("suudi arabistan pro ligi", "suudi arabistan pro league", "suudi pro lig", "saudi pro league"),
    253: ("abd mls", "abd major league soccer", "amerika mls", "major league soccer"),
    71: ("brezilya serie a", "serie a brezilya"),
    262: ("meksika liga mx", "liga mx"),
    179: ("iskocya premiership", "scottish premiership"),
    218: ("avusturya bundesliga", "bundesliga avusturya"),
    207: ("isvicre super league", "super league isvicre"),
    136: ("italya serie b", "serie b italya"),
    141: ("ispanya 2 ligi", "ispanya segunda division", "segunda division"),
    62: ("fransa ligue 2", "ligue 2 fransa"),
    197: ("yunanistan super league 1", "yunanistan super ligi", "super league 1"),
    13: ("conmebol libertadores", "guney amerika libertadores", "libertadores"),
    1: ("fifa dunya kupasi", "dunya kupasi", "fifa world cup"),
    4: ("avrupa sampiyonasi", "euro championship"),
    5: ("uefa uluslar ligi", "uefa nations league"),
    6: ("afrika uluslar kupasi", "africa cup of nations"),
    7: ("asya kupasi", "afc asian cup"),
    9: ("copa america", "amerika kupasi"),
    22: ("concacaf gold cup", "concacaf altin kupa"),
}


def _normalise_league_name(value: str) -> str:
    value = unicodedata.normalize("NFKD", value.casefold().replace("ı", "i"))
    value = "".join(char for char in value if not unicodedata.combining(char))
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value).split())


NESINE_TRACKED_LEAGUE_NAMES = {
    _normalise_league_name(name)
    for league_id in TRACKED_LEAGUE_IDS
    for name in NESINE_LEAGUE_NAMES.get(league_id, ())
}
if set(TRACKED_LEAGUE_IDS) - NESINE_LEAGUE_NAMES.keys():
    raise ValueError("Every app-tracked league needs a Nesine bulletin name mapping")
NESINE_LEAGUE_IDS_BY_NAME: dict[str, set[int]] = {}
for _league_id, _names in NESINE_LEAGUE_NAMES.items():
    for _name in _names:
        NESINE_LEAGUE_IDS_BY_NAME.setdefault(_normalise_league_name(_name), set()).add(_league_id)


def _tracked_league_ids_for_nesine_name(value: str | None) -> set[int]:
    if not value:
        return set()
    return NESINE_LEAGUE_IDS_BY_NAME.get(_normalise_league_name(value), set()) & set(TRACKED_LEAGUE_IDS)


def istanbul_day_bounds(now: datetime | None = None) -> tuple[str, str]:
    """Return the current Istanbul calendar day's UTC bounds for fixture queries."""
    local_now = now.astimezone(ISTANBUL) if now else datetime.now(ISTANBUL)
    start = datetime.combine(local_now.date(), time.min, tzinfo=ISTANBUL)
    end = start + timedelta(days=1)
    return start.astimezone(timezone.utc).isoformat(), end.astimezone(timezone.utc).isoformat()


def _is_nesine_today_label(value: Any, today: Any = None) -> bool:
    """Accept only the bulletin section explicitly labeled for Istanbul today."""
    label = unicodedata.normalize("NFKD", str(value or "").strip().casefold().replace("ı", "i"))
    label = "".join(char for char in label if not unicodedata.combining(char))
    if label == "bugun":
        return True
    match = re.fullmatch(r"(\d{1,2})[./-](\d{1,2})(?:[./-](\d{2,4}))?", label)
    if not match:
        return False
    day, month, year = match.groups()
    current = today or datetime.now(ISTANBUL).date()
    parsed_year = int(year) if year else current.year
    if parsed_year < 100:
        parsed_year += 2000
    try:
        return current == current.replace(year=parsed_year, month=int(month), day=int(day))
    except ValueError:
        return False


def _normalise(value: str) -> str:
    # Turkish dotless ı does not decompose under NFKD and was being dropped,
    # turning names such as "Iğdır" into a different string.
    value = unicodedata.normalize("NFKD", value.casefold())
    value = "".join(char for char in value if not unicodedata.combining(char)).replace("ı", "i")
    country_aliases = {
        "guney kore": "south korea", "ozbekistan": "uzbekistan",
        "kazakistan": "kazakhstan", "faroe adalari": "faroe islands",
        "danimarka": "denmark", "hollanda": "netherlands",
        "azerbaycan": "azerbaijan", "cebelitarik": "gibraltar",
        "portekiz": "portugal", "avusturya": "austria", "hindistan": "india",
        "almanya": "germany", "ispanya": "spain", "hirvatistan": "croatia",
        "ingiltere": "england", "cekya": "czechia", "arnavutluk": "albania",
        "finlandiya": "finland", "estonya": "estonia", "izlanda": "iceland",
        "isvicre": "switzerland", "slovakya": "slovakia", "slovenya": "slovenia",
        "luksemburg": "luxembourg", "bulgaristan": "bulgaria", "iskocya": "scotland",
        "fyr macedonia": "north macedonia", "karadag": "montenegro",
        "kuzey makedonya": "north macedonia", "kuzey irlanda": "northern ireland",
    }
    for local_name, canonical_name in country_aliases.items():
        value = re.sub(rf"(?<!\w){re.escape(local_name)}(?!\w)", canonical_name, value)
    text = value.casefold().replace("ø", "o").replace("æ", "ae").replace("å", "a")
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    tokens = re.sub(r"[^a-z0-9]+", " ", text.lower()).split()
    # Club-name variants that are common in the Nesine and API-Football feeds.
    # Apply these to the complete normalized name to avoid changing unrelated
    # tokens (for example, a standalone "v" is not a safe alias).
    name_aliases = {
        "v guimaraes": ("vitoria sc",),
        "vitoria de guimaraes": ("vitoria sc",),
        "cska sofya": ("cska sofia",),
        "cska 1948": ("cska 1948 sofia",),
        "us yacoub mansour": ("us yacoub el mansour",),
        "crs salmi": ("cr salmi",),
        "san roque": ("san roque de lepe",),
        "club atletico central": ("central atletico",),
        "st truidense": ("st truiden",),
        "st truidense vv": ("st truiden",),
        "lillestrom": ("lillestrom sk",),
        "molde": ("molde fk",),
        "f dusseldorf": ("fortuna dusseldorf",),
        "dusseldorf": ("fortuna dusseldorf",),
        "kaa gent ii": ("gent ii",),
        "rfc seraing": ("seraing",),
        "modena": ("modena fc",),
        "hellas verona": ("verona",),
        "dynamo moscow": ("dinamo moscow",),
        "dinamo moscow": ("dynamo moscow",),
        "lillestrom sk": ("lillestrom",),
        "molde fk": ("molde",),
        "estac troyes": ("troyes",),
        "marseille": ("marsilya",),
        "marsilya": ("marseille",),
        "atl mineiro": ("atletico mg",),
        "atletico mg": ("atl mineiro",),
        "s gijon": ("sporting gijon",),
        "sporting gijon": ("s gijon",),
        "santander": ("racing santander",),
        "racing santander": ("santander",),
        "gremio p": ("gremio",),
        "gremio": ("gremio p",),
        "al taawoun": ("al taawon",),
        "al taawon": ("al taawoun",),
        "al shabab sar": ("al shabab",),
        "al shabab": ("al shabab sar",),
        "olympiakos piraeus": ("olympiakos",),
        "olympiakos": ("olympiakos piraeus",),
        # The app's fixture feed and Nesine use the short/official city names
        # interchangeably for these same Spanish and Greek clubs.
        "real valladolid": ("valladolid",),
        "real oviedo": ("oviedo",),
        "pae ps kalamata": ("kalamata",),
        "botafogo rj": ("botafogo",),
        "newcastle utd": ("newcastle",),
        "newcastle united": ("newcastle",),
        "sabah": ("sabah fa",),
        "sabah fa": ("sabah",),
        "slavia prag": ("slavia praha",),
        "slavia praha": ("slavia prag",),
        "rc lens": ("lens",),
        "lens": ("rc lens",),
        "w bromwich": ("west brom",),
        "west brom": ("w bromwich",),
        "blackburn r": ("blackburn",),
        "blackburn": ("blackburn r",),
        "swansea city": ("swansea",),
        "swansea": ("swansea city",),
        "cardiff city": ("cardiff",),
        "cardiff": ("cardiff city",),
        "wolverhampton": ("wolves",),
        "wolves": ("wolverhampton",),
        "bayern munih": ("bayern munich",),
        "bayern munich": ("bayern munih",),
        "psv eindhoven": ("psv",),
        "psv": ("psv eindhoven",),
    }
    aliases = {"utd": "united", "munchen": "munich"}
    generic = {"fc", "afc", "cf", "sc", "sk", "fk", "rb", "sp"}
    # Feeds often shorten common club words while retaining the distinctive
    # part of a team's name (e.g. "D. Moskova" for "Dynamo Moscow").
    # Expand only well-known football abbreviations; fuzzy matching remains
    # gated by the strict team-by-team check below.
    abbreviations = {
        "d": "dynamo", "din": "dinamo", "ath": "athletic",
        "st": "saint", "ints": "internazionale",
        "inter": "internazionale", "dep": "deportivo",
        "moskova": "moscow", "moskva": "moscow",
    }
    normalized = " ".join(
        aliases.get(token, abbreviations.get(token, token))
        for token in tokens if token not in generic
    )
    variants = [normalized]
    variants.extend(name_aliases.get(normalized, ()))
    without_generic = " ".join(token for token in tokens if token not in generic)
    variants.append(" ".join(aliases.get(token, token) for token in without_generic.split()))
    # Common suffixes/prefixes and sponsored/legal forms can appear on either
    # feed. Permit the distinctive club name to match when it is the complete
    # remaining name after removing only a known club descriptor.
    descriptors = {"kv", "kaa", "rfc", "us", "crs", "cr", "sc", "vv", "sk", "fk", "fc", "cf", "afc", "rb"}
    variants.append(" ".join(token for token in tokens if token not in descriptors))
    # Expand only a complete club name from the controlled variant table.
    # This allows app aliases to be compared symmetrically without broad fuzzy
    # matches across unrelated clubs.
    return "|".join(dict.fromkeys(variants))


@dataclass(frozen=True, slots=True)
class NesineMatchQuote:
    home_team: str
    away_team: str
    kickoff_text: str | None
    home_win: str | None
    draw: str | None
    away_win: str | None
    over_2_5: str | None = None
    under_2_5: str | None = None
    btts_yes: str | None = None
    btts_no: str | None = None
    markets: tuple[dict[str, str], ...] = ()
    event_id: str | None = None
    league_name: str | None = None

    def snapshot(self) -> dict[str, str | None]:
        return {"home_win": self.home_win, "draw": self.draw, "away_win": self.away_win,
                "over_2_5": self.over_2_5, "under_2_5": self.under_2_5,
                "btts_yes": self.btts_yes, "btts_no": self.btts_no}

    def all_markets(self) -> tuple[dict[str, str], ...]:
        return self.markets

    @property
    def has_result_market(self) -> bool:
        return all(self.snapshot().get(key) for key in ("home_win", "draw", "away_win"))


def _odd(value: str) -> str | None:
    value = value.strip().replace(",", ".")
    return value if _ODD.fullmatch(value) else None


def parse_rendered_text_rows(rows: Iterable[Mapping[str, Any]]) -> list[NesineMatchQuote]:
    """Parse rows extracted from the rendered page.

    Expected row shape: ``home_team``, ``away_team``, optional ``kickoff_text``
    and a mapping/list of market values. This pure function is easy to test and
    keeps browser-specific code out of the data model.
    """
    parsed: list[NesineMatchQuote] = []
    for row in rows:
        home, away = str(row.get("home_team", "")).strip(), str(row.get("away_team", "")).strip()
        if not home or not away:
            continue
        values = row.get("markets") or row.get("odds") or {}
        if isinstance(values, Mapping):
            def get(*keys: str) -> float | None:
                return next(
                    (_odd(str(values[key])) for key in keys if values.get(key) is not None and _odd(str(values[key]))),
                    None,
                )
            result = (get("1", "home_win"), get("X", "draw"), get("2", "away_win"))
            totals = (get("Over 2.5", "over_2_5"), get("Under 2.5", "under_2_5"))
            btts = (get("Var", "Yes", "btts_yes"), get("Yok", "No", "btts_no"))
        else:
            values = [_odd(str(value)) for value in values]
            values = [value for value in values if value]
            result = tuple(values[:3]) + (None,) * max(0, 3 - len(values[:3]))
            totals, btts = (None, None), (None, None)
        raw_markets = row.get("all_markets") or ()
        markets = tuple(
            {"market_code": str(item.get("market_code", "unknown")),
             "market_name": str(item.get("market_name", "Unknown")),
             "selection_code": str(item.get("selection_code", "unknown")),
             "selection_name": str(item.get("selection_name", "Unknown")),
             "odd": str(item.get("odd"))}
            for item in raw_markets
            if isinstance(item, Mapping) and _odd(str(item.get("odd", "")))
        )
        event_id = str(row.get("event_id") or "").strip() or None
        parsed.append(NesineMatchQuote(
            home, away, row.get("kickoff_text"), *result, *totals, *btts, markets, event_id,
            str(row.get("league_name") or "").strip() or None,
        ))
    return parsed


def match_score(quote: NesineMatchQuote, match: Mapping[str, Any]) -> float:
    home = str(match.get("home_team") or match.get("home_name") or "")
    away = str(match.get("away_team") or match.get("away_name") or "")

    def similarity(left: str, right: str) -> float:
        if _same_fixture_team_name(left, right):
            return 1.0
        left_tokens, right_tokens = left.split(), right.split()
        if not left_tokens or not right_tokens:
            return 0.0
        sequence = SequenceMatcher(None, left, right).ratio()
        common = sum(min(left_tokens.count(token), right_tokens.count(token))
                     for token in set(left_tokens) | set(right_tokens))
        token_overlap = 2 * common / (len(left_tokens) + len(right_tokens))
        return max(sequence, token_overlap)

    return (similarity(_normalise(quote.home_team).split("|")[0], _normalise(home).split("|")[0]) +
            similarity(_normalise(quote.away_team).split("|")[0], _normalise(away).split("|")[0])) / 2


def _same_normalised_team(left: str, right: str) -> bool:
    left_name, right_name = _normalise(left), _normalise(right)
    left_names, right_names = left_name.split("|"), right_name.split("|")
    if set(left_names) & set(right_names):
        return True
    # Accept common harmless feed abbreviations (for example "Crew" / "Cr.")
    # only when the shorter full token is a prefix and all other tokens agree.
    for left_variant in left_names:
        for right_variant in right_names:
            left_tokens, right_tokens = left_variant.split(), right_variant.split()
            if len(left_tokens) != len(right_tokens):
                continue
            if all(a == b or (len(a) >= 2 and len(b) >= 2 and (a.startswith(b) or b.startswith(a)))
                   for a, b in zip(left_tokens, right_tokens)):
                return True
    return False


def _team_name_variants(value: str) -> set[str]:
    """Return strict normalized aliases for one complete team name."""
    variants = set(_normalise(value).split("|"))
    descriptors = {"fc", "afc", "cf", "sc", "sk", "fk", "rb", "sp", "kv", "kaa", "rfc", "us", "crs", "cr", "vv"}
    for variant in tuple(variants):
        variants.add(" ".join(token for token in variant.split() if token not in descriptors))
    return {variant for variant in variants if variant}


def _same_fixture_team_name(left: str, right: str) -> bool:
    left_variants = _team_name_variants(left)
    right_variants = _team_name_variants(right)
    if left_variants & right_variants:
        return True
    for left_variant in left_variants:
        for right_variant in right_variants:
            left_tokens, right_tokens = left_variant.split(), right_variant.split()
            if len(left_tokens) == len(right_tokens) and all(
                a == b or (len(a) >= 2 and len(b) >= 2 and (a.startswith(b) or b.startswith(a)))
                for a, b in zip(left_tokens, right_tokens)
            ):
                return True
    return False


def _team_identity_keys(value: str) -> set[str]:
    """Build normalized exact keys, including only explicitly known variants."""
    return {key for key in _normalise(value).split("|") if key}


def _quote_kickoff_matches(quote: NesineMatchQuote, match: Mapping[str, Any]) -> bool:
    raw_time = str(quote.kickoff_text or "").strip()
    raw_date = str(match.get("kickoff") or match.get("match_date") or "").strip()
    if not raw_time or not raw_date:
        return True
    time_match = re.fullmatch(r"(\d{1,2}):(\d{2})", raw_time)
    try:
        fixture_time = datetime.fromisoformat(raw_date.replace("Z", "+00:00"))
    except ValueError:
        return True
    if not time_match or fixture_time.tzinfo is None:
        return True
    local_fixture_time = fixture_time.astimezone(ISTANBUL)
    quote_minutes = int(time_match.group(1)) * 60 + int(time_match.group(2))
    fixture_minutes = local_fixture_time.hour * 60 + local_fixture_time.minute
    difference = abs(quote_minutes - fixture_minutes)
    return min(difference, 24 * 60 - difference) <= 15


def match_quotes(quotes: Iterable[NesineMatchQuote], matches: Iterable[Mapping[str, Any]], threshold: float = .82) -> list[tuple[int, NesineMatchQuote, float]]:
    """Match production quotes by league/team IDs; fuzzy names are diagnostic only.

    The legacy name-only path is retained for fixture-free tooling callers. Real
    Nesine rows include a league label and are rejected if their league or team
    identity cannot be resolved to exactly one API-Football fixture.
    """
    output: list[tuple[int, NesineMatchQuote, float]] = []
    match_list = list(matches)
    api_team_ids: dict[tuple[int, str], set[int]] = {}
    for match in match_list:
        league_id = int(match.get("league_id", -1))
        for name_key in ("home_team", "home_name", "away_team", "away_name"):
            name = str(match.get(name_key) or "").strip()
            team_id_key = "home_team_id" if name_key.startswith("home") else "away_team_id"
            if name and match.get(team_id_key) is not None:
                for name_key in _team_identity_keys(name):
                    api_team_ids.setdefault((league_id, name_key), set()).add(
                        int(match[team_id_key])
                    )

    catalog_team_ids: dict[tuple[int, str], set[int]] = {}
    for league_id, aliases in NESINE_TEAM_IDS_BY_LEAGUE.items():
        for alias, team_id in aliases.items():
            for name_key in _team_identity_keys(alias):
                catalog_team_ids.setdefault((league_id, name_key), set()).add(team_id)

    for quote in quotes:
        quote_leagues = _tracked_league_ids_for_nesine_name(quote.league_name)
        if quote.league_name:
            # A rendered league heading must be recognized; never search the
            # entire slate when the competition could not be identified.
            if len(quote_leagues) != 1:
                continue
            league_id = next(iter(quote_leagues))
            home_ids: set[int] = set()
            away_ids: set[int] = set()
            for home_key in _team_identity_keys(quote.home_team):
                home_ids.update(catalog_team_ids.get((league_id, home_key), set()))
                home_ids.update(api_team_ids.get((league_id, home_key), set()))
            for away_key in _team_identity_keys(quote.away_team):
                away_ids.update(catalog_team_ids.get((league_id, away_key), set()))
                away_ids.update(api_team_ids.get((league_id, away_key), set()))
            if len(home_ids) != 1 or len(away_ids) != 1:
                continue
            candidate_matches = [
                match for match in match_list
                if int(match.get("league_id", -1)) == league_id
                and int(match.get("home_team_id", -1)) in home_ids
                and int(match.get("away_team_id", -1)) in away_ids
                and _quote_kickoff_matches(quote, match)
            ]
            if len(candidate_matches) == 1:
                match = candidate_matches[0]
                output.append((int(match["id"]), quote, 1.0))
            continue

        # Name-only matching is available only to legacy/internal callers that
        # do not have a real bulletin league heading. It cannot be used by the
        # production collector because rendered Nesine rows carry that heading.
        candidate_matches = match_list
        candidates = sorted(((match_score(quote, match), int(match["id"]), match)
                             for match in candidate_matches), reverse=True)
        if not candidates or candidates[0][0] < threshold:
            continue
        if len(candidates) > 1 and candidates[0][0] - candidates[1][0] < .04:
            continue
        # Only accept a fixture when both team names identify the app fixture
        # exactly after shared Turkish/English and club-prefix normalization.
        if (not _same_fixture_team_name(quote.home_team, str(candidates[0][2].get("home_team") or ""))
                or not _same_fixture_team_name(quote.away_team, str(candidates[0][2].get("away_team") or ""))):
            continue
        output.append((candidates[0][1], quote, candidates[0][0]))
    return output


def _partition_quotes_to_api_slate(
    quotes: Iterable[NesineMatchQuote],
    matches: Iterable[Mapping[str, Any]],
    *,
    unrelated_name_score: float = .75,
) -> tuple[list[NesineMatchQuote], list[NesineMatchQuote]]:
    """Separate unrelated bulletin rows mislabelled under a tracked league.

    Keep uncertain rows for the strict ID matcher to diagnose. Only discard a
    row when its recognized league has no same-kickoff API fixture and even its
    closest same-league team-pair name score is low.
    """
    match_list = list(matches)
    kept: list[NesineMatchQuote] = []
    out_of_slate: list[NesineMatchQuote] = []
    for quote in quotes:
        league_ids = _tracked_league_ids_for_nesine_name(quote.league_name)
        if len(league_ids) != 1 or not quote.kickoff_text:
            kept.append(quote)
            continue
        league_id = next(iter(league_ids))
        same_league = [
            match for match in match_list
            if int(match.get("league_id", -1)) == league_id
        ]
        has_same_kickoff = any(
            _quote_kickoff_matches(quote, match) for match in same_league
        )
        best_name_score = max(
            (match_score(quote, match) for match in same_league), default=0.0
        )
        if not has_same_kickoff and best_name_score < unrelated_name_score:
            out_of_slate.append(quote)
        else:
            kept.append(quote)
    return kept, out_of_slate


def store_nesine_quotes(db: SupabaseRestClient, matched: Iterable[tuple[int, NesineMatchQuote, float]], captured_at: str | None = None) -> int:
    captured_at = captured_at or datetime.now(timezone.utc).isoformat()
    inserted = 0
    for match_id, quote, _score in matched:
        markets = quote.all_markets()
        if not quote.has_result_market and not markets:
            continue
        if quote.has_result_market:
            db.insert("odds_quote_history", [{"match_id": match_id, "bookmaker": NESINE_BOOKMAKER,
                "odds": quote.snapshot(), "source_updated_at": None, "captured_at": captured_at,
                "is_notification_reference": False}])
        if markets:
            db.insert("nesine_market_quotes", [{"match_id": match_id, **market,
                "odd": float(market["odd"]), "captured_at": captured_at,
                "source_url": DEFAULT_NESINE_URL} for market in markets])
        inserted += 1
    return inserted


def _extract_rows_from_page(
    page: Any,
    target_fixtures: set[tuple[str, str]] | None = None,
    target_league_names: set[str] | None = None,
    previous_league_name: str = "",
    previous_day_label: str = "",
) -> tuple[list[dict[str, Any]], str, str]:
    """Extract tracked league fixtures and visible 1/X/2 prices from Nesine."""
    result = page.evaluate("""({targetFixtures, targetLeagueNames, previousLeagueName, previousDayLabel}) => {
      const odd = /^([1-9]\\d?)[.,]\\d{1,3}$/;
      const clean = value => (value || '').replace(/\\s+/g, ' ').trim();
      const targets = targetFixtures || [];
      const targetLeagues = new Set(targetLeagueNames || []);
      const normalizeLeague = value => value.normalize('NFKD').replace(/[\\u0300-\\u036f]/g, '')
        .toLowerCase().replace(/ı/g, 'i').replace(/[^a-z0-9]+/g, ' ').trim().replace(/\\s+/g, ' ');
      const leagueByEvent = new Map();
      const dayByEvent = new Map();
      let currentLeague = previousLeagueName || '';
      let currentDay = previousDayLabel || '';
      for (const group of document.querySelectorAll('[id^="OCRow_"]')) {
        const lines = group.innerText.split(/\\n/).map(clean);
        const groupHeader = [...group.children]
          .map(child => child.querySelector('strong'))
          .find(element => element && !element.closest('[data-test-id^="r_"]'));
        if (groupHeader) {
          currentLeague = clean(groupHeader.innerText);
          const headingIndex = lines.indexOf(currentLeague);
          const sectionLines = headingIndex >= 0 ? lines.slice(headingIndex + 1, headingIndex + 4) : [];
          const dayLabel = sectionLines.find(line => /^(bug[uü]n|yar[iı]n|\\d{1,2}[./-]\\d{1,2}(?:[./-]\\d{2,4})?)$/i.test(line));
          if (dayLabel) currentDay = dayLabel;
        }
        const event = group.querySelector('[data-test-id^="r_"][data-code][data-test-is-live="false"]');
        if (event) {
          leagueByEvent.set(event, currentLeague);
          dayByEvent.set(event, currentDay);
        }
      }
      const supportedLeague = row => Boolean(row) && (!targetLeagues.size || targetLeagues.has(normalizeLeague(row.league_name || '')));
      const finish = rows => ({rows, lastLeagueName: currentLeague, lastDayLabel: currentDay});
      const normalizeTeam = value => value.normalize('NFKD').replace(/[\u0300-\u036f]/g, '')
        .toLowerCase().replace(/\\butd\\b/g, '').replace(/\\brj\\b/g, '')
        .replace(/\\b(munchen)\\b/g, 'munich').replace(/[^a-z0-9]+/g, ' ')
        .split(/\\s+/).filter(token => token && !['fc','afc','cf','sc','sk','fk','rb','sp'].includes(token))
        .join(' ').trim();
      const teamMatches = (raw, variants) => variants.some(value => normalizeTeam(value) === normalizeTeam(raw));
      const matchesTarget = row => Boolean(row) && (!targets.length || targets.some(target =>
        teamMatches(row.home_team, target.home) && teamMatches(row.away_team, target.away)));
      const splitPair = value => {
        const parts = clean(value).split(/\\s+[-–—]\\s+/);
        return parts.length === 2 && parts.every(part => part && part.length <= 80) ? parts : null;
      };

      // Current desktop rows expose event and selection IDs as data-test-id
      // attributes. Preserve the event ID so its expanded market panel can be
      // opened after the visible fixture list has been matched to the app.
      const codedRows = [...document.querySelectorAll('[data-test-id^="r_"][data-code][data-test-is-live="false"]')].map(node => {
        const teamLink = node.querySelector('[data-test-id="matchName"]');
        const teams = teamLink && splitPair(teamLink.innerText);
        if (!teams) return null;
        const result = {};
        for (const item of node.querySelectorAll('[data-testid]')) {
          const testId = item.getAttribute('data-testid') || '';
          if (!testId.startsWith('odd_Maç Sonucu_')) continue;
          const selection = testId.slice('odd_Maç Sonucu_'.length);
          if (['1', 'X', '2'].includes(selection)) result[selection] = clean(item.innerText).replace(',', '.');
        }
        return {event_id: node.getAttribute('data-code'), home_team: teams[0], away_team: teams[1],
          league_name: leagueByEvent.get(node) || '',
          day_label: dayByEvent.get(node) || '',
          kickoff_text: clean(node.querySelector('[data-testid^="time-"]')?.innerText),
          markets: {"1": result['1'], "X": result.X, "2": result['2']}};
      });
      const targetedRows = codedRows.filter(row => supportedLeague(row) && matchesTarget(row));
      if (targets.length) return finish(targetedRows);
      if (targetLeagues.size) return finish(targetedRows);
      if (codedRows.length) return finish(codedRows);

      // Current Nesine football fixtures are OCRow_* event cards identified
      // by id. Read only explicit 1/X/2 labels; positional buttons shift when
      // a selection is unavailable and can pull in the next market's price.
      const eventRows = [...document.querySelectorAll('[id^="OCRow_"][data-test-is-live="false"]')].map(node => {
        const teamLink = [...node.querySelectorAll('a')]
          .map(link => clean(link.innerText)).find(text => splitPair(text));
        const teams = teamLink && splitPair(teamLink);
        if (!teams) return null;
        const result = {};
        for (const item of node.querySelectorAll('[data-testid]')) {
          const testId = item.getAttribute('data-testid') || '';
          if (!testId.startsWith('odd_Maç Sonucu_')) continue;
          const selection = testId.slice('odd_Maç Sonucu_'.length);
          if (['1', 'X', '2'].includes(selection)) result[selection] = clean(item.innerText).replace(',', '.');
        }
        const eventNode = node.querySelector('[data-test-id^="r_"][data-code]');
        return {event_id: node.getAttribute('data-code') || null, home_team: teams[0], away_team: teams[1],
          league_name: (eventNode && leagueByEvent.get(eventNode)) || '',
          day_label: (eventNode && dayByEvent.get(eventNode)) || '',
          markets: {"1": result['1'], "X": result.X, "2": result['2']}};
      });
      const targetedEventRows = eventRows.filter(row => supportedLeague(row) && matchesTarget(row));
      if (targets.length) return finish(targetedEventRows);
      if (targetLeagues.size) return finish(targetedEventRows);
      if (eventRows.length) {
        const seen = new Set();
        return finish(eventRows.filter(row => {
          const key = `${row.home_team.toLowerCase()}|${row.away_team.toLowerCase()}|${row.markets['1']}|${row.markets.X}|${row.markets['2']}`;
          if (seen.has(key)) return false;
          seen.add(key);
          return true;
        }));
      }

      // Unknown layouts are skipped instead of guessing from text order.
      return finish([]);
    }""", {
      "targetFixtures": [
      {"home": _normalise(str(home)).split("|"), "away": _normalise(str(away)).split("|")}
      for home, away in (target_fixtures or set())
      ],
      "targetLeagueNames": sorted(target_league_names or ()),
      "previousLeagueName": previous_league_name,
      "previousDayLabel": previous_day_label,
    })
    return (result["rows"], str(result.get("lastLeagueName") or previous_league_name),
            str(result.get("lastDayLabel") or previous_day_label))


def _collect_all_rendered_rows(page: Any, *, max_scrolls: int = 80,
                               target_fixtures: set[tuple[str, str]] | None = None,
                               target_league_names: set[str] | None = None) -> list[dict[str, Any]]:
    """Collect quotes while scrolling Nesine's virtualized event list."""
    # Nesine locks document scrolling while its welcome/help layer is mounted.
    # Removing the scroll lock affects only this temporary browser page and lets
    # the site render the next batch of event cards as we move through the list.
    page.evaluate("""() => {
      document.body.classList.remove('stop-scrolling');
      document.body.style.overflow = 'auto';
      document.documentElement.style.overflow = 'auto';
    }""")
    rows_by_fixture: dict[tuple[str, str, str], dict[str, Any]] = {}
    previous_state: tuple[int, int] | None = None
    stagnant_bottom_steps = 0
    current_league_name = ""
    current_day_label = ""
    for _ in range(max_scrolls):
        page_rows, current_league_name, current_day_label = _extract_rows_from_page(
            page, target_fixtures, target_league_names, current_league_name, current_day_label
        )
        for row in page_rows:
            if not isinstance(row, Mapping):
                continue
            key = (_normalise_league_name(str(row.get("league_name", ""))),
                   _normalise(str(row.get("home_team", ""))),
                   _normalise(str(row.get("away_team", ""))))
            if all(key):
                rows_by_fixture[key] = row

        state = page.evaluate("""() => ({
          y: Math.round(window.scrollY),
          height: Math.round(document.body.scrollHeight),
          viewport: Math.round(window.innerHeight)
        })""")
        current_state = (state["y"], state["height"])
        at_bottom = state["y"] + state["viewport"] >= state["height"] - 5
        if at_bottom and current_state == previous_state:
            stagnant_bottom_steps += 1
        else:
            stagnant_bottom_steps = 0
        if stagnant_bottom_steps >= 3:
            break
        previous_state = current_state
        page.evaluate("window.scrollBy(0, Math.max(400, window.innerHeight * 0.75))")
        page.wait_for_timeout(350)
    return list(rows_by_fixture.values())


def _normalise_market_label(value: str) -> str:
    text = unicodedata.normalize("NFKD", value.casefold())
    text = "".join(char for char in text if not unicodedata.combining(char)).replace("ı", "i")
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _market_slug(value: str) -> str:
    return _normalise_market_label(value).replace(" ", "_")[:80] or "unknown"


def _is_requested_nesine_market(value: str, selection_name: str = "") -> bool:
    """Keep only the match, half, goal, team-total and corner markets requested."""
    name = _normalise_market_label(value)
    selection = _normalise_market_label(selection_name)
    # A half-time/full-time selection such as "1/1" is not a standalone
    # half-result price. Keep it out of the requested 1st-half result market.
    if name.startswith(("ilk yari mac sonucu", "1 yari mac sonucu")) and re.search(
        r"(?:^| )[1x2] [1x2](?: |$)", selection
    ):
        return False
    # Nesine labels the requested market as either "1. Yarı Maç Sonucu" or
    # "1. Yarı Sonucu" depending on the bulletin section.
    if name.startswith(("ilk yari mac sonucu", "1 yari mac sonucu")):
        return True
    if name in {
        "mac sonucu", "cifte sans", "ilk yari sonucu", "ikinci yari sonucu",
        "1 yari sonucu", "2 yari sonucu", "1 yarisi sonucu", "2 yarisi sonucu",
        "1 yari mac sonucu", "2 yari mac sonucu", "ilk yari mac sonucu",
        "karsilikli gol", "mac sonucu ve karsilikli gol",
        "1 yarisi 2 yarisi alt ust", "ev sahibi 1 yarisi 2 yarisi alt ust",
        "deplasman 1 yarisi 2 yarisi alt ust",
    }:
        return True
    if re.fullmatch(r"mac sonucu ve \d+ 5 alt ust", name):
        return True
    if re.fullmatch(r"[12] yari \d+ 5 (?:gol )?alt ust", name):
        return True
    if re.fullmatch(r"\d+ 5 (?:gol )?alt ust", name):
        return True
    if re.fullmatch(r"(ev sahibi|deplasman)(?: [12] (?:y|yari))? \d+ 5 (?:gol )?alt ust", name):
        return True
    if re.fullmatch(r"(?:ev sahibi |deplasman )?[12] yari [12] yari alt ust", name):
        return True
    return "korner" in name and "alt ust" in name


def _extract_requested_market_rows(page: Any, event_id: str) -> list[dict[str, str]]:
    """Read selections from Nesine's expanded all-markets panel."""
    raw = page.evaluate("""eventId => {
      const root = document.querySelector(`[data-test-id="${eventId}-all"]`);
      if (!root) return [];
      const output = [];
      // Read semantic attributes, not ordinal positions, so unavailable
      // selections cannot shift odds across rows. Some bulletin versions
      // virtualize the expanded panel, so harvest as its rendered rows change.
      const marketViewport = root.querySelector('[class*="MarketList"], [class*="marketList"], [class*="scroll"]') || root;
      let previousSignature = '';
      let stagnantPasses = 0;
      for (let pass = 0; pass < 80; pass++) {
        for (const selection of root.querySelectorAll('[data-mid][data-test-title][data-test-value]')) {
          let market = selection.parentElement;
          while (market && !market.querySelector('[data-test-m-id]')) market = market.parentElement;
          const marker = market?.querySelector('[data-test-m-id]');
          const heading = marker?.parentElement?.querySelector('span span') || marker?.parentElement;
          output.push({market_name: (heading?.innerText || '').trim(),
            selection_name: selection.getAttribute('data-test-title') || '',
            odd: selection.getAttribute('data-test-value') || ''});
        }
        const current = [...root.querySelectorAll('[data-mid][data-test-title][data-test-value]')]
          .map(node => `${node.getAttribute('data-mid')}|${node.getAttribute('data-test-title')}|${node.getAttribute('data-test-value')}`)
          .join('~');
        stagnantPasses = current === previousSignature ? stagnantPasses + 1 : 0;
        if (stagnantPasses >= 2) break;
        previousSignature = current;
        const scrollers = [marketViewport, ...root.querySelectorAll('*')]
          .filter(node => node.scrollHeight > node.clientHeight + 20);
        const scroller = scrollers.sort((a, b) => b.scrollHeight - a.scrollHeight)[0];
        if (!scroller) break;
        const before = scroller.scrollTop;
        scroller.scrollTop = before + Math.max(450, scroller.clientHeight * 0.75);
        if (scroller.scrollTop === before) {
          const last = scroller.scrollHeight - scroller.clientHeight;
          if (before <= 0) scroller.scrollTop = last;
          else if (before >= last) scroller.scrollTop = 0;
          else break;
          if (scroller.scrollTop === before) break;
        }
      }
      // Remove duplicates created by overlapping virtualized viewports.
      const seen = new Set();
      return output.filter(row => {
        const key = `${row.market_name}|${row.selection_name}|${row.odd}`;
        if (seen.has(key)) return false;
        seen.add(key);
        return true;
      });
    }""", str(event_id))
    return [
        {"market_code": _market_slug(row["market_name"]), "market_name": row["market_name"],
         "selection_code": _market_slug(row["selection_name"]), "selection_name": row["selection_name"],
         "odd": str(_odd(row["odd"]))}
        for row in raw
        if _is_requested_nesine_market(
            str(row.get("market_name", "")), str(row.get("selection_name", ""))
        )
        and row.get("selection_name") and _odd(str(row.get("odd", "")))
    ]


def collect_requested_markets_from_nesine(url: str, event_ids: Iterable[str]) -> dict[str, tuple[dict[str, str], ...]]:
    """Open each matched event's market-count panel and read the requested odds."""
    wanted_ids = list(dict.fromkeys(str(event_id) for event_id in event_ids if str(event_id).strip()))
    if not wanted_ids:
        return {}
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError("Playwright is required for Nesine collection") from exc

    collected: dict[str, tuple[dict[str, str], ...]] = {}
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            page = browser.new_page(locale="tr-TR")
            page.goto(url, wait_until="domcontentloaded", timeout=45_000)
            page.wait_for_timeout(3_000)
            page.evaluate("""() => {
              document.body.classList.remove('stop-scrolling');
              document.body.style.overflow = 'auto';
              document.documentElement.style.overflow = 'auto';
            }""")
            for index, event_id in enumerate(wanted_ids):
                try:
                    row_selector = f'[data-test-id="r_{event_id}"]'
                    count_selector = f'[data-test-id="{event_id}_m"]'
                    # The home page may initially select another sport. Click
                    # its visible football tab first, just as a user would.
                    football = page.get_by_text("Futbol", exact=True).first
                    if football.count():
                        football.evaluate("element => element.click()")
                        page.wait_for_timeout(800)
                    for _ in range(120):
                        if page.locator(row_selector).count() and page.locator(count_selector).count():
                            break
                        page.evaluate("window.scrollBy(0, Math.max(400, window.innerHeight * 0.75))")
                        page.wait_for_timeout(200)
                    if not page.locator(count_selector).count():
                        LOGGER.warning("Nesine detailed markets unavailable for event %s", event_id)
                        continue
                    counter = page.locator(count_selector).locator("span.a4cc134d0a3a892deaa6")
                    (counter if counter.count() else page.locator(count_selector)).evaluate("element => element.click()")
                    page.wait_for_timeout(1_500)
                    markets = _extract_requested_market_rows(page, event_id)
                    if not markets:
                        # Some fixtures expand directly in the row without an
                        # event-wide wrapper; extraction also checks document.
                        page.wait_for_timeout(800)
                        markets = _extract_requested_market_rows(page, event_id)
                    if markets:
                        collected[event_id] = tuple(markets)
                    LOGGER.info("Nesine detail event %s: %s selected-market odds", event_id, len(markets))
                except Exception as exc:
                    LOGGER.warning("Nesine detail skipped for event %s: %s", event_id, type(exc).__name__)
        finally:
            browser.close()
    return collected


def _extract_visible_market_rows(page: Any, home_team: str, away_team: str) -> list[dict[str, str]]:
    """Read visible market/selection/odd triplets from a match detail view."""
    return page.evaluate("""() => {
      const odd = /^([1-9]\\d?)[.,]\\d{1,3}$/;
      const clean = value => (value || '').replace(/\\s+/g, ' ').trim();
      const marketHints = /(maç sonucu|ilk yarı|toplam gol|alt\\s*\\/\\s*üst|kg|karşılıklı|korner|kart|handikap|çifte şans|gol aralığı|oyuncu)/i;
      const nodes = [...document.querySelectorAll('tr, li, [role="row"], [class*="market"], [class*="coupon"]')];
      return nodes.flatMap(node => {
        const cells = [...node.querySelectorAll('th, td, [role="cell"], button, a, span')]
          .map(item => clean(item.innerText)).filter(Boolean);
        const text = clean(node.innerText);
        const values = cells.filter(value => odd.test(value));
        if (!values.length || !marketHints.test(text)) return [];
        const market = cells.find(value => marketHints.test(value)) || text.slice(0, 120);
        return values.map((value, index) => ({
          market_code: market.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_|_$/g, '').slice(0, 80) || 'unknown',
          market_name: market,
          selection_code: String(index + 1),
          selection_name: cells[Math.max(0, cells.indexOf(value) - 1)] || String(index + 1),
          odd: value
        }));
      });
    }""")


def _collect_detail_markets(page: Any, base_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Open match details where possible and attach all visible market rows."""
    collected: list[dict[str, Any]] = []
    for row in base_rows:
        home, away = row["home_team"], row["away_team"]
        try:
            locator = page.locator("[class*='OCRow_'], tr, li, [role='row']").filter(has_text=home).filter(has_text=away).first
            if locator.count() == 0:
                collected.append(row)
                continue
            locator.click(timeout=3_000)
            page.wait_for_timeout(400)
            detailed = _extract_visible_market_rows(page, home, away)
            collected.append({**row, "all_markets": detailed})
            page.go_back(wait_until="domcontentloaded", timeout=10_000)
            page.wait_for_timeout(300)
        except Exception as exc:
            LOGGER.warning("Nesine market detail skipped for visible row: %s", type(exc).__name__)
            collected.append(row)
    return collected


def collect_from_nesine_page(url: str = DEFAULT_NESINE_URL, *, headless: bool = True,
                             target_fixtures: set[tuple[str, str]] | None = None,
                             target_league_names: set[str] | None = None) -> list[NesineMatchQuote]:
    """Open Nesine and return rendered, parseable quotes.

    Playwright is imported lazily so the rest of the pipeline remains usable in
    environments where browser binaries are not installed.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError("Playwright is required for Nesine collection") from exc
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=headless)
        try:
            page = browser.new_page(locale="tr-TR")
            page.goto(url, wait_until="domcontentloaded", timeout=45_000)
            page.wait_for_timeout(4_000)
            # The page virtualizes its event list, so a single DOM read only
            # sees the first screenful. Scroll through the feed and aggregate
            # every fixture before parsing/storing its 1/X/2 market.
            rows = _collect_all_rendered_rows(
                page, target_fixtures=target_fixtures,
                target_league_names=target_league_names,
            )
            today_rows = [row for row in rows if _is_nesine_today_label(row.get("day_label"))]
            return parse_rendered_text_rows(today_rows)
        finally:
            browser.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows-json", help="JSON file containing rendered Nesine rows")
    parser.add_argument("--url", default=DEFAULT_NESINE_URL)
    parser.add_argument("--headed", action="store_true", help="Show Chromium while collecting")
    parser.add_argument("--threshold", type=float, default=.82)
    args = parser.parse_args()
    settings = get_settings()
    db = SupabaseRestClient(settings.supabase_url, settings.supabase_service_role_key)
    local_today = datetime.now(ISTANBUL).date()
    api = ApiFootballClient(settings.api_football_key)
    # Read the Nesine bulletin first. Its league headings limit browser-side
    # extraction to competitions configured in this app; API-Football is used
    # afterwards only to attach stable fixture IDs and the app's match data.
    if args.rows_json:
        with open(args.rows_json, encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        payload = [
            {"home_team": item.home_team, "away_team": item.away_team,
             "kickoff_text": item.kickoff_text, "event_id": item.event_id,
             "league_name": item.league_name, "markets": item.snapshot()}
            for item in collect_from_nesine_page(
                args.url, headless=not args.headed,
                target_league_names=NESINE_TRACKED_LEAGUE_NAMES,
            )
        ]
    quotes = parse_rendered_text_rows(payload)
    # Complete Nesine's fixture/odds phase before querying API-Football.
    # The bulletin's first pass already contains the common result markets;
    # this pass loads the selected detailed markets for every today's fixture.
    detailed = {}
    if not args.rows_json:
        event_ids = [quote.event_id for quote in quotes if quote.event_id]
        detailed = collect_requested_markets_from_nesine(args.url, event_ids)
        quotes = [replace(quote, markets=detailed.get(quote.event_id or "", quote.all_markets()))
                  for quote in quotes]
    api_fixtures = [
        item for item in api.get(
            "fixtures", {"date": local_today.isoformat(), "timezone": "Europe/Istanbul"}
        )
        if int(item["league"]["id"]) in TRACKED_LEAGUE_IDS
    ]
    api_matches = [
        {"id": int(item["fixture"]["id"]),
         "league_id": int(item["league"]["id"]),
         "home_team": str(item["teams"]["home"]["name"]),
         "away_team": str(item["teams"]["away"]["name"]),
         "home_team_id": int(item["teams"]["home"]["id"]),
         "away_team_id": int(item["teams"]["away"]["id"]),
         "kickoff": str(item["fixture"]["date"])}
        for item in api_fixtures
    ]
    bulletin_quote_count = len(quotes)
    quotes, out_of_slate_quotes = _partition_quotes_to_api_slate(quotes, api_matches)
    # Nesine is authoritative for which fixtures are offered today. API-Football
    # supplies stable fixture/team/league IDs and the match data used by the app.
    matched = match_quotes(quotes, api_matches, args.threshold)
    selected_ids = {match_id for match_id, _quote, _score in matched}
    selected_fixtures = [
        item for item in api_fixtures if int(item["fixture"]["id"]) in selected_ids
    ]
    if selected_fixtures:
        leagues = {
            int(item["league"]["id"]): {
                "id": int(item["league"]["id"]),
                "name": str(item["league"]["name"]),
                "country": str(item["league"].get("country") or ""),
                "season": int(item["league"]["season"]),
                "is_active": True,
            }
            for item in selected_fixtures
        }
        teams, matches = transform_fixtures(selected_fixtures)
        db.upsert("leagues", list(leagues.values()), on_conflict="id")
        db.upsert("teams", teams, on_conflict="id")
        db.upsert("matches", matches, on_conflict="id")
    stored = store_nesine_quotes(db, matched)
    market_count = sum(len(quote.all_markets()) for _match_id, quote, _score in matched)
    matched_quotes = {id(quote) for _match_id, quote, _score in matched}
    unmatched_samples = []
    for quote in (quote for quote in quotes if id(quote) not in matched_quotes):
        quote_leagues = _tracked_league_ids_for_nesine_name(quote.league_name)
        league_candidates = [
            match for match in api_matches
            if not quote_leagues or int(match.get("league_id", -1)) in quote_leagues
        ]
        candidates = sorted(((match_score(quote, match), match) for match in league_candidates),
                            key=lambda item: item[0], reverse=True)
        best = candidates[0] if candidates else None
        best_league_matches = [
            match for match in league_candidates
            if best and int(match.get("league_id", -1)) == int(best[1].get("league_id", -2))
            and match_score(quote, match) >= 0.75
        ]
        kickoff_candidates = [
            match for match in best_league_matches
            if _quote_kickoff_matches(quote, match)
        ]
        unmatched_samples.append({
            "nesine_league": quote.league_name,
            "api_league_id": next(iter(quote_leagues)) if len(quote_leagues) == 1 else None,
            "nesine": f"{quote.home_team} - {quote.away_team}",
            "best_candidate": (f"{best[1].get('home_team')} - {best[1].get('away_team')}" if best else None),
            "candidate_kickoff": best[1].get("kickoff") if best else None,
            "nesine_kickoff": quote.kickoff_text,
            "close_candidate_count_same_league": len(best_league_matches),
            "close_candidate_count_same_kickoff": len(kickoff_candidates),
            "best_candidate_team_ids": (
                [best[1].get("home_team_id"), best[1].get("away_team_id")] if best else None
            ),
            "score": round(best[0], 3) if best else None,
        })
    report = {"nesine_tracked_league_fixtures": len(quotes),
              "nesine_bulletin_rows": bulletin_quote_count,
              "out_of_slate_rows_ignored": len(out_of_slate_quotes),
              "out_of_slate_samples": [
                  f"{quote.home_team} - {quote.away_team} ({quote.kickoff_text})"
                  for quote in out_of_slate_quotes[:10]
              ],
              "api_football_daily_fixtures": len(api_fixtures),
              "matched": len(matched), "fixtures_synced": len(selected_fixtures), "stored": stored,
              "market_selections": market_count, "unmatched_samples": unmatched_samples}
    # Scheduled tasks use Windows PowerShell's legacy cp1254 console encoding;
    # escaped JSON keeps team names such as Žalgiris from crashing after DB writes.
    print(json.dumps(report, ensure_ascii=True))
    if not quotes:
        raise SystemExit(
            "Nesine collector extracted zero rendered fixtures; page layout, consent overlay, "
            "or odds availability prevented collection"
        )
    if not matched:
        raise SystemExit(
            f"Nesine collector extracted {len(quotes)} fixtures but none matched the "
            f"{len(api_matches)} API-Football fixtures for {local_today}; see unmatched_samples"
        )
    if not stored:
        raise SystemExit("Nesine collector matched fixtures but stored zero quotes")
    if len(matched) != len(quotes):
        raise SystemExit(
            f"Nesine fixture identity coverage is incomplete: {len(matched)}/{len(quotes)} "
            "matched. Matched odds were stored safely; unmatched fixtures need a verified "
            "league-scoped team ID alias before analysis can continue."
        )


if __name__ == "__main__":
    main()
