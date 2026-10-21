"""Import today's IdDaa.com football bulletin and all available markets."""

from __future__ import annotations

import argparse
import logging
import re
import time
import unicodedata
from difflib import SequenceMatcher
from datetime import date, datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

import requests

from config.leagues import TRACKED_LEAGUES
from config.settings import get_settings
from db.db_client import SupabaseRestClient

BASE_URL = "https://sportsbookv2.iddaa.com/sportsbook"
ISTANBUL = ZoneInfo("Europe/Istanbul")
LOG = logging.getLogger("iddaa_odds")

# IdDaa exposes the corresponding API-Football ID in `cref` for most leagues;
# its own German/Belgian competition IDs are included as fallbacks.
IDDAA_COMPETITION_REFS: dict[int, int] = {
    39: 1,
    140: 36,
    135: 33,
    78: 8,
    61: 4,
    40: 2,
    94: 52,
    71: 83,
    203: 62,
    88: 39,
    253: 18,
    144: 22,
    307: 3708,
    235: 53,
    128: 170677,
    # Lower divisions and Mexico are present in TRACKED_LEAGUES and in the
    # daily match screen. Include their IdDaa competitions so eligible events
    # are not silently skipped by the odds importer.
    262: 28,  # Liga MX: IdDaa Meksika Primera Ligi
    141: 37,  # Segunda División: IdDaa İspanya La Liga 2
    62: 19,  # Ligue 2: IdDaa Fransa Ligue 2
}
REQUIRED_IDDAA_LEAGUES = set(IDDAA_COMPETITION_REFS)


def normalize(value: str) -> str:
    value = unicodedata.normalize("NFKD", value.casefold())
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


TEAM_SUFFIXES = {
    "fc", "cf", "sc", "ac", "afc", "club", "football", "futebol",
    "fbpa", "esporte", "clube", "de", "do", "da", "the", "rb",
    "sp", "mg", "ba", "rs", "sa", "k",
}


def normalize_team(value: str) -> str:
    """Remove diacritics and common club suffixes while keeping core name."""
    tokens = normalize(value).split()
    tokens = [token for token in tokens if token not in TEAM_SUFFIXES]
    return " ".join(tokens)


TEAM_ALIASES: dict[str, tuple[str, ...]] = {
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
        if league_id is None and own_competition_id in {1220, 1219, 1221}:
            league_id = 144
        # IdDaa's current Champions League catalogue no longer exposes the
        # API-Football league ID as cref. Match its exact football competition
        # name so the daily bulletin does not silently skip those fixtures.
        if league_id is None and normalize(str(competition.get("n", ""))) == "uefa sampiyonlar ligi":
            league_id = 2
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
) -> dict[str, int]:
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

    matches = db.select_all(
        "matches",
        columns="id,league_id,home_team_id,away_team_id,match_date,status",
        order="match_date.asc,id.asc",
    )
    teams = db.select_all("teams", columns="id,name", order="id.asc")
    team_names = {int(row["id"]): str(row["name"]) for row in teams}
    index: dict[tuple[int, date], list[dict[str, Any]]] = {}
    for match in matches:
        if int(match["league_id"]) not in target_competitions.values():
            continue
        if match.get("status") not in ("scheduled", "live"):
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
    selected_events = 0
    selected_market_events: list[dict[str, Any]] = []
    unmapped_competitions: set[int] = set()
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
            same_day_league = [
                row
                for row in matches
                if int(row["league_id"]) == league_id
                and row.get("status") in ("scheduled", "live")
                and _date(str(row["match_date"])) == kickoff.date()
            ]
            team_mismatch_candidates = []
            for row in same_day_league:
                db_home = team_names.get(int(row["home_team_id"]), "")
                db_away = team_names.get(int(row["away_team_id"]), "")
                if team_names_match(db_home, home) or team_names_match(db_home, away) or team_names_match(db_away, home) or team_names_match(db_away, away):
                    team_mismatch_candidates.append(
                        {"match_id": row["id"], "home": db_home, "away": db_away}
                    )
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
        records.append(
            {
                "match_id": int(target["id"]),
                "bookmaker": "iddaa.com",
                "odds": match_markets(event, labels),
                "source_updated_at": now,
                "captured_at": now,
            }
        )

    if not target_competitions:
        raise RuntimeError("IdDaa competitions response did not match any configured target leagues")
    if selected_events and not records:
        raise RuntimeError(
            f"IdDaa returned {selected_events} target-league events for {match_day}, but none matched app fixtures"
        )
    if records:
        db.insert("odds_quote_history", records)
    return {
        "competitions_matched": len(target_competitions),
        "events_for_day": selected_events,
        "quotes_saved": len(records),
        "events_unmatched": unmatched,
        "markets": summarize_markets(selected_market_events, labels),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", help="Istanbul calendar date (YYYY-MM-DD); defaults to today")
    parser.add_argument("--retries", type=int, default=3, help="Attempts before reporting failure")
    args = parser.parse_args()
    if args.retries < 1 or args.retries > 5:
        parser.error("--retries must be between 1 and 5")
    match_day = date.fromisoformat(args.date) if args.date else datetime.now(ISTANBUL).date()
    settings = get_settings()
    db = SupabaseRestClient(settings.supabase_url, settings.supabase_service_role_key)

    for attempt in range(1, args.retries + 1):
        try:
            summary = import_daily_odds(db, match_day=match_day)
            print({"date": match_day.isoformat(), "iddaa_odds": summary})
            return
        except Exception:
            LOG.exception("IdDaa odds import failed (attempt %s/%s)", attempt, args.retries)
            if attempt == args.retries:
                raise
            time.sleep(30 * attempt)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
