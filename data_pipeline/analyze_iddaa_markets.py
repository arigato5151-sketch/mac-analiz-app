"""Summarize today's stored IdDaa markets using odds-implied probabilities."""

from __future__ import annotations

import json
import argparse
import logging
from collections import defaultdict
from datetime import date, datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

import requests

from config.settings import get_settings
from data_pipeline.fetch_iddaa_odds import fetch_bulletin
from db.db_client import SupabaseRestClient

ISTANBUL = ZoneInfo("Europe/Istanbul")


def _as_market_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            loaded = json.loads(value)
            return loaded if isinstance(loaded, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _day(value: str) -> date:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(ISTANBUL).date()


def analyze_markets(
    quotes: list[dict[str, Any]],
    matches: list[dict[str, Any]],
    teams: list[dict[str, Any]],
    *,
    match_day: date,
    labels: dict[tuple[int, int], str] | None = None,
    source_market_groups: dict[tuple[int, int], int] | None = None,
    include_match_details: bool = False,
) -> dict[str, Any]:
    teams_by_id = {int(row["id"]): str(row["name"]) for row in teams}
    matches_by_id = {int(row["id"]): row for row in matches}
    latest_quote: dict[int, dict[str, Any]] = {}
    for row in sorted(quotes, key=lambda item: str(item.get("captured_at", ""))):
        match = matches_by_id.get(int(row["match_id"]))
        if not match or _day(str(match["match_date"])) != match_day:
            continue
        latest_quote[int(row["match_id"])] = row

    aggregate: dict[tuple[int, str, str, str, tuple[str, ...]], dict[str, Any]] = defaultdict(
        lambda: {"events": set(), "overrounds": [], "selections": defaultdict(list), "lines": set()}
    )
    labels_by_code: dict[int, set[str]] = defaultdict(set)
    for (_, market_code), label in (labels or {}).items():
        labels_by_code[market_code].add(label)
    per_match: dict[str, list[dict[str, Any]]] = {}
    for match_id, quote in latest_quote.items():
        match = matches_by_id[match_id]
        home = teams_by_id.get(int(match["home_team_id"]), f"Takım {match['home_team_id']}")
        away = teams_by_id.get(int(match["away_team_id"]), f"Takım {match['away_team_id']}")
        payload = _as_market_payload(quote.get("odds"))
        rows: list[dict[str, Any]] = []
        for market in payload.get("markets", []):
            valid = []
            incomplete_prices = False
            for selection in market.get("selections", []):
                try:
                    odds = float(selection.get("odds"))
                except (TypeError, ValueError):
                    incomplete_prices = True
                    continue
                if odds > 1.0:
                    valid.append((selection, odds))
                else:
                    incomplete_prices = True
            if len(valid) < 2 or incomplete_prices:
                continue
            raw_probs = [1 / odds for _, odds in valid]
            total = sum(raw_probs)
            fair_probs = [prob / total for prob in raw_probs]
            market_id = str(market.get("market_id", "unknown"))
            try:
                market_code = int(market_id)
                event_id = int(payload.get("event_id", -1))
            except (TypeError, ValueError):
                event_id, market_code = -1, -1
            sport_type = (source_market_groups or {}).get(
                (event_id, market_code), int(market.get("sport_type", 4))
            )
            exact_label = (labels or {}).get((sport_type, market_code))
            if not exact_label and len(labels_by_code.get(market_code, set())) == 1:
                exact_label = next(iter(labels_by_code[market_code]))
            market_label = str(exact_label or "")
            selection_names = tuple(sorted(str(item.get("name") or item.get("id") or "?") for item, _ in valid))
            line = str(market.get("line") or "")
            key = (sport_type, market_id, market_label, line, selection_names)
            bucket = aggregate[key]
            bucket["events"].add(match_id)
            bucket["overrounds"].append(total - 1)
            if line:
                bucket["lines"].add(line)
            selections = []
            for (selection, odds), probability in zip(valid, fair_probs):
                label = str(selection.get("name") or selection.get("id") or "?")
                bucket["selections"][label].append(probability)
                selections.append({"selection": label, "odds": odds, "fair_probability": probability})
            rows.append(
                {
                    "market_id": key[0],
                    "market_name": key[1] or None,
                    "line": market.get("line"),
                    "overround": total - 1,
                    "most_likely": max(selections, key=lambda item: item["fair_probability"]),
                    "selections": selections,
                }
            )
        if rows:
            per_match[f"{home} – {away}"] = rows

    summary = []
    for (sport_type, market_id, market_name, line, selection_names), bucket in aggregate.items():
        selection_means = {
            name: round(sum(values) / len(values), 4)
            for name, values in bucket["selections"].items()
        }
        summary.append(
            {
                "market_id": market_id,
                "sport_type": sport_type,
                "market_name": market_name or None,
                "selection_types": list(selection_names),
                "matches": len(bucket["events"]),
                "line": line or None,
                "mean_overround_pct": round(100 * sum(bucket["overrounds"]) / len(bucket["overrounds"]), 2),
                "mean_fair_probability_pct": {
                    name: round(value * 100, 1) for name, value in selection_means.items()
                },
            }
        )
    summary.sort(key=lambda item: (item["sport_type"], item["market_id"], item["line"] or "", ",".join(item["selection_types"])))
    return {
        "date": match_day.isoformat(),
        "matches_with_quotes": len(latest_quote),
        "market_types": summary,
        "by_match": per_match if include_match_details else None,
        "method": "odds-implied probabilities normalized within each market; not an independent model forecast",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--by-match", action="store_true", help="Include the full market breakdown for each match")
    args = parser.parse_args()
    today = datetime.now(ISTANBUL).date()
    settings = get_settings()
    db = SupabaseRestClient(settings.supabase_url, settings.supabase_service_role_key)
    quotes = db.select_all(
        "odds_quote_history",
        columns="id,match_id,bookmaker,odds,captured_at",
        filters={"bookmaker": "eq.iddaa.com"},
        order="captured_at.asc,id.asc",
    )
    matches = db.select_all(
        "matches",
        columns="id,league_id,home_team_id,away_team_id,match_date,status",
        order="match_date.asc,id.asc",
    )
    teams = db.select_all("teams", columns="id,name", order="id.asc")
    labels: dict[tuple[int, int], str] = {}
    source_market_groups: dict[tuple[int, int], int] = {}
    try:
        session = requests.Session()
        session.trust_env = False
        _, events, labels = fetch_bulletin(session)
        for event in events:
            event_id = int(event.get("i", -1))
            for market in event.get("m", []):
                source_market_groups[(event_id, int(market.get("st", -1)))] = int(market.get("t", -1))
    except (requests.RequestException, RuntimeError, ValueError):
        logging.getLogger(__name__).warning("IdDaa market labels unavailable; showing market codes")
    report = analyze_markets(
        quotes,
        matches,
        teams,
        match_day=today,
        labels=labels,
        source_market_groups=source_market_groups,
        include_match_details=args.by_match,
    )
    report["by_match_included"] = args.by_match
    print(json.dumps(report, ensure_ascii=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
