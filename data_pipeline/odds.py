"""Retrieve and normalize the limited bookmaker markets shown in notifications."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping

import numpy as np

from data_pipeline.api_client import ApiFootballClient
from db.db_client import SupabaseRestClient


DEFAULT_BOOKMAKER_ID = 8
DEFAULT_BOOKMAKER_NAME = "Bet365"


@dataclass(frozen=True, slots=True)
class MatchOdds:
    """Pre-match lines for the three markets used by the prediction model."""

    bookmaker: str
    source_updated_at: str | None = None
    home_win: str | None = None
    draw: str | None = None
    away_win: str | None = None
    over_2_5: str | None = None
    under_2_5: str | None = None
    btts_yes: str | None = None
    btts_no: str | None = None

    @property
    def has_any_market(self) -> bool:
        return any(
            (
                self.home_win,
                self.draw,
                self.away_win,
                self.over_2_5,
                self.under_2_5,
                self.btts_yes,
                self.btts_no,
            )
        )

    def as_snapshot(self) -> dict[str, str | None]:
        return {
            "home_win": self.home_win,
            "draw": self.draw,
            "away_win": self.away_win,
            "over_2_5": self.over_2_5,
            "under_2_5": self.under_2_5,
            "btts_yes": self.btts_yes,
            "btts_no": self.btts_no,
        }


def _market_values(bets: list[dict[str, Any]], market_name: str) -> dict[str, str]:
    """Return normalized selection-to-odd values for a named API-Football market."""
    market = next((item for item in bets if item.get("name") == market_name), None)
    if market is None:
        return {}
    return {
        str(value["value"]): str(value["odd"])
        for value in market.get("values", [])
        if value.get("value") is not None and value.get("odd") is not None
    }


def parse_match_odds(payload: list[dict[str, Any]]) -> MatchOdds | None:
    """Parse one filtered /odds response without assuming every market exists."""
    if not payload:
        return None
    bookmakers = payload[0].get("bookmakers", [])
    if not bookmakers:
        return None
    bookmaker = bookmakers[0]
    bets = bookmaker.get("bets", [])
    winner = _market_values(bets, "Match Winner")
    totals = _market_values(bets, "Goals Over/Under")
    btts = _market_values(bets, "Both Teams Score")
    odds = MatchOdds(
        bookmaker=str(bookmaker.get("name") or DEFAULT_BOOKMAKER_NAME),
        source_updated_at=str(payload[0].get("update") or "") or None,
        home_win=winner.get("Home"),
        draw=winner.get("Draw"),
        away_win=winner.get("Away"),
        over_2_5=totals.get("Over 2.5"),
        under_2_5=totals.get("Under 2.5"),
        btts_yes=btts.get("Yes"),
        btts_no=btts.get("No"),
    )
    return odds if odds.has_any_market else None


def fetch_match_odds(api: ApiFootballClient, *, fixture_id: int) -> MatchOdds | None:
    """Fetch the preferred bookmaker's current pre-match odds for one fixture."""
    payload = api.get("odds", {"fixture": fixture_id, "bookmaker": DEFAULT_BOOKMAKER_ID})
    return parse_match_odds(payload)


def record_odds_quote(
    db: SupabaseRestClient, *, match_id: int, odds: MatchOdds, captured_at: str,
    notification_reference: bool = False,
) -> bool:
    """Append only meaningful line changes; preserve the exact alert-time quote."""
    quote = {
        "match_id": match_id,
        "bookmaker": odds.bookmaker,
        "odds": odds.as_snapshot(),
        "source_updated_at": odds.source_updated_at,
        "captured_at": captured_at,
        "is_notification_reference": notification_reference,
    }
    if not notification_reference:
        previous = db.select(
            "odds_quote_history", columns="odds,source_updated_at",
            filters={"match_id": f"eq.{match_id}", "bookmaker": f"eq.{odds.bookmaker}"},
            limit=1, order="captured_at.desc",
        )
        if previous and previous[0].get("odds") == quote["odds"] and previous[0].get("source_updated_at") == quote["source_updated_at"]:
            return False
    db.insert("odds_quote_history", [quote])
    return True


def closing_line_value(entry_odd: object, closing_odd: object) -> float | None:
    """Positive CLV means the quoted price shortened after the reference point."""
    try:
        entry = float(entry_odd)
        closing = float(closing_odd)
    except (TypeError, ValueError):
        return None
    if entry <= 1 or closing <= 1:
        return None
    return entry / closing - 1


def vig_free_market_probabilities(odds: Mapping[str, object]) -> dict[str, float] | None:
    """Convert supported decimal odds to vig-free probabilities."""
    def normalize(keys: tuple[str, ...]) -> dict[str, float] | None:
        try:
            raw = {key: 1.0 / float(odds[key]) for key in keys}
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            return None
        if any(value <= 0 or not np.isfinite(value) for value in raw.values()):
            return None
        total = sum(raw.values())
        return {key: value / total for key, value in raw.items()}

    outcomes = normalize(("home_win", "draw", "away_win"))
    if outcomes is None:
        return None
    totals = normalize(("over_2_5", "under_2_5")) or {}
    btts = normalize(("btts_yes", "btts_no")) or {}
    return {
        "market_implied_home_win": outcomes["home_win"],
        "market_implied_draw": outcomes["draw"],
        "market_implied_away_win": outcomes["away_win"],
        "market_implied_over_2_5": totals.get("over_2_5", 0.5),
        "market_implied_btts": btts.get("btts_yes", 0.5),
    }


def attach_pre_match_odds(
    matches: list[dict[str, Any]], quotes: list[dict[str, Any]], *, observed_at: datetime | None = None,
) -> list[dict[str, Any]]:
    """Attach only the newest quote captured strictly before each fixture's kickoff."""
    kickoff_by_match = {
        int(match["id"]): datetime.fromisoformat(str(match["match_date"]).replace("Z", "+00:00"))
        for match in matches
    }
    latest: dict[int, dict[str, Any]] = {}
    for quote in sorted(quotes, key=lambda item: str(item.get("captured_at", ""))):
        if quote.get("match_id") is None or not quote.get("captured_at"):
            continue
        captured = datetime.fromisoformat(str(quote["captured_at"]).replace("Z", "+00:00"))
        if (
            int(quote["match_id"]) in kickoff_by_match
            and captured < kickoff_by_match[int(quote["match_id"])]
            and (observed_at is None or captured <= observed_at)
        ):
            latest[int(quote["match_id"])] = quote

    enriched: list[dict[str, Any]] = []
    for match in matches:
        row = dict(match)
        quote = latest.get(int(row["id"]))
        if quote:
            row["market_odds"] = quote.get("odds") or {}
            row["market_captured_at"] = quote["captured_at"]
        enriched.append(row)
    return enriched
