"""Retrieve and normalize the limited bookmaker markets shown in notifications."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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
