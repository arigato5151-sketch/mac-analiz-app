"""Retrieve and normalize the limited bookmaker markets shown in notifications."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping

import numpy as np

from data_pipeline.api_client import ApiFootballClient
from db.db_client import SupabaseRestClient


PRIMARY_BOOKMAKER_ID = 8
PRIMARY_BOOKMAKER_NAME = "Bet365"
SECONDARY_BOOKMAKER_ID = 4
SECONDARY_BOOKMAKER_NAME = "Pinnacle"
MULTI_BOOKMAKER_IDS = [PRIMARY_BOOKMAKER_ID, SECONDARY_BOOKMAKER_ID]
MULTI_BOOKMAKER_NAMES = {PRIMARY_BOOKMAKER_ID: PRIMARY_BOOKMAKER_NAME, SECONDARY_BOOKMAKER_ID: SECONDARY_BOOKMAKER_NAME}


def _parse_utc_timestamp(value: object, *, field: str) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc)


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


@dataclass(frozen=True, slots=True)
class MultiBookmakerOdds:
    """Odds from multiple bookmakers for multi-bookmaker vig-free blend."""

    bookmakers: dict[int, MatchOdds]  # bookmaker_id -> MatchOdds

    def get_primary(self) -> MatchOdds | None:
        return self.bookmakers.get(PRIMARY_BOOKMAKER_ID)

    def get_secondary(self) -> MatchOdds | None:
        return self.bookmakers.get(SECONDARY_BOOKMAKER_ID)

    def all_odds_snapshots(self) -> list[dict[str, str | None]]:
        return [odds.as_snapshot() for odds in self.bookmakers.values() if odds.has_any_market]

    def has_any_market(self) -> bool:
        return any(odds.has_any_market for odds in self.bookmakers.values())

    # Backward compatibility: allow attribute access to primary bookmaker's odds
    def __getattr__(self, name: str) -> str | None:
        primary = self.get_primary()
        if primary and hasattr(primary, name):
            return getattr(primary, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


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


def parse_match_odds(payload: list[dict[str, Any]]) -> MultiBookmakerOdds | None:
    """Parse one filtered /odds response for configured bookmakers."""
    if not payload:
        return None
    bookmakers = payload[0].get("bookmakers", [])
    if not bookmakers:
        return None

    parsed: dict[int, MatchOdds] = {}
    for bookmaker in bookmakers:
        bm_id = bookmaker.get("id")
        bm_name = bookmaker.get("name", "")
        # Match by ID or by name
        if bm_id not in MULTI_BOOKMAKER_IDS and bm_name not in MULTI_BOOKMAKER_NAMES.values():
            continue
        # If ID missing but name matches, use the ID from name
        if bm_id not in MULTI_BOOKMAKER_IDS:
            bm_id = next((k for k, v in MULTI_BOOKMAKER_NAMES.items() if v == bm_name), None)
            if bm_id is None:
                continue

        bets = bookmaker.get("bets", [])
        winner = _market_values(bets, "Match Winner")
        totals = _market_values(bets, "Goals Over/Under")
        btts = _market_values(bets, "Both Teams Score")
        odds = MatchOdds(
            bookmaker=str(bookmaker.get("name") or MULTI_BOOKMAKER_NAMES.get(bm_id, f"Bookmaker{bm_id}")),
            source_updated_at=str(payload[0].get("update") or "") or None,
            home_win=winner.get("Home"),
            draw=winner.get("Draw"),
            away_win=winner.get("Away"),
            over_2_5=totals.get("Over 2.5"),
            under_2_5=totals.get("Under 2.5"),
            btts_yes=btts.get("Yes"),
            btts_no=btts.get("No"),
        )
        if odds.has_any_market:
            parsed[bm_id] = odds

    return MultiBookmakerOdds(bookmakers=parsed) if parsed else None


def fetch_match_odds(api: ApiFootballClient, *, fixture_id: int) -> MultiBookmakerOdds | None:
    """Fetch configured bookmakers' current pre-match odds for one fixture."""
    payload = api.get("odds", {"fixture": fixture_id})
    return parse_match_odds(payload)


def record_odds_quote(
    db: SupabaseRestClient, *, match_id: int, odds: MultiBookmakerOdds, captured_at: str,
    notification_reference: bool = False,
) -> bool:
    """Append only meaningful line changes for each bookmaker; preserve the exact alert-time quote."""
    any_inserted = False
    for bm_id, odds_bm in odds.bookmakers.items():
        if not odds_bm.has_any_market:
            continue
        quote = {
            "match_id": match_id,
            "bookmaker": odds_bm.bookmaker,
            "bookmaker_id": bm_id,
            "odds": odds_bm.as_snapshot(),
            "source_updated_at": odds_bm.source_updated_at,
            "captured_at": captured_at,
            "is_notification_reference": notification_reference,
        }
        if not notification_reference:
            previous = db.select(
                "odds_quote_history", columns="odds,source_updated_at",
                filters={"match_id": f"eq.{match_id}", "bookmaker_id": f"eq.{bm_id}"},
                limit=1, order="captured_at.desc",
            )
            if previous and previous[0].get("odds") == quote["odds"] and previous[0].get("source_updated_at") == quote["source_updated_at"]:
                continue
        db.insert("odds_quote_history", [quote])
        any_inserted = True
    return any_inserted


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
    """Convert supported decimal odds to vig-free probabilities (single bookmaker)."""
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


def multi_bookmaker_vig_free_probabilities(
    multi_odds: MultiBookmakerOdds,
) -> dict[str, float] | None:
    """Compute inverse-variance weighted vig-free probabilities from multiple bookmakers.

    Each bookmaker's vig-free probabilities are weighted by inverse variance
    (approximated by 1 / vig), so sharper bookmakers (lower vig) get higher weight.
    """
    if not multi_odds.bookmakers:
        return None

    all_probs: list[dict[str, float]] = []
    weights: list[float] = []

    for bm_id, odds in multi_odds.bookmakers.items():
        probs = vig_free_market_probabilities(odds.as_snapshot())
        if probs is None:
            continue
        all_probs.append(probs)
        # Estimate vig from the raw odds to compute weight (lower vig = higher weight)
        vig_weight = 1.0  # fallback
        try:
            raw_odds = odds.as_snapshot()
            # Sum of implied probabilities - 1 = vig
            result_keys = ("home_win", "draw", "away_win")
            total_implied = sum(1.0 / float(raw_odds[k]) for k in result_keys if raw_odds.get(k))
            if total_implied > 1.0:
                vig = total_implied - 1.0
                vig_weight = 1.0 / max(vig, 0.01)  # inverse vig
        except Exception:
            pass
        weights.append(vig_weight)

    if not all_probs:
        return None

    # Weighted average
    total_weight = sum(weights)
    if total_weight <= 0:
        return None

    blended: dict[str, float] = {}
    keys = set()
    for p in all_probs:
        keys.update(p.keys())
    for key in keys:
        values = [p.get(key, 0.0) for p in all_probs]
        if len(values) != len(weights):
            continue
        blended[key] = sum(v * w for v, w in zip(values, weights)) / total_weight

    # Ensure result probabilities sum to 1
    result_keys = ("market_implied_home_win", "market_implied_draw", "market_implied_away_win")
    result_sum = sum(blended.get(k, 0.0) for k in result_keys)
    if result_sum > 0:
        for k in result_keys:
            if k in blended:
                blended[k] /= result_sum

    # Ensure binary markets are in [0, 1]
    for k in ("market_implied_over_2_5", "market_implied_btts"):
        if k in blended:
            blended[k] = float(np.clip(blended[k], 0.0, 1.0))

    return blended


def attach_pre_match_odds(
    matches: list[dict[str, Any]],
    quotes: list[dict[str, Any]],
    *,
    observed_at: datetime | None = None,
    training_lead_minutes: int = 20,
) -> list[dict[str, Any]]:
    """Attach causal opening/current quotes at the intended decision time.

    A quote is eligible only when it was captured (1) at or before the decision
    cutoff (``observed_at`` for live inference, ``kickoff - lead`` for training)
    AND (2) before kickoff. The second clause is a defensive guard: in the live
    path a caller-supplied ``observed_at`` that is later than kickoff must not
    allow a post-kickoff quote into the pre-match feature matrix. With
    ``training_lead_minutes >= 0`` the two conditions coincide.

    Returns matches enriched with both per-bookmaker odds and a multi-bookmaker
    blended vig-free probability set under ``market_odds`` / ``market_opening_odds``.
    """
    if training_lead_minutes < 0:
        raise ValueError("training_lead_minutes must not be negative")
    kickoff_by_match = {
        int(match["id"]): _parse_utc_timestamp(match["match_date"], field="match_date")
        for match in matches
    }
    decision_at = (
        _parse_utc_timestamp(observed_at, field="observed_at")
        if observed_at is not None
        else None
    )
    valid_with_captured_at = [
        quote
        for quote in quotes
        if quote.get("match_id") is not None and quote.get("captured_at")
    ]
    valid_by_match: dict[int, list[dict[str, Any]]] = {}
    for quote in sorted(
        valid_with_captured_at,
        key=lambda item: (
            _parse_utc_timestamp(item["captured_at"], field="captured_at"),
            int(item.get("id", 0)),
        ),
    ):
        try:
            captured = _parse_utc_timestamp(quote["captured_at"], field="captured_at")
            source_updated = (
                _parse_utc_timestamp(
                    quote["source_updated_at"], field="source_updated_at"
                )
                if quote.get("source_updated_at")
                else None
            )
        except (TypeError, ValueError):
            continue
        match_id = int(quote["match_id"])
        if match_id not in kickoff_by_match:
            continue
        cutoff = (
            decision_at
            if decision_at is not None
            else kickoff_by_match[match_id] - timedelta(minutes=training_lead_minutes)
        )
        if (
            captured <= cutoff
            and captured < kickoff_by_match[match_id]
            and (
                source_updated is None
                or (
                    source_updated <= cutoff
                    and source_updated < kickoff_by_match[match_id]
                )
            )
        ):
            valid_by_match.setdefault(match_id, []).append(quote)

    enriched: list[dict[str, Any]] = []
    for match in matches:
        row = dict(match)
        history = valid_by_match.get(int(row["id"]), [])
        if history:
            opening, latest = history[0], history[-1]

            # Per-bookmaker odds (support both legacy flat format and new per-bookmaker format)
            opening_odds = opening.get("odds") or {}
            latest_odds = latest.get("odds") or {}

            def _normalize_odds_snapshot(odds_dict: dict[str, Any]) -> dict[int, MatchOdds]:
                """Convert stored odds snapshot to bookmaker_id -> MatchOdds mapping."""
                if not odds_dict:
                    return {}
                # Detect format: if keys are numeric bookmaker IDs, use new format
                # Otherwise assume legacy flat format with single bookmaker (Bet365)
                first_key = next(iter(odds_dict))
                if isinstance(first_key, (int, str)) and str(first_key).isdigit():
                    # New format: {bookmaker_id: {bookmaker, home_win, ...}}
                    return {
                        int(bm_id): MatchOdds(
                            bookmaker=odds.get("bookmaker", ""),
                            source_updated_at=odds.get("source_updated_at"),
                            home_win=odds.get("home_win"),
                            draw=odds.get("draw"),
                            away_win=odds.get("away_win"),
                            over_2_5=odds.get("over_2_5"),
                            under_2_5=odds.get("under_2_5"),
                            btts_yes=odds.get("btts_yes"),
                            btts_no=odds.get("btts_no"),
                        )
                        for bm_id, odds in odds_dict.items()
                    }
                else:
                    # Legacy flat format: {home_win: "...", draw: "...", ...} -> single bookmaker
                    return {
                        PRIMARY_BOOKMAKER_ID: MatchOdds(
                            bookmaker=PRIMARY_BOOKMAKER_NAME,
                            source_updated_at=None,
                            home_win=odds_dict.get("home_win"),
                            draw=odds_dict.get("draw"),
                            away_win=odds_dict.get("away_win"),
                            over_2_5=odds_dict.get("over_2_5"),
                            under_2_5=odds_dict.get("under_2_5"),
                            btts_yes=odds_dict.get("btts_yes"),
                            btts_no=odds_dict.get("btts_no"),
                        )
                    }

            opening_multi = MultiBookmakerOdds(bookmakers=_normalize_odds_snapshot(opening_odds))
            latest_multi = MultiBookmakerOdds(bookmakers=_normalize_odds_snapshot(latest_odds))

            # Store per-bookmaker snapshots (preserve original format for backward compatibility)
            row["market_opening_odds"] = opening_odds
            row["market_odds"] = latest_odds
            row["market_captured_at"] = latest["captured_at"]

            # Add multi-bookmaker blended vig-free probabilities
            opening_blend = multi_bookmaker_vig_free_probabilities(opening_multi)
            latest_blend = multi_bookmaker_vig_free_probabilities(latest_multi)
            if opening_blend:
                row["market_opening_odds_blended"] = opening_blend
            if latest_blend:
                row["market_odds_blended"] = latest_blend
                # Also expose as market_odds for backward compatibility (flat format from primary bookmaker)
                primary_latest = latest_multi.get_primary()
                if primary_latest and primary_latest.has_any_market:
                    row["market_odds"] = primary_latest.as_snapshot()
        enriched.append(row)
    return enriched
