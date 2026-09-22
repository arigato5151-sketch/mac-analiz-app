"""Decision-board helpers: featured signals, countdowns, and honest gaps."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping

import pandas as pd

from app.components.ui import prediction_signal
from app.components.freshness import FRESHNESS_CURRENT, freshness_status
from data_pipeline.value_bets import vig_free_probabilities
from models.decision_policy import minimum_confidence_for_league
from models.market_forecast import MINIMUM_GOAL_MARKET_CONFIDENCE

# Odds-snapshot selection keys per model market, with the sibling selections
# required for the margin-free implied probability.
MARKET_SELECTIONS: dict[str, tuple[str, ...]] = {
    "home_win": ("home_win", "draw", "away_win"),
    "draw": ("home_win", "draw", "away_win"),
    "away_win": ("home_win", "draw", "away_win"),
    "over_2_5": ("over_2_5", "under_2_5"),
    "btts_yes": ("btts_yes", "btts_no"),
}

SELECTION_KEYS = {
    "Ev kazanır": "home_win",
    "Beraberlik": "draw",
    "Deplasman kazanır": "away_win",
    "Üst 2.5": "over_2_5",
    "KG Var": "btts_yes",
}


@dataclass(frozen=True)
class MatchDecision:
    match_id: int
    label: str
    kickoff: datetime
    countdown: str
    market: str
    probability: float | None
    confidence: str
    featured: bool
    prediction_ready: bool
    pas_reason: str | None
    market_gap: str | None
    data_status: str


def countdown_to(kickoff: datetime, now: datetime) -> str:
    """Human-readable time remaining until kickoff."""
    seconds = (kickoff - now).total_seconds()
    if seconds <= 0:
        return "başladı"
    hours, remainder = divmod(seconds, 3600)
    minutes = remainder / 60
    if hours >= 24:
        return f"{hours / 24:.0f} gün sonra"
    if hours >= 1:
        return f"{hours:.0f} sa {minutes:.0f} dk sonra"
    return f"{minutes:.0f} dk sonra"


def market_gap_text(
    market: str, probability: float | None, odds: Mapping[str, Any] | None
) -> str | None:
    """Model-vs-market gap for the featured selection; honest, no value claim."""
    if probability is None or not odds:
        return None
    key = SELECTION_KEYS.get(market)
    if key is None:
        return None
    implied = vig_free_probabilities(odds, MARKET_SELECTIONS[key])
    if implied is None:
        return None
    gap = probability - implied[key]
    return f"Model %{probability * 100:.1f} · piyasa %{implied[key] * 100:.1f} ({gap * 100:+.1f} puan)"


def requested_match_id(
    query: Mapping[str, Any], matches: pd.DataFrame
) -> int | None:
    """Resolve a deep-linked match id; unknown or missing ids never select."""
    raw = query.get("match_id")
    if raw is None or matches.empty or "id" not in matches:
        return None
    try:
        match_id = int(str(raw))
    except (TypeError, ValueError):
        return None
    ids = set(matches["id"].astype(int))
    return match_id if match_id in ids else None


def _published_threshold(market: str, league_id: object) -> float:
    if market in {"Ev kazanır", "Beraberlik", "Deplasman kazanır"}:
        try:
            parsed_league_id = int(league_id)
        except (TypeError, ValueError):
            parsed_league_id = None
        return minimum_confidence_for_league(parsed_league_id)
    return MINIMUM_GOAL_MARKET_CONFIDENCE


def _current_odds(
    quote: Mapping[str, Any] | None, *, now: datetime
) -> Mapping[str, Any] | None:
    if not quote:
        return None
    odds = quote.get("odds")
    captured_at = quote.get("captured_at")
    if not isinstance(odds, Mapping) or captured_at is None:
        return None
    if isinstance(captured_at, pd.Timestamp):
        captured_at = captured_at.to_pydatetime()
    elif not isinstance(captured_at, datetime):
        try:
            captured_at = datetime.fromisoformat(
                str(captured_at).replace("Z", "+00:00")
            )
        except ValueError:
            return None
    if freshness_status(captured_at, source="odds", now=now) != FRESHNESS_CURRENT:
        return None
    return odds


def build_match_decisions(
    matches: pd.DataFrame,
    odds_by_match: Mapping[int, Mapping[str, Any]] | None = None,
    *,
    now: datetime,
) -> list[MatchDecision]:
    """One decision row per match: signal, confidence, countdown and gaps."""
    decisions: list[MatchDecision] = []
    for _, row in matches.iterrows():
        market, probability, confidence = prediction_signal(row)
        prediction_ready = probability is not None
        kickoff = row["match_date"].to_pydatetime()
        match_id = int(row["id"])
        label = f"{row['home_team']} — {row['away_team']}"
        threshold = _published_threshold(market, row.get("league_id"))
        if not prediction_ready:
            pas_reason = "Model tahmini henüz hazırlanmadı"
        elif probability < threshold:
            pas_reason = (
                f"{market} güveni yayın eşiğinin altında "
                f"(%{probability * 100:.1f} < %{threshold * 100:.0f})"
            )
        else:
            pas_reason = None
        current_odds = _current_odds((odds_by_match or {}).get(match_id), now=now)
        market_gap = (
            market_gap_text(market, probability, current_odds)
            if prediction_ready and current_odds
            else None
        )
        decisions.append(
            MatchDecision(
                match_id=match_id,
                label=label,
                kickoff=kickoff,
                countdown=countdown_to(kickoff, now),
                market=market,
                probability=probability,
                confidence=confidence,
                featured=prediction_ready and pas_reason is None,
                prediction_ready=prediction_ready,
                pas_reason=pas_reason,
                market_gap=market_gap,
                data_status=(
                    "Tahmin ve güncel oran hazır"
                    if current_odds
                    else "Tahmin hazır · güncel oran yok"
                    if prediction_ready
                    else "Tahmin bekleniyor"
                ),
            )
        )
    return decisions
