"""Pure calculations for the authenticated personal decision report."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime, timedelta, timezone
from typing import Any


def summarize_decisions(
    rows: Iterable[Mapping[str, Any]], *, now: datetime | None = None
) -> dict[str, Any]:
    current = now or datetime.now(timezone.utc)
    start = current - timedelta(days=7)
    recent = []
    for row in rows:
        created = row.get("created_at")
        try:
            timestamp = datetime.fromisoformat(str(created).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            continue
        if timestamp >= start:
            recent.append(row)
    played = [row for row in recent if row.get("decision") == "played"]
    settled = [row for row in played if row.get("was_correct") is not None]
    correct = [row for row in settled if row.get("was_correct")]
    return {
        "window_start": start,
        "decision_count": len(recent),
        "played_count": len(played),
        "passed_count": sum(row.get("decision") == "passed" for row in recent),
        "settled_count": len(settled),
        "correct_count": len(correct),
        "accuracy": len(correct) / len(settled) if settled else None,
        "recent": recent,
    }


def settle_outcome_decision(
    decision: Mapping[str, Any], match: Mapping[str, Any]
) -> bool | None:
    """Settle supported 1X2 decisions without pretending to settle other markets."""
    home_score = match.get("home_score")
    away_score = match.get("away_score")
    if home_score is None or away_score is None:
        return None
    try:
        home = int(home_score)
        away = int(away_score)
    except (TypeError, ValueError):
        return None
    actual = "1" if home > away else "X" if home == away else "2"
    market = str(decision.get("selected_market") or "")
    legacy_selections = {
        "Ev kazanır": "1", "Beraberlik": "X", "Deplasman kazanır": "2",
        "Maç Sonucu 1": "1", "Maç Sonucu X": "X", "Maç Sonucu 2": "2",
    }
    selected = legacy_selections.get(market)
    return selected == actual if selected else None
