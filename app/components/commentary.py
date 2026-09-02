"""Prepare verified match context for the Gemini commentary interface."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from models.feature_engineering import TeamState


STATUS_LABELS = {
    "injured": "sakat",
    "suspended": "cezalı",
    "doubtful": "şüpheli",
}


def summarize_form(state: TeamState) -> str:
    """Summarize only completed results already held in the causal feature state."""
    results = list(state.results)[-5:]
    if not results:
        return "Son beş maç için veri yok"
    wins = sum(result.points == 3 for result in results)
    draws = sum(result.points == 1 for result in results)
    losses = len(results) - wins - draws
    goals_for = sum(result.goals_for for result in results) / len(results)
    goals_against = sum(result.goals_against for result in results) / len(results)
    return (
        f"Son {len(results)} maç: {wins}G {draws}B {losses}M; "
        f"maç başına gol {goals_for:.2f}, yenilen gol {goals_against:.2f}"
    )


def summarize_absences(
    rows: Iterable[dict[str, Any]], *, team_id: int
) -> list[str]:
    """Return bounded, provider-confirmed absences for one team."""
    absences: list[str] = []
    for row in rows:
        if int(row.get("team_id", -1)) != team_id:
            continue
        status = STATUS_LABELS.get(str(row.get("status", "")).lower())
        name = str(row.get("player_name", "")).strip()
        if status and name:
            absences.append(f"{name[:100]} ({status})")
    return absences[:12]
