from __future__ import annotations

from app.components.commentary import summarize_absences, summarize_form
from models.feature_engineering import MatchResult, TeamState


def test_form_summary_uses_only_recent_completed_results() -> None:
    state = TeamState()
    state.results.extend([
        MatchResult(2, 0, 3, True),
        MatchResult(1, 1, 1, False),
        MatchResult(0, 2, 0, True),
    ])

    assert summarize_form(state) == "Son 3 maç: 1G 1B 1M; maç başına gol 1.00, yenilen gol 1.00"


def test_absence_summary_excludes_other_teams_and_unknown_states() -> None:
    rows = [
        {"team_id": 10, "player_name": "Oyuncu A", "status": "injured"},
        {"team_id": 10, "player_name": "Oyuncu B", "status": "suspended"},
        {"team_id": 20, "player_name": "Oyuncu C", "status": "injured"},
        {"team_id": 10, "player_name": "Oyuncu D", "status": "available"},
    ]

    assert summarize_absences(rows, team_id=10) == [
        "Oyuncu A (sakat)",
        "Oyuncu B (cezalı)",
    ]
