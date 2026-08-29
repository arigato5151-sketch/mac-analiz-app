from __future__ import annotations

from datetime import datetime, timezone

from notifications.pre_match import due_matches, pre_match_message


def test_due_matches_uses_the_45_to_75_minute_window() -> None:
    now = datetime(2026, 8, 29, 12, 0, tzinfo=timezone.utc)
    matches = [
        {"id": 1, "match_date": "2026-08-29T12:44:00+00:00"},
        {"id": 2, "match_date": "2026-08-29T13:00:00+00:00"},
        {"id": 3, "match_date": "2026-08-29T13:16:00+00:00"},
    ]

    assert [row["id"] for row in due_matches(matches, now=now)] == [2]


def test_pre_match_message_shows_every_market() -> None:
    message = pre_match_message(
        {"match_date": "2026-08-29T16:00:00+00:00"},
        {"prob_home_win": 0.6, "prob_draw": 0.2, "prob_away_win": 0.2, "prob_over_2_5": 0.55, "prob_btts": 0.45},
        home_team="Ev",
        away_team="Deplasman",
        league_name="Lig",
    )

    assert "1-X-2: 1 %60 · X %20 · 2 %20" in message
    assert "Üst/Alt 2.5: Üst %55 · Alt %45" in message
    assert "KG Var/Yok: Var %45 · Yok %55" in message
