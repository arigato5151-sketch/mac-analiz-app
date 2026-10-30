from __future__ import annotations

from typing import Any

from data_pipeline.precompute_match_analysis import _load_league_history


class FakeSupabase:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.last_call: tuple[str, dict[str, Any]] | None = None

    def select_all(self, table: str, **kwargs: Any) -> list[dict[str, Any]]:
        self.last_call = (table, kwargs)
        return self.rows


def test_load_league_history_converts_all_numeric_columns() -> None:
    db = FakeSupabase(
        [
            {
                "id": "17",
                "league_id": "3",
                "home_team_id": "11",
                "away_team_id": "12",
                "match_date": "2026-10-29T18:00:00Z",
                "status": "finished",
                "home_score": "2",
                "away_score": "1",
                "home_halftime_score": "1",
                "away_halftime_score": "0",
                "home_xg": "1.7",
                "away_xg": "0.8",
                "home_corners": "6",
                "away_corners": "4",
            }
        ]
    )

    history = _load_league_history(db, 3)

    assert len(history) == 1
    assert history.iloc[0]["id"] == 17
    assert history.iloc[0]["home_score"] == 2
    assert history.iloc[0]["home_corners"] == 6
    assert history.iloc[0]["match_date"].isoformat() == "2026-10-29T18:00:00+00:00"
    assert db.last_call is not None
    assert db.last_call[0] == "matches"
    assert db.last_call[1]["filters"]["league_id"] == "eq.3"
