from __future__ import annotations

from datetime import datetime, timezone

from notifications.pre_match import due_matches, persist_production_snapshot, pre_match_message


def test_due_matches_tolerates_scheduler_delay_around_the_60_minute_target() -> None:
    now = datetime(2026, 8, 29, 12, 0, tzinfo=timezone.utc)
    matches = [
        {"id": 1, "match_date": "2026-08-29T12:09:00+00:00"},
        {"id": 2, "match_date": "2026-08-29T13:00:00+00:00"},
        {"id": 3, "match_date": "2026-08-29T13:31:00+00:00"},
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


class _SnapshotDb:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def select(self, _table: str, **_kwargs: object) -> list[dict]:
        return self.rows

    def insert(self, _table: str, rows: list[dict]) -> list[dict]:
        persisted = {"id": 42, **rows[0]}
        self.rows.append(persisted)
        return [persisted]


def test_production_snapshot_is_inserted_once() -> None:
    db = _SnapshotDb()
    prediction = {
        "id": 7,
        "predicted_at": "2026-08-30T10:00:00+00:00",
        "prob_home_win": 0.6,
        "prob_draw": 0.2,
        "prob_away_win": 0.2,
        "prob_over_2_5": 0.55,
        "prob_btts": 0.45,
    }

    first = persist_production_snapshot(
        db, prediction, match_id=11, model_version="model_v1", captured_at="2026-08-30T10:05:00+00:00"
    )
    second = persist_production_snapshot(
        db, {**prediction, "prob_home_win": 0.2}, match_id=11, model_version="model_v2", captured_at="2026-08-30T10:10:00+00:00"
    )

    assert first["id"] == second["id"] == 42
    assert len(db.rows) == 1
    assert first["prob_home_win"] == 0.6
