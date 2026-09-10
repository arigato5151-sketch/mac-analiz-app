from __future__ import annotations

from datetime import datetime, timezone

from notifications.pre_match import (
    _absence_summary,
    _telegram_commentary,
    due_matches,
    persist_production_snapshot,
    pre_match_message,
)


def test_due_matches_uses_the_full_pre_kickoff_window() -> None:
    now = datetime(2026, 8, 29, 12, 0, tzinfo=timezone.utc)
    matches = [
        {"id": 1, "match_date": "2026-08-29T12:14:00+00:00"},
        {"id": 2, "match_date": "2026-08-29T12:20:00+00:00"},
        {"id": 3, "match_date": "2026-08-29T12:26:00+00:00"},
    ]

    assert [row["id"] for row in due_matches(matches, now=now)] == [1, 2]


def test_pre_match_message_shows_every_market() -> None:
    message = pre_match_message(
        {"match_date": "2026-08-29T16:00:00+00:00"},
        {"prob_home_win": 0.6, "prob_draw": 0.2, "prob_away_win": 0.2, "prob_over_2_5": 0.55, "prob_btts": 0.45},
        home_team="Ev",
        away_team="Deplasman",
        league_name="Lig",
    )

    assert "1X2 Tahmin: Ev kazanır %60" in message
    assert "Üst 2.5: %55 · KG Var: %45" in message


def test_pre_match_message_marks_low_confidence_1x2_as_pass() -> None:
    message = pre_match_message(
        {"match_date": "2026-08-29T16:00:00+00:00"},
        {"prob_home_win": 0.4, "prob_draw": 0.32, "prob_away_win": 0.28, "prob_over_2_5": 0.55, "prob_btts": 0.45},
        home_team="Ev",
        away_team="Deplasman",
        league_name="Lig",
    )

    assert "1X2: Pas · en yüksek Ev kazanır %40" in message
    assert "Üst 2.5: %55 · KG Var: %45" in message


def test_pre_match_message_adds_only_confident_diversified_markets() -> None:
    message = pre_match_message(
        {"match_date": "2026-08-29T16:00:00+00:00"},
        {
            "prob_home_win": 0.6,
            "prob_draw": 0.2,
            "prob_away_win": 0.2,
            "prob_over_2_5": 0.55,
            "prob_btts": 0.45,
            "market_probabilities": {
                "double_chance": {"1X": 0.8, "X2": 0.4, "12": 0.8},
                "total_goals": {"over_1_5": 0.75, "under_3_5": 0.7},
                "team_goals": {"home_over_0_5": 0.8, "away_over_0_5": 0.55},
                "correct_scores": [
                    {"score": "1-0", "probability": 0.16},
                    {"score": "1-1", "probability": 0.14},
                    {"score": "2-0", "probability": 0.12},
                ],
            },
        },
        home_team="Ev",
        away_team="Deplasman",
        league_name="Lig",
    )

    assert "📊 Ek tahminler" in message
    assert "Çifte şans: 1X %80" in message
    assert "Gol çizgisi: Üst 1.5 %75 · Alt 3.5 %70" in message
    assert "Takım golü: Ev 0.5 Üst %80" in message
    assert "Dep. 0.5 Üst" not in message
    assert "Olası skorlar: 1-0 %16 · 1-1 %14 · 2-0 %12" in message


def test_pre_match_message_adds_a_bounded_ai_commentary_section() -> None:
    message = pre_match_message(
        {"match_date": "2026-08-29T16:00:00+00:00"},
        {"prob_home_win": 0.6, "prob_draw": 0.2, "prob_away_win": 0.2, "prob_over_2_5": 0.55, "prob_btts": 0.45},
        home_team="Ev",
        away_team="Deplasman",
        league_name="Lig",
        commentary="İlk paragraf.\n\nİkinci paragraf.",
    )

    assert "🧠 Maç yorumu" in message
    assert message.endswith("İkinci paragraf.")
    assert len(_telegram_commentary("kelime " * 500) or "") <= 1_401


def test_telegram_commentary_hard_cuts_a_run_on_string() -> None:
    text = "x" * 5_000

    result = _telegram_commentary(text)

    assert result is not None
    assert len(result) <= 1_401
    assert result.endswith("…")


def test_absence_summary_keeps_only_relevant_confirmed_absences() -> None:
    rows = [
        {"team_id": 1, "player_name": "A", "status": "injured"},
        {"team_id": 1, "player_name": "B", "status": "available"},
        {"team_id": 2, "player_name": "C", "status": "suspended"},
    ]

    assert _absence_summary(rows, team_id=1) == ["A (sakat)"]


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
        "market_probabilities": {"double_chance": {"1X": 0.8}},
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
    assert first["market_probabilities"] == {"double_chance": {"1X": 0.8}}
