from __future__ import annotations

from typing import Any

from app.components import data
from db.db_client import DatabaseError


class ResultsViewDb:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def select(self, table: str, **kwargs: Any) -> list[dict[str, Any]]:
        self.calls.append((table, kwargs))
        return [
            {
                "prediction_id": 3,
                "match_id": 4,
                "actual_result": "home_win",
                "was_correct": True,
                "brier_score": 0.21,
                "evaluated_at": "2026-08-29T17:00:00+00:00",
                "league_id": 39,
                "match_date": "2026-08-29T15:00:00+00:00",
                "home_score": 2,
                "away_score": 1,
                "prob_home_win": 0.61,
                "prob_draw": 0.22,
                "prob_away_win": 0.17,
                "prob_over_2_5": 0.57,
                "prob_btts": 0.48,
                "model_version": "v1",
                "predicted_at": "2026-08-29T09:00:00+00:00",
                "home_team": "Ev",
                "away_team": "Deplasman",
                "league_name": "Premier League",
            }
        ]


class DashboardDb:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def select_all(self, table: str, **kwargs: Any) -> list[dict[str, Any]]:
        self.calls.append((table, kwargs))
        if table == "matches":
            return [{
                "id": 10,
                "league_id": 39,
                "home_team_id": 1,
                "away_team_id": 2,
                "match_date": "2026-09-02T18:00:00+00:00",
                "status": "scheduled",
            }]
        if table == "predictions":
            return [{
                "match_id": 10,
                "model_version": "v1",
                "prob_home_win": 0.5,
                "prob_draw": 0.3,
                "prob_away_win": 0.2,
                "prob_over_2_5": 0.5,
                "prob_btts": 0.5,
                "market_probabilities": {"double_chance": {"1X": 0.8}},
                "predicted_at": "2026-09-02T12:00:00+00:00",
            }]
        if table == "teams":
            return [{"id": 1, "name": "Ev"}, {"id": 2, "name": "Deplasman"}]
        if table == "leagues":
            return [{"id": 39, "name": "Lig", "country": "TR"}]
        raise AssertionError(f"Unexpected table: {table}")


class PerformanceDb:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def select_all(self, table: str, **kwargs: Any) -> list[dict[str, Any]]:
        self.calls.append((table, kwargs))
        return []


def test_evaluated_predictions_use_single_database_view(monkeypatch) -> None:
    db = ResultsViewDb()
    data.load_evaluated_predictions.clear()
    monkeypatch.setattr(data, "get_db", lambda: db)

    frame = data.load_evaluated_predictions(limit=25)

    assert len(db.calls) == 1
    assert db.calls[0][0] == "evaluated_prediction_results"
    assert db.calls[0][1]["limit"] == 25
    assert frame.loc[0, "home_team"] == "Ev"
    assert str(frame.loc[0, "match_date"].tz) == "Europe/Istanbul"


def test_upcoming_dashboard_requests_predictions_only_for_visible_fixtures(monkeypatch) -> None:
    db = DashboardDb()
    data.load_upcoming_dashboard.clear()
    data.load_reference_catalog.clear()
    monkeypatch.setattr(data, "get_db", lambda: db)

    frame = data.load_upcoming_dashboard(horizon_days=3)

    prediction_call = next(call for call in db.calls if call[0] == "predictions")
    assert prediction_call[1]["filters"] == {"match_id": "in.(10)"}
    assert "market_probabilities" in prediction_call[1]["columns"]
    assert frame.loc[0, "home_team"] == "Ev"
    assert frame.loc[0, "market_probabilities"] == {"double_chance": {"1X": 0.8}}
    assert str(frame.loc[0, "predicted_at"].tz) == "Europe/Istanbul"


def test_live_performance_requests_diversified_market_fields(monkeypatch) -> None:
    db = PerformanceDb()
    data.load_prediction_performance.clear()
    monkeypatch.setattr(data, "get_db", lambda: db)

    frame = data.load_prediction_performance()

    assert frame.empty
    assert len(db.calls) == 1
    assert db.calls[0][0] == "evaluated_prediction_results"
    assert "model_version" in db.calls[0][1]["columns"]
    assert "market_probabilities" in db.calls[0][1]["columns"]
    assert "market_performance" in db.calls[0][1]["columns"]
    assert "league_name" in db.calls[0][1]["columns"]


class SquadContextDb:
    """player_availability works; the team snapshot table is not exposed."""

    def select_all(self, table: str, **kwargs: Any) -> list[dict[str, Any]]:
        if table == "player_availability":
            return [
                {
                    "team_id": 1,
                    "player_name": "A",
                    "status": "injured",
                    "updated_at": "2026-09-20T10:00:00+00:00",
                },
                {
                    "team_id": 1,
                    "player_name": "A",
                    "status": "doubtful",
                    "updated_at": "2026-09-21T10:00:00+00:00",
                },
                {
                    "team_id": 2,
                    "player_name": "B",
                    "status": "suspended",
                    "updated_at": "2026-09-19T10:00:00+00:00",
                },
            ]
        raise DatabaseError(
            "Supabase GET team_availability_status failed (404): not exposed"
        )


def test_match_availability_deduplicates_and_derives_freshness(monkeypatch) -> None:
    data.load_match_availability.clear()
    monkeypatch.setattr(data, "get_db", lambda: SquadContextDb())

    players, snapshots = data.load_match_availability(1, 2)

    assert len(players) == 2
    repeated = players[players["player_name"] == "A"].iloc[0]
    assert repeated["status"] == "doubtful"
    assert not snapshots.empty
    assert snapshots["refreshed_at"].notna().all()


def test_match_availability_excludes_other_fixture_rows(monkeypatch) -> None:
    class FixtureSquadDb:
        def select_all(self, table: str, **kwargs: Any) -> list[dict[str, Any]]:
            if table == "player_availability":
                assert kwargs["filters"]["match_id"] == "eq.101"
                return [
                    {
                        "match_id": 101,
                        "team_id": 1,
                        "player_name": "Current",
                        "status": "injured",
                        "updated_at": "2026-09-21T10:00:00+00:00",
                    },
                    {
                        "match_id": 202,
                        "team_id": 1,
                        "player_name": "Other fixture",
                        "status": "injured",
                        "updated_at": "2026-09-21T10:00:00+00:00",
                    },
                ]
            if table == "team_availability_status":
                return []
            raise AssertionError(table)

    data.load_match_availability.clear()
    monkeypatch.setattr(data, "get_db", lambda: FixtureSquadDb())

    players, _ = data.load_match_availability(1, 2, 101)

    assert players["player_name"].tolist() == ["Current"]
