from __future__ import annotations

from typing import Any

from app.components import data


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
                "predicted_at": "2026-09-02T12:00:00+00:00",
            }]
        if table == "teams":
            return [{"id": 1, "name": "Ev"}, {"id": 2, "name": "Deplasman"}]
        if table == "leagues":
            return [{"id": 39, "name": "Lig", "country": "TR"}]
        raise AssertionError(f"Unexpected table: {table}")


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
    assert frame.loc[0, "home_team"] == "Ev"
