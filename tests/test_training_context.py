from __future__ import annotations

from models.train_model import attach_historical_context, load_historical_matches


def test_historical_loader_requests_only_base_match_columns() -> None:
    calls: list[dict[str, object]] = []

    class _Db:
        def select_all(self, table: str, **kwargs: object) -> list[dict]:
            calls.append({"table": table, **kwargs})
            return [{
                "id": 10,
                "league_id": 1,
                "home_team_id": 1,
                "away_team_id": 2,
                "match_date": "2026-01-10T12:00:00+00:00",
                "status": "finished",
                "home_score": 2,
                "away_score": 1,
                "home_xg": 1.8,
                "away_xg": 0.9,
                "home_xa": None,
                "away_xa": None,
            }]

    rows = load_historical_matches(_Db())

    assert rows[0]["home_score"] == 2
    assert len(calls) == 1
    assert calls[0]["table"] == "matches"
    asserted_columns = str(calls[0]["columns"])
    # Inference only consumes team ids, scores, xg, league and date. Quote,
    # availability and lineup history are deliberately not transferred here.
    assert "odds_quote_history" not in str(calls[0].get("table"))
    assert all(
        column in asserted_columns
        for column in ("home_team_id", "away_team_id", "home_score", "away_score", "match_date")
    )


def test_historical_context_uses_only_pre_kickoff_observations() -> None:
    matches = [{
        "id": 10,
        "home_team_id": 1,
        "away_team_id": 2,
        "match_date": "2026-01-10T12:00:00+00:00",
    }]
    availability = [
        {"team_id": 1, "refreshed_at": "2026-01-09T10:00:00+00:00", "available_count": 19},
        {"team_id": 1, "refreshed_at": "2026-01-11T10:00:00+00:00", "available_count": 22},
        {"team_id": 2, "refreshed_at": "2026-01-09T10:00:00+00:00", "available_count": 21},
    ]
    lineups = [
        {"match_id": 10, "team_id": 1, "confirmed_at": "2026-01-10T11:00:00+00:00"},
        {"match_id": 10, "team_id": 2, "confirmed_at": "2026-01-10T11:50:00+00:00"},
    ]

    row = attach_historical_context(
        matches, availability_history=availability, lineups=lineups
    )[0]

    assert row["home_available_count"] == 19
    assert row["home_unavailable_count"] == 3
    assert row["away_available_count"] == 21
    assert row["home_lineup_confirmed"] is True
    assert row["away_lineup_confirmed"] is False
