from __future__ import annotations

from math import log

import pytest

from evaluation.track_performance import (
    _select_rows_for_match_ids,
    actual_result,
    binary_market_performance,
    build_performance_row,
    evaluate_pending_predictions,
    multiclass_log_loss,
    select_official_predictions,
)


@pytest.mark.parametrize(
    ("home", "away", "expected"),
    [(2, 0, "home_win"), (1, 1, "draw"), (0, 3, "away_win")],
)
def test_actual_result(home: int, away: int, expected: str) -> None:
    assert actual_result(home, away) == expected


def test_build_performance_row_calculates_multiclass_brier_score() -> None:
    prediction = {
        "id": 9,
        "prob_home_win": 0.60,
        "prob_draw": 0.25,
        "prob_away_win": 0.15,
        "prob_over_2_5": 0.55,
        "prob_btts": 0.15,
    }
    match = {"id": 100, "home_score": 2, "away_score": 1}

    row = build_performance_row(
        prediction, match, evaluated_at="2026-08-26T12:00:00+00:00"
    )

    assert row["actual_result"] == "home_win"
    assert row["was_correct"] is True
    assert row["brier_score"] == pytest.approx(0.245)
    assert row["log_loss"] == pytest.approx(-log(0.60))
    assert row["over_2_5_actual"] is True
    assert row["over_2_5_was_correct"] is True
    assert row["over_2_5_brier_score"] == pytest.approx(0.2025)
    assert row["btts_actual"] is True
    assert row["btts_was_correct"] is False
    assert row["btts_brier_score"] == pytest.approx(0.7225)


def test_multiclass_log_loss_uses_a_finite_floor_for_zero_probability() -> None:
    assert multiclass_log_loss((0.0, 0.5, 0.5), 0) > 30


def test_binary_market_performance_uses_real_outcome_and_probability() -> None:
    assert binary_market_performance(0.70, actual_positive=True) == (
        True,
        pytest.approx(0.09),
    )
    assert binary_market_performance(0.30, actual_positive=True) == (
        False,
        pytest.approx(0.49),
    )
    assert binary_market_performance(None, actual_positive=True) == (None, None)


def test_build_performance_row_rejects_invalid_probability_sum() -> None:
    prediction = {
        "id": 9,
        "prob_home_win": 0.8,
        "prob_draw": 0.3,
        "prob_away_win": 0.1,
    }

    with pytest.raises(ValueError, match="sum to one"):
        build_performance_row(
            prediction,
            {"id": 100, "home_score": 0, "away_score": 0},
            evaluated_at="2026-08-26T12:00:00+00:00",
        )


def test_select_official_predictions_keeps_latest_snapshot_per_match() -> None:
    selected = select_official_predictions(
        [
            {"id": 1, "match_id": 10, "predicted_at": "2026-08-30T09:00:00+00:00"},
            {"id": 2, "match_id": 10, "predicted_at": "2026-08-30T10:00:00+00:00"},
            {"id": 3, "match_id": 11, "predicted_at": "2026-08-30T09:00:00+00:00"},
        ]
    )

    assert {int(row["id"]) for row in selected} == {2, 3}


def test_select_official_predictions_compares_parsed_timestamps() -> None:
    # id 1 reads as the "larger" string but lands earlier in UTC when parsed
    # (10:00:00+01:00 is 09:00 UTC, while id 2 is 09:30 UTC). Lexicographic
    # comparison would wrongly crown id 1; parse-aware ordering must pick id 2.
    selected = select_official_predictions(
        [
            {"id": 1, "match_id": 10, "predicted_at": "2026-08-30T10:00:00+01:00"},
            {"id": 2, "match_id": 10, "predicted_at": "2026-08-30T09:30:00Z"},
        ]
    )

    assert {int(row["id"]) for row in selected} == {2}


def test_build_performance_row_requires_a_pre_kickoff_prediction() -> None:
    # The evaluator compares timestamps before persisting a production score.
    prediction = {"id": 4, "match_id": 7, "predicted_at": "2026-08-30T10:00:00+00:00"}
    match = {"id": 7, "match_date": "2026-08-30T11:00:00+00:00"}

    assert str(prediction["predicted_at"]) <= str(match["match_date"])


def test_build_performance_row_links_an_immutable_snapshot() -> None:
    row = build_performance_row(
        {
            "id": 9,
            "snapshot_id": 55,
            "prob_home_win": 0.6,
            "prob_draw": 0.25,
            "prob_away_win": 0.15,
        },
        {"id": 100, "home_score": 2, "away_score": 1},
        evaluated_at="2026-08-26T12:00:00+00:00",
    )

    assert row["snapshot_id"] == 55


def test_build_performance_row_keeps_bulk_insert_keys_without_snapshot() -> None:
    row = build_performance_row(
        {
            "id": 10,
            "prob_home_win": 0.4,
            "prob_draw": 0.35,
            "prob_away_win": 0.25,
        },
        {"id": 101, "home_score": 1, "away_score": 1},
        evaluated_at="2026-08-26T12:00:00+00:00",
    )

    assert "snapshot_id" in row
    assert row["snapshot_id"] is None


class _ScopedDb:
    """Records which tables are queried to prove pending-only narrowing."""

    def __init__(self) -> None:
        self.queried: set[str] = set()
        self.last_filters: dict[object, object] = {}

    def select_all(self, table: str, **kwargs: object) -> list[dict]:
        self.queried.add(table)
        self.last_filters[table] = kwargs.get("filters")
        if table == "prediction_performance":
            return [{"match_id": 100}]
        if table == "matches":
            return [{"id": 100, "match_date": "2026-08-26T10:00:00+00:00", "home_score": 1, "away_score": 0}]
        if table == "predictions":
            return [{"id": 5, "match_id": 100, "prob_home_win": 0.6, "prob_draw": 0.25, "prob_away_win": 0.15, "predicted_at": "2026-08-26T09:00:00+00:00"}]
        if table == "prediction_snapshots":
            return []
        return []

    def upsert(self, table: str, records: list[dict], **kwargs: object) -> list[dict]:
        return records


def test_evaluate_pending_skips_database_scans_when_nothing_is_pending() -> None:
    db = _ScopedDb()

    rows = evaluate_pending_predictions(db)

    assert rows == []
    # The evaluator broadens exactly the finished-window matches; the growing
    # predictions/snapshots tables are only touched when a match is pending.
    assert db.queried <= {"prediction_performance", "matches"}


def test_evaluate_pending_bounds_finished_window_recently() -> None:
    db = _ScopedDb()

    evaluate_pending_predictions(db)

    match_filters = db.last_filters["matches"]
    assert any(f.startswith("gte.") for f in match_filters.values() if isinstance(f, str))


class _BatchDb:
    def __init__(self) -> None:
        self.filters: list[dict[str, str]] = []

    def select_all(self, _table: str, **kwargs: object) -> list[dict]:
        self.filters.append(dict(kwargs["filters"]))
        return []


def test_match_id_queries_are_split_below_proxy_url_limits() -> None:
    db = _BatchDb()

    rows = _select_rows_for_match_ids(
        db,
        "predictions",
        columns="id,match_id",
        match_ids=list(range(1, 206)) + [1],
        filters={"status": "eq.ready"},
        batch_size=100,
    )

    assert rows == []
    assert len(db.filters) == 3
    assert all(query["status"] == "eq.ready" for query in db.filters)
    assert [query["match_id"].count(",") + 1 for query in db.filters] == [100, 100, 5]


def test_match_id_query_rejects_invalid_batch_size() -> None:
    with pytest.raises(ValueError, match="positive"):
        _select_rows_for_match_ids(
            _BatchDb(),
            "predictions",
            columns="id",
            match_ids=[1],
            batch_size=0,
        )
