from __future__ import annotations

from models.shadow import (
    MINIMUM_PROMOTION_SAMPLE,
    evaluate_shadow_predictions,
    promotion_decision,
)


class _NoFinishedMatchesDb:
    def __init__(self) -> None:
        self.queried: set[str] = set()

    def select_all(self, table: str, **kwargs: object) -> list[dict]:
        self.queried.add(table)
        if table == "shadow_prediction_performance":
            return []
        if table == "matches":
            return []
        if table == "shadow_predictions":
            raise AssertionError("shadow_predictions must not be scanned with no finished fixtures")
        return []

    def upsert(self, table: str, records: list[dict], **kwargs: object) -> list[dict]:
        return records


def test_shadow_evaluation_skips_scan_when_no_finished_matches() -> None:
    db = _NoFinishedMatchesDb()

    assert evaluate_shadow_predictions(db) == []
    assert db.queried <= {"shadow_prediction_performance", "matches"}


def test_shadow_candidate_needs_a_sufficient_same_match_sample() -> None:
    accepted, reason = promotion_decision(
        candidate_brier=0.40,
        candidate_accuracy=0.60,
        production_brier=0.50,
        production_accuracy=0.55,
        sample_size=MINIMUM_PROMOTION_SAMPLE - 1,
    )

    assert accepted is False
    assert "yetersiz" in reason


def test_shadow_candidate_requires_better_brier_and_no_accuracy_regression() -> None:
    assert promotion_decision(
        candidate_brier=0.42,
        candidate_accuracy=0.60,
        production_brier=0.50,
        production_accuracy=0.60,
        sample_size=MINIMUM_PROMOTION_SAMPLE,
    )[0] is True
    assert promotion_decision(
        candidate_brier=0.50,
        candidate_accuracy=0.65,
        production_brier=0.50,
        production_accuracy=0.60,
        sample_size=MINIMUM_PROMOTION_SAMPLE,
    )[0] is False
    assert promotion_decision(
        candidate_brier=0.42,
        candidate_accuracy=0.59,
        production_brier=0.50,
        production_accuracy=0.60,
        sample_size=MINIMUM_PROMOTION_SAMPLE,
    )[0] is False
