from __future__ import annotations

import app.components.data as data
from app.components.shadow_status import summarize_shadow_candidate
from models.shadow import MINIMUM_PROMOTION_SAMPLE


def _candidate(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "model_version": "candidate-v1",
        "status": "shadow",
        "paired_matches": MINIMUM_PROMOTION_SAMPLE,
        "prediction_count": MINIMUM_PROMOTION_SAMPLE + 20,
        "candidate_accuracy": 0.60,
        "production_accuracy": 0.55,
        "candidate_brier": 0.42,
        "production_brier": 0.50,
        "offline_log_loss": 1.00,
        "offline_raw_log_loss": 1.02,
        "offline_baseline_log_loss": 1.08,
    }
    row.update(overrides)
    return row


def test_shadow_status_reports_sample_collection_progress() -> None:
    summary = summarize_shadow_candidate(_candidate(paired_matches=25))

    assert summary.decision == "Veri topluyor"
    assert summary.reason == f"Aynı maç örneklemi 25/{MINIMUM_PROMOTION_SAMPLE}"
    assert summary.progress == 0.25


def test_shadow_status_marks_candidate_ready_only_after_full_gate() -> None:
    summary = summarize_shadow_candidate(_candidate())

    assert summary.decision == "Terfiye hazır"
    assert summary.progress == 1.0


def test_shadow_status_explains_failed_gate() -> None:
    summary = summarize_shadow_candidate(_candidate(candidate_brier=0.51))

    assert summary.decision == "Eşiği geçemedi"
    assert "Brier" in summary.reason


def test_shadow_status_preserves_terminal_lifecycle_state() -> None:
    assert summarize_shadow_candidate(_candidate(status="promoted")).decision == "Yayına alındı"
    assert summarize_shadow_candidate(_candidate(status="rejected")).decision == "Reddedildi"


def test_shadow_status_loader_uses_sanitized_public_view(monkeypatch) -> None:
    class PublicViewDb:
        def __init__(self) -> None:
            self.table = ""
            self.columns = ""

        def select_all(self, table: str, **kwargs: object) -> list[dict[str, object]]:
            self.table = table
            self.columns = str(kwargs.get("columns") or "")
            return [_candidate()]

    db = PublicViewDb()
    data.load_shadow_model_status.clear()
    monkeypatch.setattr(data, "get_db", lambda: db)

    frame = data.load_shadow_model_status()

    assert db.table == "shadow_model_status"
    assert "candidate_accuracy" in db.columns
    assert "prob_home_win" not in db.columns
    assert frame.loc[0, "model_version"] == "candidate-v1"
    data.load_shadow_model_status.clear()
