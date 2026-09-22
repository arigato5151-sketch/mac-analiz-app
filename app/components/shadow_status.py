"""User-facing shadow-model status derived from same-match evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from models.shadow import MINIMUM_PROMOTION_SAMPLE, promotion_decision


@dataclass(frozen=True)
class ShadowCandidateStatus:
    model_version: str
    lifecycle_status: str
    decision: str
    reason: str
    paired_matches: int
    prediction_count: int
    candidate_accuracy: float | None
    production_accuracy: float | None
    candidate_brier: float | None
    production_brier: float | None

    @property
    def progress(self) -> float:
        return min(self.paired_matches / MINIMUM_PROMOTION_SAMPLE, 1.0)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def summarize_shadow_candidate(row: Mapping[str, Any]) -> ShadowCandidateStatus:
    """Turn one aggregate database row into an explicit promotion state."""
    model_version = str(row.get("model_version") or "Bilinmeyen sürüm")
    lifecycle_status = str(row.get("status") or "shadow")
    paired_matches = int(row.get("paired_matches") or 0)
    prediction_count = int(row.get("prediction_count") or 0)
    candidate_accuracy = _optional_float(row.get("candidate_accuracy"))
    production_accuracy = _optional_float(row.get("production_accuracy"))
    candidate_brier = _optional_float(row.get("candidate_brier"))
    production_brier = _optional_float(row.get("production_brier"))

    base = {
        "model_version": model_version,
        "lifecycle_status": lifecycle_status,
        "paired_matches": paired_matches,
        "prediction_count": prediction_count,
        "candidate_accuracy": candidate_accuracy,
        "production_accuracy": production_accuracy,
        "candidate_brier": candidate_brier,
        "production_brier": production_brier,
    }
    if lifecycle_status == "promoted":
        return ShadowCandidateStatus(
            **base, decision="Yayına alındı", reason="Terfi tamamlandı"
        )
    if lifecycle_status == "rejected":
        return ShadowCandidateStatus(
            **base, decision="Reddedildi", reason="Aday model reddedildi"
        )
    if paired_matches < MINIMUM_PROMOTION_SAMPLE:
        return ShadowCandidateStatus(
            **base,
            decision="Veri topluyor",
            reason=f"Aynı maç örneklemi {paired_matches}/{MINIMUM_PROMOTION_SAMPLE}",
        )
    if None in (
        candidate_accuracy,
        production_accuracy,
        candidate_brier,
        production_brier,
    ):
        return ShadowCandidateStatus(
            **base,
            decision="Karar verilemiyor",
            reason="Aday ve üretim için eşleşmiş metrik eksik",
        )

    accepted, reason = promotion_decision(
        candidate_brier=candidate_brier,
        candidate_accuracy=candidate_accuracy,
        production_brier=production_brier,
        production_accuracy=production_accuracy,
        sample_size=paired_matches,
        candidate_calibrated_log_loss=_optional_float(row.get("offline_log_loss")),
        candidate_raw_log_loss=_optional_float(row.get("offline_raw_log_loss")),
        candidate_ece=_optional_float(row.get("offline_ece")),
        candidate_log_loss=_optional_float(row.get("offline_log_loss")),
        candidate_baseline_log_loss=_optional_float(
            row.get("offline_baseline_log_loss")
        ),
    )
    return ShadowCandidateStatus(
        **base,
        decision="Terfiye hazır" if accepted else "Eşiği geçemedi",
        reason=reason,
    )
