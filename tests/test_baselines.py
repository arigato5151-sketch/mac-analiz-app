from __future__ import annotations

import numpy as np
import pytest

from models.baselines import (
    audit_class_distribution,
    compare_to_baselines,
    detect_upcoming_collapse,
    given_probabilities_baseline,
    home_pick_baseline,
    majority_class_baseline,
)


def test_majority_and_home_baselines_score_the_same_rows() -> None:
    labels = np.array([0, 0, 2, 1, 0])

    majority = majority_class_baseline(labels)
    home = home_pick_baseline(labels)

    assert majority.accuracy == pytest.approx(3 / 5)
    assert home.accuracy == pytest.approx(3 / 5)


def test_given_probabilities_baseline_rejects_misaligned_matrices() -> None:
    labels = np.array([0, 1, 2])

    with pytest.raises(ValueError, match="aligned"):
        given_probabilities_baseline(labels, np.zeros((2, 3)), "Poisson")


def test_compare_to_baselines_flags_a_model_that_cannot_beat_them() -> None:
    labels = np.array([0, 0, 0, 0, 1])
    model = np.tile(np.array([0.55, 0.25, 0.20]), (len(labels), 1))

    comparison = compare_to_baselines(labels, model)

    assert comparison["beats_all_available"] is False


def test_compare_to_baselines_confirms_a_strong_model() -> None:
    labels = np.array([0, 0, 2, 1, 0])
    one_hot = np.eye(3)[labels]
    model = one_hot * 0.85 + 0.05

    comparison = compare_to_baselines(labels, model)

    assert comparison["beats_all_available"] is True
    assert comparison["model"].accuracy == pytest.approx(1.0)


def test_baseline_metrics_reject_non_normalized_probabilities() -> None:
    labels = np.array([0, 1, 2])

    with pytest.raises(ValueError, match="sum to one"):
        given_probabilities_baseline(
            labels,
            np.tile(np.array([0.5, 0.4, 0.2]), (len(labels), 1)),
            "Bozuk",
        )


def test_baseline_metrics_normalize_harmless_decimal_drift() -> None:
    labels = np.array([0, 1, 2])
    probabilities = np.array(
        [
            [0.6000004, 0.25, 0.15],
            [0.20, 0.5000004, 0.30],
            [0.15, 0.25, 0.6000004],
        ]
    )

    result = given_probabilities_baseline(labels, probabilities, "Yuvarlanmış")

    assert np.isfinite(result.log_loss)


def test_market_baseline_is_only_scored_when_provided() -> None:
    labels = np.array([0, 0, 2, 1, 0])
    model = np.tile(np.array([0.5, 0.3, 0.2]), (len(labels), 1))

    without_market = compare_to_baselines(labels, model)
    market = np.tile(np.array([0.45, 0.27, 0.28]), (len(labels), 1))
    with_market = compare_to_baselines(labels, model, market_probabilities=market)

    baseline_names = [baseline.name for baseline in without_market["baselines"]]
    assert "Piyasa (marjsız)" not in baseline_names
    assert "Piyasa (marjsız)" in [
        baseline.name for baseline in with_market["baselines"]
    ]


def test_audit_detects_a_collapsed_draw_class() -> None:
    labels = np.array([0, 0, 0, 2, 2])
    probs = np.tile(np.array([0.48, 0.04, 0.48]), (5, 1))

    audit = audit_class_distribution(labels, probs)

    assert audit.collapsed_class == "draw"
    assert audit.warning is not None


def test_audit_flags_single_class_concentration() -> None:
    labels = np.zeros(10, dtype=int)
    probs = np.tile(np.array([0.50, 0.25, 0.25]), (10, 1))

    audit = audit_class_distribution(labels, probs)

    assert audit.collapsed_class is None
    assert audit.top_pick_concentration == pytest.approx(1.0)
    assert audit.warning is not None


def test_audit_passes_a_healthy_distribution() -> None:
    labels = np.array([0, 0, 2, 1, 0, 1, 2, 0, 1, 2])
    probs = np.array(
        [
            [0.50, 0.25, 0.25],
            [0.45, 0.30, 0.25],
            [0.25, 0.25, 0.50],
            [0.25, 0.50, 0.25],
            [0.48, 0.27, 0.25],
            [0.20, 0.45, 0.35],
            [0.25, 0.30, 0.45],
            [0.52, 0.24, 0.24],
            [0.24, 0.48, 0.28],
            [0.26, 0.26, 0.48],
        ]
    )

    audit = audit_class_distribution(labels, probs)

    assert audit.collapsed_class is None
    assert audit.warning is None


def test_upcoming_collapse_checks_label_less_distributions() -> None:
    healthy = np.array(
        [
            [0.45, 0.27, 0.28],
            [0.25, 0.30, 0.45],
            [0.30, 0.45, 0.25],
            [0.50, 0.24, 0.26],
            [0.22, 0.28, 0.50],
            [0.26, 0.48, 0.26],
            [0.48, 0.26, 0.26],
            [0.24, 0.26, 0.50],
            [0.28, 0.46, 0.26],
            [0.46, 0.28, 0.26],
        ]
    )
    collapsed = np.tile(np.array([0.47, 0.05, 0.48]), (20, 1))

    assert detect_upcoming_collapse(healthy) is None
    assert "çöktü" in (detect_upcoming_collapse(collapsed) or "")
    assert detect_upcoming_collapse(np.zeros((0, 3))) is None
