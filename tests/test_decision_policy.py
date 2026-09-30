from __future__ import annotations

import numpy as np
import pytest

from models.decision_policy import (
    LEAGUE_CONFIDENCE_OVERRIDES,
    minimum_confidence_for_league,
    select_1x2,
    select_threshold_from_calibration,
)


def test_select_1x2_applies_the_publish_threshold() -> None:
    assert select_1x2((0.55, 0.25, 0.20)) == (0, 0.55, True)
    assert select_1x2((0.52, 0.28, 0.20)) == (0, 0.52, False)
    assert select_1x2((0.40, 0.32, 0.28)) == (0, 0.40, False)


def test_select_1x2_rejects_invalid_probabilities() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        select_1x2((0.8, 0.3, 0.1))


def test_leagues_without_calibration_override_use_global_gate() -> None:
    assert LEAGUE_CONFIDENCE_OVERRIDES == {}
    assert minimum_confidence_for_league(62) == 0.55
    assert minimum_confidence_for_league(253) == 0.55
    assert select_1x2((0.57, 0.25, 0.18), threshold=minimum_confidence_for_league(39))[2] is True


def test_unknown_or_missing_league_falls_back_to_the_default_gate() -> None:
    assert minimum_confidence_for_league(None) == 0.55
    assert minimum_confidence_for_league(99999) == 0.55


def test_threshold_selection_uses_calibration_rows_only():
    labels = [0, 1, 2] * 20
    probabilities = np.tile(np.array([[.70, .20, .10], [.20, .60, .20], [.10, .20, .70]]), (20, 1))
    threshold, metrics = select_threshold_from_calibration(labels, probabilities)
    assert .50 <= threshold <= .70
    assert metrics["coverage"] >= .20
