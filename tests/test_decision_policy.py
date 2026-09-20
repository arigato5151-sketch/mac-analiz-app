from __future__ import annotations

import pytest

from models.decision_policy import (
    LEAGUE_CONFIDENCE_OVERRIDES,
    minimum_confidence_for_league,
    select_1x2,
)


def test_select_1x2_applies_the_publish_threshold() -> None:
    assert select_1x2((0.55, 0.25, 0.20)) == (0, 0.55, True)
    assert select_1x2((0.52, 0.28, 0.20)) == (0, 0.52, False)
    assert select_1x2((0.40, 0.32, 0.28)) == (0, 0.40, False)


def test_select_1x2_rejects_invalid_probabilities() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        select_1x2((0.8, 0.3, 0.1))


def test_weak_leagues_carry_a_tightened_publish_gate() -> None:
    assert LEAGUE_CONFIDENCE_OVERRIDES[39] > 0.55
    assert minimum_confidence_for_league(62) == 0.60
    assert minimum_confidence_for_league(253) == 0.60
    assert select_1x2((0.57, 0.25, 0.18), threshold=minimum_confidence_for_league(39))[
        2
    ] is False


def test_unknown_or_missing_league_falls_back_to_the_default_gate() -> None:
    assert minimum_confidence_for_league(None) == 0.55
    assert minimum_confidence_for_league(99999) == 0.55
