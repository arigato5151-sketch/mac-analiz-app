from __future__ import annotations

import pytest

from models.decision_policy import select_1x2


def test_select_1x2_applies_the_publish_threshold() -> None:
    assert select_1x2((0.55, 0.25, 0.20)) == (0, 0.55, True)
    assert select_1x2((0.40, 0.32, 0.28)) == (0, 0.40, False)


def test_select_1x2_rejects_invalid_probabilities() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        select_1x2((0.8, 0.3, 0.1))
