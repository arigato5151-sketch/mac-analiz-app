from config.settings import PRE_MATCH_DECISION_LEAD_MINUTES
from notifications.pre_match import WINDOW_END_MINUTES


def test_pre_match_window_uses_single_decision_deadline_constant():
    assert WINDOW_END_MINUTES == PRE_MATCH_DECISION_LEAD_MINUTES == 20
