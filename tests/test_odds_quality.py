from app.odds_quality import inconsistent_total_market_keys
from app.market_mapping import (
    is_compound_selection,
    standard_market_price,
    total_goals_outcome,
)


def _market(line: float, over: float, under: float) -> dict[str, object]:
    return {
        "sport_type": 4,
        "market_id": 60,
        "market_name": "Toplam Gol Alt/Üst",
        "line": line,
        "selections": [
            {"name": "Üst", "display_odds": over, "odds": over},
            {"name": "Alt", "display_odds": under, "odds": under},
        ],
    }


def test_flags_cross_line_inconsistency_like_reported_odds() -> None:
    markets = [_market(1.5, 1.16, 3.05), _market(2.5, 10.55, 2.11), _market(3.5, 3.11, 1.15)]

    assert inconsistent_total_market_keys(markets) == {
        ("4", "60", "2.5"),
        ("4", "60", "3.5"),
    }


def test_accepts_ordered_total_lines() -> None:
    markets = [_market(1.5, 1.2, 4.1), _market(2.5, 1.8, 2.0), _market(3.5, 3.1, 1.3)]

    assert inconsistent_total_market_keys(markets) == set()


def test_compares_same_total_family_across_market_codes() -> None:
    lower = _market(2.5, 1.8, 2.0)
    upper = _market(3.5, 1.4, 3.2)
    upper["market_id"] = 61

    assert inconsistent_total_market_keys([lower, upper]) == {
        ("4", "60", "2.5"),
        ("4", "61", "3.5"),
    }


def test_does_not_compare_different_market_families() -> None:
    lower = _market(2.5, 1.8, 2.0)
    upper = _market(3.5, 1.2, 3.3)
    upper["market_name"] = "Toplam Korner Alt/Üst"

    assert inconsistent_total_market_keys([lower, upper]) == set()


def test_combo_outcomes_cannot_be_used_as_single_over_under_quotes() -> None:
    assert is_compound_selection("Üst ve Yok") is True
    assert is_compound_selection("Alt ve Var") is True
    assert is_compound_selection("1 ve Üst") is True
    assert is_compound_selection("Üst") is False
    assert is_compound_selection("Alt") is False


def test_double_chance_is_not_misclassified_as_a_combo_market() -> None:
    assert is_compound_selection("1 ve 0") is False
    assert is_compound_selection("0 ve 2") is False


def test_only_standalone_total_outcomes_map_to_over_under() -> None:
    assert total_goals_outcome("Üst") == "over"
    assert total_goals_outcome("Alt") == "under"
    assert total_goals_outcome("Üst ve Yok") is None
    assert total_goals_outcome("Alt ve Var") is None


def test_standard_iddaa_odd_is_preferred_to_promotional_wodd() -> None:
    assert standard_market_price({"odds": 1.84, "display_odds": 1.77}) == 1.84
    assert standard_market_price({"display_odds": 1.84}) == 1.84
