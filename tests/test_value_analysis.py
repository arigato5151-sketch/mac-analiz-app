from models.value_analysis import MIN_VALUE_EV, assess_market_value, best_value_assessment, evaluate_flat_stakes, fractional_kelly_stake, value_confidence_status


def test_normalizes_bookmaker_margin_and_calculates_positive_value():
    result = assess_market_value(
        {"home_win": 0.50, "draw": 0.25, "away_win": 0.25},
        {"home_win": 2.20, "draw": 3.50, "away_win": 3.50},
    )

    home = next(item for item in result if item.key == "home_win")
    assert round(home.fair_market_probability, 4) == round((1 / 2.2) / ((1 / 2.2) + (1 / 3.5) + (1 / 3.5)), 4)
    assert round(home.expected_value, 4) == 0.10
    assert home.has_value


def test_invalid_or_missing_quotes_are_ignored():
    result = assess_market_value(
        {"home_win": 0.55, "draw": None, "away_win": 0.20},
        {"home_win": 1.80, "draw": 1.0, "away_win": 3.20, "unused": 9.0},
    )

    assert [item.key for item in result] == ["home_win", "away_win"]


def test_flat_stake_performance_uses_decimal_odds():
    assessments = assess_market_value(
        {"home_win": 0.60, "draw": 0.20, "away_win": 0.20},
        {"home_win": 2.00, "draw": 3.50, "away_win": 4.00},
    )
    home = next(item for item in assessments if item.key == "home_win")
    performance = evaluate_flat_stakes([(home, True), (home, False)])

    assert performance.bets == 2
    assert performance.wins == 1
    assert performance.profit == 0.0
    assert performance.roi == 0.0


def test_small_positive_edge_is_not_marked_as_value():
    assessments = assess_market_value(
        {"home_win": 0.501, "draw": 0.2495, "away_win": 0.2495},
        {"home_win": 1.99, "draw": 4.00, "away_win": 4.00},
    )
    home = next(item for item in assessments if item.key == "home_win")
    assert home.expected_value < MIN_VALUE_EV
    assert not home.has_value


def test_value_confidence_requires_sample_and_positive_roi():
    assessments = assess_market_value(
        {"home_win": 0.60, "draw": 0.20, "away_win": 0.20},
        {"home_win": 2.00, "draw": 3.50, "away_win": 4.00},
    )
    home = next(item for item in assessments if item.key == "home_win")
    performance = evaluate_flat_stakes([(home, True)])
    assert value_confidence_status(performance, minimum_sample=2) == "Yetersiz örneklem"


def test_fractional_kelly_is_capped():
    assessments = assess_market_value(
        {"home_win": 0.70, "draw": 0.15, "away_win": 0.15},
        {"home_win": 2.00, "draw": 5.00, "away_win": 5.00},
    )
    home = next(item for item in assessments if item.key == "home_win")
    assert fractional_kelly_stake(home) == 0.05


def test_best_value_applies_the_same_ev_threshold():
    best = best_value_assessment(
        {"home_win": 0.60, "draw": 0.20, "away_win": 0.20},
        {"home_win": 2.00, "draw": 3.50, "away_win": 4.00},
    )
    assert best is not None
    assert best.key == "home_win"
    assert best.expected_value >= MIN_VALUE_EV
