from types import SimpleNamespace

from models.coupon_policy import diversified_coupon_rows


def test_coupon_does_not_require_every_market_family():
    def item(key, ev):
        return SimpleNamespace(key=key, expected_value=ev, model_probability=.6, odds=2.0)

    rows = [
        {"match_id": 1, "assessments": [item("home_win", .40), item("over_2_5", .20)], "best": item("home_win", .40)},
        {"match_id": 2, "assessments": [item("away_win", .35), item("btts_yes", .15)], "best": item("away_win", .35)},
        {"match_id": 3, "assessments": [item("under_2_5", .30)], "best": item("under_2_5", .30)},
    ]
    selected = diversified_coupon_rows(rows, max_items=3, min_ev=0.0)
    assert {row["best"].key for row in selected} == {"home_win", "away_win", "under_2_5"}


def test_coupon_rejects_probability_below_fifty_percent():
    def item(probability):
        return SimpleNamespace(key="home_win", expected_value=.40, model_probability=probability, odds=3.0)

    rows = [{"match_id": 1, "assessments": [item(.49)], "best": item(.49)}]
    assert diversified_coupon_rows(rows, min_ev=0.0) == []


def test_coupon_handles_empty_assessments_without_key_error():
    assert diversified_coupon_rows([{"match_id": 1, "assessments": []}], min_ev=0.0) == []


def test_coupon_requires_stronger_confidence_for_result_markets():
    item = SimpleNamespace(key="home_win", expected_value=.40, model_probability=.57, odds=3.0)
    rows = [{"match_id": 1, "assessments": [item], "best": item}]
    assert diversified_coupon_rows(rows, min_ev=0.0) == []


def test_coupon_rejects_absurd_odds():
    item = SimpleNamespace(key="away_win", expected_value=20.0, model_probability=.6, odds=46.0)
    rows = [{"match_id": 1, "assessments": [item], "best": item}]
    assert diversified_coupon_rows(rows, min_ev=0.0) == []


def test_coupon_requires_higher_threshold_for_btts():
    item = SimpleNamespace(key="btts_yes", expected_value=.40, model_probability=.52, odds=3.5)
    rows = [{"match_id": 1, "assessments": [item], "best": item}]
    assert diversified_coupon_rows(rows, min_ev=0.0) == []


def test_coupon_rejects_extreme_value_outlier():
    item = SimpleNamespace(key="home_win", expected_value=9.85, model_probability=.62, odds=17.5)
    rows = [{"match_id": 1, "assessments": [item]}]
    assert diversified_coupon_rows(rows, min_ev=0.0) == []


def test_coupon_value_table_candidates_exclude_negative_ev() -> None:
    negative = SimpleNamespace(key="home_win", expected_value=-.397, model_probability=.60, odds=1.15)
    positive = SimpleNamespace(key="away_win", expected_value=.04, model_probability=.60, odds=1.80)
    rows = [
        {"match_id": 1, "assessments": [negative], "best": negative},
        {"match_id": 2, "assessments": [positive], "best": positive},
    ]
    assert [row["match_id"] for row in diversified_coupon_rows(rows, min_ev=0.0)] == [2]


def test_same_match_can_use_different_market_families():
    def item(key):
        return SimpleNamespace(key=key, expected_value=.40, model_probability=.6, odds=2.0)

    rows = [
        {"match_id": 1, "assessments": [item("home_win"), item("btts_yes")], "best": item("home_win")},
    ]
    selected = diversified_coupon_rows(rows, min_ev=0.0)
    assert {row["best"].key for row in selected} == {"home_win", "btts_yes"}


def test_same_match_can_use_multiple_distinct_secondary_markets():
    def item(key):
        return SimpleNamespace(key=key, expected_value=.20, model_probability=.6, odds=2.0)

    rows = [{"match_id": 1, "assessments": [
        item("first_half_home_win"), item("corners_over_9_5"), item("home_over_1_5")
    ]}]
    selected = diversified_coupon_rows(rows, min_ev=0.0)
    assert {row["best"].key for row in selected} == {
        "first_half_home_win", "corners_over_9_5", "home_over_1_5"
    }


def test_excluded_match_family_allows_other_market_for_same_match():
    def item(key):
        return SimpleNamespace(key=key, expected_value=.40, model_probability=.6, odds=3.0)

    rows = [{"match_id": 1, "assessments": [item("home_win"), item("btts_yes")], "best": item("home_win")}]
    selected = diversified_coupon_rows(
        rows, min_ev=0.0, excluded_match_families={(1, "result")}
    )
    assert [row["best"].key for row in selected] == ["btts_yes"]
