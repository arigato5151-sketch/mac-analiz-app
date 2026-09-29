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


def test_coupon_rejects_absurd_odds():
    item = SimpleNamespace(key="away_win", expected_value=20.0, model_probability=.6, odds=46.0)
    rows = [{"match_id": 1, "assessments": [item], "best": item}]
    assert diversified_coupon_rows(rows, min_ev=0.0) == []


def test_coupon_requires_higher_threshold_for_btts():
    item = SimpleNamespace(key="btts_yes", expected_value=.40, model_probability=.52, odds=3.5)
    rows = [{"match_id": 1, "assessments": [item], "best": item}]
    assert diversified_coupon_rows(rows, min_ev=0.0) == []
