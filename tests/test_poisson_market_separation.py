from models.feature_engineering import CausalFeatureState


def test_missing_market_quotes_do_not_overwrite_market_features_with_poisson():
    state = CausalFeatureState()
    row = {
        "league_id": 1,
        "match_date": "2026-01-01T18:00:00Z",
        "home_team_id": 1,
        "away_team_id": 2,
    }

    features = state.feature_row(row)

    assert features["market_odds_available"] == 0
    assert features["market_implied_home_win"] == 1 / 3
    assert features["poisson_home_win"] != features["market_implied_home_win"]
