from models.feature_engineering import CausalFeatureState


def test_infeasible_league_rho_falls_back_for_this_fixture_only():
    state = CausalFeatureState(league_rhos={1: 0.5})
    state.league_goals[1].append((6, 6))
    row = {"league_id": 1, "home_team_id": 1, "away_team_id": 2}

    prediction = state.poisson_baseline(row)

    assert prediction.dixon_coles_rho == 0.0
    assert prediction.score_matrix.sum() == 1.0
    assert state.league_rhos[1] == 0.5


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


def test_partial_market_quotes_keep_missing_binary_markets_neutral():
    state = CausalFeatureState()
    row = {
        "league_id": 1,
        "match_date": "2026-01-01T18:00:00Z",
        "home_team_id": 1,
        "away_team_id": 2,
        "market_odds": {
            "home_win": 2.0,
            "draw": 4.0,
            "away_win": 4.0,
        },
    }

    features = state.feature_row(row)

    assert features["market_odds_available"] == 1
    assert features["market_implied_over_2_5"] == 0.5
    assert features["market_implied_btts"] == 0.5
    assert features["market_over_2_5_move"] == 0.0
    assert features["market_btts_move"] == 0.0


def test_partial_opening_market_quotes_keep_missing_moves_neutral():
    state = CausalFeatureState()
    row = {
        "league_id": 1,
        "match_date": "2026-01-01T18:00:00Z",
        "home_team_id": 1,
        "away_team_id": 2,
        "market_odds": {
            "home_win": 2.0,
            "draw": 4.0,
            "away_win": 4.0,
        },
        "market_opening_odds": {
            "home_win": 2.2,
            "draw": 3.8,
            "away_win": 3.8,
        },
    }

    features = state.feature_row(row)

    assert features["market_over_2_5_move"] == 0.0
    assert features["market_btts_move"] == 0.0
