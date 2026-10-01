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
