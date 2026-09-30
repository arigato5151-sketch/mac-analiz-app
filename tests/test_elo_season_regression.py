from models.feature_engineering import CausalFeatureState


def _match(date: str, home: int, away: int, season: str) -> dict:
    return {
        "league_id": 7,
        "season": season,
        "match_date": date,
        "home_team_id": home,
        "away_team_id": away,
        "home_score": 3,
        "away_score": 0,
        "home_xg": None,
        "away_xg": None,
    }


def test_new_season_regresses_returning_teams_to_league_initial_elo():
    state = CausalFeatureState(league_initial_elos={7: 1600.0})
    first = _match("2025-05-01T18:00:00Z", 1, 2, "2024-25")
    state.update(first)
    assert state.states[1].elo > 1600
    previous_elo = state.states[1].elo

    next_season = _match("2025-08-01T18:00:00Z", 1, 2, "2025-26")
    features = state.feature_row(next_season)
    assert features["home_elo"] == 1600.0 + 0.7 * (previous_elo - 1600.0)
