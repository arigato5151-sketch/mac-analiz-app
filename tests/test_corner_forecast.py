from data_pipeline.fetch_results import extract_corner_counts
from models.feature_engineering import CausalFeatureState, build_upcoming_corner_predictions
from models.market_forecast import derive_corner_probabilities


def test_extract_corner_counts_from_fixture_statistics():
    payload = [
        {"team": {"id": 1}, "statistics": [{"type": "Corner Kicks", "value": 7}]},
        {"team": {"id": 2}, "statistics": [{"type": "Corner Kicks", "value": 4}]},
    ]
    assert extract_corner_counts(payload, home_team_id=1, away_team_id=2) == (7, 4)
    assert extract_corner_counts(payload[:1], home_team_id=1, away_team_id=2) is None


def test_corner_poisson_probabilities_are_valid_and_monotonic():
    probabilities = derive_corner_probabilities(5.2, 4.1)
    assert 0 < probabilities["over_8_5"] < 1
    assert probabilities["over_7_5"] > probabilities["over_8_5"]
    assert probabilities["over_8_5"] + probabilities["under_8_5"] == 1


def test_corner_forecast_uses_only_prior_matches_and_requires_three_per_team():
    history = []
    for fixture_id in range(1, 4):
        history.append({
            "id": fixture_id, "league_id": 1, "home_team_id": 10,
            "away_team_id": 20 + fixture_id, "match_date": f"2026-01-0{fixture_id}T12:00:00Z",
            "home_score": 1, "away_score": 0, "home_corners": 6, "away_corners": 3,
        })
        history.append({
            "id": fixture_id + 10, "league_id": 1, "home_team_id": 30 + fixture_id,
            "away_team_id": 40, "match_date": f"2026-01-0{fixture_id}T14:00:00Z",
            "home_score": 0, "away_score": 1, "home_corners": 2, "away_corners": 8,
        })
    target = {"id": 50, "home_team_id": 10, "away_team_id": 40,
              "match_date": "2026-01-05T12:00:00Z"}
    result = build_upcoming_corner_predictions(history, [target])
    assert 50 in result
    assert result[50]["home_expected_corners"] == 4.0
    assert result[50]["away_expected_corners"] == 5.5

    state = CausalFeatureState()
    assert state.corner_baseline({"home_team_id": 10, "away_team_id": 40}) is None
