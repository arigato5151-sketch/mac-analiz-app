import pytest

from models.feature_engineering import (
    FEATURE_COLUMNS,
    build_training_dataset,
    build_upcoming_features,
)


def completed_match(
    match_id: int,
    date: str,
    home_score: int,
    away_score: int,
) -> dict[str, object]:
    return {
        "id": match_id,
        "league_id": 39,
        "home_team_id": 1,
        "away_team_id": 2,
        "match_date": date,
        "status": "finished",
        "home_score": home_score,
        "away_score": away_score,
    }


def test_features_are_causal_and_updates_apply_to_next_match() -> None:
    features, labels = build_training_dataset(
        [
            completed_match(1, "2026-01-01T12:00:00+00:00", 3, 0),
            completed_match(2, "2026-01-08T12:00:00+00:00", 1, 1),
        ]
    )

    assert tuple(features.columns) == FEATURE_COLUMNS
    assert features.iloc[0]["home_win_rate_5"] == 0.33
    assert features.iloc[1]["home_win_rate_5"] == 1.0
    assert features.iloc[1]["elo_diff"] > 0
    assert features.iloc[1]["home_rest_days"] == 7.0
    assert labels["result"].tolist() == [0, 1]


def test_upcoming_matches_do_not_mutate_state_without_results() -> None:
    history = [completed_match(1, "2026-01-01T12:00:00+00:00", 3, 0)]
    upcoming = [
        {
            **completed_match(2, "2026-01-08T12:00:00+00:00", 0, 0),
            "home_score": None,
            "away_score": None,
        },
        {
            **completed_match(3, "2026-01-09T12:00:00+00:00", 0, 0),
            "home_score": None,
            "away_score": None,
        },
    ]

    features = build_upcoming_features(history, upcoming)

    assert features.loc[2, "home_win_rate_5"] == 1.0
    assert features.loc[3, "home_win_rate_5"] == 1.0
    assert features.loc[2, "elo_diff"] == features.loc[3, "elo_diff"]


def test_upcoming_features_use_latest_live_team_form() -> None:
    history = [completed_match(1, "2026-01-01T12:00:00+00:00", 1, 0)]
    upcoming = [
        {
            **completed_match(2, "2026-01-08T12:00:00+00:00", 0, 0),
            "home_score": None,
            "away_score": None,
        }
    ]
    team_forms = {
        1: {
            "elo_rating": 1610,
            "avg_goals_scored_last5": 2.2,
            "avg_goals_conceded_last5": 0.8,
            "win_rate_last5": 0.8,
            "home_away_split": {"home_win_rate": 0.75},
        },
        2: {
            "elo_rating": 1475,
            "avg_goals_scored_last5": 1.0,
            "avg_goals_conceded_last5": 1.4,
            "win_rate_last5": 0.2,
            "home_away_split": '{"away_win_rate": 0.25}',
        },
    }

    features = build_upcoming_features(
        history, upcoming, team_form_by_id=team_forms
    )

    assert features.loc[2, "home_win_rate_5"] == 0.8
    assert features.loc[2, "away_goal_diff_5"] == pytest.approx(-0.4)
    assert features.loc[2, "home_venue_win_rate_5"] == 0.75
    assert features.loc[2, "away_venue_win_rate_5"] == 0.25
    assert features.loc[2, "elo_diff"] == 135
