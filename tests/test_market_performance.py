from app.components.market_performance import summarize_diversified_market_performance


def test_summary_scores_only_markets_that_cleared_publish_thresholds() -> None:
    rows = [{
        "market_probabilities": {
            "double_chance": {"1X": 0.8, "X2": 0.4, "12": 0.8},
            "total_goals": {"over_1_5": 0.75, "under_3_5": 0.64},
            "team_goals": {"home_over_0_5": 0.8, "away_over_0_5": 0.55},
        },
        "market_performance": {
            "double_chance": {
                "1X": {"actual": True, "brier_score": 0.04},
                "12": {"actual": True, "brier_score": 0.04},
            },
            "total_goals": {
                "over_1_5": {"actual": True, "brier_score": 0.0625},
                "under_3_5": {"actual": True, "brier_score": 0.1296},
            },
            "team_goals": {
                "home_over_0_5": {"actual": True, "brier_score": 0.04},
                "away_over_0_5": {"actual": False, "brier_score": 0.3025},
            },
            "correct_score": {"actual": "1-0", "top_3_hit": True},
        },
    }]

    summary = summarize_diversified_market_performance(rows)
    labels = {row["Pazar"] for row in summary}

    assert labels == {"Çifte şans", "Üst 1.5", "Ev 0.5 Üst", "İlk 3 olası skor"}
    assert all(row["İsabet"] == 1.0 for row in summary)


def test_summary_accepts_postgrest_json_strings() -> None:
    rows = [{
        "market_probabilities": '{"double_chance":{"1X":0.75}}',
        "market_performance": '{"double_chance":{"1X":{"actual":false,"brier_score":0.5625}}}',
    }]

    summary = summarize_diversified_market_performance(rows)

    assert summary == [{"Pazar": "Çifte şans", "Örneklem": 1, "İsabet": 0.0, "Brier": 0.5625}]
