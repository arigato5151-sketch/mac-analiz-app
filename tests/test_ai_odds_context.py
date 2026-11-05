from app.ai_match_analysis import _iddaa_odds_context


def test_iddaa_odds_context_preserves_prices_and_removes_market_margin() -> None:
    quote = {
            "captured_at": "2026-11-05T10:00:00+00:00",
            "source_updated_at": "2026-11-05T10:00:00+00:00",
            "odds": {
                "source": "iddaa.com",
                "event_id": 123,
                "home_team": "Arsenal",
                "away_team": "Chelsea",
                "markets": [
                    {
                        "market_id": 36,
                        "sport_type": 2,
                        "market_name": "Maç Sonucu",
                        "selections": [
                            {"name": "1", "odds": 2.0},
                            {"name": "X", "odds": 3.5},
                            {"name": "2", "odds": 4.0},
                        ],
                    },
                    {
                        "market_id": 12,
                        "sport_type": 2,
                        "market_name": "Toplam Korner Alt/Üst",
                        "selections": [
                            {"name": "Alt", "odds": 1.9},
                            {"name": "Üst", "odds": 1.9},
                        ],
                    },
                ],
            },
        }
    context = _iddaa_odds_context(quote, fixture_home="Arsenal", fixture_away="Chelsea")

    assert context is not None
    assert context["event_home_team"] == "Arsenal"
    assert len(context["markets"]) == 1
    assert context["markets"][0]["kind"] == "result"
    assert context["markets"][0]["selections"][0]["odds"] == 2.0
    assert round(
        sum(item["no_vig_market_probability"] for item in context["markets"][0]["selections"]),
        10,
    ) == 1.0


def test_odds_context_rejects_non_iddaa_quotes() -> None:
    assert _iddaa_odds_context({"odds": {"source": "other", "markets": []}}) is None


def test_odds_context_rejects_old_quote_with_reversed_teams() -> None:
    quote = {
        "odds": {
            "source": "iddaa.com",
            "home_team": "Chelsea",
            "away_team": "Arsenal",
            "markets": [],
        }
    }
    assert _iddaa_odds_context(
        quote, fixture_home="Arsenal", fixture_away="Chelsea"
    ) is None


def test_iddaa_generic_match_result_over_under_label_is_total_goals() -> None:
    context = _iddaa_odds_context(
        {
            "odds": {
                "source": "iddaa.com",
                "markets": [
                    {
                        "market_id": 101,
                        "sport_type": 2,
                        "market_name": "Maç Sonucu Alt/Üst",
                        "line": "2.5",
                        "selections": [
                            {"name": "Alt", "odds": 1.9},
                            {"name": "Üst", "odds": 1.9},
                        ],
                    }
                ],
            }
        }
    )

    assert context is not None
    assert context["markets"][0]["kind"] == "total_goals"


def test_iddaa_both_teams_to_score_market_is_recognized() -> None:
    context = _iddaa_odds_context(
        {
            "odds": {
                "source": "iddaa.com",
                "markets": [
                    {
                        "market_id": 89,
                        "sport_type": 2,
                        "market_name": "Karşılıklı Gol / Toplam",
                        "selections": [
                            {"name": "Var", "odds": 1.7},
                            {"name": "Yok", "odds": 2.0},
                        ],
                    }
                ],
            }
        }
    )

    assert context is not None
    assert context["markets"][0]["kind"] == "both_teams_to_score"
