from data_pipeline.nesine_odds import match_quotes, parse_rendered_text_rows


def test_parse_and_match_nesine_result_market():
    quotes = parse_rendered_text_rows([{
        "home_team": "Columbus Cr.", "away_team": "Inter Miami",
        "kickoff_text": "02:00", "markets": {"1": "2.66", "X": "3.54", "2": "1.93", "Over 2.5": "1.27", "Under 2.5": "2.56"},
    }])
    matched = match_quotes(quotes, [{"id": 10, "home_team": "Columbus Crew", "away_team": "Inter Miami"}])
    assert len(matched) == 1
    assert matched[0][0] == 10
    assert matched[0][1].home_win == "2.66"


def test_ambiguous_or_incomplete_rows_are_not_usable():
    quotes = parse_rendered_text_rows([{"home_team": "A", "away_team": "B", "markets": {"1": "1.90"}}])
    assert not quotes[0].has_result_market


def test_detailed_markets_are_preserved():
    quotes = parse_rendered_text_rows([{
        "home_team": "A", "away_team": "B",
        "markets": {"1": "2.00", "X": "3.20", "2": "3.40"},
        "all_markets": [{"market_code": "corners_total", "market_name": "Toplam Korner",
                         "selection_code": "1", "selection_name": "Üst 9.5", "odd": "1.88"}],
    }])
    assert quotes[0].all_markets()[0]["market_code"] == "corners_total"
