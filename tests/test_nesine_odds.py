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


def test_missing_home_selection_keeps_draw_and_away_prices_in_their_columns():
    quotes = parse_rendered_text_rows([{
        "home_team": "Albania", "away_team": "San Marino",
        "markets": {"1": "-", "X": "17.50", "2": "17.50", "Over 2.5": "3.73"},
    }])

    assert quotes[0].home_win is None
    assert quotes[0].draw == "17.50"
    assert quotes[0].away_win == "17.50"
    assert not quotes[0].has_result_market


def test_only_exact_app_fixtures_are_matched():
    quotes = parse_rendered_text_rows([
        {"home_team": "Hull City U21", "away_team": "Wigan Ath U21",
         "markets": {"1": "2.00", "X": "3.20", "2": "3.40"}},
        {"home_team": "Columbus Crew", "away_team": "Inter Miami",
         "markets": {"1": "2.00", "X": "3.20", "2": "3.40"}},
    ])
    matches = [{"id": 10, "home_team": "Hull City", "away_team": "Everton"},
               {"id": 11, "home_team": "Columbus Crew", "away_team": "Inter Miami"}]
    matched = match_quotes(quotes, matches)
    assert [item[0] for item in matched] == [11]


def test_detailed_markets_are_preserved():
    quotes = parse_rendered_text_rows([{
        "home_team": "A", "away_team": "B",
        "markets": {"1": "2.00", "X": "3.20", "2": "3.40"},
        "all_markets": [{"market_code": "corners_total", "market_name": "Toplam Korner",
                         "selection_code": "1", "selection_name": "Üst 9.5", "odd": "1.88"}],
    }])
    assert quotes[0].all_markets()[0]["market_code"] == "corners_total"
