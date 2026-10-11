from data_pipeline.nesine_odds import (
    _extract_rows_from_page,
    istanbul_day_bounds,
    match_quotes,
    parse_rendered_text_rows,
)
from datetime import datetime, timezone


def test_nesine_fixture_query_bounds_are_istanbul_calendar_day():
    start, end = istanbul_day_bounds(datetime(2026, 10, 10, 12, tzinfo=timezone.utc))
    assert start == "2026-10-09T21:00:00+00:00"
    assert end == "2026-10-10T21:00:00+00:00"


class _RenderedPage:
    def __init__(self, rows):
        self.rows = rows

    def evaluate(self, _script, *_args):
        return self.rows


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


def test_rendered_page_falls_back_to_catalog_when_fixture_accents_differ():
    rows = [{
        "home_team": "Galatasaray",
        "away_team": "Kasımpaşa",
        "markets": {"1": "1.50", "X": "4.00", "2": "6.00"},
    }]
    extracted = _extract_rows_from_page(
        _RenderedPage(rows), {("galatasaray", "kasimpasa")}
    )
    assert extracted == rows
    quotes = parse_rendered_text_rows(extracted)
    matched = match_quotes(
        quotes,
        [{"id": 21, "home_team": "Galatasaray", "away_team": "Kasımpaşa"}],
    )
    assert [item[0] for item in matched] == [21]


def test_page_extractor_selects_only_upcoming_fixtures():
    rows = [{
        "home_team": "Upcoming FC", "away_team": "Visitor FC",
        "markets": {"1": "2.00", "X": "3.20", "2": "3.40"},
    }]
    extracted = _extract_rows_from_page(
        _RenderedPage(rows), {("upcoming fc", "visitor fc")}
    )
    assert extracted == rows


def test_detailed_markets_are_preserved():
    quotes = parse_rendered_text_rows([{
        "home_team": "A", "away_team": "B",
        "markets": {"1": "2.00", "X": "3.20", "2": "3.40"},
        "all_markets": [{"market_code": "corners_total", "market_name": "Toplam Korner",
                         "selection_code": "1", "selection_name": "Üst 9.5", "odd": "1.88"}],
    }])
    assert quotes[0].all_markets()[0]["market_code"] == "corners_total"


def test_matches_bragantino_and_mirassol_despite_nesine_suffixes():
    quotes = parse_rendered_text_rows([{
        "home_team": "Bragantino", "away_team": "Mirassol SP",
        "markets": {"1": "1.60", "X": "3.39", "2": "3.89"},
    }])

    matched = match_quotes(
        quotes,
        [{"id": 1492392, "home_team": "RB Bragantino", "away_team": "Mirassol"}],
    )

    assert [item[0] for item in matched] == [1492392]


def test_matches_known_shortened_dynamo_name():
    quotes = parse_rendered_text_rows([{
        "home_team": "Rubin Kazan", "away_team": "D. Moskova",
        "markets": {"1": "2.00", "X": "3.20", "2": "3.40"},
    }])

    matched = match_quotes(
        quotes,
        [{"id": 88, "home_team": "Rubin Kazan", "away_team": "Dynamo Moscow"}],
    )

    assert [item[0] for item in matched] == [88]


def test_known_short_name_does_not_match_unrelated_club():
    quotes = parse_rendered_text_rows([{
        "home_team": "Rubin Kazan", "away_team": "D. Moskova",
        "markets": {"1": "2.00", "X": "3.20", "2": "3.40"},
    }])

    matched = match_quotes(
        quotes,
        [{"id": 88, "home_team": "Rubin Kazan", "away_team": "Dunav Moscow"}],
    )

    assert matched == []


def test_matches_vitoria_guimaraes_feed_name():
    quotes = parse_rendered_text_rows([{
        "home_team": "Benfica", "away_team": "V. Guimaraes",
        "markets": {"1": "1.50", "X": "4.00", "2": "6.00"},
    }])

    assert [item[0] for item in match_quotes(quotes, [
        {"id": 91, "home_team": "Benfica", "away_team": "Vitória SC"},
    ])] == [91]


def test_matches_cska_sofia_variants_without_confusing_cska_teams():
    quotes = parse_rendered_text_rows([
        {"home_team": "CSKA 1948 Sofia", "away_team": "CSKA Sofya",
         "markets": {"1": "2.00", "X": "3.00", "2": "3.50"}},
    ])

    assert [item[0] for item in match_quotes(quotes, [
        {"id": 92, "home_team": "CSKA 1948", "away_team": "CSKA Sofia"},
    ])] == [92]


def test_matches_moroccan_and_spanish_feed_name_variants():
    quotes = parse_rendered_text_rows([
        {"home_team": "US Yacoub Mansour", "away_team": "CRS Salmi",
         "markets": {"1": "2.00", "X": "3.00", "2": "3.50"}},
        {"home_team": "Xerez Deportivo", "away_team": "San Roque",
         "markets": {"1": "2.00", "X": "3.00", "2": "3.50"}},
    ])

    matched = match_quotes(quotes, [
        {"id": 93, "home_team": "US Yacoub El Mansour", "away_team": "CR Salmi"},
        {"id": 94, "home_team": "Xerez Deportivo", "away_team": "San Roque de Lepe"},
    ])

    assert [item[0] for item in matched] == [93, 94]


def test_matches_more_current_feed_variants():
    quotes = parse_rendered_text_rows([
        {"home_team": "Mechelen", "away_team": "St. Truidense",
         "markets": {"1": "2.00", "X": "3.00", "2": "3.50"}},
        {"home_team": "Lillestrom", "away_team": "Molde",
         "markets": {"1": "2.00", "X": "3.00", "2": "3.50"}},
        {"home_team": "F. Dusseldorf", "away_team": "Verl",
         "markets": {"1": "2.00", "X": "3.00", "2": "3.50"}},
        {"home_team": "KAA Gent II", "away_team": "RFC Seraing",
         "markets": {"1": "2.00", "X": "3.00", "2": "3.50"}},
        {"home_team": "Modena", "away_team": "Hellas Verona",
         "markets": {"1": "2.00", "X": "3.00", "2": "3.50"}},
    ])

    matched = match_quotes(quotes, [
        {"id": 101, "home_team": "KV Mechelen", "away_team": "St. Truiden"},
        {"id": 102, "home_team": "Lillestrøm SK", "away_team": "Molde FK"},
        {"id": 103, "home_team": "Fortuna Düsseldorf", "away_team": "SC Verl"},
        {"id": 104, "home_team": "Gent II", "away_team": "Seraing"},
        {"id": 105, "home_team": "Modena FC", "away_team": "Verona"},
    ])

    assert [item[0] for item in matched] == [101, 102, 103, 104, 105]


def test_matches_live_nesine_names_to_daily_app_fixtures():
    cases = [
        ("Olympiakos", "Panathinaikos", "Olympiakos Piraeus", "Panathinaikos"),
        ("AL Shabab (Sar)", "AL Taawoun", "Al Shabab", "Al Taawon"),
        ("Troyes", "Marsilya", "Estac Troyes", "Marseille"),
        ("Atl Mineiro", "Santos SP", "Atletico-MG", "Santos"),
        ("Cadiz", "S. Gijon", "Cadiz", "Sporting Gijon"),
        ("Santander", "Valencia", "Racing Santander", "Valencia"),
        ("Gremio P.", "Internacional", "Gremio", "Internacional"),
    ]
    quotes = parse_rendered_text_rows([
        {"home_team": home, "away_team": away,
         "markets": {"1": "2.00", "X": "3.00", "2": "3.50"}}
        for home, away, _app_home, _app_away in cases
    ])
    matches = [
        {"id": 200 + index, "home_team": app_home, "away_team": app_away}
        for index, (_home, _away, app_home, app_away) in enumerate(cases)
    ]

    assert [item[0] for item in match_quotes(quotes, matches)] == list(range(200, 207))
