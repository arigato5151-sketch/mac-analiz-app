from data_pipeline.nesine_odds import (
    _extract_rows_from_page,
    _partition_quotes_to_api_slate,
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
        return {
            "rows": self.rows,
            "lastLeagueName": "",
            "lastDayLabel": "",
        }


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
    extracted, _, _ = _extract_rows_from_page(
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
    extracted, _, _ = _extract_rows_from_page(
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


def test_matches_verified_uefa_nesine_labels_by_league_team_ids_and_kickoff():
    cases = [
        (3, "UEFA AVRUPA LİGİ", "Salzburg", "AC Milan", 571, 489, "Red Bull Salzburg", "AC Milan"),
        (3, "UEFA AVRUPA LİGİ", "AZ Alkmaar", "HB Sheva", 201, 563, "AZ Alkmaar", "Hapoel Beer Sheva"),
        (3, "UEFA AVRUPA LİGİ", "Lech Poznan", "B. Leverkusen", 347, 168, "Lech Poznan", "Bayer Leverkusen"),
        (3, "UEFA AVRUPA LİGİ", "Sparta Prag", "Lillestrom", 628, 321, "Sparta Praha", "Lillestrom"),
        (3, "UEFA AVRUPA LİGİ", "NK Celje", "Omonia", 4360, 3402, "Celje", "Omonia Nicosia"),
        (3, "UEFA AVRUPA LİGİ", "U. S. Gilloise", "R. Sociedad", 1393, 548, "Union St. Gilloise", "Real Sociedad"),
        (3, "UEFA AVRUPA LİGİ", "D. Zagreb", "Anderlecht", 620, 554, "Dinamo Zagreb", "Anderlecht"),
        (3, "UEFA AVRUPA LİGİ", "NEC Nijmegen", "Levski Sofya", 413, 646, "NEC Nijmegen", "Levski Sofia"),
        (3, "UEFA AVRUPA LİGİ", "Ferencvaros", "Viktoria Plzen", 651, 567, "Ferencvarosi TC", "Plzen"),
        (3, "UEFA AVRUPA LİGİ", "Hoffenheim", "Beşiktaş", 167, 549, "1899 Hoffenheim", "Beşiktaş"),
        (3, "UEFA AVRUPA LİGİ", "Rennes", "Ofi Crete", 94, 1124, "Rennes", "OFI"),
        (848, "UEFA KONFERANS LİGİ", "Kuopion", "Trabzonspor", 1165, 998, "KuPS", "Trabzonspor"),
        (848, "UEFA KONFERANS LİGİ", "Riga", "K. Almaty", 10124, 664, "Riga", "Kairat Almaty"),
        (848, "UEFA KONFERANS LİGİ", "Panathinaikos", "Banja Luka", 617, 3364, "Panathinaikos", "Borac Banja Luka"),
        (848, "UEFA KONFERANS LİGİ", "Lugano", "Kızıl Yıldız", 606, 598, "FC Lugano", "FK Crvena Zvezda"),
        (848, "UEFA KONFERANS LİGİ", "KF Egnatia R.", "Midtjylland", 3327, 397, "Egnatia Rrogozhinë", "FC Midtjylland"),
        (848, "UEFA KONFERANS LİGİ", "Un. Craiova", "Getafe", 632, 546, "Universitatea Craiova", "Getafe"),
        (848, "UEFA KONFERANS LİGİ", "KAA Gent", "AGF Aarhus", 631, 406, "Gent", "Aarhus"),
        (848, "UEFA KONFERANS LİGİ", "Hajduk Split", "Ajax", 608, 194, "HNK Hajduk Split", "Ajax"),
        (848, "UEFA KONFERANS LİGİ", "Mjallby", "Inter Escaldes", 2240, 3342, "Mjallby AIF", "Inter Club d'Escaldes"),
        (848, "UEFA KONFERANS LİGİ", "St. Truidense", "Iberia 1999", 735, 3502, "St. Truiden", "FC Iberia 1999"),
        (848, "UEFA KONFERANS LİGİ", "Brann", "Lincoln Red", 319, 667, "Brann", "Lincoln Red Imps FC"),
        (848, "UEFA KONFERANS LİGİ", "Brighton", "Zalgiris Kaunas", 51, 3872, "Brighton", "Kauno Žalgiris"),
        (848, "UEFA KONFERANS LİGİ", "Hearts", "Nordsjælland", 254, 398, "Heart Of Midlothian", "FC Nordsjaelland"),
        (848, "UEFA KONFERANS LİGİ", "Kopenhag", "Braga", 400, 217, "FC Copenhagen", "SC Braga"),
    ]
    quotes = parse_rendered_text_rows([
        {"home_team": home, "away_team": away, "league_name": league,
         "kickoff_text": "19:45", "markets": {"1": "2.00", "X": "3.00", "2": "3.50"}}
        for _league_id, league, home, away, *_ in cases
    ])
    matches = [
        {"id": index, "league_id": league_id, "home_team_id": home_id, "away_team_id": away_id,
         "home_team": api_home, "away_team": api_away,
         "kickoff": "2026-10-15T19:45:00+03:00"}
        for index, (league_id, _league, _home, _away, home_id, away_id, api_home, api_away)
        in enumerate(cases, start=1)
    ]

    assert [item[0] for item in match_quotes(quotes, matches)] == list(range(1, len(cases) + 1))


def test_unrelated_bulletin_rows_are_excluded_but_uncertain_slate_rows_remain():
    quotes = parse_rendered_text_rows([
        {"home_team": "BG Pathum Utd", "away_team": "Adelaide Utd",
         "league_name": "UEFA KONFERANS LİGİ", "kickoff_text": "15:15",
         "markets": {"1": "2.00", "X": "3.00", "2": "3.50"}},
        {"home_team": "Kopenhag", "away_team": "Braga",
         "league_name": "UEFA KONFERANS LİGİ", "kickoff_text": "22:00",
         "markets": {"1": "2.00", "X": "3.00", "2": "3.50"}},
    ])
    api_matches = [
        {"id": 1, "league_id": 848, "home_team": "Egnatia Rrogozhinë",
         "away_team": "FC Midtjylland", "home_team_id": 3327, "away_team_id": 397,
         "kickoff": "2026-10-15T19:45:00+03:00"},
        {"id": 2, "league_id": 848, "home_team": "FC Copenhagen",
         "away_team": "SC Braga", "home_team_id": 400, "away_team_id": 217,
         "kickoff": "2026-10-15T22:00:00+03:00"},
    ]

    kept, out_of_slate = _partition_quotes_to_api_slate(quotes, api_matches)

    assert [(quote.home_team, quote.away_team) for quote in out_of_slate] == [
        ("BG Pathum Utd", "Adelaide Utd")
    ]
    assert [(quote.home_team, quote.away_team) for quote in kept] == [("Kopenhag", "Braga")]
