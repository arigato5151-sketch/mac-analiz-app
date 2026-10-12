"""Turkish names for countries and football competitions used in the UI."""

COUNTRY_NAMES_TR = {
    "austria": "Avusturya",
    "belgium": "Belçika",
    "brazil": "Brezilya",
    "england": "İngiltere",
    "france": "Fransa",
    "germany": "Almanya",
    "greece": "Yunanistan",
    "italy": "İtalya",
    "mexico": "Meksika",
    "netherlands": "Hollanda",
    "portugal": "Portekiz",
    "saudi-arabia": "Suudi Arabistan",
    "saudi arabia": "Suudi Arabistan",
    "scotland": "İskoçya",
    "spain": "İspanya",
    "switzerland": "İsviçre",
    "turkey": "Türkiye",
    "usa": "ABD",
    "united states": "ABD",
    "world": "Uluslararası",
}

LEAGUE_NAMES_TR = {
    ("england", "premier league"): "Premier Lig",
    ("england", "championship"): "İngiltere Championship",
    ("spain", "la liga"): "La Liga",
    ("spain", "segunda división"): "İspanya 2. Ligi",
    ("spain", "segunda division"): "İspanya 2. Ligi",
    ("italy", "serie a"): "İtalya Serie A",
    ("italy", "serie b"): "İtalya Serie B",
    ("brazil", "serie a"): "Brezilya Série A",
    ("germany", "bundesliga"): "Bundesliga",
    ("france", "ligue 1"): "Fransa Ligue 1",
    ("france", "ligue 2"): "Fransa Ligue 2",
    ("portugal", "primeira liga"): "Portekiz Primeira Ligi",
    ("netherlands", "eredivisie"): "Hollanda Eredivisie",
    ("belgium", "jupiler pro league"): "Belçika Pro Ligi",
    ("saudi-arabia", "pro league"): "Suudi Arabistan Pro Ligi",
    ("saudi arabia", "pro league"): "Suudi Arabistan Pro Ligi",
    ("usa", "major league soccer"): "ABD MLS",
    ("united states", "major league soccer"): "ABD MLS",
    ("mexico", "liga mx"): "Meksika Liga MX",
    ("scotland", "premiership"): "İskoçya Premiership",
    ("austria", "bundesliga"): "Avusturya Bundesliga",
    ("switzerland", "super league"): "İsviçre Süper Ligi",
    ("greece", "super league 1"): "Yunanistan Süper Ligi",
    ("turkey", "super lig"): "Türkiye Süper Ligi",
    ("turkey", "süper lig"): "Türkiye Süper Ligi",
    ("world", "uefa champions league"): "UEFA Şampiyonlar Ligi",
    ("world", "uefa europa league"): "UEFA Avrupa Ligi",
    ("world", "uefa europa conference league"): "UEFA Konferans Ligi",
    ("world", "conmebol libertadores"): "Libertadores Kupası",
    ("world", "fifa world cup"): "Dünya Kupası",
    ("world", "euro championship"): "Avrupa Şampiyonası",
    ("world", "uefa nations league"): "UEFA Uluslar Ligi",
    ("world", "africa cup of nations"): "Afrika Uluslar Kupası",
    ("world", "asian cup"): "Asya Kupası",
    ("world", "copa america"): "Amerika Kupası",
    ("world", "concacaf gold cup"): "CONCACAF Altın Kupa",
}


def country_name_tr(country: object) -> str:
    """Translate a catalog country name, preserving unknown names as fallback."""
    if not isinstance(country, str) or not country.strip():
        return "Ülke bilinmiyor"
    normalized = country.strip().casefold()
    return COUNTRY_NAMES_TR.get(normalized, country.strip())


def league_name_tr(country: object, league: object) -> str:
    """Translate known leagues using their country to disambiguate same names."""
    if not isinstance(league, str) or not league.strip():
        return "Lig bilinmiyor"
    country_key = country.strip().casefold() if isinstance(country, str) else ""
    league_name = league.strip()
    return LEAGUE_NAMES_TR.get((country_key, league_name.casefold()), league_name)
