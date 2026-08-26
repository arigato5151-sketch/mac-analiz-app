"""API-Football leagues tracked by the application.

The identifiers and active seasons were verified against ``GET /leagues`` on
2026-08-25. The tuple order is the display/priority order used by the app.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LeagueConfig:
    id: int
    name: str
    country: str
    season: int


TRACKED_LEAGUES: tuple[LeagueConfig, ...] = (
    LeagueConfig(39, "Premier League", "England", 2026),
    LeagueConfig(140, "La Liga", "Spain", 2026),
    LeagueConfig(135, "Serie A", "Italy", 2026),
    LeagueConfig(78, "Bundesliga", "Germany", 2026),
    LeagueConfig(61, "Ligue 1", "France", 2026),
    LeagueConfig(2, "UEFA Champions League", "World", 2026),
    LeagueConfig(3, "UEFA Europa League", "World", 2026),
    LeagueConfig(848, "UEFA Europa Conference League", "World", 2026),
    LeagueConfig(203, "Süper Lig", "Turkey", 2026),
    LeagueConfig(94, "Primeira Liga", "Portugal", 2026),
    LeagueConfig(88, "Eredivisie", "Netherlands", 2026),
    LeagueConfig(144, "Jupiler Pro League", "Belgium", 2026),
    LeagueConfig(40, "Championship", "England", 2026),
    LeagueConfig(307, "Pro League", "Saudi-Arabia", 2026),
    LeagueConfig(253, "Major League Soccer", "USA", 2026),
    LeagueConfig(71, "Serie A", "Brazil", 2026),
    LeagueConfig(262, "Liga MX", "Mexico", 2026),
    LeagueConfig(179, "Premiership", "Scotland", 2026),
    LeagueConfig(218, "Bundesliga", "Austria", 2026),
    LeagueConfig(207, "Super League", "Switzerland", 2026),
    LeagueConfig(136, "Serie B", "Italy", 2026),
    LeagueConfig(141, "Segunda División", "Spain", 2026),
    LeagueConfig(62, "Ligue 2", "France", 2026),
    LeagueConfig(197, "Super League 1", "Greece", 2026),
    LeagueConfig(13, "CONMEBOL Libertadores", "World", 2026),
)

LEAGUES_BY_ID: dict[int, LeagueConfig] = {
    league.id: league for league in TRACKED_LEAGUES
}
TRACKED_LEAGUE_IDS: tuple[int, ...] = tuple(LEAGUES_BY_ID)


def validate_league_config() -> None:
    """Fail fast when an edit introduces duplicate or invalid entries."""
    if len(TRACKED_LEAGUES) != 25:
        raise ValueError("Exactly 25 tracked leagues are required")
    if len(LEAGUES_BY_ID) != len(TRACKED_LEAGUES):
        raise ValueError("Tracked API-Football league IDs must be unique")
    if any(league.id <= 0 or league.season < 2000 for league in TRACKED_LEAGUES):
        raise ValueError("League IDs and seasons must be valid positive values")


validate_league_config()
