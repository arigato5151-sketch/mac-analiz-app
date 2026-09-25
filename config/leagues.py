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
    # National-team competitions. API-Football exposes these through the
    # same fixtures endpoint as club competitions, so they can share the
    # existing sync, prediction, and dashboard pipeline.
    LeagueConfig(1, "FIFA World Cup", "World", 2026),
    LeagueConfig(4, "Euro Championship", "World", 2026),
    LeagueConfig(5, "UEFA Nations League", "World", 2026),
    LeagueConfig(6, "Africa Cup of Nations", "World", 2026),
    LeagueConfig(7, "Asian Cup", "World", 2026),
    LeagueConfig(9, "Copa America", "World", 2026),
    LeagueConfig(22, "CONCACAF Gold Cup", "World", 2026),
)

LEAGUES_BY_ID: dict[int, LeagueConfig] = {
    league.id: league for league in TRACKED_LEAGUES
}

# Re-estimated from 25,911 completed historical matches on 2026-09-21:
# realized home expected score converted to Elo points per league. Every
# tracked league carries its own measured value so nothing falls back to the
# global prior silently.
HOME_ADVANTAGE_BY_LEAGUE: dict[int, float] = {
    1: 52, 2: 59, 3: 69, 4: 52, 5: 52, 6: 52, 7: 52, 9: 52,
    13: 107, 22: 52, 39: 36, 40: 49, 61: 41, 62: 40,
    71: 80, 78: 34, 88: 47, 94: 38, 135: 29, 136: 48, 140: 61,
    141: 66, 144: 45, 179: 47, 197: 35, 203: 61, 207: 53, 218: 33,
    253: 61, 262: 67, 307: 32, 848: 65,
}
DEFAULT_HOME_ADVANTAGE = 52


def home_advantage_for_league(league_id: int) -> float:
    return HOME_ADVANTAGE_BY_LEAGUE.get(league_id, DEFAULT_HOME_ADVANTAGE)
TRACKED_LEAGUE_IDS: tuple[int, ...] = tuple(LEAGUES_BY_ID)


def validate_league_config() -> None:
    """Fail fast when an edit introduces duplicate or invalid entries."""
    if len(TRACKED_LEAGUES) != 32:
        raise ValueError("Exactly 32 tracked leagues are required")
    if len(LEAGUES_BY_ID) != len(TRACKED_LEAGUES):
        raise ValueError("Tracked API-Football league IDs must be unique")
    if any(league.id <= 0 or league.season < 2000 for league in TRACKED_LEAGUES):
        raise ValueError("League IDs and seasons must be valid positive values")


validate_league_config()
