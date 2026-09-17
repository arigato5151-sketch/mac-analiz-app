"""Isolated adapter for StatsBomb Open Data.

Used strictly for tactical research, retrospective match analysis, and feature exploration.
Never used directly for live API-Football pre-match predictions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from config.settings import PROJECT_ROOT

LOGGER = logging.getLogger(__name__)

CACHE_DIR = PROJECT_ROOT / "data" / "statsbomb_cache"

try:
    from statsbombpy import sb
    STATSBOMB_AVAILABLE = True
except ImportError:
    sb = None
    STATSBOMB_AVAILABLE = False


class StatsBombProviderProtocol(Protocol):
    """Abstract protocol for StatsBomb data providers."""

    def get_competitions(self) -> pd.DataFrame:
        ...

    def get_matches(self, competition_id: int, season_id: int) -> pd.DataFrame:
        ...

    def get_events(self, match_id: int) -> pd.DataFrame:
        ...


class StatsBombAdapter:
    """Adapter to fetch and cache StatsBomb Open Data safely with graceful fallbacks."""

    def __init__(self, cache_dir: Path = CACHE_DIR) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def is_available(self) -> bool:
        return STATSBOMB_AVAILABLE

    def get_competitions(self) -> pd.DataFrame:
        """Fetch available open competitions."""
        if not STATSBOMB_AVAILABLE:
            LOGGER.warning("statsbombpy is not available in the environment")
            return pd.DataFrame()

        cache_file = self.cache_dir / "competitions.parquet"
        if cache_file.exists():
            try:
                return pd.read_parquet(cache_file)
            except Exception:
                pass

        try:
            competitions = sb.competitions()
            if isinstance(competitions, pd.DataFrame) and not competitions.empty:
                try:
                    competitions.to_parquet(cache_file)
                except Exception:
                    pass
                return competitions
            return pd.DataFrame()
        except Exception as exc:
            LOGGER.warning("Error fetching StatsBomb competitions: %s", exc)
            return pd.DataFrame()

    def get_matches(self, competition_id: int, season_id: int) -> pd.DataFrame:
        """Fetch matches for a specific open competition and season."""
        if not STATSBOMB_AVAILABLE:
            return pd.DataFrame()

        cache_file = self.cache_dir / f"matches_{competition_id}_{season_id}.parquet"
        if cache_file.exists():
            try:
                return pd.read_parquet(cache_file)
            except Exception:
                pass

        try:
            matches = sb.matches(competition_id=competition_id, season_id=season_id)
            if isinstance(matches, pd.DataFrame) and not matches.empty:
                try:
                    matches.to_parquet(cache_file)
                except Exception:
                    pass
                return matches
            return pd.DataFrame()
        except Exception as exc:
            LOGGER.warning(
                "Error fetching StatsBomb matches for comp=%s season=%s: %s",
                competition_id,
                season_id,
                exc,
            )
            return pd.DataFrame()

    def get_events(self, match_id: int) -> pd.DataFrame:
        """Fetch event-level data for a single StatsBomb match."""
        if not STATSBOMB_AVAILABLE:
            return pd.DataFrame()

        cache_file = self.cache_dir / f"events_{match_id}.parquet"
        if cache_file.exists():
            try:
                return pd.read_parquet(cache_file)
            except Exception:
                pass

        try:
            events = sb.events(match_id=match_id)
            if isinstance(events, pd.DataFrame) and not events.empty:
                try:
                    events.to_parquet(cache_file)
                except Exception:
                    pass
                return events
            return pd.DataFrame()
        except Exception as exc:
            LOGGER.warning("Error fetching StatsBomb events for match %s: %s", match_id, exc)
            return pd.DataFrame()

    def find_match_by_teams(
        self, home_team: str, away_team: str, year: int | None = None
    ) -> dict[str, Any] | None:
        """Search available open data matches for matching team names.

        Returns match metadata dict if found, else None.
        """
        if not STATSBOMB_AVAILABLE:
            return None

        comps = self.get_competitions()
        if comps.empty:
            return None

        norm_home = home_team.strip().lower()
        norm_away = away_team.strip().lower()

        # Scan cached matches or competitions
        for _, comp_row in comps.iterrows():
            comp_id = int(comp_row["competition_id"])
            season_id = int(comp_row["season_id"])
            matches = self.get_matches(comp_id, season_id)
            if matches.empty:
                continue

            for _, match in matches.iterrows():
                m_home = str(match.get("home_team", "")).strip().lower()
                m_away = str(match.get("away_team", "")).strip().lower()
                if (norm_home in m_home or m_home in norm_home) and (
                    norm_away in m_away or m_away in norm_away
                ):
                    return match.to_dict()

        return None

