"""Isolated adapter for StatsBomb Open Data.

Used strictly for tactical research, retrospective match analysis, and feature exploration.
Never used directly for live API-Football pre-match predictions.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import re
import unicodedata
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

    def competitions(self) -> pd.DataFrame:
        ...

    def matches(self, competition_id: int, season_id: int) -> pd.DataFrame:
        ...

    def events(self, match_id: int) -> pd.DataFrame:
        ...


def _normalize_name(name: str) -> str:
    """Normalize a team name for reliable fuzzy/token comparison."""
    if not name:
        return ""
    # Normalize unicode accents
    norm = unicodedata.normalize("NFKD", name)
    norm = "".join(c for c in norm if not unicodedata.combining(c))
    # Remove punctuation and lowercase
    norm = re.sub(r"[^\w\s]", " ", norm.lower())
    return " ".join(norm.split())


def _team_names_match(query: str, candidate: str) -> bool:
    """Compare team names using normalized token overlap or containment."""
    q_norm = _normalize_name(query)
    c_norm = _normalize_name(candidate)
    if not q_norm or not c_norm:
        return False
    if q_norm == c_norm or q_norm in c_norm or c_norm in q_norm:
        return True
    # Check significant word token overlap
    q_tokens = set(q_norm.split()) - {"fc", "cf", "sc", "afc", "united", "city", "de", "the"}
    c_tokens = set(c_norm.split()) - {"fc", "cf", "sc", "afc", "united", "city", "de", "the"}
    return bool(q_tokens and q_tokens.issubset(c_tokens))


class StatsBombAdapter:
    """Adapter to fetch and cache StatsBomb Open Data safely with graceful fallbacks.

    Supports dependency injection for testing via the ``provider`` argument.
    """

    def __init__(
        self,
        cache_dir: Path = CACHE_DIR,
        provider: Any | None = None,
        timeout_seconds: float = 10.0,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout_seconds = timeout_seconds

        if provider is not None:
            self._provider = provider
            self._available = True
        elif STATSBOMB_AVAILABLE and sb is not None:
            self._provider = sb
            self._available = True
        else:
            self._provider = None
            self._available = False

    @property
    def is_available(self) -> bool:
        return self._available and self._provider is not None

    def _call_with_timeout(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        """Call a provider function with a timeout to prevent hanging UI."""
        """Call a provider function with a timeout to prevent hanging UI.

        Explicitly manages ThreadPoolExecutor so that when a timeout occurs,
        the UI thread returns immediately without waiting for the blocked worker.
        """
        if not self.is_available:
            return None
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func, *args, **kwargs)
            try:
                return future.result(timeout=self.timeout_seconds)
            except concurrent.futures.TimeoutError:
                LOGGER.warning("StatsBomb call %s timed out after %.1fs", getattr(func, "__name__", "call"), self.timeout_seconds)
                return None
            except Exception as exc:
                LOGGER.warning("StatsBomb call failed: %s", exc)
                return None
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=self.timeout_seconds)
            res = future.result(timeout=self.timeout_seconds)
            executor.shutdown(wait=False, cancel_futures=True)
            return res
        except concurrent.futures.TimeoutError:
            LOGGER.warning(
                "StatsBomb call %s timed out after %.1fs",
                getattr(func, "__name__", "call"),
                self.timeout_seconds,
            )
            executor.shutdown(wait=False, cancel_futures=True)
            return None
        except Exception as exc:
            LOGGER.warning("StatsBomb call failed: %s", exc)
            executor.shutdown(wait=False, cancel_futures=True)
            return None
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    def get_competitions(self) -> pd.DataFrame:
        """Fetch available open competitions with disk caching."""
        cache_file = self.cache_dir / "competitions.parquet"
        if cache_file.exists():
            try:
                return pd.read_parquet(cache_file)
            except Exception as exc:
                LOGGER.debug("Cache read error for competitions: %s", exc)

        if not self.is_available:
            LOGGER.debug("StatsBomb provider not available, returning empty competitions")
            return pd.DataFrame()

        try:
            competitions = self._call_with_timeout(self._provider.competitions)
            if isinstance(competitions, pd.DataFrame) and not competitions.empty:
                try:
                    competitions.to_parquet(cache_file)
                except Exception as exc:
                    LOGGER.debug("Cache write error for competitions: %s", exc)
                return competitions
            return pd.DataFrame()
        except Exception as exc:
            LOGGER.warning("Error fetching StatsBomb competitions: %s", exc)
            return pd.DataFrame()

    def get_matches(self, competition_id: int, season_id: int) -> pd.DataFrame:
        """Fetch matches for a specific open competition and season with disk caching."""
        cache_file = self.cache_dir / f"matches_{competition_id}_{season_id}.parquet"
        if cache_file.exists():
            try:
                return pd.read_parquet(cache_file)
            except Exception as exc:
                LOGGER.debug("Cache read error for matches: %s", exc)

        if not self.is_available:
            return pd.DataFrame()

        try:
            matches = self._call_with_timeout(
                self._provider.matches,
                competition_id=competition_id,
                season_id=season_id,
            )
            if isinstance(matches, pd.DataFrame) and not matches.empty:
                try:
                    matches.to_parquet(cache_file)
                except Exception as exc:
                    LOGGER.debug("Cache write error for matches: %s", exc)
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
        """Fetch event-level data for a single StatsBomb match with disk caching."""
        cache_file = self.cache_dir / f"events_{match_id}.parquet"
        if cache_file.exists():
            try:
                return pd.read_parquet(cache_file)
            except Exception as exc:
                LOGGER.debug("Cache read error for events: %s", exc)

        if not self.is_available:
            return pd.DataFrame()

        try:
            events = self._call_with_timeout(self._provider.events, match_id=match_id)
            if isinstance(events, pd.DataFrame) and not events.empty:
                try:
                    events.to_parquet(cache_file)
                except Exception as exc:
                    LOGGER.debug("Cache write error for events: %s", exc)
                return events
            return pd.DataFrame()
        except Exception as exc:
            LOGGER.warning("Error fetching StatsBomb events for match %s: %s", match_id, exc)
            return pd.DataFrame()

    def find_match_by_teams(
        self, home_team: str, away_team: str, year: int | None = None
    ) -> dict[str, Any] | None:
        """Search available open data matches for matching team names.

        Applies year filtering when requested to avoid scanning irrelevant tournaments.
        Uses normalized token comparison.
        Returns match metadata dict if found, else None.
        """
        if not self.is_available:
            return None

        # Check lookup cache with normalized names and year
        norm_h = _normalize_name(home_team).replace(" ", "_")
        norm_a = _normalize_name(away_team).replace(" ", "_")
        lookup_cache_file = self.cache_dir / f"match_lookup_{norm_h}_{norm_a}_{year or 'all'}.json"
        if lookup_cache_file.exists():
            try:
                cached_data = json.loads(lookup_cache_file.read_text(encoding="utf-8"))
                return cached_data.get("match")
            except Exception:
                pass

        comps = self.get_competitions()
        if comps.empty:
            return None

        # If year is specified, filter competitions/seasons first
        if year is not None and "season_name" in comps.columns:
            year_str = str(year)
            year_filtered = comps[comps["season_name"].astype(str).str.contains(year_str, na=False)]
            if not year_filtered.empty:
                comps = year_filtered

        found_match: dict[str, Any] | None = None
        for _, comp_row in comps.iterrows():
            comp_id = int(comp_row["competition_id"])
            season_id = int(comp_row["season_id"])
            matches = self.get_matches(comp_id, season_id)
            if matches.empty:
                continue

            # Check year at match level if match_date exists
            if year is not None and "match_date" in matches.columns:
                matches_year = matches[matches["match_date"].astype(str).str.startswith(str(year))]
                if not matches_year.empty:
                    matches = matches_year

            for _, match in matches.iterrows():
                m_home = str(match.get("home_team", ""))
                m_away = str(match.get("away_team", ""))
                if _team_names_match(home_team, m_home) and _team_names_match(away_team, m_away):
                    return match.to_dict()
                    found_match = match.to_dict()
                    break

        return None
            if found_match is not None:
                break

        # Persist lookup cache
        try:
            lookup_cache_file.write_text(
                json.dumps({"match": found_match}, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
        except Exception:
            pass

        return found_match
