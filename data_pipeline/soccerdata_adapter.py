"""
soccerdata Adapter — Phase 5
============================
Wraps the ``soccerdata`` library behind a clean, network-isolated interface
with:
  - Automatic disk caching (Parquet) so network is only hit once per season.
  - Configurable request timeout and retry logic.
  - Graceful degradation: if ``soccerdata`` is not installed or the network
    is unavailable, all public functions return empty DataFrames and log a
    warning rather than raising.

``soccerdata`` supports multiple data sources (FBref, Sofascore, ESPN, …).
This adapter currently targets **FBref** for schedule / standings data.

Usage
-----
    from data_pipeline.soccerdata_adapter import SoccerdataAdapter
    adapter = SoccerdataAdapter(league="ENG-Premier League", season="2324")
    schedule = adapter.get_schedule()
    standings = adapter.get_standings()
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency guard
# ---------------------------------------------------------------------------
try:
    import soccerdata as sd  # type: ignore[import]

    SOCCERDATA_AVAILABLE = True
    logger.info("soccerdata available — live data scraping enabled")
except ImportError:
    SOCCERDATA_AVAILABLE = False
    logger.warning(
        "soccerdata not installed. Live schedule/standings scraping disabled. "
        "Install via: pip install soccerdata  (use requirements-research.txt)"
    )

# ---------------------------------------------------------------------------
# Default settings
# ---------------------------------------------------------------------------
DEFAULT_CACHE_DIR = Path("cache") / "soccerdata"
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 2.0  # seconds


# ---------------------------------------------------------------------------
# Adapter class
# ---------------------------------------------------------------------------

class SoccerdataAdapter:
    """Network-isolated adapter around ``soccerdata.FBref``.

    Parameters
    ----------
    league : str
        soccerdata league string, e.g. ``"ENG-Premier League"``,
        ``"ESP-La Liga"``, ``"GER-Bundesliga"``.
    season : str
        Season code accepted by soccerdata, e.g. ``"2324"`` (2023-24).
    cache_dir : Path | str | None
        Directory for Parquet cache files.  Defaults to ``cache/soccerdata``.
    max_retries : int
        Number of HTTP retry attempts before returning an empty DataFrame.
    retry_delay : float
        Seconds to wait between retries.
    no_cache : bool
        If True, bypass the disk cache and always fetch from the network.
    """

    def __init__(
        self,
        league: str = "ENG-Premier League",
        season: str = "2324",
        cache_dir: Path | str | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_delay: float = DEFAULT_RETRY_DELAY,
        no_cache: bool = False,
    ) -> None:
        self.league = league
        self.season = season
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.no_cache = no_cache
        self._fbref: Any = None  # lazy init

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _cache_path(self, stem: str) -> Path:
        safe_league = self.league.replace(" ", "_").replace("-", "_")
        return self.cache_dir / f"{safe_league}_{self.season}_{stem}.parquet"

    def _load_cache(self, stem: str) -> pd.DataFrame | None:
        if self.no_cache:
            return None
        p = self._cache_path(stem)
        if p.exists():
            try:
                df = pd.read_parquet(p)
                logger.debug("soccerdata cache hit: %s (%d rows)", p.name, len(df))
                return df
            except Exception as exc:  # noqa: BLE001
                logger.warning("Cache read failed %s: %s", p, exc)
        return None

    def _save_cache(self, df: pd.DataFrame, stem: str) -> None:
        if self.no_cache or df.empty:
            return
        p = self._cache_path(stem)
        p.parent.mkdir(parents=True, exist_ok=True)
        try:
            df.to_parquet(p, index=False)
            logger.debug("soccerdata cached: %s", p.name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Cache write failed %s: %s", p, exc)

    def _get_fbref(self) -> Any:
        """Lazy-initialise the FBref scraper."""
        if self._fbref is None:
            if not SOCCERDATA_AVAILABLE:
                raise ImportError("soccerdata is not installed")
            self._fbref = sd.FBref(leagues=[self.league], seasons=[self.season])
        return self._fbref

    def _fetch_with_retry(self, fetch_fn: Any, stem: str) -> pd.DataFrame:
        """Wrap a network fetch call with retry/back-off logic."""
        cached = self._load_cache(stem)
        if cached is not None:
            return cached

        if not SOCCERDATA_AVAILABLE:
            logger.warning("soccerdata unavailable — returning empty DataFrame for %s", stem)
            return pd.DataFrame()

        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                df = fetch_fn()
                if isinstance(df, pd.DataFrame) and not df.empty:
                    # Flatten MultiIndex columns if present
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = ["_".join(str(c) for c in col if c).strip("_")
                                      for col in df.columns]
                    df = df.reset_index()
                    self._save_cache(df, stem)
                    return df
                return pd.DataFrame()
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                logger.warning(
                    "soccerdata fetch attempt %d/%d failed for %s: %s",
                    attempt, self.max_retries, stem, exc,
                )
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)

        logger.error("All retries exhausted for %s: %s", stem, last_exc)
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_schedule(self) -> pd.DataFrame:
        """Fetch the league schedule (fixtures + results).

        Returns
        -------
        pd.DataFrame
            Columns vary by season availability; typically includes
            ``date``, ``home_team``, ``away_team``, ``home_score``,
            ``away_score``.
            Returns an empty DataFrame if data is unavailable.
        """
        def _fetch() -> pd.DataFrame:
            fbref = self._get_fbref()
            return fbref.read_schedule()

        return self._fetch_with_retry(_fetch, "schedule")

    def get_standings(self) -> pd.DataFrame:
        """Fetch the league standings table.

        Returns
        -------
        pd.DataFrame
            Typically includes ``rank``, ``team``, ``played``, ``won``,
            ``drawn``, ``lost``, ``gf``, ``ga``, ``gd``, ``points``.
            Returns an empty DataFrame if data is unavailable.
        """
        def _fetch() -> pd.DataFrame:
            fbref = self._get_fbref()
            return fbref.read_league_table()

        return self._fetch_with_retry(_fetch, "standings")

    def get_player_season_stats(self, stat_type: str = "standard") -> pd.DataFrame:
        """Fetch per-player season statistics.

        Parameters
        ----------
        stat_type : str
            FBref stat category: ``"standard"``, ``"shooting"``,
            ``"passing"``, ``"defense"``, ``"keeper"``, etc.

        Returns
        -------
        pd.DataFrame
            Player stats with MultiIndex columns flattened.
        """
        stem = f"player_stats_{stat_type}"

        def _fetch() -> pd.DataFrame:
            fbref = self._get_fbref()
            return fbref.read_player_season_stats(stat_type=stat_type)

        return self._fetch_with_retry(_fetch, stem)

    def get_team_season_stats(self, stat_type: str = "standard") -> pd.DataFrame:
        """Fetch per-team season statistics.

        Parameters
        ----------
        stat_type : str
            FBref stat category (same options as :meth:`get_player_season_stats`).

        Returns
        -------
        pd.DataFrame
        """
        stem = f"team_stats_{stat_type}"

        def _fetch() -> pd.DataFrame:
            fbref = self._get_fbref()
            return fbref.read_team_season_stats(stat_type=stat_type)

        return self._fetch_with_retry(_fetch, stem)

    def clear_cache(self) -> int:
        """Delete all cached Parquet files for this league/season.

        Returns
        -------
        int
            Number of files deleted.
        """
        safe_league = self.league.replace(" ", "_").replace("-", "_")
        prefix = f"{safe_league}_{self.season}_"
        count = 0
        if self.cache_dir.exists():
            for p in self.cache_dir.glob(f"{prefix}*.parquet"):
                p.unlink(missing_ok=True)
                count += 1
        logger.info("Cleared %d soccerdata cache files", count)
        return count

    def list_available_leagues(self) -> list[str]:
        """Return the list of leagues soccerdata supports (FBref)."""
        if not SOCCERDATA_AVAILABLE:
            return []
        try:
            # soccerdata exposes this as a class attribute on the scraper
            return list(sd.FBref.available_leagues())  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            return []

    # ------------------------------------------------------------------
    # Convenience: match result lookup
    # ------------------------------------------------------------------

    def find_match(
        self,
        home_team: str,
        away_team: str,
    ) -> pd.Series | None:
        """Look up a single match by team names in the cached schedule.

        Matching is case-insensitive and uses ``str.contains``.

        Returns
        -------
        pd.Series | None
            The first matching row, or ``None`` if not found.
        """
        schedule = self.get_schedule()
        if schedule.empty:
            return None

        home_col = next(
            (c for c in schedule.columns if "home" in c.lower() and "team" in c.lower()),
            None,
        )
        away_col = next(
            (c for c in schedule.columns if "away" in c.lower() and "team" in c.lower()),
            None,
        )
        if home_col is None or away_col is None:
            logger.warning("Could not identify home/away team columns in schedule")
            return None

        mask = (
            schedule[home_col].str.contains(home_team, case=False, na=False)
            & schedule[away_col].str.contains(away_team, case=False, na=False)
        )
        matches = schedule[mask]
        return matches.iloc[0] if not matches.empty else None

