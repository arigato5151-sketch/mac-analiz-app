"""Resilient API-Football HTTP client."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from time import perf_counter
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class ApiFootballError(RuntimeError):
    """Raised for transport, quota, or API response errors."""


LOG = logging.getLogger(__name__)


class ApiFootballClient:
    BASE_URL = "https://v3.football.api-sports.io"

    def __init__(
        self,
        api_key: str,
        *,
        timeout: float = 30.0,
        session: requests.Session | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("API-Football key is required")

        self._timeout = timeout
        self._session = session or self._build_session()
        self._headers = {
            "x-apisports-key": api_key,
            "User-Agent": "mac-analiz-app/1.0",
        }
        self.last_rate_limit: dict[str, str] = {}
        self.request_count = 0
        self.request_failures = 0
        self.last_request_seconds: float | None = None

    @staticmethod
    def _build_session() -> requests.Session:
        retry = Retry(
            total=4,
            backoff_factor=0.75,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET"}),
            respect_retry_after_header=True,
        )
        session = requests.Session()
        session.mount("https://", HTTPAdapter(max_retries=retry))
        return session

    def get(
        self, endpoint: str, params: Mapping[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        self.request_count += 1
        started_at = perf_counter()
        try:
            response = self._session.get(
                f"{self.BASE_URL}/{endpoint.lstrip('/')}",
                params=params,
                headers=self._headers,
                timeout=self._timeout,
            )
        except requests.RequestException as exc:
            self.request_failures += 1
            self.last_request_seconds = perf_counter() - started_at
            LOG.warning(
                "API-Football GET %s transport error after %.3fs (%s)",
                endpoint,
                self.last_request_seconds,
                type(exc).__name__,
            )
            raise
        self.last_request_seconds = perf_counter() - started_at
        self.last_rate_limit = {
            key: value
            for key, value in response.headers.items()
            if key.lower().startswith("x-ratelimit")
        }
        if not response.ok:
            self.request_failures += 1
            LOG.warning(
                "API-Football GET %s returned HTTP %d after %.3fs",
                endpoint,
                response.status_code,
                self.last_request_seconds,
            )
            raise ApiFootballError(
                f"API-Football request failed ({response.status_code}): "
                f"{response.text[:500]}"
            )

        try:
            payload = response.json()
        except ValueError as exc:
            self.request_failures += 1
            LOG.warning(
                "API-Football GET %s returned invalid JSON after %.3fs",
                endpoint,
                self.last_request_seconds,
            )
            raise ApiFootballError("API-Football returned invalid JSON") from exc
        if not isinstance(payload, dict):
            self.request_failures += 1
            LOG.warning(
                "API-Football GET %s returned an invalid response shape after %.3fs",
                endpoint,
                self.last_request_seconds,
            )
            raise ApiFootballError("Unexpected API-Football response shape")
        errors = payload.get("errors")
        if errors:
            self.request_failures += 1
            LOG.warning(
                "API-Football GET %s returned provider errors after %.3fs",
                endpoint,
                self.last_request_seconds,
            )
            raise ApiFootballError(f"API-Football returned errors: {errors}")
        data = payload.get("response", [])
        if not isinstance(data, list):
            self.request_failures += 1
            LOG.warning(
                "API-Football GET %s returned an invalid response shape after %.3fs",
                endpoint,
                self.last_request_seconds,
            )
            raise ApiFootballError("Unexpected API-Football response shape")
        LOG.info(
            "API-Football GET %s completed in %.3fs (%d rows)",
            endpoint,
            self.last_request_seconds,
            len(data),
        )
        return data

    def diagnostics(self) -> dict[str, Any]:
        """Return logical request, latency, and quota diagnostics for job logs."""
        normalized = {key.lower(): value for key, value in self.last_rate_limit.items()}
        remaining = next(
            (
                value
                for key, value in normalized.items()
                if "remaining" in key
            ),
            None,
        )
        warning = None
        try:
            if remaining is not None and int(remaining) <= 50:
                warning = "API-Football remaining quota is low"
        except ValueError:
            warning = "API-Football returned a non-numeric remaining quota header"
        return {
            "requests": self.request_count,
            "failures": self.request_failures,
            "last_request_seconds": self.last_request_seconds,
            "rate_limit": self.last_rate_limit,
            "warning": warning,
        }
