"""Resilient API-Football HTTP client."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class ApiFootballError(RuntimeError):
    """Raised for transport, quota, or API response errors."""


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
        response = self._session.get(
            f"{self.BASE_URL}/{endpoint.lstrip('/')}",
            params=params,
            headers=self._headers,
            timeout=self._timeout,
        )
        self.last_rate_limit = {
            key: value
            for key, value in response.headers.items()
            if key.lower().startswith("x-ratelimit")
        }
        if not response.ok:
            raise ApiFootballError(
                f"API-Football request failed ({response.status_code}): "
                f"{response.text[:500]}"
            )

        payload = response.json()
        errors = payload.get("errors")
        if errors:
            raise ApiFootballError(f"API-Football returned errors: {errors}")
        data = payload.get("response", [])
        if not isinstance(data, list):
            raise ApiFootballError("Unexpected API-Football response shape")
        return data
