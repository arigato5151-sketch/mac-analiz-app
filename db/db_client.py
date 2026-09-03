"""Small, testable Supabase PostgREST client."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class DatabaseError(RuntimeError):
    """Raised when Supabase rejects or cannot complete an operation."""


def _build_session() -> requests.Session:
    retry = Retry(
        total=4,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "POST", "PATCH", "DELETE"}),
        respect_retry_after_header=True,
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


class SupabaseRestClient:
    def __init__(
        self,
        url: str,
        service_key: str,
        *,
        timeout: float = 30.0,
        session: requests.Session | None = None,
    ) -> None:
        if not url or not service_key:
            raise ValueError("Supabase URL and service key are required")

        self._base_url = f"{url.rstrip('/')}/rest/v1"
        self._timeout = timeout
        self._session = session or _build_session()
        self._headers = {
            "apikey": service_key,
            "Content-Type": "application/json",
        }
        # Legacy service_role JWTs require Authorization. New sb_secret keys
        # authenticate through the apikey header and must not be used as JWTs.
        if service_key.count(".") == 2:
            self._headers["Authorization"] = f"Bearer {service_key}"

    def _request(
        self,
        method: str,
        table: str,
        *,
        params: Mapping[str, Any] | None = None,
        json: Any = None,
        headers: Mapping[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        response = self._session.request(
            method,
            f"{self._base_url}/{table}",
            params=params,
            json=json,
            headers={**self._headers, **(headers or {})},
            timeout=self._timeout,
        )
        if not response.ok:
            detail = response.text[:500]
            raise DatabaseError(
                f"Supabase {method} {table} failed ({response.status_code}): {detail}"
            )
        if not response.content:
            return []
        payload = response.json()
        return payload if isinstance(payload, list) else [payload]

    def upsert(
        self,
        table: str,
        records: Sequence[Mapping[str, Any]],
        *,
        on_conflict: str | None = None,
    ) -> list[dict[str, Any]]:
        if not records:
            return []

        params = {"on_conflict": on_conflict} if on_conflict else None
        return self._request(
            "POST",
            table,
            params=params,
            json=list(records),
            headers={"Prefer": "resolution=merge-duplicates,return=representation"},
        )

    def insert(
        self,
        table: str,
        records: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        """Insert records without allowing a conflict to overwrite history."""
        if not records:
            return []
        return self._request(
            "POST",
            table,
            json=list(records),
            headers={"Prefer": "return=representation"},
        )

    def select(
        self,
        table: str,
        *,
        columns: str = "*",
        filters: Mapping[str, str | int | float | bool] | None = None,
        limit: int | None = None,
        offset: int | None = None,
        order: str | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"select": columns}
        if filters:
            params.update(filters)
        if limit is not None:
            params["limit"] = limit
        if offset is not None:
            params["offset"] = offset
        if order:
            params["order"] = order
        return self._request("GET", table, params=params)

    def select_all(
        self,
        table: str,
        *,
        columns: str = "*",
        filters: Mapping[str, str | int | float | bool] | None = None,
        order: str | None = None,
        page_size: int = 1_000,
    ) -> list[dict[str, Any]]:
        if page_size < 1 or page_size > 1_000:
            raise ValueError("page_size must be between 1 and 1000")

        rows: list[dict[str, Any]] = []
        offset = 0
        while True:
            page = self.select(
                table,
                columns=columns,
                filters=filters,
                limit=page_size,
                offset=offset,
                order=order,
            )
            rows.extend(page)
            if len(page) < page_size:
                return rows
            offset += page_size

    def delete(
        self,
        table: str,
        *,
        filters: Mapping[str, str | int | float | bool],
    ) -> None:
        if not filters:
            raise ValueError("Delete requires at least one filter")
        self._request("DELETE", table, params=filters)


class PublicSupabaseRestClient(SupabaseRestClient):
    """Query-only client for the deployed Streamlit UI."""

    def insert(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        raise PermissionError("Public Supabase client is read-only")

    def upsert(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        raise PermissionError("Public Supabase client is read-only")

    def delete(self, *args: Any, **kwargs: Any) -> None:
        raise PermissionError("Public Supabase client is read-only")
