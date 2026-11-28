from __future__ import annotations

from typing import Any

import pytest
import requests

from data_pipeline.api_client import ApiFootballClient, ApiFootballError


class FakeResponse:
    ok = True
    headers = {"x-ratelimit-requests-remaining": "500"}

    @staticmethod
    def json() -> dict[str, Any]:
        return {"response": [{"fixture": 1}], "errors": []}


class FakeSession:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls = 0

    def get(self, *_: Any, **__: Any) -> FakeResponse:
        self.calls += 1
        if self.error:
            raise self.error
        return FakeResponse()


def test_request_diagnostics_count_transport_failures() -> None:
    session = FakeSession(requests.Timeout("timed out"))
    client = ApiFootballClient("test-key", session=session)

    with pytest.raises(requests.Timeout):
        client.get("fixtures")

    diagnostics = client.diagnostics()
    assert diagnostics["requests"] == 1
    assert diagnostics["failures"] == 1
    assert isinstance(diagnostics["last_request_seconds"], float)


def test_request_diagnostics_include_success_duration() -> None:
    session = FakeSession()
    client = ApiFootballClient("test-key", session=session)

    assert client.get("fixtures") == [{"fixture": 1}]

    diagnostics = client.diagnostics()
    assert diagnostics["requests"] == 1
    assert diagnostics["failures"] == 0
    assert isinstance(diagnostics["last_request_seconds"], float)


def test_invalid_json_counts_as_a_failed_request() -> None:
    class InvalidJsonResponse(FakeResponse):
        @staticmethod
        def json() -> dict[str, Any]:
            raise ValueError("invalid json")

    class InvalidJsonSession(FakeSession):
        def get(self, *_: Any, **__: Any) -> InvalidJsonResponse:
            return InvalidJsonResponse()

    client = ApiFootballClient("test-key", session=InvalidJsonSession())

    with pytest.raises(ApiFootballError, match="invalid JSON"):
        client.get("fixtures")

    assert client.diagnostics()["failures"] == 1
