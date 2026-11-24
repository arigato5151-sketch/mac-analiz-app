from __future__ import annotations

from typing import Any

from app.ai_match_analysis import generate_match_analysis


class FakeResponse:
    ok = True

    def json(self) -> dict[str, Any]:
        return {"choices": [{"message": {"content": '{"matches": []}'}}]}


def test_openai_compatible_provider_builds_json_chat_request(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: Any) -> FakeResponse:
        captured["url"] = url
        captured.update(kwargs)
        return FakeResponse()

    monkeypatch.setattr("app.ai_match_analysis.requests.post", fake_post)

    result = generate_match_analysis(
        {"matches": []},
        api_key="test-key",
        model="test-model",
        instructions_override="Return only JSON.",
        json_mode=True,
        max_output_tokens=12000,
        provider="openai_compatible",
        base_url="https://example.test/openai/v1/",
    )

    assert result == '{"matches": []}'
    assert captured["url"] == "https://example.test/openai/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert captured["json"]["model"] == "test-model"
    assert captured["json"]["max_tokens"] == 8192
    assert captured["json"]["response_format"] == {"type": "json_object"}
