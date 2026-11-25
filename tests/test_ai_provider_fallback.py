from __future__ import annotations

import json
from typing import Any

from app.ai_match_analysis import (
    MatchAnalysisAIError,
    generate_match_ai_packages,
    generate_match_analysis,
)


class FakeResponse:
    ok = True
    status_code = 200
    headers: dict[str, str] = {}

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
        strict_json_schema=True,
        reasoning_effort="low",
        extra_body={"chat_template_kwargs": {"enable_thinking": True}},
        response_schema={
            "type": "OBJECT",
            "properties": {"matches": {"type": "ARRAY", "items": {"type": "OBJECT"}}},
            "required": ["matches"],
        },
    )

    assert result == '{"matches": []}'
    assert captured["url"] == "https://example.test/openai/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert captured["json"]["model"] == "test-model"
    assert captured["json"]["max_tokens"] == 8192
    assert captured["json"]["reasoning_effort"] == "low"
    assert captured["json"]["chat_template_kwargs"] == {"enable_thinking": True}
    response_format = captured["json"]["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    assert response_format["json_schema"]["schema"]["type"] == "object"
    assert response_format["json_schema"]["schema"]["additionalProperties"] is False


def test_fallback_splits_match_batches_after_primary_quota_error(monkeypatch: Any) -> None:
    fallback_calls: list[list[int]] = []

    def fake_generate(context: dict[str, Any], **kwargs: Any) -> str:
        if kwargs["provider"] == "google":
            raise MatchAnalysisAIError("HTTP 429 quota exhausted")
        matches = context["matches"]
        match_ids = [int(item["match_id"]) for item in matches]
        fallback_calls.append(match_ids)
        return json.dumps(
            {
                "matches": [
                    {"match_id": match_id, "analysis": "Kısa analiz", "risk_selections": {}}
                    for match_id in match_ids
                ]
            }
        )

    monkeypatch.setattr("app.ai_match_analysis.generate_match_analysis", fake_generate)
    monkeypatch.setattr("app.ai_match_analysis.time.sleep", lambda _: None)
    contexts = [
        {"match_id": match_id, "model": {"candidate_markets": {"Beraberlik": 0.3}}}
        for match_id in (11, 12)
    ]

    packages = generate_match_ai_packages(
        contexts,
        api_key="google-key",
        fallback_providers=[
            {
                "provider": "openai_compatible",
                "name": "Groq",
                "api_key": "groq-key",
                "base_url": "https://example.test/v1",
                "model": "test-model",
            }
        ],
    )

    assert fallback_calls == [[11], [12]]
    assert set(packages) == {11, 12}
