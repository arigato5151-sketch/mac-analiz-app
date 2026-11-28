"""Send one small, non-persistent request to each configured AI fallback."""

from __future__ import annotations

import json

from app.ai_match_analysis import MatchAnalysisAIError, generate_match_analysis
from config.settings import get_optional_setting


def main() -> None:
    providers = (
        {
            "name": "Groq",
            "key": get_optional_setting("GROQ_API_KEY"),
            "model": get_optional_setting("GROQ_MODEL", "openai/gpt-oss-20b"),
            "base_url": "https://api.groq.com/openai/v1",
            "strict_json_schema": True,
            "reasoning_effort": "low",
            "max_output_tokens": 1600,
            "request_timeout_seconds": 20,
        },
        {
            "name": "NVIDIA NIM",
            "key": get_optional_setting("NVIDIA_API_KEY"),
            "model": get_optional_setting(
                "NVIDIA_MODEL", "nvidia/nemotron-3-ultra-550b-a55b"
            ),
            "base_url": "https://integrate.api.nvidia.com/v1",
            "strict_json_schema": False,
            "extra_body": {
                "chat_template_kwargs": {
                    "enable_thinking": False,
                    "force_nonempty_content": True,
                }
            },
            "max_output_tokens": 1800,
            "request_timeout_seconds": 90,
        },
    )
    failures = 0
    for provider in providers:
        key = "".join(provider["key"].replace("\ufeff", "").split())
        if not key:
            print(f"{provider['name']}: API key missing")
            failures += 1
            continue
        try:
            output = generate_match_analysis(
                {"provider_check": "Return the JSON health result."},
                api_key=key,
                model=provider["model"],
                instructions_override='Return exactly this JSON: {"ok":true}',
                json_mode=True,
                max_output_tokens=provider["max_output_tokens"],
                provider="openai_compatible",
                base_url=provider["base_url"],
                strict_json_schema=bool(provider["strict_json_schema"]),
                reasoning_effort=provider.get("reasoning_effort"),
                extra_body=provider.get("extra_body"),
                request_timeout_seconds=provider["request_timeout_seconds"],
                response_schema={
                    "type": "OBJECT",
                    "properties": {"ok": {"type": "BOOLEAN"}},
                    "required": ["ok"],
                },
            )
            if json.loads(output).get("ok") is not True:
                raise MatchAnalysisAIError("API returned an unexpected health response.")
            print(f"{provider['name']}: API key and model are working")
        except (MatchAnalysisAIError, ValueError, AttributeError) as exc:
            print(f"{provider['name']}: validation failed: {exc}")
            failures += 1
    if failures:
        raise SystemExit(f"{failures} AI provider check(s) failed.")


if __name__ == "__main__":
    main()
