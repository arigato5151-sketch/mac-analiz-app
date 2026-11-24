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
        },
        {
            "name": "NVIDIA NIM",
            "key": get_optional_setting("NVIDIA_API_KEY"),
            "model": get_optional_setting(
                "NVIDIA_MODEL", "nvidia/nemotron-3.5-lightning-30b-a3b"
            ),
            "base_url": "https://integrate.api.nvidia.com/v1",
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
                max_output_tokens=64,
                provider="openai_compatible",
                base_url=provider["base_url"],
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
