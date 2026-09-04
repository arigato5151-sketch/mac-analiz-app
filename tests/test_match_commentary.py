from __future__ import annotations

from pathlib import Path

import pytest

from data_pipeline.match_commentary import (
    MatchCommentaryError,
    _ensure_readable_paragraphs,
    _provider_failure_reason,
    _request_timeout_ms,
    build_match_commentary_prompt,
    generate_match_commentary,
    get_gemini_api_key,
)


def _arguments() -> dict[str, object]:
    return {
        "home_team": "Ev Takımı",
        "away_team": "Deplasman Takımı",
        "home_xg": 1.55,
        "away_xg": 1.12,
        "home_absences": ["Orta saha oyuncusu (sakat)"],
        "away_absences": None,
        "home_form": {"son_5": "3G 1B 1M"},
        "away_form": "2G 2B 1M",
        "home_win_probability": 0.48,
        "draw_probability": 0.27,
        "away_win_probability": 0.25,
    }


def test_key_is_read_from_dotenv_without_overwriting_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("GEMINI_API_KEY=from-env-file\n", encoding="utf-8")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    assert get_gemini_api_key(env_file=env_file, secrets={}) == "from-env-file"
    monkeypatch.setenv("GEMINI_API_KEY", "from-environment")
    assert get_gemini_api_key(env_file=env_file, secrets={}) == "from-environment"


def test_key_falls_back_to_streamlit_secret_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    assert get_gemini_api_key(
        env_file=Path("does-not-exist"), secrets={"GEMINI_API_KEY": "from-secrets"}
    ) == "from-secrets"


def test_key_accepts_google_api_key_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert get_gemini_api_key(
        env_file=Path("does-not-exist"), secrets={"GOOGLE_API_KEY": "from-google-secret"}
    ) == "from-google-secret"


def test_prompt_contains_metrics_but_rejects_invalid_probability_sum() -> None:
    prompt = build_match_commentary_prompt(**_arguments())  # type: ignore[arg-type]

    assert "Ev sahibi xG: 1.55" in prompt
    assert "Model olasılıkları: ev sahibi %48.0" in prompt
    invalid = _arguments() | {"away_win_probability": 0.50}
    with pytest.raises(ValueError, match="sum"):
        build_match_commentary_prompt(**invalid)  # type: ignore[arg-type]


def test_single_block_commentary_is_split_only_at_sentence_boundaries() -> None:
    assert _ensure_readable_paragraphs("Birinci cümle. İkinci cümle. Üçüncü cümle. Dördüncü cümle.") == (
        "Birinci cümle. İkinci cümle.\n\nÜçüncü cümle. Dördüncü cümle."
    )


def test_generation_uses_requested_model_and_keeps_key_out_of_errors() -> None:
    class FakeModels:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def generate_content(
            self, *, model: str, contents: str, config: dict[str, object]
        ) -> object:
            self.calls.append({"model": model, "contents": contents, "config": config})
            return type("Response", (), {"text": "İlk paragraf.\n\nİkinci paragraf."})()

    class FakeClient:
        def __init__(self, key: str) -> None:
            self.key = key
            self.models = FakeModels()

    created: list[FakeClient] = []

    def factory(key: str) -> FakeClient:
        client = FakeClient(key)
        created.append(client)
        return client

    commentary = generate_match_commentary(
        **_arguments(),  # type: ignore[arg-type]
        api_key="private-key",
        model_name="gemini-3.6-flash",
        client_factory=factory,
    )

    assert commentary.startswith("İlk paragraf")
    assert created[0].key == "private-key"
    assert created[0].models.calls[0]["model"] == "gemini-3.6-flash"
    assert created[0].models.calls[0]["config"] == {"max_output_tokens": 3_072}


def test_generation_retries_once_when_the_first_response_hits_token_limit() -> None:
    class FakeModels:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def generate_content(
            self, *, model: str, contents: str, config: dict[str, object]
        ) -> object:
            self.calls.append({"model": model, "contents": contents, "config": config})
            finish_reason = "MAX_TOKENS" if len(self.calls) == 1 else "STOP"
            candidate = type("Candidate", (), {"finish_reason": finish_reason})()
            return type("Response", (), {"text": "Tam yorum.", "candidates": [candidate]})()

    class FakeClient:
        def __init__(self) -> None:
            self.models = FakeModels()

    client = FakeClient()
    assert generate_match_commentary(
        **_arguments(),  # type: ignore[arg-type]
        api_key="private-key",
        client_factory=lambda _: client,
    ) == "Tam yorum."
    assert [call["config"] for call in client.models.calls] == [
        {"max_output_tokens": 3_072},
        {"max_output_tokens": 4_096},
    ]


def test_generation_falls_back_when_deployed_model_setting_is_stale() -> None:
    class FakeModels:
        def __init__(self) -> None:
            self.models: list[str] = []

        def generate_content(
            self, *, model: str, contents: str, config: dict[str, object]
        ) -> object:
            self.models.append(model)
            return type("Response", (), {"text": "Tam yorum."})()

    class FakeClient:
        def __init__(self) -> None:
            self.models = FakeModels()

    client = FakeClient()
    assert generate_match_commentary(
        **_arguments(),  # type: ignore[arg-type]
        api_key="private-key",
        model_name="obsolete-model",
        client_factory=lambda _: client,
    ) == "Tam yorum."
    assert client.models.models == ["obsolete-model"]


def test_generation_continues_after_model_error() -> None:
    class ModelError(Exception):
        code = 404

    class FakeModels:
        def __init__(self) -> None:
            self.models: list[str] = []

        def generate_content(
            self, *, model: str, contents: str, config: dict[str, object]
        ) -> object:
            self.models.append(model)
            if len(self.models) == 1:
                raise ModelError("model not found")
            return type("Response", (), {"text": "Tam yorum."})()

    class FakeClient:
        def __init__(self) -> None:
            self.models = FakeModels()

    client = FakeClient()
    assert generate_match_commentary(
        **_arguments(),  # type: ignore[arg-type]
        api_key="private-key",
        model_name="obsolete-model",
        client_factory=lambda _: client,
    ) == "Tam yorum."
    assert client.models.models == ["obsolete-model", "gemini-2.5-flash"]


def test_generation_retries_complete_response_when_first_ends_unfinished() -> None:
    class FakeModels:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def generate_content(
            self, *, model: str, contents: str, config: dict[str, object]
        ) -> object:
            self.calls.append({"model": model, "contents": contents, "config": config})
            text = "Tam yorum." if len(self.calls) > 1 else "Abha ve Al-Ettifaq, son 5 maçlık performanslarında 1 galibiyet,"
            candidate = type("Candidate", (), {"finish_reason": "STOP"})()
            return type("Response", (), {"text": text, "candidates": [candidate]})()

    class FakeClient:
        def __init__(self) -> None:
            self.models = FakeModels()

    client = FakeClient()
    assert generate_match_commentary(
        **_arguments(),  # type: ignore[arg-type]
        api_key="private-key",
        client_factory=lambda _: client,
    ) == "Tam yorum."
    assert [call["config"] for call in client.models.calls] == [
        {"max_output_tokens": 3_072},
        {"max_output_tokens": 4_096},
    ]


def test_generation_retries_after_empty_response() -> None:
    class FakeModels:
        def __init__(self) -> None:
            self.calls = 0

        def generate_content(
            self, *, model: str, contents: str, config: dict[str, object]
        ) -> object:
            self.calls += 1
            text = "" if self.calls == 1 else "Tam yorum."
            return type("Response", (), {"text": text})()

    class FakeClient:
        def __init__(self) -> None:
            self.models = FakeModels()

    client = FakeClient()
    assert generate_match_commentary(
        **_arguments(),  # type: ignore[arg-type]
        api_key="private-key",
        client_factory=lambda _: client,
    ) == "Tam yorum."
    assert client.models.calls == 2


def test_generation_wraps_provider_error_without_secret() -> None:
    def failing_factory(_: str) -> object:
        raise RuntimeError("private-key")

    with pytest.raises(MatchCommentaryError, match="RuntimeError") as error:
        generate_match_commentary(
            **_arguments(),  # type: ignore[arg-type]
            api_key="private-key",
            client_factory=failing_factory,
        )

    assert "private-key" not in str(error.value)


def test_provider_error_is_classified_without_keeping_response_text() -> None:
    class QuotaError(Exception):
        status_code = 429

    class ModelError(Exception):
        code = 400

    class NotFoundModelError(Exception):
        status = 404

    class TimeoutError(Exception):
        pass

    assert _provider_failure_reason(QuotaError("private-provider-detail")) == "quota"
    assert _provider_failure_reason(ModelError("private-provider-detail")) == "model"
    assert _provider_failure_reason(NotFoundModelError("model not found")) == "model"
    assert _provider_failure_reason(TimeoutError("private-provider-detail")) == "timeout"


def test_timeout_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_TIMEOUT_MS", "999999")
    assert _request_timeout_ms() == 60_000
    monkeypatch.setenv("GEMINI_TIMEOUT_MS", "invalid")
    assert _request_timeout_ms() == 60_000
