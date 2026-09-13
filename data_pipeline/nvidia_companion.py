"""NVIDIA NIM free-tier fallback for match commentary via OpenAI-compatible API."""

from __future__ import annotations

import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import requests

from config.settings import PROJECT_ROOT, _load_env_file

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_NVIDIA_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"
FALLBACK_NVIDIA_MODELS = ("nvidia/nemotron-3-super-120b-a12b", "nvidia/nemotron-3-ultra-550b-a55b")
NVIDIA_REQUEST_TIMEOUT = 60.0
NVIDIA_MAX_RETRIES = 2
NVIDIA_RETRY_DELAY_S = 2.0


class NvidiaCompanionError(RuntimeError):
    """Raised when the NVIDIA NIM companion provider fails."""

    def __init__(self, message: str, *, reason: str = "provider") -> None:
        super().__init__(message)
        self.reason = reason


def get_nvidia_api_key(
    *, env_file: Path | None = None, secrets: Mapping[str, object] | None = None
) -> str:
    """Read the key from environment / .env, then Streamlit secrets when available."""
    _load_env_file(env_file or PROJECT_ROOT / ".env")
    env_key = os.getenv("NVIDIA_API_KEY", "").strip()
    if env_key:
        return env_key

    if secrets is not None:
        configured = str(secrets.get("NVIDIA_API_KEY", "")).strip()
        if configured:
            return configured

    try:
        import streamlit as st

        configured = str(st.secrets.get("NVIDIA_API_KEY", "")).strip()
        if configured:
            return configured
    except Exception:  # Streamlit is optional for background pipeline execution.
        pass

    raise NvidiaCompanionError(
        "NVIDIA_API_KEY is not configured in the environment, .env, or Streamlit secrets",
        reason="configuration",
    )


def _provider_failure_reason(error: Exception) -> str:
    status = getattr(error, "status_code", None)
    if status is None:
        status = getattr(error, "code", None)
    if status is None:
        status = getattr(error, "status", None)
    if status is None:
        response = getattr(error, "response", None)
        status = getattr(response, "status_code", None)
    if status in {401, 403}:
        return "authentication"
    if status == 429:
        return "quota"
    if status in {400, 404}:
        return "model"
    if status in {408, 504}:
        return "timeout"
    error_text = str(error).lower()
    if "api key" in error_text or "authentication" in error_text:
        return "authentication"
    return "timeout" if "timeout" in str(type(error).__name__).lower() else "provider"


def _nvidia_model_candidates(configured_model: str) -> list[str]:
    return list(dict.fromkeys([configured_model, DEFAULT_NVIDIA_MODEL, *FALLBACK_NVIDIA_MODELS]))


def generate_nvidia_commentary(
    *,
    prompt: str,
    api_key: str | None = None,
    model_name: str | None = None,
    http_session: requests.Session | None = None,
) -> str:
    """Send a prompt to the NVIDIA NIM chat completions endpoint and return the response text."""
    from data_pipeline.match_commentary import (
        COMMENTARY_OUTPUT_BUDGETS,
        _ensure_readable_paragraphs,
        _looks_unfinished,
    )

    configured_key = (api_key or get_nvidia_api_key()).strip()
    if not configured_key:
        raise NvidiaCompanionError("NVIDIA_API_KEY is empty", reason="configuration")

    configured_model = (model_name or os.getenv("NVIDIA_MODEL", DEFAULT_NVIDIA_MODEL)).strip()
    if not configured_model:
        raise NvidiaCompanionError("NVIDIA model name is empty", reason="configuration")

    headers = {
        "Authorization": f"Bearer {configured_key}",
        "Content-Type": "application/json",
    }
    session = http_session or requests.Session()

    models = _nvidia_model_candidates(configured_model)
    last_error: NvidiaCompanionError | None = None

    for attempt in range(NVIDIA_MAX_RETRIES):
        max_output_tokens = COMMENTARY_OUTPUT_BUDGETS[
            min(attempt, len(COMMENTARY_OUTPUT_BUDGETS) - 1)
        ]
        request_model = models[min(attempt, len(models) - 1)]
        payload: dict[str, Any] = {
            "model": request_model,
            "messages": [
                {"role": "system", "content": "Sen tarafsız bir futbol veri analistisin. Yalnızca Türkçe yaz."},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_output_tokens,
            "temperature": 0.3,
        }
        try:
            resp = session.post(
                f"{NVIDIA_BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
                timeout=NVIDIA_REQUEST_TIMEOUT,
            )
            if resp.status_code == 429:
                last_error = NvidiaCompanionError(
                    f"NVIDIA NIM quota hit for {request_model}",
                    reason="quota",
                )
                time.sleep(NVIDIA_RETRY_DELAY_S * (attempt + 1))
                continue
            if resp.status_code in {401, 403}:
                raise NvidiaCompanionError(
                    f"NVIDIA NIM authentication failed: {resp.status_code}",
                    reason="authentication",
                )
            if resp.status_code == 404:
                last_error = NvidiaCompanionError(
                    f"NVIDIA NIM model not found: {request_model}",
                    reason="model",
                )
                continue
            resp.raise_for_status()

            data = resp.json()
            text = data["choices"][0]["message"]["content"].strip()
            if text and not _looks_unfinished(text):
                return _ensure_readable_paragraphs(text)
            last_error = NvidiaCompanionError(
                "NVIDIA NIM returned an incomplete response",
                reason="provider",
            )
        except NvidiaCompanionError:
            raise
        except requests.exceptions.Timeout:
            last_error = NvidiaCompanionError(
                f"NVIDIA NIM request timed out for {request_model}",
                reason="timeout",
            )
        except Exception as error:
            reason = _provider_failure_reason(error)
            if reason in {"authentication", "quota"}:
                raise NvidiaCompanionError(
                    f"NVIDIA NIM request failed: {type(error).__name__}",
                    reason=reason,
                ) from error
            last_error = NvidiaCompanionError(
                f"NVIDIA NIM request failed: {type(error).__name__}",
                reason=reason,
            )
        if attempt < NVIDIA_MAX_RETRIES - 1:
            time.sleep(NVIDIA_RETRY_DELAY_S * (attempt + 1))

    raise last_error or NvidiaCompanionError(
        "NVIDIA NIM commentary could not be generated", reason="provider"
    )
