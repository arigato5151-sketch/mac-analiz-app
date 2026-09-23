from __future__ import annotations

import pytest

from config.settings import ConfigurationError, get_supabase_admin_settings


def test_supabase_admin_settings_do_not_require_api_football_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-role")
    monkeypatch.delenv("API_FOOTBALL_KEY", raising=False)

    settings = get_supabase_admin_settings()

    assert settings.supabase_url == "https://example.supabase.co"
    assert settings.supabase_service_role_key == "service-role"


def test_supabase_admin_settings_report_missing_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)

    with pytest.raises(ConfigurationError, match="SUPABASE_URL"):
        get_supabase_admin_settings()
