from __future__ import annotations

import pytest

from config.settings import ConfigurationError, get_public_supabase_settings
from db.db_client import PublicSupabaseRestClient


def test_public_settings_require_anon_key(tmp_path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("SUPABASE_URL=https://example.supabase.co\n", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="SUPABASE_ANON_KEY"):
        get_public_supabase_settings(env_file)


def test_public_client_cannot_mutate() -> None:
    client = PublicSupabaseRestClient("https://example.supabase.co", "anon-key")

    with pytest.raises(PermissionError, match="read-only"):
        client.upsert("matches", [{"id": 1}])
    with pytest.raises(PermissionError, match="read-only"):
        client.delete("matches", filters={"id": "eq.1"})
