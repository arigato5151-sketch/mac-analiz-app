"""Environment-backed application settings."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ConfigurationError(RuntimeError):
    """Raised when a required environment setting is missing."""


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass(frozen=True, slots=True)
class Settings:
    api_football_key: str
    supabase_url: str
    supabase_service_role_key: str
    request_timeout_seconds: float = 30.0


def get_settings(env_file: Path | None = None) -> Settings:
    _load_env_file(env_file or PROJECT_ROOT / ".env")

    values = {
        "API_FOOTBALL_KEY": os.getenv("API_FOOTBALL_KEY", "").strip(),
        "SUPABASE_URL": os.getenv("SUPABASE_URL", "").strip().rstrip("/"),
        "SUPABASE_SERVICE_ROLE_KEY": os.getenv(
            "SUPABASE_SERVICE_ROLE_KEY", ""
        ).strip(),
    }
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise ConfigurationError(
            f"Missing required environment settings: {', '.join(missing)}"
        )

    return Settings(
        api_football_key=values["API_FOOTBALL_KEY"],
        supabase_url=values["SUPABASE_URL"],
        supabase_service_role_key=values["SUPABASE_SERVICE_ROLE_KEY"],
    )
