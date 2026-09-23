"""Single source of truth for model version roles across all UI screens.

Every screen that names a model version must resolve its role through this
module. When a production artifact has been hydrated locally,
``latest.joblib`` identifies it; artifacts trained after it are candidates
awaiting the promotion gate; everything older is retired. Without a hydrated
artifact, the UI must not invent a production role.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

from config.settings import PROJECT_ROOT

LOGGER = logging.getLogger(__name__)

STATUS_PRODUCTION = "production"
STATUS_CANDIDATE = "candidate"
STATUS_RETIRED = "retired"

STATUS_LABELS_TR = {
    STATUS_PRODUCTION: "Üretim",
    STATUS_CANDIDATE: "Aday",
    STATUS_RETIRED: "Eski sürüm",
}

STATUS_UNKNOWN_TR = "Durum bilinmiyor"

REGISTRY_TTL_SECONDS = 3_600


@dataclass(frozen=True)
class ModelVersionInfo:
    version: str
    status: str
    trained_at: datetime | None


def _parse_timestamp(value: object) -> datetime | None:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def _version_from_filename(name: str) -> datetime | None:
    """model_v20260908T184437Z -> 2026-09-08T18:44:37+00:00"""
    try:
        raw = name.removeprefix("model_v").removesuffix("Z")
        return datetime.strptime(raw, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


@st.cache_data(ttl=REGISTRY_TTL_SECONDS, show_spinner=False)
def load_model_registry() -> tuple[ModelVersionInfo, ...]:
    """Derive version roles from local artifacts without touching the database."""
    model_dir = PROJECT_ROOT / "models" / "saved_models"
    latest_path = model_dir / "latest.joblib"
    production_version = ""
    production_trained_at: datetime | None = None
    if latest_path.is_file():
        import joblib

        bundle = joblib.load(latest_path)
        production_version = str(bundle.get("model_version", "")).strip()
        production_trained_at = _parse_timestamp(bundle.get("training_end"))

    entries: list[ModelVersionInfo] = []
    for path in sorted(model_dir.glob("model_v*.json")):
        version = path.stem
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            LOGGER.warning("Model metadata could not be read: %s (%s)", path.name, exc)
            meta = {}
        trained_at = (
            _parse_timestamp(meta.get("training_end"))
            or _parse_timestamp(meta.get("trained_at"))
            or _version_from_filename(version)
        )
        if version == production_version:
            status = STATUS_PRODUCTION
        elif (
            production_version
            and production_trained_at is not None
            and trained_at is not None
            and trained_at > production_trained_at
        ):
            status = STATUS_CANDIDATE
        else:
            status = STATUS_RETIRED
        entries.append(ModelVersionInfo(version=version, status=status, trained_at=trained_at))
    return tuple(entries)


def registry_by_version() -> dict[str, ModelVersionInfo]:
    return {info.version: info for info in load_model_registry()}


def status_label_tr(version: str) -> str:
    """Turkish role label for one version; unknown versions never fake a role."""
    info = registry_by_version().get(version)
    if info is None:
        return STATUS_UNKNOWN_TR
    return STATUS_LABELS_TR.get(info.status, STATUS_UNKNOWN_TR)


def version_label(version: str, *, sample_size: int | None = None) -> str:
    """Version label with its role and optional sample size for filter labels."""
    label = f"{version} · {status_label_tr(version)}"
    if sample_size is None:
        return label
    return f"{label} · n={sample_size}"


def production_version_for(model_versions: list[str] | None) -> str | None:
    """Return the production version when it appears among the given versions."""
    registry = registry_by_version()
    for version in model_versions or []:
        info = registry.get(version)
        if info is not None and info.status == STATUS_PRODUCTION:
            return version
    return None


def active_production_version() -> str | None:
    """The production version even when no data rows reference it."""
    registry = registry_by_version()
    for info in registry.values():
        if info.status == STATUS_PRODUCTION:
            return info.version
    return None
