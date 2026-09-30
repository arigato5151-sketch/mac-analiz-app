"""Persist trained model artifacts in Supabase Storage instead of git.

Binary model files (joblib) are bulky and change on every retrain, so committing
them to the repository inflates history and—when a scheduled workflow holds write
permission—creates a supply-chain risk. This module moves artifact persistence to
Supabase Storage over the REST API using the same service-role credential as the
rest of the pipeline.

Local model paths remain the primary source so the UI and ad-hoc scripts work
offline; Storage is a fallback that CI uses to publish and later re-hydrate
artifacts without any repository write permission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import requests

from config.settings import get_settings

LOGGER = logging.getLogger(__name__)
STORAGE_BUCKET = "models"
DEFAULT_PREFIX = "model_artifacts"


class ArtifactStoreError(RuntimeError):
    """Raised when a storage operation fails and no local fallback exists."""


class ArtifactIntegrityError(ArtifactStoreError):
    """Raised when an artifact lacks a trusted manifest or fails verification."""


MANIFEST_NAME = "artifact_manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_artifact_manifest(directory: Path | str) -> dict[str, object]:
    root = Path(directory)
    return {
        "version": 1,
        "artifacts": {
            path.name: _sha256(path)
            for path in sorted(root.glob("*.joblib"))
            if path.is_file()
        },
    }


def load_verified_joblib(path: Path | str, *, manifest_path: Path | str | None = None) -> dict[str, Any]:
    """Verify a joblib against its manifest before deserializing it."""
    artifact_path = Path(path)
    manifest = Path(manifest_path) if manifest_path else artifact_path.parent / MANIFEST_NAME
    if not manifest.is_file():
        raise ArtifactIntegrityError(f"Artifact manifest is missing: {manifest}")
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        expected = payload["artifacts"][artifact_path.name]
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ArtifactIntegrityError(f"No valid manifest entry for {artifact_path.name}") from exc
    actual = _sha256(artifact_path)
    if actual != expected:
        raise ArtifactIntegrityError(f"Artifact checksum mismatch for {artifact_path.name}")
    return joblib.load(artifact_path)


def _storage_url() -> str:
    settings = get_settings()
    return f"{settings.supabase_url.rstrip('/')}/storage/v1"


def _headers() -> dict[str, str]:
    settings = get_settings()
    headers = {
        "apikey": settings.supabase_service_role_key,
        "Authorization": f"Bearer {settings.supabase_service_role_key}",
    }
    return headers


def object_path(name: str, *, prefix: str = DEFAULT_PREFIX) -> str:
    """Return the canonical object name inside the bucket."""
    cleaned = name.strip("/")
    return f"{prefix.strip('/')}/{cleaned}" if prefix else cleaned


def ensure_bucket() -> None:
    """Create the model bucket if it does not exist (idempotent)."""
    url = f"{_storage_url()}/bucket"
    headers = _headers()
    response = requests.get(url, headers=headers, timeout=30.0)
    existing = {bucket.get("name") for bucket in response.json() if isinstance(bucket, dict)} if response.ok else set()
    if STORAGE_BUCKET in existing:
        return
    body = {"name": STORAGE_BUCKET, "public": False}
    created = requests.post(url, headers=headers, json=body, timeout=30.0)
    if not created.ok and "already" not in created.text.lower():
        raise ArtifactStoreError(
            f"Could not ensure storage bucket '{STORAGE_BUCKET}': "
            f"{created.status_code} {created.text[:300]}"
        )


def upload_model(
    local_path: Path | str,
    *,
    name: str | None = None,
    upsert: bool = True,
) -> str:
    """Upload one model artifact and return its storage object name."""
    path = Path(local_path)
    if not path.is_file():
        raise ArtifactStoreError(f"Artifact file not found: {path}")
    object_name = object_path(name or path.name)
    ensure_bucket()
    headers = _headers()
    headers["Content-Type"] = "application/octet-stream"
    if upsert:
        headers["x-upsert"] = "true"
    url = f"{_storage_url()}/object/{STORAGE_BUCKET}/{object_name}"
    with path.open("rb") as stream:
        response = requests.put(url, headers=headers, data=stream, timeout=120.0)
    if not response.ok:
        raise ArtifactStoreError(
            f"Storage upload failed for '{object_name}': "
            f"{response.status_code} {response.text[:300]}"
        )
    return object_name


def download_model(
    name: str,
    *,
    local_path: Path | str | None = None,
    dest_dir: Path | str | None = None,
) -> Path:
    """Download an artifact and write it locally; return the written path.

    ``local_path`` wins over ``dest_dir`` (which defaults to the caller's working
    directory). Raises ``ArtifactStoreError`` on failure so callers can fall back
    to a local copy.
    """
    object_name = object_path(name)
    if local_path is not None:
        target = Path(local_path)
    elif dest_dir is not None:
        target = Path(dest_dir) / Path(name).name
    else:
        target = Path(name).name
    url = f"{_storage_url()}/object/{STORAGE_BUCKET}/{object_name}"
    response = requests.get(url, headers=_headers(), timeout=120.0)
    if not response.ok:
        raise ArtifactStoreError(
            f"Storage download failed for '{object_name}': "
            f"{response.status_code} {response.text[:300]}"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(response.content)
    return target


def list_models(prefix: str = DEFAULT_PREFIX) -> list[str]:
    """Return artifact object names under ``prefix`` (empty when none exist)."""
    url = f"{_storage_url()}/object/list/{STORAGE_BUCKET}"
    body: dict[str, Any] = {"prefix": prefix, "limit": 1000, "offset": 0}
    response = requests.post(url, headers=_headers(), json=body, timeout=30.0)
    if not response.ok:
        raise ArtifactStoreError(
            f"Storage list failed: {response.status_code} {response.text[:300]}"
        )
    return [
        str(item.get("name"))
        for item in response.json()
        if isinstance(item, dict) and item.get("name")
    ]


def push_local_models(*, dest_dir: Path | str) -> list[str]:
    """Upload model artifacts, metadata, and feature snapshots from this directory."""
    directory = Path(dest_dir)
    manifest_path = directory / MANIFEST_NAME
    manifest_path.write_text(
        json.dumps(build_artifact_manifest(directory), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    uploads: list[str] = []
    # 1. Joblib models and sibling JSON metadata
    for path in sorted(directory.glob("*.joblib")):
        uploads.append(upload_model(path, name=path.name))
        sibling = path.with_suffix(".json")
        if sibling.is_file():
            uploads.append(upload_model(sibling, name=sibling.name))

    # 2. Parquet snapshots and their companion JSON metadata
    for path in sorted(directory.glob("feature_snapshot_*.parquet")):
        uploads.append(upload_model(path, name=path.name))
        sibling = path.with_suffix(".json")
        if sibling.is_file():
            uploads.append(upload_model(sibling, name=sibling.name))

    uploads.append(upload_model(manifest_path, name=MANIFEST_NAME))

    return uploads


def download_model_artifacts(
    model_version: str = "latest",
    *,
    dest_dir: Path | str,
) -> dict[str, Path]:
    """Download model joblib, json metadata, and associated parquet snapshots from storage.

    Returns a dict mapping artifact kind ('model', 'metadata', 'ref_snapshot', 'current_snapshot')
    to local Path for artifacts successfully downloaded.
    If storage is unreachable, logs an actionable warning and raises ArtifactStoreError so caller
    can gracefully fall back to local copies.
    """
    directory = Path(dest_dir)
    directory.mkdir(parents=True, exist_ok=True)
    downloaded: dict[str, Path] = {}

    clean_version = model_version.strip()
    base_name = "latest" if clean_version in ("", "latest") else clean_version
    joblib_name = f"{base_name}.joblib"
    json_name = f"{base_name}.json"

    try:
        downloaded["model"] = download_model(joblib_name, dest_dir=directory)
    except ArtifactStoreError as exc:
        LOGGER.warning(
            "Remote storage unreachable or artifact '%s' missing: %s. Falling back to local files.",
            joblib_name,
            exc,
        )
        raise

    try:
        downloaded["manifest"] = download_model(MANIFEST_NAME, dest_dir=directory)
    except ArtifactStoreError as exc:
        raise ArtifactIntegrityError(f"Artifact manifest '{MANIFEST_NAME}' is unavailable") from exc

    snapshot_version = base_name
    if base_name == "latest":
        try:
            bundle = load_verified_joblib(downloaded["model"], manifest_path=downloaded["manifest"])
            stored_version = str(bundle.get("model_version", "")).strip()
            if stored_version:
                snapshot_version = stored_version
        except Exception as exc:
            LOGGER.warning("Could not resolve version from latest model bundle: %s", exc)

    # Companion files are best-effort. Prefer stable aliases, then fall back to
    # versioned objects produced by older training runs.
    candidate_groups: dict[str, list[str]] = {
        "metadata": [json_name, f"{snapshot_version}.json"],
        "ref_snapshot": [
            "feature_snapshot_ref.parquet",
            f"feature_snapshot_ref_{snapshot_version}.parquet",
        ],
        "ref_metadata": [
            "feature_snapshot_ref.json",
            f"feature_snapshot_ref_{snapshot_version}.json",
        ],
        "current_snapshot": [
            "feature_snapshot_current.parquet",
            f"feature_snapshot_current_{snapshot_version}.parquet",
        ],
        "current_metadata": [
            "feature_snapshot_current.json",
            f"feature_snapshot_current_{snapshot_version}.json",
        ],
    }
    for key, candidate_names in candidate_groups.items():
        for candidate_name in dict.fromkeys(candidate_names):
            try:
                downloaded[key] = download_model(candidate_name, dest_dir=directory)
                break
            except ArtifactStoreError:
                continue

    return downloaded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--push",
        type=Path,
        help="Upload model artifacts, metadata, and feature snapshots from this directory",
    )
    action.add_argument("--pull", help="Download one stored artifact to the current directory")
    action.add_argument(
        "--pull-bundle",
        metavar="MODEL_VERSION",
        help="Download a model and its metadata and feature snapshots",
    )
    action.add_argument("--list", action="store_true", help="List stored artifact names")
    parser.add_argument(
        "--dest-dir",
        type=Path,
        default=Path("models/saved_models"),
        help="Destination directory used by --pull-bundle",
    )
    args = parser.parse_args()

    if args.push is not None:
        for name in push_local_models(dest_dir=args.push):
            print(f"uploaded {name}")
    elif args.pull:
        path = download_model(args.pull)
        print(f"downloaded {path}")
    elif args.pull_bundle:
        downloaded = download_model_artifacts(
            args.pull_bundle,
            dest_dir=args.dest_dir,
        )
        for kind, path in downloaded.items():
            print(f"downloaded {kind}: {path}")
    elif args.list:
        for name in list_models():
            print(name)


if __name__ == "__main__":
    main()
