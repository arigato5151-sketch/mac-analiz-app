from __future__ import annotations

from pathlib import Path

import pytest

from models import artifact_store
from models.artifact_store import (
    ArtifactStoreError,
    download_model,
    list_models,
    upload_model,
)


@pytest.fixture(autouse=True)
def _storage_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_FOOTBALL_KEY", "test-api-key")
    monkeypatch.setenv("SUPABASE_URL", "https://demo.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "test-service-role-key")
    # The module caches nothing, so env-driven helpers pick up these values.
    artifact_store.__dict__.pop("_settings_cache", None)


class _FakeBucket:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.list_calls = 0

    def put(self, name: str, content: bytes) -> None:
        self.objects[name] = content

    def get(self, name: str) -> bytes:
        return self.objects[name]


def test_upload_and_download_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bucket = _FakeBucket()

    class Response:
        ok = True

        def __init__(self, content: bytes = b"") -> None:
            self.content = content
            self.text = ""

        def json(self) -> object:
            return []  # Callers that parse JSON expect a list here.

    class FakeRequests:
        def __init__(self) -> None:
            self.published_bucket = False

        def get(self, url: str, **kwargs: object) -> Response:
            if url.endswith("/storage/v1/bucket"):
                from models.artifact_store import STORAGE_BUCKET

                return Response(b'[{"name": "' + STORAGE_BUCKET.encode() + b'"}]')
            name = url.rsplit("/", 1)[-1]
            return Response(bucket.objects.get(name, b"missing"))

        def post(self, url: str, **kwargs: object) -> Response:
            return Response(b"{}")

        def put(self, url: str, **kwargs: object) -> Response:
            name = url.rsplit("/", 1)[-1]
            body = kwargs.get("data")
            bucket.put(name, body.read() if hasattr(body, "read") else body)
            return Response(b"{}")

    fake = FakeRequests()
    monkeypatch.setattr(artifact_store.requests, "get", fake.get)
    monkeypatch.setattr(artifact_store.requests, "put", fake.put)
    monkeypatch.setattr(artifact_store.requests, "post", fake.post)

    local_path = tmp_path / "latest.joblib"
    local_path.write_bytes(b"binary-model-payload")
    name = upload_model(local_path, name="latest.joblib")
    assert name.endswith("/latest.joblib")

    target = tmp_path / "out" / "latest.joblib"
    downloaded = download_model("latest.joblib", local_path=target)
    assert downloaded.read_bytes() == b"binary-model-payload"


def test_upload_raises_when_file_missing(tmp_path: Path) -> None:
    with pytest.raises(ArtifactStoreError):
        upload_model(tmp_path / "nope.joblib")


def test_list_models_returns_object_names(monkeypatch: pytest.MonkeyPatch) -> None:
    class Response:
        ok = True
        text = ""

        def json(self) -> object:
            return [{"name": "model_artifacts/latest.joblib"}]

    monkeypatch.setattr(
        artifact_store.requests, "post", lambda *args, **kwargs: Response()
    )

    assert list_models() == ["model_artifacts/latest.joblib"]