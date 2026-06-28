from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from lingtai.services.file_io import NoKVFileIOBackend
from lingtai.services.storage_config import resolve_storage_config
from tests.test_nokv_services import FakeNoKVClient


def _enabled_storage(mounts: list[str] | None = None) -> dict:
    return {
        "enabled": True,
        "backend": "nokv",
        "nokv": {
            "namespace_root": "/lingtai/projects/${project_hash}/agents/${agent_name}",
            "metadata_addr_env": "NOKV_METADATA_ADDR",
            "bucket_env": "NOKV_BUCKET",
            "endpoint_env": "NOKV_ENDPOINT",
            "access_key_id_env": "AWS_ACCESS_KEY_ID",
            "secret_access_key_env": "AWS_SECRET_ACCESS_KEY",
            "region_env": "AWS_REGION",
        },
        "mounts": mounts or ["artifacts", "reports", "checkpoints", "knowledge"],
    }


def _set_nokv_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NOKV_METADATA_ADDR", "127.0.0.1:7777")
    monkeypatch.setenv("NOKV_BUCKET", "nokv")
    monkeypatch.setenv("NOKV_ENDPOINT", "http://127.0.0.1:9000")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "access")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setenv("AWS_REGION", "us-west-2")


def test_storage_config_defaults_to_local_when_absent(tmp_path: Path):
    cfg = resolve_storage_config(None, agent_dir=tmp_path / ".lingtai" / "main")

    assert cfg.enabled is False
    assert cfg.routes == []
    assert cfg.status_document()["backend"] == "local"


def test_storage_config_expands_mounts_and_secret_free_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    project_dir = tmp_path / "project"
    agent_dir = project_dir / ".lingtai" / "main"
    _set_nokv_env(monkeypatch)

    cfg = resolve_storage_config(
        _enabled_storage(["artifacts", "knowledge"]),
        agent_dir=agent_dir,
        nokv_backend=NoKVFileIOBackend(FakeNoKVClient()),
    )

    project_hash = hashlib.sha256(str(project_dir).encode("utf-8")).hexdigest()[:12]
    assert cfg.enabled is True
    assert [route.mount for route in cfg.routes] == ["artifacts", "knowledge"]
    assert cfg.routes[0].local_root == agent_dir / "artifacts"
    assert cfg.routes[0].remote_root == f"/lingtai/projects/{project_hash}/agents/main/artifacts"

    status = cfg.status_document()
    assert status["schema"] == "lingtai.storage.resolved/v1"
    assert status["enabled"] is True
    assert status["routes"][1]["mount"] == "knowledge"
    assert status["nokv"]["metadata_addr"] == "127.0.0.1:7777"
    assert status["nokv"]["bucket"] == "nokv"
    assert status["nokv"]["endpoint"] == "http://127.0.0.1:9000"
    assert "secret" not in str(status).lower()
    assert "access" not in str(status).lower()


def test_storage_config_rejects_unapproved_mount_and_literal_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _set_nokv_env(monkeypatch)
    with pytest.raises(ValueError, match="unsupported storage mount"):
        resolve_storage_config(
            _enabled_storage(["logs"]),
            agent_dir=tmp_path / ".lingtai" / "main",
            nokv_backend=NoKVFileIOBackend(FakeNoKVClient()),
        )

    raw = _enabled_storage()
    raw["nokv"]["secret_access_key"] = "literal-secret"
    with pytest.raises(ValueError, match="stores env var names"):
        resolve_storage_config(
            raw,
            agent_dir=tmp_path / ".lingtai" / "main",
            nokv_backend=NoKVFileIOBackend(FakeNoKVClient()),
        )


def test_storage_config_enabled_requires_env_and_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv("NOKV_METADATA_ADDR", raising=False)
    monkeypatch.delenv("NOKV_BUCKET", raising=False)
    monkeypatch.delenv("NOKV_ENDPOINT", raising=False)
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.delenv("AWS_REGION", raising=False)

    with pytest.raises(ValueError, match="NOKV_METADATA_ADDR"):
        resolve_storage_config(
            _enabled_storage(),
            agent_dir=tmp_path / ".lingtai" / "main",
            nokv_backend=NoKVFileIOBackend(FakeNoKVClient()),
        )

    _set_nokv_env(monkeypatch)
    with pytest.raises(ValueError, match="NoKV backend"):
        resolve_storage_config(
            _enabled_storage(),
            agent_dir=tmp_path / ".lingtai" / "main",
        )


def test_storage_config_enabled_requires_public_nokv_env_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _set_nokv_env(monkeypatch)
    raw = _enabled_storage()
    del raw["nokv"]["bucket_env"]

    with pytest.raises(ValueError, match="storage.nokv.bucket_env"):
        resolve_storage_config(
            raw,
            agent_dir=tmp_path / ".lingtai" / "main",
            nokv_backend=NoKVFileIOBackend(FakeNoKVClient()),
        )
