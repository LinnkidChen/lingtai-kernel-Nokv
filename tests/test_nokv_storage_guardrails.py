from __future__ import annotations

import json

import pytest

from lingtai.agent import Agent
from tests._service_helpers import make_gemini_mock_service as make_mock_service


def _enabled_storage_init() -> dict:
    return {
        "storage": {
            "enabled": True,
            "backend": "nokv",
            "nokv": {
                "namespace_root": "/lingtai/projects/testproj/agents/alice",
                "metadata_addr_env": "NOKV_METADATA_ADDR",
                "bucket_env": "NOKV_BUCKET",
                "endpoint_env": "NOKV_ENDPOINT",
                "access_key_id_env": "AWS_ACCESS_KEY_ID",
                "secret_access_key_env": "AWS_SECRET_ACCESS_KEY",
                "region_env": "AWS_REGION",
            },
            "mounts": ["artifacts", "reports", "checkpoints", "knowledge"],
        }
    }


def test_enabled_storage_direct_boot_with_injected_nokv_client_routes_and_writes_status(
    tmp_path, monkeypatch
):
    from tests.test_nokv_services import FakeNoKVClient

    workdir = tmp_path / ".lingtai" / "alice"
    monkeypatch.setenv("NOKV_METADATA_ADDR", "127.0.0.1:7777")
    monkeypatch.setenv("NOKV_BUCKET", "nokv")
    monkeypatch.setenv("NOKV_ENDPOINT", "http://127.0.0.1:9000")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "visible-access-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "hidden-secret-value")
    monkeypatch.setenv("AWS_REGION", "us-west-2")
    workdir.mkdir(parents=True)
    (workdir / "init.json").write_text(
        json.dumps(
            {
                "storage": {
                    "enabled": True,
                    "backend": "nokv",
                    "nokv": {
                        "namespace_root": "/lingtai/projects/testproj/agents/alice",
                        "metadata_addr_env": "NOKV_METADATA_ADDR",
                        "bucket_env": "NOKV_BUCKET",
                        "endpoint_env": "NOKV_ENDPOINT",
                        "access_key_id_env": "AWS_ACCESS_KEY_ID",
                        "secret_access_key_env": "AWS_SECRET_ACCESS_KEY",
                        "region_env": "AWS_REGION",
                    },
                    "mounts": ["artifacts", "reports", "checkpoints", "knowledge"],
                }
            }
        ),
        encoding="utf-8",
    )

    fake_client = FakeNoKVClient()
    monkeypatch.setattr(Agent, "_nokv_storage_client_factory", staticmethod(lambda storage: fake_client))

    agent = Agent(
        service=make_mock_service(),
        agent_name="alice",
        working_dir=workdir,
        capabilities=["write", "read"],
    )
    try:
        agent._tool_handlers["write"]({
            "file_path": "knowledge/topic/KNOWLEDGE.md",
            "content": "remote body",
        })
        agent._tool_handlers["write"]({
            "file_path": "logs/events.jsonl",
            "content": "local runtime",
        })

        status = json.loads(
            (workdir / "system" / "storage.resolved.json").read_text(encoding="utf-8")
        )
        status_text = json.dumps(status, sort_keys=True)

        assert agent._file_io.is_routed_to_nokv("knowledge/topic/KNOWLEDGE.md")
        assert not agent._file_io.is_routed_to_nokv("logs/events.jsonl")
        assert fake_client.objects[
            "/lingtai/projects/testproj/agents/alice/knowledge/topic/KNOWLEDGE.md"
        ].content == "remote body"
        assert (workdir / "logs" / "events.jsonl").read_text(encoding="utf-8") == "local runtime"
        assert status["enabled"] is True
        assert status["backend"] == "routed"
        assert {route["mount"] for route in status["routes"]} == {
            "artifacts",
            "reports",
            "checkpoints",
            "knowledge",
        }
        assert "hidden-secret-value" not in status_text
        assert "visible-access-key" not in status_text
        assert "AWS_SECRET_ACCESS_KEY" not in status_text
    finally:
        agent.stop(timeout=1.0)


def test_enabled_storage_direct_boot_without_nokv_client_fails_before_status(
    tmp_path, monkeypatch
):
    from lingtai.services.nokv import NoKVUnsupportedError

    workdir = tmp_path / ".lingtai" / "alice"
    monkeypatch.setenv("NOKV_METADATA_ADDR", "127.0.0.1:7777")
    monkeypatch.setenv("NOKV_BUCKET", "nokv")
    monkeypatch.setenv("NOKV_ENDPOINT", "http://127.0.0.1:9000")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "visible-access-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "hidden-secret-value")
    monkeypatch.setenv("AWS_REGION", "us-west-2")
    workdir.mkdir(parents=True)
    (workdir / "init.json").write_text(json.dumps(_enabled_storage_init()), encoding="utf-8")

    with pytest.raises(NoKVUnsupportedError, match="NoKV backend/client"):
        Agent(
            service=make_mock_service(),
            agent_name="alice",
            working_dir=workdir,
            capabilities={"knowledge": None, "skills": None},
        )

    assert not (workdir / "system" / "storage.resolved.json").exists()


def test_disabled_storage_preserves_default_local_file_io_behavior(tmp_path):
    workdir = tmp_path / ".lingtai" / "alice"
    workdir.mkdir(parents=True)
    (workdir / "init.json").write_text(
        json.dumps({"storage": {"enabled": False}}),
        encoding="utf-8",
    )

    agent = Agent(
        service=make_mock_service(),
        agent_name="alice",
        working_dir=workdir,
        capabilities=["write", "read"],
    )
    try:
        agent._tool_handlers["write"]({
            "file_path": "artifacts/local.md",
            "content": "still local",
        })
        assert (workdir / "artifacts" / "local.md").read_text(encoding="utf-8") == "still local"
        assert not (workdir / "system" / "storage.resolved.json").exists()
    finally:
        agent.stop(timeout=1.0)


def test_factory_with_injected_nokv_backend_writes_secret_free_routed_status(
    tmp_path, monkeypatch
):
    from lingtai.services.file_io import LocalFileIOBackend, LocalFileIOService
    from lingtai.services.file_io_factory import build_routed_file_io_service
    from lingtai.services.storage_config import parse_storage_config
    from tests.test_routed_file_io_backend import RecordingNoKVBackend

    workdir = tmp_path / ".lingtai" / "alice"
    workdir.mkdir(parents=True)
    monkeypatch.setenv("NOKV_METADATA_ADDR", "127.0.0.1:7777")
    monkeypatch.setenv("NOKV_BUCKET", "nokv")
    monkeypatch.setenv("NOKV_ENDPOINT", "http://127.0.0.1:9000")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "visible-access-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "hidden-secret-value")
    monkeypatch.setenv("AWS_REGION", "us-west-2")
    (workdir / "init.json").write_text(
        json.dumps(_enabled_storage_init()),
        encoding="utf-8",
    )
    storage = parse_storage_config(
        _enabled_storage_init(),
        agent_dir=workdir,
        project_root=tmp_path,
        project_hash="testproj",
        agent_name="alice",
        environ=dict(
            NOKV_METADATA_ADDR="127.0.0.1:7777",
            NOKV_BUCKET="nokv",
            NOKV_ENDPOINT="http://127.0.0.1:9000",
            AWS_ACCESS_KEY_ID="visible-access-key",
            AWS_SECRET_ACCESS_KEY="hidden-secret-value",
            AWS_REGION="us-west-2",
        ),
    )

    service = build_routed_file_io_service(
        agent_dir=workdir,
        local_service=LocalFileIOService(backend=LocalFileIOBackend(root=workdir)),
        storage=storage,
        nokv_backend=RecordingNoKVBackend(),
    )
    service.write("knowledge/topic/KNOWLEDGE.md", "body")

    status_text = json.dumps(storage.to_status(), indent=2, sort_keys=True)
    assert service.is_routed_to_nokv("knowledge/topic/KNOWLEDGE.md")
    assert "hidden-secret-value" not in status_text
    assert "visible-access-key" not in status_text
    assert "AWS_SECRET_ACCESS_KEY" not in status_text


def test_disabled_storage_refresh_removes_stale_resolved_status(tmp_path):
    workdir = tmp_path / ".lingtai" / "alice"
    workdir.mkdir(parents=True)
    (workdir / "init.json").write_text(
        json.dumps({"storage": {"enabled": False}}),
        encoding="utf-8",
    )
    status_path = workdir / "system" / "storage.resolved.json"
    status_path.parent.mkdir(parents=True)
    status_path.write_text("stale\n", encoding="utf-8")

    agent = Agent(
        service=make_mock_service(),
        agent_name="alice",
        working_dir=workdir,
        capabilities={"knowledge": None, "skills": None},
    )
    try:
        agent._configure_storage({"storage": {"enabled": False}})

        assert not status_path.exists()
    finally:
        agent.stop(timeout=1.0)


def test_repeated_enabled_storage_refresh_does_not_nest_routed_backend(tmp_path, monkeypatch):
    from lingtai.services.file_io import LocalFileIOBackend, LocalFileIOService, RoutedFileIOBackend
    from lingtai.services.file_io_factory import build_routed_file_io_service
    from lingtai.services.storage_config import parse_storage_config
    from tests.test_routed_file_io_backend import RecordingNoKVBackend

    workdir = tmp_path / ".lingtai" / "alice"
    workdir.mkdir(parents=True)
    monkeypatch.setenv("NOKV_METADATA_ADDR", "127.0.0.1:7777")
    monkeypatch.setenv("NOKV_BUCKET", "nokv")
    monkeypatch.setenv("NOKV_ENDPOINT", "http://127.0.0.1:9000")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "visible-access-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "hidden-secret-value")
    monkeypatch.setenv("AWS_REGION", "us-west-2")
    enabled = _enabled_storage_init()
    (workdir / "init.json").write_text(json.dumps(enabled), encoding="utf-8")
    storage = parse_storage_config(
        enabled,
        agent_dir=workdir,
        project_root=tmp_path,
        project_hash="testproj",
        agent_name="alice",
        environ=dict(
            NOKV_METADATA_ADDR="127.0.0.1:7777",
            NOKV_BUCKET="nokv",
            NOKV_ENDPOINT="http://127.0.0.1:9000",
            AWS_ACCESS_KEY_ID="visible-access-key",
            AWS_SECRET_ACCESS_KEY="hidden-secret-value",
            AWS_REGION="us-west-2",
        ),
    )
    base_file_io = LocalFileIOService(backend=LocalFileIOBackend(root=workdir))

    first = build_routed_file_io_service(
        agent_dir=workdir,
        local_service=base_file_io,
        storage=storage,
        nokv_backend=RecordingNoKVBackend(),
    )
    second = build_routed_file_io_service(
        agent_dir=workdir,
        local_service=base_file_io,
        storage=storage,
        nokv_backend=RecordingNoKVBackend(),
    )

    assert isinstance(first._backend, RoutedFileIOBackend)
    assert isinstance(second._backend, RoutedFileIOBackend)
    assert not isinstance(second._backend._local_backend, RoutedFileIOBackend)


def test_enabled_storage_rejects_runtime_mounts_before_agent_starts(tmp_path):
    from lingtai.services.storage_config import StorageConfigError

    workdir = tmp_path / ".lingtai" / "alice"
    workdir.mkdir(parents=True)
    (workdir / "init.json").write_text(
        json.dumps(
            {
                "storage": {
                    "enabled": True,
                    "backend": "nokv",
                    "nokv": {
                        "namespace_root": "/lingtai/projects/testproj/agents/alice",
                        "metadata_addr_env": "NOKV_METADATA_ADDR",
                        "bucket_env": "NOKV_BUCKET",
                        "endpoint_env": "NOKV_ENDPOINT",
                    },
                    "mounts": ["knowledge", "mailbox"],
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(StorageConfigError, match="mailbox"):
        Agent(
            service=make_mock_service(),
            agent_name="alice",
            working_dir=workdir,
            capabilities={"knowledge": None, "skills": None},
        )
