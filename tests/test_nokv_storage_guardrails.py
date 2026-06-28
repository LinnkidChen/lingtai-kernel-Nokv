from __future__ import annotations

import json
from pathlib import Path

import pytest

from lingtai.agent import Agent
from lingtai.services.file_io import NoKVFileIOBackend
from tests._service_helpers import make_gemini_mock_service as make_mock_service
from tests.test_nokv_services import FakeNoKVClient


def _write_init(agent_dir: Path, storage: dict | None = None) -> None:
    init = {
        "manifest": {
            "agent_name": "main",
            "language": "en",
            "llm": {"provider": "gemini", "model": "gemini-test", "api_key": "test"},
            "capabilities": {"file": {}},
            "soul": {"delay": 120},
            "stamina": 3600,
            "max_turns": 50,
            "admin": {"karma": True},
            "streaming": False,
        },
        "principle": "p",
        "covenant": "c",
        "pad": "",
        "prompt": "",
    }
    if storage is not None:
        init["storage"] = storage
    agent_dir.mkdir(parents=True, exist_ok=True)
    (agent_dir / "init.json").write_text(json.dumps(init), encoding="utf-8")


def _storage() -> dict:
    return {
        "enabled": True,
        "backend": "nokv",
        "nokv": {
            "namespace_root": "/lingtai/projects/test/agents/${agent_name}",
            "metadata_addr_env": "NOKV_METADATA_ADDR",
            "bucket_env": "NOKV_BUCKET",
            "endpoint_env": "NOKV_ENDPOINT",
        },
        "mounts": ["artifacts", "reports", "checkpoints", "knowledge"],
    }


def _set_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NOKV_METADATA_ADDR", "127.0.0.1:7777")
    monkeypatch.setenv("NOKV_BUCKET", "nokv")
    monkeypatch.setenv("NOKV_ENDPOINT", "http://127.0.0.1:9000")


def test_agent_storage_disabled_preserves_local_file_io(tmp_path: Path):
    agent_dir = tmp_path / ".lingtai" / "main"
    _write_init(agent_dir, storage={"enabled": False})

    agent = Agent(service=make_mock_service(), agent_name="main", working_dir=agent_dir)
    try:
        agent._setup_from_init()
        agent._file_io.write("artifacts/a.md", "local")

        assert (agent_dir / "artifacts" / "a.md").read_text() == "local"
        status = json.loads((agent_dir / "system" / "storage.resolved.json").read_text())
        assert status["enabled"] is False
        assert status["backend"] == "local"
    finally:
        agent.stop(timeout=1.0)


def test_agent_storage_enabled_fails_fast_without_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    agent_dir = tmp_path / ".lingtai" / "main"
    _write_init(agent_dir, storage=_storage())
    _set_env(monkeypatch)

    agent = Agent(service=make_mock_service(), agent_name="main", working_dir=agent_dir)
    try:
        with pytest.raises(ValueError, match="NoKV backend"):
            agent._setup_from_init()
        assert not (agent_dir / "system" / "storage.resolved.json").exists()
    finally:
        agent.stop(timeout=1.0)


def test_agent_storage_enabled_routes_selected_mounts_with_injected_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    agent_dir = tmp_path / ".lingtai" / "main"
    _write_init(agent_dir, storage=_storage())
    _set_env(monkeypatch)
    fake = FakeNoKVClient()

    agent = Agent(service=make_mock_service(), agent_name="main", working_dir=agent_dir)
    agent._nokv_backend = NoKVFileIOBackend(fake)
    try:
        agent._setup_from_init()
        agent._file_io.write("reports/r.md", "remote report")
        agent._file_io.write("mailbox/inbox/msg.json", "{}")

        assert fake.write_calls[0][0] == "/lingtai/projects/test/agents/main/reports/r.md"
        assert (agent_dir / "mailbox" / "inbox" / "msg.json").read_text() == "{}"
        status = json.loads((agent_dir / "system" / "storage.resolved.json").read_text())
        assert status["enabled"] is True
        assert status["routes"][0]["backend"] == "nokv"
        assert "secret" not in str(status).lower()
    finally:
        agent.stop(timeout=1.0)
