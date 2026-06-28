from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from lingtai.agent import Agent
from lingtai.services.file_io import NoKVFileIOBackend
from tests._service_helpers import make_gemini_mock_service as make_mock_service
from tests.test_nokv_services import FakeNoKVClient


def _write_init(agent_dir: Path) -> None:
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
        "storage": {
            "enabled": True,
            "backend": "nokv",
            "nokv": {
                "namespace_root": "/lingtai/projects/test/agents/${agent_name}",
                "metadata_addr_env": "NOKV_METADATA_ADDR",
                "bucket_env": "NOKV_BUCKET",
                "endpoint_env": "NOKV_ENDPOINT",
            },
            "mounts": ["knowledge"],
            "streams": ["logs/events", "history/chat_history", "logs/token_ledger"],
        },
    }
    agent_dir.mkdir(parents=True, exist_ok=True)
    (agent_dir / "init.json").write_text(json.dumps(init), encoding="utf-8")


def _set_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NOKV_METADATA_ADDR", "127.0.0.1:7777")
    monkeypatch.setenv("NOKV_BUCKET", "nokv")
    monkeypatch.setenv("NOKV_ENDPOINT", "http://127.0.0.1:9000")


class FailingMirrorNoKVClient:
    def read(self, path: str) -> dict:
        raise FileNotFoundError(path)

    def write(self, path: str, content: str, metadata: dict | None = None) -> dict:
        raise RuntimeError("mirror unavailable SECRET_TOKEN=abc123")


def test_agent_mirrors_events_chat_history_and_token_ledger_streams(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent_dir = tmp_path / ".lingtai" / "main"
    _write_init(agent_dir)
    _set_env(monkeypatch)
    fake = FakeNoKVClient()

    agent = Agent(service=make_mock_service(), agent_name="main", working_dir=agent_dir)
    agent._nokv_backend = NoKVFileIOBackend(fake)
    try:
        agent._setup_from_init()
        agent._log("tool_result", tool_call_id="tc-1", tool_name="read", result={"ok": True})
        agent.get_chat_state = lambda: {  # type: ignore[method-assign]
            "messages": [{"role": "user", "content": "hello"}]
        }
        agent._last_usage = SimpleNamespace(
            input_tokens=10,
            output_tokens=5,
            thinking_tokens=1,
            cached_tokens=2,
            extra={"api_call_id": "api-1"},
        )
        agent._save_chat_history()

        local_events = agent_dir / "logs" / "events.jsonl"
        local_history = agent_dir / "history" / "chat_history.jsonl"
        local_ledger = agent_dir / "logs" / "token_ledger.jsonl"
        assert local_events.is_file()
        assert local_history.is_file()
        assert local_ledger.is_file()

        remote_paths = set(fake.objects)
        assert "/lingtai/projects/test/agents/main/logs/events/manifest.json" in remote_paths
        assert "/lingtai/projects/test/agents/main/history/chat_history/manifest.json" in remote_paths
        assert "/lingtai/projects/test/agents/main/logs/token_ledger/manifest.json" in remote_paths

        status = json.loads((agent_dir / "system" / "storage.resolved.json").read_text())
        assert [stream["stream"] for stream in status["streams"]] == [
            "logs/events",
            "history/chat_history",
            "logs/token_ledger",
        ]
        assert all(stream["mode"] == "mirror" for stream in status["streams"])
    finally:
        agent.stop(timeout=1.0)


def test_agent_persists_secret_free_degraded_health_when_stream_mirror_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent_dir = tmp_path / ".lingtai" / "main"
    _write_init(agent_dir)
    _set_env(monkeypatch)

    agent = Agent(service=make_mock_service(), agent_name="main", working_dir=agent_dir)
    agent._nokv_backend = NoKVFileIOBackend(FailingMirrorNoKVClient())
    try:
        agent._setup_from_init()
        agent._log("tool_result", tool_call_id="tc-1", tool_name="read", result={"ok": True})

        local_events = agent_dir / "logs" / "events.jsonl"
        assert local_events.is_file()
        assert "tool_result" in local_events.read_text(encoding="utf-8")

        status_text = (agent_dir / "system" / "storage.resolved.json").read_text(
            encoding="utf-8"
        )
        status = json.loads(status_text)
        assert status["health"]["status"] == "degraded"
        assert status["health"]["backend"] == "mirror"
        assert status["health"]["last_error_stream"] == "logs/events"
        assert "RuntimeError" in status["health"]["last_error"]
        assert "SECRET_TOKEN" not in status_text
        assert "abc123" not in status_text
        assert "secret_access_key" not in status_text
        assert not (agent_dir / "mailbox").exists()
        assert not (agent_dir / ".notification").exists()
    finally:
        agent.stop(timeout=1.0)
