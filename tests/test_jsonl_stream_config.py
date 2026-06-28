from __future__ import annotations

from pathlib import Path

import pytest

from lingtai.services.file_io import NoKVFileIOBackend
from lingtai.services.storage_config import resolve_storage_config
from tests.test_nokv_services import FakeNoKVClient
from tests.test_storage_config import _enabled_storage, _set_nokv_env


def test_storage_config_resolves_explicit_jsonl_streams_without_mounting_logs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_nokv_env(monkeypatch)
    raw = _enabled_storage(["knowledge"])
    raw["streams"] = ["logs/events", "history/chat_history", "logs/token_ledger"]

    cfg = resolve_storage_config(
        raw,
        agent_dir=tmp_path / "project" / ".lingtai" / "main",
        nokv_backend=NoKVFileIOBackend(FakeNoKVClient()),
    )

    assert [route.mount for route in cfg.routes] == ["knowledge"]
    assert [stream.stream for stream in cfg.streams] == [
        "logs/events",
        "history/chat_history",
        "logs/token_ledger",
    ]
    assert cfg.streams[0].local_path.name == "events.jsonl"
    assert cfg.streams[0].mode == "mirror"
    assert cfg.streams[0].remote_root.endswith("/logs/events")
    status = cfg.status_document()
    assert status["streams"][0]["stream"] == "logs/events"
    assert status["streams"][0]["mode"] == "mirror"
    assert "logs" not in [route["mount"] for route in status["routes"]]


def test_storage_config_rejects_streams_outside_feature_04_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_nokv_env(monkeypatch)
    raw = _enabled_storage(["knowledge"])
    raw["streams"] = ["mailbox/inbox"]

    with pytest.raises(ValueError, match="unsupported storage stream"):
        resolve_storage_config(
            raw,
            agent_dir=tmp_path / ".lingtai" / "main",
            nokv_backend=NoKVFileIOBackend(FakeNoKVClient()),
        )
