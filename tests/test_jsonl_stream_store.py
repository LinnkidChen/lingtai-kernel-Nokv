from __future__ import annotations

import json
from pathlib import Path

from lingtai_kernel.jsonl_stream import (
    FilesystemJsonlStreamStore,
    MirrorJsonlStreamStore,
    NoKVSegmentedJsonlStreamStore,
)
from lingtai_kernel.tool_result_recovery import recover_tool_result_block_from_events
from tests.test_nokv_services import FakeNoKVClient


def _jsonl_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_filesystem_store_preserves_jsonl_append_tail_range_and_export(tmp_path: Path) -> None:
    store = FilesystemJsonlStreamStore(tmp_path)

    pos1 = store.append("logs/events", {"type": "tool_call", "tool_call_id": "tc-1"})
    pos2 = store.append("logs/events", {"type": "tool_result", "tool_call_id": "tc-2"})

    assert pos1.seq == 1
    assert pos1.source_offset == 0
    assert pos2.seq == 2
    assert _jsonl_records(tmp_path / "logs" / "events.jsonl") == [
        {"type": "tool_call", "tool_call_id": "tc-1"},
        {"type": "tool_result", "tool_call_id": "tc-2"},
    ]
    assert store.tail("logs/events", 1) == [{"type": "tool_result", "tool_call_id": "tc-2"}]
    assert list(store.iter_range("logs/events", 1, 1)) == [
        {"type": "tool_call", "tool_call_id": "tc-1"}
    ]

    export_path = tmp_path / "export" / "events.jsonl"
    result = store.export_jsonl("logs/events", export_path)
    assert result.record_count == 2
    assert export_path.read_text(encoding="utf-8") == (
        tmp_path / "logs" / "events.jsonl"
    ).read_text(encoding="utf-8")
    assert store.health().status == "ok"


def test_nokv_segmented_store_appends_tail_range_find_and_exports(tmp_path: Path) -> None:
    fake = FakeNoKVClient()
    remote_root = "/lingtai/projects/test/agents/main/logs/events"
    store = NoKVSegmentedJsonlStreamStore(
        tmp_path,
        {"logs/events": remote_root},
        fake,
        segment_max_records=1,
    )

    store.append("logs/events", {"type": "tool_call", "tool_call_id": "tc-1"})
    store.append(
        "logs/events",
        {"type": "tool_result", "tool_call_id": "tc-2", "result": {"ok": True}},
    )

    manifest = json.loads(fake.objects[f"{remote_root}/manifest.json"].content)
    assert manifest["schema_version"] == 1
    assert manifest["stream"] == "logs/events"
    assert manifest["record_count"] == 2
    assert manifest["first_seq"] == 1
    assert manifest["last_seq"] == 2
    assert manifest["generation"] == 2
    assert len(manifest["segments"]) == 2
    assert all(segment["name"].startswith("g") for segment in manifest["segments"])
    assert all(f"{remote_root}/segments/{segment['name']}" in fake.objects for segment in manifest["segments"])

    assert store.tail("logs/events", 1) == [
        {"type": "tool_result", "tool_call_id": "tc-2", "result": {"ok": True}}
    ]
    assert list(store.iter_range("logs/events", 1, 1)) == [
        {"type": "tool_call", "tool_call_id": "tc-1"}
    ]
    assert store.find("logs/events", tool_call_id="tc-2") == [
        {"type": "tool_result", "tool_call_id": "tc-2", "result": {"ok": True}}
    ]

    export_path = tmp_path / "events-export.jsonl"
    result = store.export_jsonl("logs/events", export_path)
    assert result.record_count == 2
    assert _jsonl_records(export_path) == [
        {"type": "tool_call", "tool_call_id": "tc-1"},
        {"type": "tool_result", "tool_call_id": "tc-2", "result": {"ok": True}},
    ]


def test_mirror_store_keeps_local_jsonl_when_nokv_mirror_fails(tmp_path: Path) -> None:
    class FailingClient:
        def read(self, path: str) -> dict:
            raise FileNotFoundError(path)

        def write(self, path: str, content: str, metadata: dict | None = None) -> dict:
            raise RuntimeError("NoKV unavailable")

    local = FilesystemJsonlStreamStore(tmp_path)
    remote = NoKVSegmentedJsonlStreamStore(
        tmp_path,
        {"logs/events": "/lingtai/projects/test/agents/main/logs/events"},
        FailingClient(),
    )
    mirror = MirrorJsonlStreamStore(local, remote)

    position = mirror.append("logs/events", {"type": "heartbeat_start"})

    assert position.seq == 1
    assert _jsonl_records(tmp_path / "logs" / "events.jsonl") == [
        {"type": "heartbeat_start"}
    ]
    health = mirror.health()
    assert health.status == "degraded"
    assert health.last_error == "RuntimeError: mirror write failed"


def test_mirror_find_falls_back_to_nokv_when_local_primary_misses(tmp_path: Path) -> None:
    fake = FakeNoKVClient()
    remote_root = "/lingtai/projects/test/agents/main/logs/events"
    local = FilesystemJsonlStreamStore(tmp_path, streams=["logs/events"])
    remote = NoKVSegmentedJsonlStreamStore(tmp_path, {"logs/events": remote_root}, fake)
    mirror = MirrorJsonlStreamStore(local, remote)

    remote.append(
        "logs/events",
        {
            "type": "tool_result",
            "tool_call_id": "tc-mirror",
            "result": {"content": "from mirror"},
        },
    )

    assert mirror.find("logs/events", tool_call_id="tc-mirror") == [
        {
            "type": "tool_result",
            "tool_call_id": "tc-mirror",
            "result": {"content": "from mirror"},
        }
    ]


def test_segmented_store_replace_from_file_preserves_export_equivalence(tmp_path: Path) -> None:
    local_history = tmp_path / "history" / "chat_history.jsonl"
    local_history.parent.mkdir(parents=True)
    local_history.write_text(
        "\n".join([
            json.dumps({"role": "user", "content": "hello"}, ensure_ascii=False),
            json.dumps({"role": "assistant", "content": "hi"}, ensure_ascii=False),
        ])
        + "\n",
        encoding="utf-8",
    )

    fake = FakeNoKVClient()
    store = NoKVSegmentedJsonlStreamStore(
        tmp_path,
        {"history/chat_history": "/lingtai/projects/test/agents/main/history/chat_history"},
        fake,
        segment_max_records=1,
    )

    result = store.replace_from_file("history/chat_history", local_history)
    export_path = tmp_path / "history-export.jsonl"
    store.export_jsonl("history/chat_history", export_path)

    assert result.record_count == 2
    assert export_path.read_text(encoding="utf-8") == local_history.read_text(encoding="utf-8")


def test_tool_result_recovery_can_read_stream_store_when_local_events_missing(tmp_path: Path) -> None:
    fake = FakeNoKVClient()
    store = NoKVSegmentedJsonlStreamStore(
        tmp_path,
        {"logs/events": "/lingtai/projects/test/agents/main/logs/events"},
        fake,
    )
    store.append(
        "logs/events",
        {
            "type": "tool_result",
            "tool_call_id": "tc-stream",
            "tool_name": "read",
            "result": {"content": "from stream"},
        },
    )

    block = recover_tool_result_block_from_events(
        tmp_path,
        tool_call_id="tc-stream",
        tool_name="read",
        stream_store=store,
    )

    assert block is not None
    assert block.id == "tc-stream"
    assert block.name == "read"
    assert block.content == {"content": "from stream"}
