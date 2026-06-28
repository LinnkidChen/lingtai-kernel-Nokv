from __future__ import annotations

from pathlib import Path

import pytest

from lingtai.services.file_io import GrepMatch, LocalFileIOBackend, NoKVFileIOBackend
from lingtai.services.file_io_factory import build_file_io_service
from lingtai.services.storage_config import StorageRoute
from tests.test_nokv_services import FakeNoKVClient


def _service(tmp_path: Path, fake: FakeNoKVClient):
    agent_dir = tmp_path / ".lingtai" / "main"
    routes = [
        StorageRoute(
            mount="artifacts",
            local_root=agent_dir / "artifacts",
            remote_root="/remote/main/artifacts",
        ),
        StorageRoute(
            mount="reports",
            local_root=agent_dir / "reports",
            remote_root="/remote/main/reports",
        ),
        StorageRoute(
            mount="checkpoints",
            local_root=agent_dir / "checkpoints",
            remote_root="/remote/main/checkpoints",
        ),
        StorageRoute(
            mount="knowledge",
            local_root=agent_dir / "knowledge",
            remote_root="/remote/main/knowledge",
        ),
    ]
    return build_file_io_service(
        root=agent_dir,
        routes=routes,
        local_backend=LocalFileIOBackend(root=agent_dir),
        nokv_backend=NoKVFileIOBackend(fake),
    )


def test_routed_file_io_routes_only_selected_mounts(tmp_path: Path):
    fake = FakeNoKVClient()
    svc = _service(tmp_path, fake)

    svc.write("artifacts/a.md", "remote artifact")
    svc.write("logs/events.jsonl", "local log")

    assert fake.write_calls[0][0] == "/remote/main/artifacts/a.md"
    assert svc.read("artifacts/a.md") == "remote artifact"
    assert (tmp_path / ".lingtai" / "main" / "logs" / "events.jsonl").read_text() == "local log"
    assert "/remote/main/logs/events.jsonl" not in fake.objects


def test_routed_file_io_edit_preserves_unique_replace_contract(tmp_path: Path):
    fake = FakeNoKVClient()
    svc = _service(tmp_path, fake)
    svc.write("reports/r.md", "alpha beta")

    assert svc.edit("reports/r.md", "beta", "gamma") == "alpha gamma"
    assert svc.read("reports/r.md") == "alpha gamma"

    svc.write("reports/r.md", "dup dup")
    with pytest.raises(ValueError, match="appears 2 times"):
        svc.edit("reports/r.md", "dup", "x")


def test_routed_glob_and_grep_return_virtual_local_paths(tmp_path: Path):
    fake = FakeNoKVClient()
    svc = _service(tmp_path, fake)
    agent_dir = tmp_path / ".lingtai" / "main"

    svc.write("knowledge/retry/KNOWLEDGE.md", "---\nname: retry\ndescription: retry storms\n---\nneedle\n")
    svc.write("notes/local.md", "local needle\n")

    assert svc.glob("**/*.md", root=str(agent_dir)) == [
        str(agent_dir / "knowledge" / "retry" / "KNOWLEDGE.md"),
        str(agent_dir / "notes" / "local.md"),
    ]
    assert svc.grep("needle", path=str(agent_dir), max_results=10) == [
        GrepMatch(str(agent_dir / "knowledge" / "retry" / "KNOWLEDGE.md"), 5, "needle"),
        GrepMatch(str(agent_dir / "notes" / "local.md"), 1, "local needle"),
    ]
    assert "remote" not in svc.glob("**/*.md", root=str(agent_dir / "knowledge"))[0]


def test_routed_glob_and_grep_respect_subdirectory_roots(tmp_path: Path):
    fake = FakeNoKVClient()
    svc = _service(tmp_path, fake)
    agent_dir = tmp_path / ".lingtai" / "main"

    svc.write("knowledge/retry/KNOWLEDGE.md", "needle retry\n")
    svc.write("knowledge/other/KNOWLEDGE.md", "needle other\n")

    retry_root = str(agent_dir / "knowledge" / "retry")

    assert svc.glob("*.md", root=retry_root) == [
        str(agent_dir / "knowledge" / "retry" / "KNOWLEDGE.md")
    ]
    assert svc.grep("needle", path=retry_root, max_results=10) == [
        GrepMatch(str(agent_dir / "knowledge" / "retry" / "KNOWLEDGE.md"), 1, "needle retry")
    ]
