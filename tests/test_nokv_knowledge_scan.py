from __future__ import annotations

from pathlib import Path

from lingtai.agent import Agent
from lingtai.services.file_io import LocalFileIOBackend, NoKVFileIOBackend
from lingtai.services.file_io_factory import build_file_io_service
from lingtai.services.storage_config import StorageRoute
from tests._service_helpers import make_gemini_mock_service as make_mock_service
from tests.test_nokv_services import FakeNoKVClient


def test_knowledge_scan_uses_file_io_for_routed_knowledge(tmp_path: Path):
    agent_dir = tmp_path / ".lingtai" / "main"
    fake = FakeNoKVClient()
    fake.write(
        "/remote/main/knowledge/retry/KNOWLEDGE.md",
        "---\nname: retry-storm\ndescription: Handles retry storms.\n---\n\nBody must stay out.\n",
    )
    file_io = build_file_io_service(
        root=agent_dir,
        routes=[
            StorageRoute(
                mount="knowledge",
                local_root=agent_dir / "knowledge",
                remote_root="/remote/main/knowledge",
            )
        ],
        local_backend=LocalFileIOBackend(root=agent_dir),
        nokv_backend=NoKVFileIOBackend(fake),
    )

    agent = Agent(
        service=make_mock_service(),
        agent_name="main",
        working_dir=agent_dir,
        file_io=file_io,
        capabilities={"knowledge": {}},
    )
    try:
        result = agent._tool_handlers["knowledge"]({"action": "info"})
        prompt = agent._prompt_manager.read_section("knowledge") or ""

        assert result["catalog_size"] == 1
        assert "retry-storm" in prompt
        assert "Handles retry storms." in prompt
        assert "Body must stay out" not in prompt
        assert str(agent_dir / "knowledge" / "retry" / "KNOWLEDGE.md") in prompt
    finally:
        agent.stop(timeout=1.0)
