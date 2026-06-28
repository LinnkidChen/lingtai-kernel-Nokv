from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lingtai.core import knowledge


@dataclass
class _Config:
    language: str = "en"


class FakeNoKVKnowledgeFileIO:
    def __init__(self, files: dict[str, str], knowledge_root: Path):
        self.files = files
        self.knowledge_root = knowledge_root
        self.glob_calls: list[tuple[str, str | None]] = []
        self.read_calls: list[str] = []
        self.knowledge_backed_by_nokv = True

    def is_routed_to_nokv(self, path: str | Path) -> bool:
        return Path(path) == self.knowledge_root

    def glob(self, pattern: str, root: str | None = None) -> list[str]:
        self.glob_calls.append((pattern, root))
        assert root == str(self.knowledge_root)
        assert pattern == "**/KNOWLEDGE.md"
        return sorted(self.files)

    def read(self, path: str) -> str:
        self.read_calls.append(path)
        return self.files[path]


class FakeAgent:
    def __init__(self, working_dir: Path, file_io: FakeNoKVKnowledgeFileIO):
        self._working_dir = working_dir
        self._file_io = file_io
        self._config = _Config()
        self.prompt_sections: dict[str, str] = {}

    def update_system_prompt(self, section: str, content: str, *, protected: bool = False):
        assert protected is True
        self.prompt_sections[section] = content


def test_nokv_backed_knowledge_scan_uses_file_io_and_catalogs_metadata_only(tmp_path):
    knowledge_root = tmp_path / "agent" / "knowledge"
    entry_path = str(knowledge_root / "retry-storm" / "KNOWLEDGE.md")
    body_sentinel = "BODY_SENTINEL_must_not_enter_prompt"
    file_io = FakeNoKVKnowledgeFileIO(
        {
            entry_path: (
                "---\n"
                "name: retry-storm\n"
                "description: Retry storm diagnosis.\n"
                "---\n\n"
                f"{body_sentinel}\n"
            )
        },
        knowledge_root,
    )
    agent = FakeAgent(tmp_path / "agent", file_io)

    result = knowledge._reconcile(agent)

    assert file_io.glob_calls == [("**/KNOWLEDGE.md", str(knowledge_root))]
    assert file_io.read_calls == [entry_path]
    assert result["catalog_size"] == 1
    assert result["problems"] == []
    prompt = agent.prompt_sections["knowledge"]
    assert "- name: retry-storm" in prompt
    assert "Retry storm diagnosis." in prompt
    assert f"location: {entry_path}" in prompt
    assert body_sentinel not in prompt


def test_nokv_backed_knowledge_invalid_frontmatter_becomes_problem(tmp_path):
    knowledge_root = tmp_path / "agent" / "knowledge"
    bad_path = str(knowledge_root / "bad-entry" / "KNOWLEDGE.md")
    file_io = FakeNoKVKnowledgeFileIO(
        {bad_path: "---\nname: bad-entry\n---\n\nmissing description\n"},
        knowledge_root,
    )
    agent = FakeAgent(tmp_path / "agent", file_io)

    result = knowledge._reconcile(agent)

    assert result["catalog_size"] == 0
    assert any("description" in problem["reason"] for problem in result["problems"])
    assert agent.prompt_sections["knowledge"] == ""


def test_nokv_backed_knowledge_skips_legacy_json_migration(tmp_path):
    workdir = tmp_path / "agent"
    knowledge_root = workdir / "knowledge"
    legacy_json = knowledge_root / "knowledge.json"
    legacy_json.parent.mkdir(parents=True)
    legacy_json.write_text(
        '{"entries": [{"id": "legacy", "title": "Legacy", "summary": "Do not migrate"}]}',
        encoding="utf-8",
    )
    file_io = FakeNoKVKnowledgeFileIO({}, knowledge_root)
    agent = FakeAgent(workdir, file_io)

    result = knowledge._reconcile(agent)

    assert result["catalog_size"] == 0
    assert result["problems"] == []
    assert legacy_json.is_file()
    assert not (knowledge_root / "legacy" / "KNOWLEDGE.md").exists()
    assert not (knowledge_root / "knowledge.json.migrated").exists()
