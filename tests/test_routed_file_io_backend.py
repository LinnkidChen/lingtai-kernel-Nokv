from __future__ import annotations

from pathlib import Path

from lingtai.services.file_io import (
    FileIOBackend,
    GrepMatch,
    LocalFileIOBackend,
    LocalFileIOService,
    TraversalStats,
)


class RecordingNoKVBackend(FileIOBackend):
    def __init__(self):
        self.objects: dict[str, str] = {}
        self.calls: list[tuple[str, str]] = []
        self.last_traversal = TraversalStats()

    def read(self, path: str) -> str:
        self.calls.append(("read", path))
        return self.objects[path]

    def write(self, path: str, content: str) -> None:
        self.calls.append(("write", path))
        self.objects[path] = content

    def edit(self, path: str, old_string: str, new_string: str) -> str:
        content = self.read(path)
        if content.count(old_string) != 1:
            raise ValueError("old_string must appear exactly once")
        updated = content.replace(old_string, new_string, 1)
        self.write(path, updated)
        return updated

    def glob(
        self,
        pattern: str,
        root: str | None = None,
        *,
        exclude_dirs=None,
        walltime_s=None,
        max_visited=None,
        max_results=None,
    ) -> list[str]:
        self.calls.append(("glob", root or ""))
        prefix = (root or "").rstrip("/") + "/"
        return sorted(path for path in self.objects if path.startswith(prefix))

    def grep(
        self,
        pattern: str,
        path: str | None = None,
        max_results: int = 50,
        *,
        exclude_dirs=None,
        walltime_s=None,
        max_visited=None,
        max_file_bytes=None,
    ) -> list[GrepMatch]:
        self.calls.append(("grep", path or ""))
        prefix = (path or "").rstrip("/") + "/"
        matches: list[GrepMatch] = []
        for object_path, content in sorted(self.objects.items()):
            if not object_path.startswith(prefix):
                continue
            for line_no, line in enumerate(content.splitlines(), 1):
                if pattern in line:
                    matches.append(GrepMatch(object_path, line_no, line))
        return matches[:max_results]


class LeakyNoKVBackend(RecordingNoKVBackend):
    def glob(
        self,
        pattern: str,
        root: str | None = None,
        *,
        exclude_dirs=None,
        walltime_s=None,
        max_visited=None,
        max_results=None,
    ) -> list[str]:
        self.calls.append(("glob", root or ""))
        return sorted(self.objects)

    def grep(
        self,
        pattern: str,
        path: str | None = None,
        max_results: int = 50,
        *,
        exclude_dirs=None,
        walltime_s=None,
        max_visited=None,
        max_file_bytes=None,
    ) -> list[GrepMatch]:
        self.calls.append(("grep", path or ""))
        matches: list[GrepMatch] = []
        for object_path, content in sorted(self.objects.items()):
            for line_no, line in enumerate(content.splitlines(), 1):
                if pattern in line:
                    matches.append(GrepMatch(object_path, line_no, line))
        return matches[:max_results]


class MaxVisitedNoKVBackend(RecordingNoKVBackend):
    def glob(
        self,
        pattern: str,
        root: str | None = None,
        *,
        exclude_dirs=None,
        walltime_s=None,
        max_visited=None,
        max_results=None,
    ) -> list[str]:
        self.calls.append(("glob", root or ""))
        prefix = (root or "").rstrip("/") + "/"
        results: list[str] = []
        stats = TraversalStats()
        for path in sorted(self.objects):
            if not path.startswith(prefix):
                continue
            if max_visited is not None and stats.visited >= max_visited:
                stats.truncated_reason = "visited"
                break
            stats.visited += 1
            results.append(path)
        self.last_traversal = stats
        if max_results is not None:
            return results[:max_results]
        return results

    def grep(
        self,
        pattern: str,
        path: str | None = None,
        max_results: int = 50,
        *,
        exclude_dirs=None,
        walltime_s=None,
        max_visited=None,
        max_file_bytes=None,
    ) -> list[GrepMatch]:
        self.calls.append(("grep", path or ""))
        prefix = (path or "").rstrip("/") + "/"
        matches: list[GrepMatch] = []
        stats = TraversalStats()
        for object_path, content in sorted(self.objects.items()):
            if not object_path.startswith(prefix):
                continue
            if max_visited is not None and stats.visited >= max_visited:
                stats.truncated_reason = "visited"
                break
            stats.visited += 1
            for line_no, line in enumerate(content.splitlines(), 1):
                if pattern in line:
                    matches.append(GrepMatch(object_path, line_no, line))
                    if len(matches) >= max_results:
                        stats.truncated_reason = "max_results"
                        self.last_traversal = stats
                        return matches
        self.last_traversal = stats
        return matches


class BudgetSensitiveLocalBackend(LocalFileIOBackend):
    def __init__(self, root: Path):
        super().__init__(root=root)
        self.glob_exclude_dirs = None
        self.grep_exclude_dirs = None

    def glob(
        self,
        pattern: str,
        root: str | None = None,
        *,
        exclude_dirs=None,
        walltime_s=None,
        max_visited=None,
        max_results=None,
    ) -> list[str]:
        self.glob_exclude_dirs = exclude_dirs
        if exclude_dirs is None or "knowledge" not in exclude_dirs:
            self.last_traversal = TraversalStats(visited=max_visited or 0, truncated_reason="visited")
            return []
        return super().glob(
            pattern,
            root=root,
            exclude_dirs=exclude_dirs,
            walltime_s=walltime_s,
            max_visited=max_visited,
            max_results=max_results,
        )

    def grep(
        self,
        pattern: str,
        path: str | None = None,
        max_results: int = 50,
        *,
        exclude_dirs=None,
        walltime_s=None,
        max_visited=None,
        max_file_bytes=None,
    ) -> list[GrepMatch]:
        self.grep_exclude_dirs = exclude_dirs
        if exclude_dirs is None or "knowledge" not in exclude_dirs:
            self.last_traversal = TraversalStats(visited=max_visited or 0, truncated_reason="visited")
            return []
        return super().grep(
            pattern,
            path=path,
            max_results=max_results,
            exclude_dirs=exclude_dirs,
            walltime_s=walltime_s,
            max_visited=max_visited,
            max_file_bytes=max_file_bytes,
        )


def _routed_service(tmp_path: Path):
    from lingtai.services.file_io_factory import StorageRoute
    from lingtai.services.file_io import RoutedFileIOBackend

    agent_dir = tmp_path / ".lingtai" / "alice"
    fake_nokv = RecordingNoKVBackend()
    routes = [
        StorageRoute(
            mount=mount,
            local_root=agent_dir / mount,
            remote_root=f"/lingtai/projects/abc123/agents/alice/{mount}",
            backend="nokv",
        )
        for mount in ("artifacts", "reports", "checkpoints", "knowledge")
    ]
    backend = RoutedFileIOBackend(
        agent_dir=agent_dir,
        local_backend=LocalFileIOBackend(root=agent_dir),
        nokv_backend=fake_nokv,
        routes=routes,
    )
    return LocalFileIOService(backend=backend), agent_dir, fake_nokv


def _routed_service_with_local_backend(
    tmp_path: Path,
    local_backend: FileIOBackend,
    fake_nokv: FileIOBackend,
):
    from lingtai.services.file_io_factory import StorageRoute
    from lingtai.services.file_io import RoutedFileIOBackend

    agent_dir = tmp_path / ".lingtai" / "alice"
    routes = [
        StorageRoute(
            mount=mount,
            local_root=agent_dir / mount,
            remote_root=f"/lingtai/projects/abc123/agents/alice/{mount}",
            backend="nokv",
        )
        for mount in ("artifacts", "reports", "checkpoints", "knowledge")
    ]
    backend = RoutedFileIOBackend(
        agent_dir=agent_dir,
        local_backend=local_backend,
        nokv_backend=fake_nokv,
        routes=routes,
    )
    return LocalFileIOService(backend=backend), agent_dir, fake_nokv


def _routed_service_with_backend(tmp_path: Path, fake_nokv: FileIOBackend):
    from lingtai.services.file_io_factory import StorageRoute
    from lingtai.services.file_io import RoutedFileIOBackend

    agent_dir = tmp_path / ".lingtai" / "alice"
    routes = [
        StorageRoute(
            mount=mount,
            local_root=agent_dir / mount,
            remote_root=f"/lingtai/projects/abc123/agents/alice/{mount}",
            backend="nokv",
        )
        for mount in ("artifacts", "reports", "checkpoints", "knowledge")
    ]
    backend = RoutedFileIOBackend(
        agent_dir=agent_dir,
        local_backend=LocalFileIOBackend(root=agent_dir),
        nokv_backend=fake_nokv,
        routes=routes,
    )
    return LocalFileIOService(backend=backend), agent_dir, fake_nokv


def test_selected_agent_facing_subtrees_route_to_nokv(tmp_path):
    service, agent_dir, fake_nokv = _routed_service(tmp_path)

    service.write("artifacts/brief.md", "artifact")
    service.write("reports/root-cause.md", "report")
    service.write("checkpoints/run-001/meta.json", "{}")
    service.write("knowledge/retry/KNOWLEDGE.md", "---\nname: retry\n---\n")

    assert sorted(fake_nokv.objects) == [
        "/lingtai/projects/abc123/agents/alice/artifacts/brief.md",
        "/lingtai/projects/abc123/agents/alice/checkpoints/run-001/meta.json",
        "/lingtai/projects/abc123/agents/alice/knowledge/retry/KNOWLEDGE.md",
        "/lingtai/projects/abc123/agents/alice/reports/root-cause.md",
    ]
    assert not (agent_dir / "artifacts" / "brief.md").exists()


def test_runtime_control_paths_stay_local(tmp_path):
    service, agent_dir, fake_nokv = _routed_service(tmp_path)

    local_paths = [
        "mailbox/inbox/msg/message.json",
        "logs/events.jsonl",
        ".agent.heartbeat",
        ".status.json",
        ".notification/event.json",
        ".interrupt",
        ".refresh",
    ]
    for rel in local_paths:
        service.write(rel, f"local:{rel}")

    assert fake_nokv.objects == {}
    for rel in local_paths:
        assert (agent_dir / rel).read_text(encoding="utf-8") == f"local:{rel}"


def test_relative_and_absolute_selected_paths_resolve_to_same_remote_object(tmp_path):
    service, agent_dir, fake_nokv = _routed_service(tmp_path)

    service.write("artifacts/relative.md", "relative")
    service.write(str(agent_dir / "artifacts" / "absolute.md"), "absolute")

    assert service.read("artifacts/relative.md") == "relative"
    assert service.read(str(agent_dir / "artifacts" / "absolute.md")) == "absolute"
    assert sorted(fake_nokv.objects) == [
        "/lingtai/projects/abc123/agents/alice/artifacts/absolute.md",
        "/lingtai/projects/abc123/agents/alice/artifacts/relative.md",
    ]


def test_glob_and_grep_return_virtual_local_paths(tmp_path):
    service, agent_dir, fake_nokv = _routed_service(tmp_path)
    fake_nokv.objects[
        "/lingtai/projects/abc123/agents/alice/knowledge/retry/KNOWLEDGE.md"
    ] = "needle\n"
    fake_nokv.objects[
        "/lingtai/projects/abc123/agents/alice/knowledge/other/KNOWLEDGE.md"
    ] = "other\n"

    matches = service.glob("**/KNOWLEDGE.md", root="knowledge")
    grep_matches = service.grep("needle", path=str(agent_dir / "knowledge"))

    assert matches == [
        str(agent_dir / "knowledge" / "other" / "KNOWLEDGE.md"),
        str(agent_dir / "knowledge" / "retry" / "KNOWLEDGE.md"),
    ]
    assert grep_matches == [
        GrepMatch(str(agent_dir / "knowledge" / "retry" / "KNOWLEDGE.md"), 1, "needle")
    ]


def test_broad_glob_from_agent_root_includes_selected_nokv_mount_and_local_results(tmp_path):
    service, agent_dir, fake_nokv = _routed_service(tmp_path)
    fake_nokv.objects[
        "/lingtai/projects/abc123/agents/alice/knowledge/remote/KNOWLEDGE.md"
    ] = "remote"
    service.write("notes/KNOWLEDGE.md", "local")
    (agent_dir / "knowledge" / "stale" / "KNOWLEDGE.md").parent.mkdir(parents=True)
    (agent_dir / "knowledge" / "stale" / "KNOWLEDGE.md").write_text("stale", encoding="utf-8")

    matches = service.glob("**/KNOWLEDGE.md", root=str(agent_dir))

    assert matches == [
        str(agent_dir / "knowledge" / "remote" / "KNOWLEDGE.md"),
        str(agent_dir / "notes" / "KNOWLEDGE.md"),
    ]


def test_mount_qualified_broad_glob_skips_unrelated_selected_routes(tmp_path):
    service, agent_dir, fake_nokv = _routed_service(tmp_path)
    fake_nokv.objects[
        "/lingtai/projects/abc123/agents/alice/knowledge/remote/guide.md"
    ] = "remote"
    fake_nokv.objects[
        "/lingtai/projects/abc123/agents/alice/reports/knowledge/false-positive.md"
    ] = "false positive"

    matches = service.glob("knowledge/**/*.md", root=str(agent_dir))

    assert matches == [
        str(agent_dir / "knowledge" / "remote" / "guide.md"),
    ]


def test_broad_grep_from_default_root_includes_selected_nokv_mount_and_local_results(tmp_path):
    service, agent_dir, fake_nokv = _routed_service(tmp_path)
    fake_nokv.objects[
        "/lingtai/projects/abc123/agents/alice/knowledge/remote/KNOWLEDGE.md"
    ] = "needle remote"
    service.write("notes/local.md", "needle local")
    (agent_dir / "knowledge" / "stale.md").parent.mkdir(parents=True)
    (agent_dir / "knowledge" / "stale.md").write_text("needle stale", encoding="utf-8")

    matches = service.grep("needle")

    assert matches == [
        GrepMatch(str(agent_dir / "knowledge" / "remote" / "KNOWLEDGE.md"), 1, "needle remote"),
        GrepMatch(str(agent_dir / "notes" / "local.md"), 1, "needle local"),
    ]


def test_broad_glob_excludes_routed_mounts_before_local_budgeting(tmp_path):
    agent_dir = tmp_path / ".lingtai" / "alice"
    local_backend = BudgetSensitiveLocalBackend(root=agent_dir)
    fake_nokv = RecordingNoKVBackend()
    service, agent_dir, fake_nokv = _routed_service_with_local_backend(
        tmp_path,
        local_backend,
        fake_nokv,
    )
    fake_nokv.objects[
        "/lingtai/projects/abc123/agents/alice/knowledge/remote/KNOWLEDGE.md"
    ] = "remote"
    (agent_dir / "notes").mkdir(parents=True)
    (agent_dir / "notes" / "KNOWLEDGE.md").write_text("local", encoding="utf-8")
    (agent_dir / "knowledge" / "stale").mkdir(parents=True)
    (agent_dir / "knowledge" / "stale" / "KNOWLEDGE.md").write_text("stale", encoding="utf-8")

    matches = service.glob(
        "**/KNOWLEDGE.md",
        root=str(agent_dir),
        exclude_dirs={"custom-cache"},
        max_visited=4,
    )

    assert local_backend.glob_exclude_dirs >= {
        "custom-cache",
        "artifacts",
        "reports",
        "checkpoints",
        "knowledge",
    }
    assert matches == [
        str(agent_dir / "knowledge" / "remote" / "KNOWLEDGE.md"),
        str(agent_dir / "notes" / "KNOWLEDGE.md"),
    ]


def test_broad_grep_excludes_routed_mounts_before_local_budgeting(tmp_path):
    agent_dir = tmp_path / ".lingtai" / "alice"
    local_backend = BudgetSensitiveLocalBackend(root=agent_dir)
    fake_nokv = RecordingNoKVBackend()
    service, agent_dir, fake_nokv = _routed_service_with_local_backend(
        tmp_path,
        local_backend,
        fake_nokv,
    )
    fake_nokv.objects[
        "/lingtai/projects/abc123/agents/alice/knowledge/remote.md"
    ] = "needle remote"
    (agent_dir / "notes").mkdir(parents=True)
    (agent_dir / "notes" / "local.md").write_text("needle local", encoding="utf-8")
    (agent_dir / "knowledge").mkdir(parents=True)
    (agent_dir / "knowledge" / "stale.md").write_text("needle stale", encoding="utf-8")

    matches = service.grep(
        "needle",
        path=str(agent_dir),
        max_results=5,
        exclude_dirs={"custom-cache"},
        max_visited=4,
    )

    assert local_backend.grep_exclude_dirs >= {
        "custom-cache",
        "artifacts",
        "reports",
        "checkpoints",
        "knowledge",
    }
    assert matches == [
        GrepMatch(str(agent_dir / "knowledge" / "remote.md"), 1, "needle remote"),
        GrepMatch(str(agent_dir / "notes" / "local.md"), 1, "needle local"),
    ]


def test_broad_glob_reports_remote_route_max_visited_truncation(tmp_path):
    service, agent_dir, fake_nokv = _routed_service_with_backend(
        tmp_path,
        MaxVisitedNoKVBackend(),
    )
    fake_nokv.objects[
        "/lingtai/projects/abc123/agents/alice/knowledge/a/KNOWLEDGE.md"
    ] = "remote a"
    fake_nokv.objects[
        "/lingtai/projects/abc123/agents/alice/knowledge/b/KNOWLEDGE.md"
    ] = "remote b"

    matches = service.glob("**/KNOWLEDGE.md", root=str(agent_dir), max_visited=1)

    assert matches == [str(agent_dir / "knowledge" / "a" / "KNOWLEDGE.md")]
    assert service.last_traversal.truncated_reason == "visited"
    assert service.last_traversal.visited >= 1


def test_broad_grep_reports_remote_route_max_visited_truncation(tmp_path):
    service, agent_dir, fake_nokv = _routed_service_with_backend(
        tmp_path,
        MaxVisitedNoKVBackend(),
    )
    fake_nokv.objects[
        "/lingtai/projects/abc123/agents/alice/knowledge/a.md"
    ] = "needle remote a"
    fake_nokv.objects[
        "/lingtai/projects/abc123/agents/alice/knowledge/b.md"
    ] = "needle remote b"

    matches = service.grep("needle", path=str(agent_dir), max_results=5, max_visited=1)

    assert matches == [
        GrepMatch(str(agent_dir / "knowledge" / "a.md"), 1, "needle remote a")
    ]
    assert service.last_traversal.truncated_reason == "visited"
    assert service.last_traversal.visited >= 1


def test_broad_glob_uses_one_max_visited_budget_across_local_and_nokv(tmp_path):
    service, agent_dir, fake_nokv = _routed_service_with_backend(
        tmp_path,
        MaxVisitedNoKVBackend(),
    )
    service.write("notes/KNOWLEDGE.md", "local")
    fake_nokv.objects[
        "/lingtai/projects/abc123/agents/alice/knowledge/a/KNOWLEDGE.md"
    ] = "remote a"
    fake_nokv.objects[
        "/lingtai/projects/abc123/agents/alice/knowledge/b/KNOWLEDGE.md"
    ] = "remote b"

    matches = service.glob("**/KNOWLEDGE.md", root=str(agent_dir), max_visited=4)

    assert matches == [
        str(agent_dir / "knowledge" / "a" / "KNOWLEDGE.md"),
        str(agent_dir / "notes" / "KNOWLEDGE.md"),
    ]
    assert service.last_traversal.truncated_reason == "visited"
    assert service.last_traversal.visited == 4


def test_broad_grep_uses_one_max_visited_budget_across_local_and_nokv(tmp_path):
    service, agent_dir, fake_nokv = _routed_service_with_backend(
        tmp_path,
        MaxVisitedNoKVBackend(),
    )
    service.write("notes/local.md", "needle local")
    fake_nokv.objects[
        "/lingtai/projects/abc123/agents/alice/knowledge/a.md"
    ] = "needle remote a"
    fake_nokv.objects[
        "/lingtai/projects/abc123/agents/alice/knowledge/b.md"
    ] = "needle remote b"

    matches = service.grep("needle", path=str(agent_dir), max_results=10, max_visited=4)

    assert matches == [
        GrepMatch(str(agent_dir / "knowledge" / "a.md"), 1, "needle remote a"),
        GrepMatch(str(agent_dir / "notes" / "local.md"), 1, "needle local"),
    ]
    assert service.last_traversal.truncated_reason == "visited"
    assert service.last_traversal.visited == 4


def test_routed_glob_filters_nokv_results_outside_active_route(tmp_path):
    service, agent_dir, fake_nokv = _routed_service_with_backend(tmp_path, LeakyNoKVBackend())
    fake_nokv.objects[
        "/lingtai/projects/abc123/agents/alice/knowledge/good/KNOWLEDGE.md"
    ] = "good"
    fake_nokv.objects["/lingtai/projects/abc123/agents/alice/reports/leak.md"] = "leak"
    fake_nokv.objects["/other/raw/namespace/KNOWLEDGE.md"] = "leak"

    matches = service.glob("**/*.md", root="knowledge")

    assert matches == [str(agent_dir / "knowledge" / "good" / "KNOWLEDGE.md")]


def test_routed_glob_filters_traversal_segments_under_active_route(tmp_path):
    service, agent_dir, fake_nokv = _routed_service_with_backend(tmp_path, LeakyNoKVBackend())
    fake_nokv.objects[
        "/lingtai/projects/abc123/agents/alice/knowledge/good/KNOWLEDGE.md"
    ] = "good"
    fake_nokv.objects[
        "/lingtai/projects/abc123/agents/alice/knowledge/../mailbox/msg.json"
    ] = "leak"
    fake_nokv.objects[
        "/lingtai/projects/abc123/agents/alice/knowledge/../../outside/secret.md"
    ] = "leak"

    matches = service.glob("**/*.md", root="knowledge")

    assert matches == [str(agent_dir / "knowledge" / "good" / "KNOWLEDGE.md")]
    assert all(".." not in Path(match).parts for match in matches)


def test_routed_grep_filters_nokv_results_outside_active_route(tmp_path):
    service, agent_dir, fake_nokv = _routed_service_with_backend(tmp_path, LeakyNoKVBackend())
    fake_nokv.objects[
        "/lingtai/projects/abc123/agents/alice/knowledge/good/KNOWLEDGE.md"
    ] = "needle good"
    fake_nokv.objects["/lingtai/projects/abc123/agents/alice/reports/leak.md"] = "needle leak"
    fake_nokv.objects["/other/raw/namespace/KNOWLEDGE.md"] = "needle leak"

    matches = service.grep("needle", path="knowledge")

    assert matches == [
        GrepMatch(str(agent_dir / "knowledge" / "good" / "KNOWLEDGE.md"), 1, "needle good")
    ]


def test_routed_grep_filters_traversal_segments_under_active_route(tmp_path):
    service, agent_dir, fake_nokv = _routed_service_with_backend(tmp_path, LeakyNoKVBackend())
    fake_nokv.objects[
        "/lingtai/projects/abc123/agents/alice/knowledge/good/KNOWLEDGE.md"
    ] = "needle good"
    fake_nokv.objects[
        "/lingtai/projects/abc123/agents/alice/knowledge/../mailbox/msg.json"
    ] = "needle leak"
    fake_nokv.objects[
        "/lingtai/projects/abc123/agents/alice/knowledge/../../outside/secret.md"
    ] = "needle leak"

    matches = service.grep("needle", path="knowledge")

    assert matches == [
        GrepMatch(str(agent_dir / "knowledge" / "good" / "KNOWLEDGE.md"), 1, "needle good")
    ]
    assert all(".." not in Path(match.path).parts for match in matches)
