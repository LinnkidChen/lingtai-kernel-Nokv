"""Factory and selected-subtree router for LingTai FileIO."""
from __future__ import annotations

import fnmatch
import os
import re
import time
from pathlib import Path
from typing import Iterable

from .file_io import (
    DEFAULT_EXCLUDED_DIRS,
    DEFAULT_MAX_FILE_BYTES,
    DEFAULT_MAX_VISITED,
    DEFAULT_WALLTIME_S,
    FileIOBackend,
    GrepMatch,
    LocalFileIOService,
    NoKVFileIOBackend,
    TraversalStats,
)
from .storage_config import StorageRoute


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


class RoutedFileIOBackend(FileIOBackend):
    """Route selected local agent subtrees to NoKV while keeping all else local."""

    def __init__(
        self,
        *,
        local_backend: FileIOBackend,
        nokv_backend: NoKVFileIOBackend,
        routes: Iterable[StorageRoute],
        root: str | Path | None = None,
    ) -> None:
        self._local_backend = local_backend
        self._nokv_backend = nokv_backend
        self._routes = sorted(list(routes), key=lambda r: len(str(r.local_root)), reverse=True)
        self._root = Path(root).resolve() if root is not None else None
        self._last_backend: FileIOBackend = local_backend
        self._last_traversal = TraversalStats()

    @property
    def local_backend(self) -> FileIOBackend:
        return self._local_backend

    @property
    def last_traversal(self) -> TraversalStats:
        return self._last_traversal

    @last_traversal.setter
    def last_traversal(self, value: TraversalStats) -> None:
        self._last_traversal = value

    def is_routed_path(self, path: str | Path) -> bool:
        return self._route_for_local_path(path) is not None

    def _resolve_local(self, path: str | Path | None) -> Path:
        if path is None:
            return self._root or Path(".").resolve()
        p = Path(path)
        if not p.is_absolute() and self._root is not None:
            p = self._root / p
        return p.resolve()

    def _route_for_local_path(self, path: str | Path | None) -> StorageRoute | None:
        local = self._resolve_local(path)
        for route in self._routes:
            if local == route.local_root or _is_relative_to(local, route.local_root):
                return route
        return None

    def _remote_for_local(self, path: str | Path) -> tuple[StorageRoute, str]:
        local = self._resolve_local(path)
        route = self._route_for_local_path(local)
        if route is None:
            raise ValueError(f"path is not routed to NoKV: {path}")
        rel = local.relative_to(route.local_root).as_posix()
        if rel == ".":
            return route, route.remote_root
        return route, f"{route.remote_root.rstrip('/')}/{rel}"

    def _local_for_remote(self, route: StorageRoute, remote_path: str) -> str | None:
        normalized_remote = "/" + "/".join(part for part in remote_path.split("/") if part)
        root = route.remote_root.rstrip("/")
        if normalized_remote == root:
            return str(route.local_root)
        prefix = root + "/"
        if not normalized_remote.startswith(prefix):
            return None
        rel = normalized_remote[len(prefix):]
        return str(route.local_root / Path(*rel.split("/")))

    def _routes_under(self, root: Path) -> list[StorageRoute]:
        out: list[StorageRoute] = []
        for route in self._routes:
            if route.local_root == root or _is_relative_to(route.local_root, root):
                out.append(route)
        return out

    def _filter_local_results(self, paths: list[str]) -> list[str]:
        filtered: list[str] = []
        for path in paths:
            if self._route_for_local_path(path) is None:
                filtered.append(path)
        return filtered

    def read(self, path: str) -> str:
        route = self._route_for_local_path(path)
        if route is None:
            self._last_backend = self._local_backend
            result = self._local_backend.read(path)
            self._last_traversal = self._local_backend.last_traversal
            return result
        self._last_backend = self._nokv_backend
        _, remote = self._remote_for_local(path)
        result = self._nokv_backend.read(remote)
        self._last_traversal = self._nokv_backend.last_traversal
        return result

    def write(self, path: str, content: str) -> None:
        route = self._route_for_local_path(path)
        if route is None:
            self._last_backend = self._local_backend
            self._local_backend.write(path, content)
            self._last_traversal = self._local_backend.last_traversal
            return
        self._last_backend = self._nokv_backend
        _, remote = self._remote_for_local(path)
        self._nokv_backend.write(remote, content)
        self._last_traversal = self._nokv_backend.last_traversal

    def edit(self, path: str, old_string: str, new_string: str) -> str:
        route = self._route_for_local_path(path)
        if route is None:
            self._last_backend = self._local_backend
            result = self._local_backend.edit(path, old_string, new_string)
            self._last_traversal = self._local_backend.last_traversal
            return result
        self._last_backend = self._nokv_backend
        _, remote = self._remote_for_local(path)
        result = self._nokv_backend.edit(remote, old_string, new_string)
        self._last_traversal = self._nokv_backend.last_traversal
        return result

    def _remote_glob(
        self,
        route: StorageRoute,
        pattern: str,
        *,
        search_root: Path,
        max_results: int | None,
    ) -> list[str]:
        results: list[str] = []
        search_remote_root = self._remote_root_for_search(route, search_root)
        for entry in self._nokv_backend.list(search_remote_root):
            self._last_traversal.visited += 1
            local = self._local_for_remote(route, entry["path"])
            if local is None:
                continue
            try:
                rel = os.path.relpath(local, search_root)
            except ValueError:
                rel = local
            if fnmatch.fnmatch(rel, pattern):
                results.append(local)
                if max_results is not None and len(results) >= max_results:
                    self._last_traversal.truncated_reason = "max_results"
                    break
        return results

    def _remote_root_for_search(self, route: StorageRoute, search_root: Path) -> str:
        if search_root == route.local_root:
            return route.remote_root
        if _is_relative_to(search_root, route.local_root):
            rel = search_root.relative_to(route.local_root).as_posix()
            return f"{route.remote_root.rstrip('/')}/{rel}"
        return route.remote_root

    def glob(
        self,
        pattern: str,
        root: str | None = None,
        *,
        exclude_dirs: frozenset[str] | set[str] | None = None,
        walltime_s: float | None = DEFAULT_WALLTIME_S,
        max_visited: int | None = DEFAULT_MAX_VISITED,
        max_results: int | None = 2000,
    ) -> list[str]:
        search_root = self._resolve_local(root)
        self._last_traversal = TraversalStats()
        route = self._route_for_local_path(search_root)
        route_roots = [route] if route is not None else self._routes_under(search_root)

        local_results = []
        if route is None:
            local_results = self._filter_local_results(
                self._local_backend.glob(
                    pattern,
                    root=str(search_root),
                    exclude_dirs=exclude_dirs,
                    walltime_s=walltime_s,
                    max_visited=max_visited,
                    max_results=max_results,
                )
            )
            self._last_traversal.visited += self._local_backend.last_traversal.visited
            self._last_traversal.files_skipped_binary += self._local_backend.last_traversal.files_skipped_binary
            self._last_traversal.files_skipped_size += self._local_backend.last_traversal.files_skipped_size
            self._last_traversal.dirs_pruned += self._local_backend.last_traversal.dirs_pruned
            self._last_traversal.truncated_reason = self._local_backend.last_traversal.truncated_reason

        remote_results: list[str] = []
        for remote_route in route_roots:
            remaining = None if max_results is None else max_results - len(local_results) - len(remote_results)
            if remaining is not None and remaining <= 0:
                self._last_traversal.truncated_reason = self._last_traversal.truncated_reason or "max_results"
                break
            remote_results.extend(
                self._remote_glob(
                    remote_route,
                    pattern,
                    search_root=search_root,
                    max_results=remaining,
                )
            )
        results = sorted(local_results + remote_results)
        if max_results is not None and len(results) > max_results:
            self._last_traversal.truncated_reason = self._last_traversal.truncated_reason or "max_results"
            results = results[:max_results]
        return results

    def grep(
        self,
        pattern: str,
        path: str | None = None,
        max_results: int = 50,
        *,
        exclude_dirs: frozenset[str] | set[str] | None = None,
        walltime_s: float | None = DEFAULT_WALLTIME_S,
        max_visited: int | None = DEFAULT_MAX_VISITED,
        max_file_bytes: int | None = DEFAULT_MAX_FILE_BYTES,
    ) -> list[GrepMatch]:
        regex = re.compile(pattern)
        search_root = self._resolve_local(path)
        self._last_traversal = TraversalStats()
        route = self._route_for_local_path(search_root)
        route_roots = [route] if route is not None else self._routes_under(search_root)
        start = time.monotonic()

        local_results: list[GrepMatch] = []
        if route is None:
            local_results = [
                match for match in self._local_backend.grep(
                    pattern,
                    path=str(search_root),
                    max_results=max_results,
                    exclude_dirs=exclude_dirs,
                    walltime_s=walltime_s,
                    max_visited=max_visited,
                    max_file_bytes=max_file_bytes,
                )
                if self._route_for_local_path(match.path) is None
            ]
            self._last_traversal.visited += self._local_backend.last_traversal.visited
            self._last_traversal.files_skipped_binary += self._local_backend.last_traversal.files_skipped_binary
            self._last_traversal.files_skipped_size += self._local_backend.last_traversal.files_skipped_size
            self._last_traversal.dirs_pruned += self._local_backend.last_traversal.dirs_pruned
            self._last_traversal.truncated_reason = self._local_backend.last_traversal.truncated_reason

        results = list(local_results)
        for remote_route in route_roots:
            remote_search_root = self._remote_root_for_search(remote_route, search_root)
            for entry in self._nokv_backend.list(remote_search_root):
                if len(results) >= max_results:
                    self._last_traversal.truncated_reason = self._last_traversal.truncated_reason or "max_results"
                    return sorted(results, key=lambda m: (m.path, m.line_number))
                if walltime_s is not None and (time.monotonic() - start) > walltime_s:
                    self._last_traversal.truncated_reason = self._last_traversal.truncated_reason or "walltime"
                    return sorted(results, key=lambda m: (m.path, m.line_number))
                self._last_traversal.visited += 1
                if max_visited is not None and self._last_traversal.visited > max_visited:
                    self._last_traversal.truncated_reason = self._last_traversal.truncated_reason or "visited"
                    return sorted(results, key=lambda m: (m.path, m.line_number))
                local = self._local_for_remote(remote_route, entry["path"])
                if local is None:
                    continue
                try:
                    text = self._nokv_backend.read(entry["path"])
                except Exception:
                    self._last_traversal.files_skipped_binary += 1
                    continue
                if max_file_bytes is not None and len(text.encode("utf-8")) > max_file_bytes:
                    self._last_traversal.files_skipped_size += 1
                    continue
                for i, line in enumerate(text.splitlines(), 1):
                    if regex.search(line):
                        results.append(GrepMatch(local, i, line))
                        if len(results) >= max_results:
                            self._last_traversal.truncated_reason = self._last_traversal.truncated_reason or "max_results"
                            return sorted(results, key=lambda m: (m.path, m.line_number))
        return sorted(results, key=lambda m: (m.path, m.line_number))


def build_file_io_service(
    *,
    root: str | Path,
    routes: list[StorageRoute] | None = None,
    local_backend: FileIOBackend | None = None,
    nokv_backend: NoKVFileIOBackend | None = None,
) -> LocalFileIOService:
    if not routes:
        if local_backend is not None:
            return LocalFileIOService(backend=local_backend)
        from .file_io_sidecar import default_file_io_service
        return default_file_io_service(root=root)
    if nokv_backend is None:
        raise ValueError("enabled NoKV storage requires an injected NoKV backend")
    if local_backend is None:
        from .file_io_sidecar import default_file_io_service
        local_service = default_file_io_service(root=root)
        local_backend = getattr(local_service, "_backend")
    backend = RoutedFileIOBackend(
        root=root,
        local_backend=local_backend,
        nokv_backend=nokv_backend,
        routes=routes,
    )
    return LocalFileIOService(backend=backend)
