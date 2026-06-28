"""Factories and shared route types for LingTai file I/O storage routing."""
from __future__ import annotations

from pathlib import Path

from .nokv import NoKVUnsupportedError
from .storage_config import ResolvedStorageConfig, StorageRoute


def build_routed_file_io_service(
    *,
    agent_dir: Path | str,
    local_service,
    storage: ResolvedStorageConfig,
    nokv_backend=None,
):
    """Wrap an existing local FileIO service with selected-subtree routing."""
    from .file_io import LocalFileIOBackend, LocalFileIOService, NoKVFileIOBackend, RoutedFileIOBackend

    if storage.enabled and nokv_backend is None:
        raise NoKVUnsupportedError(
            "NoKV backend/client is required before enabling selected-subtree storage"
        )
    local_backend = getattr(local_service, "_backend", None)
    if local_backend is None:
        local_backend = LocalFileIOBackend(root=agent_dir)
    backend = RoutedFileIOBackend(
        agent_dir=Path(agent_dir),
        local_backend=local_backend,
        nokv_backend=nokv_backend or NoKVFileIOBackend(),
        routes=list(storage.routes),
    )
    return LocalFileIOService(backend=backend)


__all__ = [
    "ResolvedStorageConfig",
    "StorageRoute",
    "build_routed_file_io_service",
]
