"""Explicit selected-subtree storage configuration.

The first NoKV integration surface is deliberately narrow: only configured
agent-local ``artifacts/``, ``reports/``, ``checkpoints/``, and ``knowledge/``
mounts may route to NoKV. Runtime control paths stay local.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Any

from .nokv import DEFAULT_NOKV_SELECTED_SUBTREES, normalize_nokv_path

_ALLOWED_MOUNTS = frozenset(DEFAULT_NOKV_SELECTED_SUBTREES)
_ALLOWED_STREAMS: dict[str, str] = {
    "logs/events": "logs/events",
    "history/chat_history": "history/chat_history",
    "logs/token_ledger": "logs/token_ledger",
}
_NOKV_PUBLIC_STATUS_KEYS = {
    "metadata_addr_env": "metadata_addr",
    "bucket_env": "bucket",
    "endpoint_env": "endpoint",
}
_REQUIRED_NOKV_ENV_KEYS = frozenset(_NOKV_PUBLIC_STATUS_KEYS)
_ALLOWED_NOKV_KEYS = {
    "namespace_root",
    "metadata_addr_env",
    "bucket_env",
    "endpoint_env",
    "access_key_id_env",
    "secret_access_key_env",
    "region_env",
}
_LITERAL_SECRET_KEYS = {
    "metadata_addr",
    "bucket",
    "endpoint",
    "access_key_id",
    "secret_access_key",
    "region",
}


@dataclass(frozen=True)
class StorageRoute:
    mount: str
    local_root: Path
    remote_root: str
    backend: str = "nokv"


@dataclass(frozen=True)
class StorageStreamRoute:
    stream: str
    local_path: Path
    remote_root: str
    backend: str = "nokv"
    mode: str = "mirror"


@dataclass(frozen=True)
class ResolvedStorageConfig:
    enabled: bool
    backend: str
    routes: list[StorageRoute]
    streams: list[StorageStreamRoute]
    nokv_status: dict[str, str]
    nokv_backend: Any | None = None

    def status_document(self, *, health: Mapping[str, Any] | None = None) -> dict[str, Any]:
        default_health: dict[str, Any] = {
            "status": "ok",
            "backend": "mirror" if self.streams else self.backend,
            "streams": [stream.stream for stream in self.streams],
        }
        return {
            "schema": "lingtai.storage.resolved/v1",
            "schema_version": 1,
            "source": "kernel",
            "enabled": self.enabled,
            "backend": self.backend,
            "routes": [
                {
                    "mount": route.mount,
                    "local_root": str(route.local_root),
                    "backend": route.backend,
                    "remote_root": route.remote_root,
                }
                for route in self.routes
            ],
            "streams": [
                {
                    "stream": stream.stream,
                    "local_path": str(stream.local_path),
                    "backend": stream.backend,
                    "remote_root": stream.remote_root,
                    "mode": stream.mode,
                }
                for stream in self.streams
            ],
            "health": dict(health or default_health),
            "nokv": dict(self.nokv_status),
        }


def _project_root_for_agent(agent_dir: Path) -> Path:
    parent = agent_dir.parent
    if parent.name == ".lingtai":
        return parent.parent
    return parent


def _project_hash(project_root: Path) -> str:
    return hashlib.sha256(str(project_root).encode("utf-8")).hexdigest()[:12]


def _agent_name(agent_dir: Path) -> str:
    return agent_dir.name


def _expand_namespace_root(template: str, *, agent_dir: Path) -> str:
    project_root = _project_root_for_agent(agent_dir)
    expanded = (
        template
        .replace("${project_hash}", _project_hash(project_root))
        .replace("${agent_name}", _agent_name(agent_dir))
    )
    if not expanded.startswith("/"):
        raise ValueError("storage.nokv.namespace_root must be an absolute NoKV namespace path")
    return normalize_nokv_path(expanded)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _mounts(raw_mounts: Any) -> list[str]:
    if raw_mounts is None:
        raw_mounts = list(DEFAULT_NOKV_SELECTED_SUBTREES)
    if not isinstance(raw_mounts, list) or not all(isinstance(m, str) and m for m in raw_mounts):
        raise ValueError("storage.mounts must be a list of non-empty strings")
    out: list[str] = []
    seen: set[str] = set()
    for mount in raw_mounts:
        if mount not in _ALLOWED_MOUNTS:
            raise ValueError(
                f"unsupported storage mount {mount!r}; allowed mounts: {sorted(_ALLOWED_MOUNTS)}"
            )
        if mount not in seen:
            out.append(mount)
            seen.add(mount)
    return out


def _streams(raw_streams: Any) -> list[str]:
    if raw_streams is None:
        return []
    if not isinstance(raw_streams, list) or not all(
        isinstance(stream, str) and stream for stream in raw_streams
    ):
        raise ValueError("storage.streams must be a list of non-empty strings")
    out: list[str] = []
    seen: set[str] = set()
    for stream in raw_streams:
        normalized = stream.strip().strip("/")
        if normalized not in _ALLOWED_STREAMS:
            raise ValueError(
                f"unsupported storage stream {stream!r}; allowed streams: {sorted(_ALLOWED_STREAMS)}"
            )
        if normalized not in seen:
            out.append(normalized)
            seen.add(normalized)
    return out


def _resolve_env_requirements(
    nokv: Mapping[str, Any],
    env: Mapping[str, str],
) -> dict[str, str]:
    unknown = sorted(set(nokv) - _ALLOWED_NOKV_KEYS - _LITERAL_SECRET_KEYS)
    if unknown:
        raise ValueError(f"unknown storage.nokv field: {unknown[0]}")
    literal = sorted(set(nokv) & _LITERAL_SECRET_KEYS)
    if literal:
        raise ValueError("storage.nokv stores env var names, not literal values")
    missing = sorted(_REQUIRED_NOKV_ENV_KEYS - set(nokv))
    if missing:
        raise ValueError(f"storage.nokv.{missing[0]} is required")

    resolved_public: dict[str, str] = {}
    for key, value in nokv.items():
        if key == "namespace_root":
            continue
        if not key.endswith("_env"):
            continue
        if not isinstance(value, str) or not value:
            raise ValueError(f"storage.nokv.{key} must be a non-empty env var name")
        env_value = env.get(value)
        if not env_value:
            raise ValueError(f"storage.nokv.{key} references missing env var {value}")
        status_key = _NOKV_PUBLIC_STATUS_KEYS.get(key)
        if status_key:
            resolved_public[status_key] = env_value
    return resolved_public


def resolve_storage_config(
    raw: Mapping[str, Any] | None,
    *,
    agent_dir: str | Path,
    nokv_backend: Any | None = None,
    env: Mapping[str, str] | None = None,
) -> ResolvedStorageConfig:
    agent_path = Path(agent_dir).expanduser().resolve()
    if not raw or raw.get("enabled") is False:
        return ResolvedStorageConfig(
            enabled=False,
            backend="local",
            routes=[],
            streams=[],
            nokv_status={},
            nokv_backend=None,
        )
    if not isinstance(raw, Mapping):
        raise ValueError("storage must be an object")
    if raw.get("enabled") is not True:
        return ResolvedStorageConfig(
            enabled=False,
            backend="local",
            routes=[],
            streams=[],
            nokv_status={},
            nokv_backend=None,
        )
    backend = raw.get("backend", "nokv")
    if backend != "nokv":
        raise ValueError("storage.backend must be 'nokv' when storage.enabled is true")
    if nokv_backend is None:
        raise ValueError("enabled NoKV storage requires an injected NoKV backend or client")

    nokv = raw.get("nokv")
    if not isinstance(nokv, Mapping):
        raise ValueError("storage.nokv must be an object when storage.enabled is true")
    namespace_template = nokv.get("namespace_root")
    if not isinstance(namespace_template, str) or not namespace_template:
        raise ValueError("storage.nokv.namespace_root is required")

    env_map = os.environ if env is None else env
    nokv_status = _resolve_env_requirements(nokv, env_map)
    namespace_root = _expand_namespace_root(namespace_template, agent_dir=agent_path)

    routes: list[StorageRoute] = []
    for mount in _mounts(raw.get("mounts")):
        local_root = (agent_path / mount).resolve()
        if not _is_relative_to(local_root, agent_path):
            raise ValueError(f"storage route {mount!r} must stay under the agent working directory")
        routes.append(
            StorageRoute(
                mount=mount,
                local_root=local_root,
                remote_root=normalize_nokv_path(f"{namespace_root}/{mount}"),
            )
        )

    streams: list[StorageStreamRoute] = []
    stream_paths = {
        "logs/events": agent_path / "logs" / "events.jsonl",
        "history/chat_history": agent_path / "history" / "chat_history.jsonl",
        "logs/token_ledger": agent_path / "logs" / "token_ledger.jsonl",
    }
    for stream in _streams(raw.get("streams")):
        local_path = stream_paths[stream].resolve()
        if not _is_relative_to(local_path, agent_path):
            raise ValueError(f"storage stream {stream!r} must stay under the agent working directory")
        streams.append(
            StorageStreamRoute(
                stream=stream,
                local_path=local_path,
                remote_root=normalize_nokv_path(f"{namespace_root}/{_ALLOWED_STREAMS[stream]}"),
            )
        )

    return ResolvedStorageConfig(
        enabled=True,
        backend="routed",
        routes=routes,
        streams=streams,
        nokv_status=nokv_status,
        nokv_backend=nokv_backend,
    )
