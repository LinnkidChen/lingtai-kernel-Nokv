"""Storage config parsing for selected NoKV-backed agent subtrees."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any, Mapping
from urllib.parse import urlsplit

from .nokv import DEFAULT_NOKV_SELECTED_SUBTREES, normalize_nokv_path


class StorageConfigError(ValueError):
    """Raised when an enabled storage config is invalid."""


_ALLOWED_MOUNTS = frozenset(DEFAULT_NOKV_SELECTED_SUBTREES)
_SECRET_ENV_KEYS = frozenset({"secret_access_key_env"})
_PUBLIC_NOKV_ENV_KEYS = frozenset({
    "metadata_addr_env",
    "bucket_env",
    "endpoint_env",
    "region_env",
})
_REQUIRED_NOKV_ENV_KEYS = _PUBLIC_NOKV_ENV_KEYS | frozenset({
    "access_key_id_env",
    "secret_access_key_env",
})
_SENSITIVE_ENV_NAME_PARTS = frozenset({
    "SECRET",
    "TOKEN",
    "PASSWORD",
    "PASSWD",
    "PWD",
    "CREDENTIAL",
    "CREDENTIALS",
    "KEY",
    "AUTH",
    "BEARER",
    "SESSION",
    "COOKIE",
})
_SENSITIVE_ENV_NAME_PHRASES = (
    "ACCESS_KEY",
    "API_KEY",
    "PRIVATE_KEY",
    "SECRET_KEY",
)
_MISSING = object()


@dataclass(frozen=True)
class StorageRoute:
    """Resolved route from an agent-local mount to a NoKV namespace root."""

    mount: str
    local_root: Path
    remote_root: str
    backend: str = "nokv"

    def to_status(self) -> dict[str, str]:
        return {
            "mount": self.mount,
            "local_root": str(self.local_root),
            "backend": self.backend,
            "remote_root": self.remote_root,
        }


@dataclass(frozen=True)
class ResolvedStorageConfig:
    """Secret-free resolved storage configuration."""

    enabled: bool
    backend: str
    routes: list[StorageRoute]
    nokv: dict[str, str]

    def to_status(self) -> dict[str, Any]:
        if not self.enabled:
            return {
                "schema": "lingtai.storage.resolved/v1",
                "enabled": False,
                "backend": "local",
                "routes": [],
            }
        return {
            "schema": "lingtai.storage.resolved/v1",
            "enabled": True,
            "backend": "routed",
            "routes": [route.to_status() for route in self.routes],
            "nokv": dict(self.nokv),
        }


def _disabled() -> ResolvedStorageConfig:
    return ResolvedStorageConfig(
        enabled=False,
        backend="local",
        routes=[],
        nokv={},
    )


def _storage_block(data: Mapping[str, Any]) -> Mapping[str, Any]:
    raw = data.get("storage")
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise StorageConfigError("storage: expected object")
    return raw


def _require_mapping(raw: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        raise StorageConfigError(f"{label}: expected object")
    return raw


def _safe_env_error(key: str, env_name: str) -> StorageConfigError:
    if key in _SECRET_ENV_KEYS or "SECRET" in env_name.upper():
        return StorageConfigError("required secret environment variable is not set")
    return StorageConfigError(f"required environment variable is not set: {env_name}")


def _looks_secret_env_name(env_name: str) -> bool:
    normalized = "".join(
        character if character.isalnum() else "_"
        for character in env_name.upper()
    )
    parts = frozenset(part for part in normalized.split("_") if part)
    if parts & _SENSITIVE_ENV_NAME_PARTS:
        return True
    return any(phrase in normalized for phrase in _SENSITIVE_ENV_NAME_PHRASES)


def _env_value(
    nokv: Mapping[str, Any],
    key: str,
    environ: Mapping[str, str],
) -> tuple[str, str]:
    env_name = nokv.get(key)
    if not isinstance(env_name, str) or not env_name:
        if key in _SECRET_ENV_KEYS:
            raise StorageConfigError("nokv secret environment setting is required")
        raise StorageConfigError(f"nokv.{key}: expected non-empty environment variable name")
    if key in _PUBLIC_NOKV_ENV_KEYS and _looks_secret_env_name(env_name):
        raise StorageConfigError(
            "nokv public status environment settings must not reference secret variables"
        )
    value = environ.get(env_name)
    if value is None or value == "":
        raise _safe_env_error(key, env_name)
    return env_name, value


def _validate_public_env_value(key: str, value: str) -> None:
    guarded_labels = {
        "endpoint_env": "endpoint",
        "metadata_addr_env": "metadata address",
    }
    label = guarded_labels.get(key)
    if label is None:
        return
    parsed = urlsplit(value)
    authority = parsed if parsed.netloc else urlsplit(f"//{value}")
    has_userinfo = authority.username is not None or authority.password is not None
    if has_userinfo or authority.query or authority.fragment:
        raise StorageConfigError(
            f"nokv {label} environment value must not contain credentials, query, or fragment"
        )


def _normalize_mounts(raw: Any) -> list[str]:
    if raw is None:
        mounts = list(DEFAULT_NOKV_SELECTED_SUBTREES)
    elif isinstance(raw, list):
        mounts = raw
    else:
        raise StorageConfigError("storage.mounts: expected list[str]")

    resolved: list[str] = []
    for mount in mounts:
        if not isinstance(mount, str) or not mount:
            raise StorageConfigError("storage.mounts: expected list[str]")
        if mount not in _ALLOWED_MOUNTS:
            raise StorageConfigError(
                f"storage mount {mount!r} is not supported; allowed mounts: "
                + ", ".join(sorted(_ALLOWED_MOUNTS))
            )
        if mount not in resolved:
            resolved.append(mount)
    return resolved


def _assert_under_agent(local_root: Path, agent_dir: Path, mount: str) -> None:
    try:
        local_root.resolve(strict=False).relative_to(agent_dir.resolve(strict=False))
    except ValueError as exc:
        raise StorageConfigError(
            f"storage mount {mount!r} local_root must stay under agent_dir"
        ) from exc


def parse_storage_config(
    data: Mapping[str, Any],
    *,
    agent_dir: Path | str,
    project_root: Path | str,
    project_hash: str,
    agent_name: str,
    environ: Mapping[str, str],
) -> ResolvedStorageConfig:
    """Parse top-level ``storage`` from init data.

    Missing or disabled storage is intentionally local-only. Enabled NoKV must
    be explicit and complete so agents do not silently fall back to local paths.
    """
    del project_root  # reserved for future validation; routes are agent-rooted.

    storage = _storage_block(data)
    if not storage:
        return _disabled()
    enabled = storage.get("enabled", _MISSING)
    if enabled is _MISSING:
        return _disabled()
    if not isinstance(enabled, bool):
        raise StorageConfigError("storage.enabled: expected boolean")
    if not enabled:
        return _disabled()

    backend = storage.get("backend")
    if backend != "nokv":
        raise StorageConfigError("enabled storage requires backend='nokv'")

    nokv = _require_mapping(storage.get("nokv"), "storage.nokv")
    namespace_root = nokv.get("namespace_root")
    if not isinstance(namespace_root, str) or not namespace_root:
        raise StorageConfigError("nokv.namespace_root: expected non-empty absolute path")

    try:
        expanded_root = Template(namespace_root).substitute(
            project_hash=project_hash,
            agent_name=agent_name,
        )
    except (KeyError, ValueError) as exc:
        raise StorageConfigError(
            "nokv.namespace_root contains an unsupported template variable; "
            "allowed variables: ${agent_name}, ${project_hash}"
        ) from exc
    remote_base = normalize_nokv_path(expanded_root)
    if not remote_base.startswith("/"):
        raise StorageConfigError("nokv.namespace_root must be absolute")
    if namespace_root.strip() and not namespace_root.strip().startswith("/"):
        raise StorageConfigError("nokv.namespace_root must be absolute")

    agent_path = Path(agent_dir).resolve(strict=False)
    mounts = _normalize_mounts(storage.get("mounts"))

    env_values: dict[str, str] = {}
    for key in sorted(_REQUIRED_NOKV_ENV_KEYS):
        _, value = _env_value(nokv, key, environ)
        _validate_public_env_value(key, value)
        if key in _PUBLIC_NOKV_ENV_KEYS:
            env_values[key[:-4]] = value

    routes: list[StorageRoute] = []
    for mount in mounts:
        local_root = agent_path / mount
        _assert_under_agent(local_root, agent_path, mount)
        routes.append(
            StorageRoute(
                mount=mount,
                local_root=local_root,
                remote_root=f"{remote_base.rstrip('/')}/{mount}",
                backend="nokv",
            )
        )

    return ResolvedStorageConfig(
        enabled=True,
        backend="nokv",
        routes=routes,
        nokv=env_values,
    )
