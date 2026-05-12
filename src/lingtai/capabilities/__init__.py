"""Composable agent capabilities — add via Agent(capabilities=[...])."""
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lingtai_kernel.base_agent import BaseAgent

# Registry of built-in capability names → module paths.
# Entries starting with "." are relative to this package (lingtai.capabilities);
# absolute paths point at lingtai.core (the always-on agent floor). Both forms
# work because importlib.import_module honors the `package=` kwarg only for
# relative names.
_BUILTIN: dict[str, str] = {
    # Always-on floor (lingtai.core)
    "knowledge": "lingtai.core.library",
    "skills": "lingtai.core.skills",
    # Deprecated compatibility names. Agent config normalization maps these
    # before setup; direct callers still resolve for one release window.
    "library": "lingtai.core.library",
    "codex": "lingtai.core.library",
    "bash": "lingtai.core.bash",
    "avatar": "lingtai.core.avatar",
    "daemon": "lingtai.core.daemon",
    "mcp": "lingtai.core.mcp",
    "read": "lingtai.core.read",
    "write": "lingtai.core.write",
    "edit": "lingtai.core.edit",
    "glob": "lingtai.core.glob",
    "grep": "lingtai.core.grep",
    # Optional/multimodal capabilities (this package)
    "vision": ".vision",
    "web_search": ".web_search",
}

# Group names that expand to multiple capabilities.
_GROUPS: dict[str, list[str]] = {
    "file": ["read", "write", "edit", "glob", "grep"],
}


def canonical_capability_name(name: str) -> str:
    """Return the canonical post-rename capability name for *name*.

    Compatibility mapping:
    - ``knowledge`` is canonical for the private durable knowledge store.
    - ``library`` and ``codex`` are compatibility spellings for the same store
      when they clearly mean durable knowledge.
    - Raw init/preset manifests that predate the library->skills split are
      normalized by ``normalize_capabilities`` below so old ``library.paths``
      becomes ``skills.paths`` before setup.
    """
    return "knowledge" if name in {"library", "codex"} else name


def normalize_capabilities(capabilities: dict[str, dict]) -> dict[str, dict]:
    """Normalize old capability names to the post-rename surface.

    ``knowledge`` is the canonical private durable knowledge capability.
    ``codex`` is the oldest compatibility name. ``library`` is ambiguous:
    before the library->skills split it meant the skill catalog, and during the
    transition it also meant durable knowledge. Normalization keeps old
    skill-catalog manifests working while routing explicit durable-knowledge
    manifests to ``knowledge``.

    Rules:
    - ``knowledge`` always means private durable knowledge.
    - ``codex`` always means private durable knowledge (compatibility alias).
    - bare/list ``library`` or ``library: {}`` without explicit ``skills`` means
      the old skill catalog.
    - ``library: {paths: [...]}`` without explicit ``skills`` means old
      ``skills.paths``.
    - ``library`` with explicit knowledge kwargs, or any ``library`` alongside
      explicit ``skills``, means private durable knowledge.
    """
    out: dict[str, dict] = {}

    def is_explicit_knowledge_config(value: object) -> bool:
        """True when a config clearly means durable knowledge."""
        return isinstance(value, dict) and any(
            key in value for key in ("knowledge_limit", "library_limit", "codex_limit")
        )

    def merge_dict(dst: str, value: object) -> None:
        if value is None:
            value = {}
        if dst not in out:
            out[dst] = value if isinstance(value, dict) else value  # type: ignore[assignment]
            return
        if isinstance(out[dst], dict) and isinstance(value, dict):
            merged = dict(value)
            merged.update(out[dst])
            # Preserve skill path extras from either spelling.
            if dst == "skills":
                paths = []
                seen = set()
                for source in (value.get("paths", []), out[dst].get("paths", [])):
                    if not isinstance(source, list):
                        continue
                    for p in source:
                        if isinstance(p, str) and p not in seen:
                            paths.append(p)
                            seen.add(p)
                if paths:
                    merged["paths"] = paths
            out[dst] = merged

    for name, kwargs in capabilities.items():
        if name in {"knowledge", "codex"}:
            target = "knowledge"
        elif (
            name == "library"
            and "skills" not in capabilities
            and not is_explicit_knowledge_config(kwargs)
        ):
            # Legacy ambiguity: before the rename, a bare ``library`` entry was
            # the skill catalog. Keep old ``library``/``library: {}``/``library.paths``
            # manifests on the skill-catalog path. A library entry with
            # durable-knowledge-only kwargs is treated as explicit knowledge.
            # New configs that want both meanings spell them explicitly as
            # ``knowledge`` + ``skills`` (or transitional ``library`` + ``skills``).
            target = "skills"
        elif name == "library":
            target = "knowledge"
        else:
            target = name
        merge_dict(target, kwargs)
    return out


def expand_groups(names: list[str]) -> list[str]:
    """Expand group names (e.g. 'file') into individual capability names."""
    result = []
    for name in names:
        if name in _GROUPS:
            result.extend(_GROUPS[name])
        else:
            result.append(name)
    return result


def setup_capability(agent: "BaseAgent", name: str, **kwargs: Any) -> Any:
    """Look up a capability by *name* and call its ``setup(agent, **kwargs)``.

    Returns whatever the capability's ``setup`` function returns (typically
    a manager instance).

    Raises ``ValueError`` if the name is unknown or the module lacks ``setup``.
    """
    module_path = _BUILTIN.get(name)
    if module_path is None:
        raise ValueError(
            f"Unknown capability: {name!r}. "
            f"Available: {', '.join(sorted(_BUILTIN))}. "
            f"Groups: {', '.join(sorted(_GROUPS))}"
        )
    mod = importlib.import_module(module_path, package=__package__)
    setup_fn = getattr(mod, "setup", None)
    if setup_fn is None:
        raise ValueError(
            f"Capability module {name!r} does not export a setup() function"
        )
    return setup_fn(agent, **kwargs)


def get_all_providers() -> dict[str, dict]:
    """Return provider metadata for all user-facing capabilities.

    Returns a dict mapping capability name to
    ``{"providers": [...], "default": ... }``.
    Used by ``lingtai-agent check-caps`` CLI.
    """
    _USER_FACING: dict[str, str] = {
        "file": "lingtai.core.read",
        "bash": "lingtai.core.bash",
        "web_search": ".web_search",
        "knowledge": "lingtai.core.library",
        "library": "lingtai.core.library",
        "skills": "lingtai.core.skills",
        "codex": "lingtai.core.library",
        "vision": ".vision",
        "avatar": "lingtai.core.avatar",
        "daemon": "lingtai.core.daemon",
    }
    result = {}
    for name, module_path in _USER_FACING.items():
        mod = importlib.import_module(module_path, package=__package__)
        providers = getattr(mod, "PROVIDERS", None)
        if providers is not None:
            result[name] = dict(providers)
        else:
            result[name] = {"providers": [], "default": "builtin"}
    return result
