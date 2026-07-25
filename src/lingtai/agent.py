"""Agent — BaseAgent + composable capabilities.

Anatomy leaf: docs/plans/drafts/2026-04-30-anatomy-tree/leaves/core/preset-materialization/

Layer 2 of the three-layer hierarchy:
    BaseAgent (kernel) → Agent (capabilities) → CustomAgent (domain)

Capabilities are declared at construction and sealed before start().
"""
from __future__ import annotations

import contextlib
import copy
from dataclasses import dataclass
import json
from pathlib import Path
import threading
from types import SimpleNamespace
from typing import Any

from lingtai.kernel.base_agent import BaseAgent
from lingtai.kernel.base_agent.prompt import _refresh_meta_guidance_section
from lingtai.kernel._frontmatter import strip_frontmatter as _strip_frontmatter
from lingtai.kernel.config import AgentConfig, THINKING_PROVIDERS
from lingtai.llm.service import LLMService, build_provider_defaults_from_manifest_llm
from lingtai.kernel.prompt import build_system_prompt


@dataclass(frozen=True)
class MCPActivationOutcome:
    """Exact committed client identity returned by an activation transaction."""

    client: Any
    tool_names: tuple[str, ...]


@dataclass(frozen=True)
class _MCPPredecessorProjection:
    """Validated public/private projection captured before retirement."""

    spec_name: str
    client: Any
    tool_names: frozenset[str]


def _detached_spawn_kwargs() -> dict[str, Any]:
    """Platform kwargs for launching a detached agent-run child.

    This wrapper is a composition root, so the one platform branch for the
    CPR relaunch mechanism lives here rather than in kernel Core. POSIX
    detaches into a new session; Windows uses the shared detached creation
    flags (new process group, no window — ``start_new_session`` is
    POSIX-only and Windows children survive parent exit by default).
    """
    import os as _os

    if _os.name == "nt":
        from lingtai.adapters.windows._win32 import DETACHED_CREATIONFLAGS

        return {"creationflags": DETACHED_CREATIONFLAGS, "close_fds": True}
    return {"start_new_session": True}


def _run_preset_library_migrations(directory: Path) -> None:
    """Retained callback slot; production preset reads are migration-free.

    Preset data is read as authored. The old migration registry remains only for
    explicit historical/test callers and is not a runtime dependency of boot,
    refresh, or preset selection.
    """
    return


def load_preset(name: str, working_dir: "Path | None" = None) -> dict:
    """Wrapper-level preset-loader: Core ``load_preset`` wired to the POSIX runner.
    The single shared implementation — CLI, each `Agent`'s ``_preset_loader`` hook,
    and `Agent`'s own preset paths call it, so nothing else builds a workspace.
    """
    from lingtai.kernel.presets import load_preset as _core_load_preset

    return _core_load_preset(
        name, working_dir=working_dir, run_migrations=_run_preset_library_migrations
    )


def build_agent_config(manifest: dict[str, Any], *, max_rpm: int) -> AgentConfig:
    """Overlay host manifest values onto AgentConfig defaults."""
    defaults = AgentConfig()
    soul = manifest.get("soul", {})
    llm = manifest.get("llm", {})

    return AgentConfig(
        soul_delay=soul.get("delay", defaults.soul_delay),
        consultation_past_count=soul.get(
            "consultation_past_count", defaults.consultation_past_count
        ),
        soul_voice=soul.get("voice", defaults.soul_voice),
        soul_voice_prompt=soul.get("voice_prompt", defaults.soul_voice_prompt),
        # ``manifest.max_turns`` is a legacy/resolved-manifest field and is no
        # longer the authoritative tool-loop guard source. ACTIVE-turn
        # tool-call safety is kernel-owned in ``lingtai.kernel.safety_limits``.
        # Keep AgentConfig.max_turns at its default for API compatibility, but
        # deliberately ignore stale init.json values here.
        language=manifest.get("language", defaults.language),
        activeness=manifest.get("activeness", defaults.activeness),
        context_limit=manifest.get("context_limit", defaults.context_limit),
        # Soft per-molt/session cache-miss token budget (default 1_000_000).
        # Validated as a positive int in init_schema; hydrated verbatim here.
        cache_miss_budget=manifest.get(
            "cache_miss_budget", defaults.cache_miss_budget
        ),
        # Codex-family providers own their omitted-thinking default at the
        # adapter (omitted -> reasoning.effort "xhigh"), so an omitted manifest
        # value stays the "default" sentinel for them instead of being promoted
        # to the legacy cross-provider "high" main-session default.
        thinking=llm.get(
            "thinking",
            "default"
            if str(llm.get("provider") or "").lower() in THINKING_PROVIDERS
            else defaults.thinking,
        ),
        # Molt thresholds and the context.molt message are kernel-fixed runtime
        # constants and are NOT agent-configurable. Stale manifest
        # molt_notice/molt_pressure/molt_urgency/molt_prompt values are
        # deliberately ignored.
        snapshot_interval=manifest.get(
            "snapshot_interval", defaults.snapshot_interval
        ),
        time_awareness=manifest.get("time_awareness", defaults.time_awareness),
        timezone_awareness=manifest.get(
            "timezone_awareness", defaults.timezone_awareness
        ),
        aed_timeout=manifest.get("aed_timeout", defaults.aed_timeout),
        max_aed_attempts=manifest.get(
            "max_aed_attempts", defaults.max_aed_attempts
        ),
        max_rpm=max_rpm,
    )


class Agent(BaseAgent):
    """BaseAgent with composable capabilities.

    Args:
        capabilities: Capability names to enable. Either a list of strings
            (no kwargs) or a dict mapping names to kwargs dicts.
            Each capability dict may include ``"provider"`` to route that
            capability to a specific LLM provider (e.g. ``"gemini"``, ``"minimax"``).
            Group names (e.g. ``"file"``) expand to individual capabilities.
        *args, **kwargs: Passed through to BaseAgent.
    """

    def __init__(
        self,
        *args: Any,
        capabilities: list[str] | dict[str, dict] | None = None,
        addons: list[str] | None = None,
        combo_name: str | None = None,
        disable: list[str] | None = None,
        **kwargs: Any,
    ):
        # MCP transport startup, validation, predecessor retirement, and
        # publication form one wrapper-owned transaction.  Candidates stay
        # outside every public registry until the transaction commits.
        self._mcp_activation_lock = threading.RLock()
        self._mcp_lifecycle_generation = 0
        self._mcp_lifecycle_state = "active"
        self._mcp_lifecycle_barrier = threading.Event()
        self._mcp_stop_requested = threading.Event()
        self._mcp_refresh_owner_thread: int | None = None
        self._mcp_refresh_handoff_committed = False
        self._mcp_pending_retirements: dict[str, Any] = {}
        self._mcp_reserved_activation_tokens: set[object] = set()
        self._mcp_clients: list[Any] = []
        self._mcp_clients_by_tool: dict[str, Any] = {}
        self._mcp_tool_names: set[str] = set()

        # Default karma authority for the primary agent (本我)
        kwargs.setdefault("admin", {"karma": True})

        # Inject the built-in intrinsic tool registry. The kernel owns the tool
        # machinery, not the concrete tools: it accepts intrinsics as injection
        # and a bare BaseAgent has none. lingtai.Agent is the composing layer, so
        # it supplies the five mandatory intrinsics here. ``setdefault`` lets a
        # host override (e.g. a test injecting a subset).
        from lingtai.tools.registry import INTRINSICS
        kwargs.setdefault("intrinsics", INTRINSICS)

        # The outer wrapper is a composition root for direct ``lingtai.Agent``
        # callers.  Explicit ``event_journal=None`` remains an honest opt-out.
        owned_event_journal = None
        if "event_journal" not in kwargs and "working_dir" in kwargs:
            from lingtai.adapters.posix.event_journal import (
                PosixJsonlEventJournalAdapter,
            )

            config = kwargs.get("config")
            ensure_ascii = bool(getattr(config, "ensure_ascii", False))
            owned_event_journal = PosixJsonlEventJournalAdapter(
                kwargs["working_dir"],
                ensure_ascii=ensure_ascii,
            )
            kwargs["event_journal"] = owned_event_journal

        try:
            # BaseAgent requires an explicit workdir lease. As the composition
            # root for direct ``lingtai.Agent`` callers, select and construct the
            # production adapter for the running platform (fail-loud on
            # unsupported platforms). A caller may inject its own lease.
            if "workdir_lease" not in kwargs and "working_dir" in kwargs:
                from lingtai.adapters.workdir_lease import select_workdir_lease

                kwargs["workdir_lease"] = select_workdir_lease(kwargs["working_dir"])

            # BaseAgent requires a notification store. As the composition root,
            # construct the production POSIX adapter. A caller may inject its own.
            if "notification_store" not in kwargs and "working_dir" in kwargs:
                from lingtai.adapters.posix.notification_store import (
                    PosixNotificationStoreAdapter,
                )

                kwargs["notification_store"] = PosixNotificationStoreAdapter(
                    kwargs["working_dir"]
                )

            # BaseAgent requires an agent-presence store bound to its own
            # working directory. As the composition root, construct the
            # production POSIX adapter. A caller may inject its own.
            if "agent_presence" not in kwargs and "working_dir" in kwargs:
                from lingtai.adapters.posix.agent_presence import (
                    PosixAgentPresenceStoreAdapter,
                )

                kwargs["agent_presence"] = PosixAgentPresenceStoreAdapter(
                    kwargs["working_dir"]
                )

            # BaseAgent requires a lifecycle clock. As the composition root for
            # direct ``lingtai.Agent`` callers, construct the portable system
            # adapter (no working_dir needed). A caller may inject its own.
            if "lifecycle_clock" not in kwargs:
                from lingtai.adapters.lifecycle_clock import (
                    SystemLifecycleClockAdapter,
                )

                kwargs["lifecycle_clock"] = SystemLifecycleClockAdapter()

            # BaseAgent requires a refresh-watcher Port for the detached relaunch
            # handoff. As the composition root, select the production capability
            # adapter for the running platform. A caller may inject its own.
            if kwargs.get("refresh_watcher") is None:
                from lingtai.adapters.refresh_watcher import select_refresh_watcher

                kwargs["refresh_watcher"] = select_refresh_watcher()

            # Compose the two required snapshot/revision capabilities outside Core.
            # Separate instances intentionally target the workdir and running source.
            if "snapshot_port" not in kwargs and "working_dir" in kwargs:
                from lingtai.adapters.posix.git_cli import PosixGitCliAdapter

                kwargs["snapshot_port"] = PosixGitCliAdapter(kwargs["working_dir"])
            if "source_revision_port" not in kwargs:
                from lingtai.adapters.posix.git_cli import PosixGitCliAdapter

                kwargs["source_revision_port"] = PosixGitCliAdapter(
                    Path(__file__).resolve().parent
                )

            # Store combo name before super().__init__ (not forwarded to BaseAgent)
            self._combo_name = combo_name
            super().__init__(*args, **kwargs)
            # Compose the preset-loader hook so daemon/system tools resolve
            # presets through the wrapper implementation instead of importing
            # Core load_preset or constructing a migration workspace adapter.
            self._preset_loader = load_preset
        except Exception:
            if owned_event_journal is not None:
                with contextlib.suppress(Exception):
                    owned_event_journal.close()
            raise

        # Persist LLM config for revive (self-sufficient agents contract)
        self._persist_llm_config()

        # Auto-create FileIOService if not provided by host. Uses the
        # ``default_file_io_service`` factory so the Rust sidecar gets
        # picked up automatically when a wheel-bundled or env-provided
        # binary is available, with transparent pure-Python fallback.
        # See LINGTAI_FILE_IO_BACKEND in services/file_io_sidecar.py.
        if self._file_io is None:
            from .services.file_io_sidecar import default_file_io_service
            self._file_io = default_file_io_service(root=self._working_dir)

        # Expand groups and normalize to dict
        if isinstance(capabilities, list):
            from lingtai.tools.registry import expand_groups, normalize_capabilities
            expanded = expand_groups(capabilities)
            capabilities = normalize_capabilities({name: {} for name in expanded})
        elif isinstance(capabilities, dict):
            from lingtai.tools.registry import _GROUPS, normalize_capabilities
            expanded_dict: dict[str, dict] = {}
            for name, cap_kwargs in capabilities.items():
                if name in _GROUPS:
                    for sub in _GROUPS[name]:
                        expanded_dict[sub] = {}
                elif cap_kwargs is None:
                    expanded_dict[name] = None  # propagate disable-sentinel
                else:
                    expanded_dict[name] = cap_kwargs
            capabilities = normalize_capabilities(
                {n: (v if v is not None else {}) for n, v in expanded_dict.items()}
            )
            # Preserve null sentinels for apply_core_defaults to interpret as opt-out.
            for n, v in expanded_dict.items():
                if v is None:
                    capabilities[n] = None  # type: ignore[assignment]

        # Apply core defaults — the always-on tool floor boots on every agent
        # unless explicitly disabled via `disable=[...]` or `"name": null` in
        # the capabilities dict. init.json kwargs override default kwargs.
        from lingtai.tools.registry import apply_core_defaults
        capabilities = apply_core_defaults(capabilities, disable=disable)

        # Track for avatar replay
        self._capabilities: list[tuple[str, dict]] = []
        self._capability_managers: dict[str, Any] = {}
        # Names registered by parent MCP clients. Daemon uses this only to avoid
        # leaking parent MCP tools through tasks[].tools; task MCP access must
        # come from complete per-task registrations.
        self._mcp_tool_names: set[str] = set()

        # Decompress addons BEFORE capability setup so the `mcp` capability
        # sees the populated registry on its first reconcile.
        if addons:
            try:
                from .services.mcp_registry import decompress_addons
                report = decompress_addons(self._working_dir, addons)
                self._log("mcp_decompress", **report)
            except Exception as e:
                self._log("mcp_decompress_failed", reason=str(e))

        # Register capabilities — provider kwarg flows through to setup() naturally
        if capabilities:
            for name, cap_kwargs in capabilities.items():
                try:
                    self._setup_capability(name, **cap_kwargs)
                except (ValueError, ImportError, TypeError) as e:
                    self._log("capability_skipped", capability=name, reason=str(e))

        # Install intrinsic manuals (wipe-and-rewrite .library/intrinsic/)
        # from the bundles shipped with each enabled capability.
        self._install_intrinsic_manuals()

        # Auto-load MCP servers from working directory.
        # Runs AFTER addon decompression so init.json mcp entries can reference
        # newly-decompressed registry records.
        self._load_mcp_from_workdir()

        # Re-write manifest now that capabilities are registered
        if self._capabilities:
            self._workdir.write_manifest(self._build_manifest())

    def _persist_llm_config(self) -> None:
        """Persist LLM config to llm.json for agent revive.

        Extracted from __init__ to avoid duplication.
        """
        _service = getattr(self, "service", None)
        if _service is None:
            return
        try:
            import json as _json
            llm_config: dict[str, Any] = {
                "provider": _service.provider,
                "model": _service.model,
            }
            _base_url = getattr(_service, "_base_url", None)
            if isinstance(_base_url, str) and _base_url:
                llm_config["base_url"] = _base_url
            llm_dir = self._working_dir / "system"
            llm_dir.mkdir(exist_ok=True)
            (llm_dir / "llm.json").write_text(
                _json.dumps(llm_config, ensure_ascii=False)
            )
        except (TypeError, AttributeError, OSError):
            pass  # LLM config not available (e.g., mock service in tests)

    def _setup_capability(self, name: str, **kwargs: Any) -> Any:
        """Load a named capability.

        ``None`` from setup means success without a manager object. A setup that
        cannot register must return ``CAPABILITY_UNAVAILABLE`` before adding any
        tools so this wrapper can leave the capability absent from public
        registration surfaces.

        Not directly sealed — but setup() calls add_tool() which checks the seal.
        Must only be called from __init__ (before start()).
        """
        from lingtai.tools.registry import CAPABILITY_UNAVAILABLE, setup_capability

        serializable_kw = {
            k: v for k, v in kwargs.items()
            if isinstance(v, (str, int, float, bool, type(None), list, dict))
        }
        self._capabilities.append((name, serializable_kw))
        try:
            mgr = setup_capability(self, name, **kwargs)
        except Exception:
            # Roll back the entry so _capabilities only lists registered caps.
            self._capabilities.pop()
            raise
        if mgr is CAPABILITY_UNAVAILABLE:
            self._capabilities.pop()
            return None
        self._capability_managers[name] = mgr
        return mgr

    def _install_intrinsic_manuals(self) -> None:
        """Wipe and rewrite ``.library/intrinsic/`` from kernel-shipped manuals.

        Runs near the end of ``__init__`` and ``_setup_from_init``. Installs
        every capability's ``manual/`` bundle into
        ``.library/intrinsic/capabilities/<name>/``, **regardless of whether
        this agent enabled the capability**. The library is kernel-shipped
        documentation — agents should be able to read about a capability
        before they configure it.

        Never touches ``.library/custom/``. That is the agent's territory.
        """
        import shutil
        import lingtai.tools as tools_pkg
        import lingtai.intrinsic_skills as skills_pkg

        library_dir = self._working_dir / ".library"
        intrinsic_dir = library_dir / "intrinsic"

        (library_dir / "custom").mkdir(parents=True, exist_ok=True)

        if intrinsic_dir.exists():
            shutil.rmtree(intrinsic_dir)
        (intrinsic_dir / "capabilities").mkdir(parents=True, exist_ok=True)

        def install_from(pkg, subdir: str) -> None:
            pkg_file = getattr(pkg, "__file__", None)
            if not pkg_file:
                return
            pkg_root = Path(pkg_file).parent
            for entry in sorted(pkg_root.iterdir()):
                if not entry.is_dir() or entry.name.startswith("_"):
                    continue
                src = entry / "manual"
                if src.is_dir():
                    # The retained implementation directory is ``bash``; its
                    # agent-facing manual is installed under canonical ``shell``.
                    destination_name = "shell" if entry.name == "bash" else entry.name
                    shutil.copytree(src, intrinsic_dir / subdir / destination_name)

        def install_skills_from(pkg, subdir: str) -> None:
            """Install standalone skill bundles (no companion code, no manual/ wrapper).

            Each ``<pkg>/<entry>/`` directory IS the skill — copied verbatim into
            ``intrinsic/<subdir>/<entry>/`` (manuals plus any sidecar scripts/assets,
            e.g. the ``lingtai-kernel-anatomy`` checker and benchmark). Used for
            skills that don't belong to any single tool.
            """
            pkg_file = getattr(pkg, "__file__", None)
            if not pkg_file:
                return
            pkg_root = Path(pkg_file).parent
            for entry in sorted(pkg_root.iterdir()):
                if not entry.is_dir() or entry.name.startswith("_"):
                    continue
                shutil.copytree(entry, intrinsic_dir / subdir / entry.name)

        # Every tool package with a manual/ installs into
        # intrinsic/capabilities/<name>/ — agents see one flat capability
        # namespace. Scanning the consolidated ``lingtai.tools`` package replaces the
        # former core/ + capabilities/ dual scan; tools without a manual/ (the
        # file tools, the non-email intrinsics whose manuals ship as
        # intrinsic_skills bundles below) are simply skipped.
        install_from(tools_pkg, "capabilities")
        install_skills_from(skills_pkg, "capabilities")

        # If the skills capability is loaded, re-run its reconcile now that
        # the manuals are on disk — so the injected catalog reflects them on
        # the very first turn (skills.setup()'s initial _reconcile ran BEFORE
        # install, when the manual dir was empty).
        for cap_name, cap_kwargs in self._capabilities:
            if cap_name == "skills":
                try:
                    from lingtai.tools import skills as skillsmod
                    skillsmod._reconcile(self, list(cap_kwargs.get("paths", []) or []))
                except Exception as e:
                    self._log("skills_reconcile_failed", reason=str(e))
                break

    _SENSITIVE_KEYS = {"api_key", "api_key_env", "api_secret", "token", "password"}

    #: Safelist for the public ``llm`` block surfaced in .agent.json. Mirrors
    #: ``base_agent.identity._LLM_PUBLIC_KEYS`` and exists at the wrapper layer
    #: as defense-in-depth — init.json's ``manifest.llm`` may carry api_key /
    #: api_key_env values that must never reach the on-disk manifest or the
    #: system prompt's identity section.
    _LLM_PUBLIC_KEYS = ("provider", "model", "base_url", "api_compat", "context_limit")

    #: Safelist for the public ``preset`` block. ``active`` and ``default`` are
    #: path strings, ``allowed`` is a list of path strings — none of these
    #: carry secrets, but pinning the safelist guards against future preset
    #: schema growth that might introduce sensitive fields.
    _PRESET_PUBLIC_KEYS = ("active", "default", "allowed")

    def _build_manifest(self) -> dict:
        """Extend kernel manifest with capabilities, preset, and combo.

        Strips sensitive fields (api_key, etc.) from capability kwargs
        so they don't leak into the system prompt or outgoing mail identity.
        Adds a sanitized ``preset`` block (active/default/allowed) and
        re-applies the ``llm`` safelist for defense-in-depth — even if a
        future LLMService grew a sensitive attribute, the manifest never
        carries anything outside ``_LLM_PUBLIC_KEYS``.
        """
        data = super()._build_manifest()
        caps = getattr(self, "_capabilities", None)
        if caps:
            data["capabilities"] = [
                (name, {k: v for k, v in kw.items() if k not in self._SENSITIVE_KEYS})
                for name, kw in caps
            ]
        if self._combo_name:
            data["combo"] = self._combo_name

        # Enforce the llm safelist a second time — the kernel layer already
        # filters, but a subclass override or future service shape might add
        # a non-safelisted attribute. Doing it here means anything written
        # to disk is guaranteed safelist-only.
        if isinstance(data.get("llm"), dict):
            data["llm"] = {
                k: v for k, v in data["llm"].items()
                if k in self._LLM_PUBLIC_KEYS
            }
            if not data["llm"]:
                del data["llm"]

        preset = self._read_preset_from_init()
        if preset:
            data["preset"] = preset

        return data

    def _read_preset_from_init(self) -> dict:
        """Read ``manifest.preset`` from init.json and sanitize.

        Returns ``{}`` if init.json is missing, unreadable, or has no preset
        block — bare init.json (e.g. tests) and pre-preset deployments both
        silently fall through. Never raises: a corrupt init.json must not
        break manifest writes.

        Filters to ``_PRESET_PUBLIC_KEYS`` and string/list-of-string values so
        the disk manifest never carries anything the safelist doesn't explicitly
        allow.
        """
        import json

        init_path = self._working_dir / "init.json"
        if not init_path.is_file():
            return {}
        try:
            raw = json.loads(init_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        manifest = raw.get("manifest") if isinstance(raw, dict) else None
        if not isinstance(manifest, dict):
            return {}
        preset = manifest.get("preset")
        if not isinstance(preset, dict):
            return {}
        clean: dict = {}
        for key in self._PRESET_PUBLIC_KEYS:
            if key not in preset:
                continue
            val = preset[key]
            if key in {"active", "default"}:
                if isinstance(val, str) and val:
                    clean[key] = val
                continue
            if key == "allowed" and isinstance(val, list):
                allowed = [item for item in val if isinstance(item, str) and item]
                if allowed:
                    clean[key] = allowed
        return clean

    def _build_system_prompt(self) -> str:
        """Override kernel's prompt builder to inject app tool descriptions.

        ``base_prompt`` is the init-prompt contract's third-party (application /
        recipe / preset) injection point, resolved from init.json by
        ``_reload_prompt_sections`` into ``self._base_prompt``. The kernel
        builder renders it right after the raw ``principle`` section and before
        the rest of Batch 1.
        """
        self._refresh_tool_inventory_section()
        _refresh_meta_guidance_section(self)
        return build_system_prompt(
            prompt_manager=self._prompt_manager,
            base_prompt=getattr(self, "_base_prompt", ""),
            language=self._config.language,
            activeness=self._config.activeness,
        )

    def _build_system_prompt_batches(self) -> list[str]:
        """Override kernel's batched builder to inject app tool descriptions."""
        from lingtai.kernel.prompt import build_system_prompt_batches
        self._refresh_tool_inventory_section()
        _refresh_meta_guidance_section(self)
        return build_system_prompt_batches(
            prompt_manager=self._prompt_manager,
            base_prompt=getattr(self, "_base_prompt", ""),
            language=self._config.language,
            activeness=self._config.activeness,
        )

    def _build_mcp_launch_env(self, name: str, cfg: dict) -> dict:
        """Merge user env without allowing runtime routing-key overrides."""
        runtime_owned = {"LINGTAI_AGENT_DIR", "LINGTAI_MCP_NAME"}
        user_env = {
            key: value
            for key, value in (cfg.get("env") or {}).items()
            if key.upper() not in runtime_owned
        }
        return {
            **user_env,
            "LINGTAI_AGENT_DIR": str(self._working_dir),
            "LINGTAI_MCP_NAME": name,
        }

    def _load_mcp_from_workdir(self) -> None:
        """Auto-load MCP servers from two sources, in order:

        1. ``working_dir/mcp/servers.json`` — legacy, ungated. Loaded as-is.
        2. ``init.json`` top-level ``mcp`` field — gated by the per-agent
           registry at ``working_dir/mcp_registry.jsonl``. An init.json mcp
           entry whose name is not in the registry is skipped with a warning.

        Both sources accept stdio and HTTP entries:

            {
              "vision-server": {
                "type": "stdio",
                "command": "npx",
                "args": ["-y", "@z_ai/mcp-server"],
                "env": {"Z_AI_API_KEY": "...", "Z_AI_MODE": "ZAI"}
              },
              "web-search": {
                "type": "http",
                "url": "https://api.z.ai/api/mcp/web_search_prime/mcp",
                "headers": {"Authorization": "Bearer ..."}
              }
            }

        The ``type`` field defaults to ``"stdio"`` if omitted.

        Side effect: every init.json mcp entry whose name was registered
        is recorded in ``self._mcp_init_specs``. The ``_retry_failed_mcps``
        helper consults this dict on ``system(action="refresh")`` to detect
        and re-spawn MCPs whose subprocess died (issue #34).
        """
        import json

        from lingtai.kernel.logging import get_logger
        logger = get_logger()

        # Per-name tracking of init.json MCP launches. Populated below and
        # consulted by `_retry_failed_mcps`. Reset on every load so that
        # entries removed from init.json drop out of the retry pool.
        self._mcp_init_specs: dict[str, dict] = {}
        # Parent MCP tool names are tracked only so daemon can prevent
        # tasks[].tools from leaking parent MCP tools. Task-level daemon MCP
        # access must come from complete registrations in tasks[].mcp.
        self._mcp_tool_names: set[str] = set()

        # LICC env injection — every spawned MCP gets these so it can
        # locate the agent's working dir + know its own registry name and
        # write events into the LICC inbox. User-supplied env in cfg wins.
        def _spawn(
            name: str,
            cfg: dict,
            source: str,
            *,
            init_spec_name: str | None = None,
            reserved_provenance: object | None = None,
        ) -> object | None:
            """Return the MCPClient/HTTPMCPClient on success, None on failure."""
            try:
                server_type = cfg.get("type", "stdio")
                if server_type == "http":
                    if "url" not in cfg:
                        return None
                    connector = self.connect_mcp_http
                    connector_args = {
                        "url": cfg["url"],
                        "headers": cfg.get("headers"),
                    }
                else:
                    if "command" not in cfg:
                        return None
                    # Merge: LICC defaults < per-MCP env (user-supplied).
                    # Add LINGTAI_MCP_NAME per-spawn so each MCP knows its
                    # own registry name without needing to be told elsewhere.
                    merged_env = self._build_mcp_launch_env(name, cfg)
                    connector = self.connect_mcp
                    connector_args = {
                        "command": cfg["command"],
                        "args": cfg.get("args"),
                        "env": merged_env,
                    }
                reserved_token = (
                    object() if reserved_provenance is not None else None
                )
                if reserved_token is not None:
                    self._mcp_reserved_activation_tokens.add(reserved_token)
                try:
                    outcome = connector(
                        **connector_args,
                        _init_spec_name=init_spec_name,
                        _predecessor=None,
                        _return_activation_outcome=True,
                        _reserved_activation_token=reserved_token,
                    )
                finally:
                    if reserved_token is not None:
                        self._mcp_reserved_activation_tokens.discard(
                            reserved_token
                        )
                if not isinstance(outcome, MCPActivationOutcome):
                    raise RuntimeError(
                        "MCP loader received no activation outcome"
                    )
                if init_spec_name is not None:
                    self._assert_mcp_activation_outcome(
                        init_spec_name, outcome
                    )
                tools = list(outcome.tool_names)
                loaded_client = outcome.client
                logger.info("[%s] MCP %s (%s): loaded %d tools (%s)",
                            self.agent_name, name, source, len(tools),
                            ", ".join(tools))
                return loaded_client
            except Exception as e:
                logger.warning("[%s] MCP %s (%s): failed to load: %s",
                               self.agent_name, name, source, e)
                return None

        # Source 1: legacy mcp/servers.json
        legacy_config = self._working_dir / "mcp" / "servers.json"
        if legacy_config.is_file():
            try:
                servers = json.loads(legacy_config.read_text(encoding="utf-8"))
                if isinstance(servers, dict):
                    for name, cfg in servers.items():
                        if isinstance(cfg, dict):
                            _spawn(name, cfg, source="mcp/servers.json")
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("[%s] mcp/servers.json: failed to read: %s",
                               self.agent_name, e)

        # Source 2: init.json top-level mcp section, gated by registry.
        init_path = self._working_dir / "init.json"
        if not init_path.is_file():
            return
        try:
            init_data = json.loads(init_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return

        init_mcp = init_data.get("mcp")
        if not isinstance(init_mcp, dict) or not init_mcp:
            return

        # Cross-reference against the registry.
        try:
            from .services.mcp_registry import (
                materialize_curated_provenance,
                read_registry,
            )
            registered, _problems = read_registry(self._working_dir)
            registered_records = {r["name"]: r for r in registered}
        except Exception as e:
            logger.warning("[%s] mcp registry read failed: %s",
                           self.agent_name, e)
            registered_records = {}

        for name, cfg in init_mcp.items():
            if not isinstance(cfg, dict):
                continue
            if name not in registered_records:
                logger.warning(
                    "[%s] init.json mcp %r: skipped — not in mcp_registry.jsonl. "
                    "Register it first (see mcp-manual skill).",
                    self.agent_name, name,
                )
                continue
            # Record every registered init.json mcp entry — failures (client
            # is None) and successes alike — so `_retry_failed_mcps` can
            # tell which ones to re-attempt vs leave alone.
            reserved_provenance = materialize_curated_provenance(
                name,
                cfg,
                "init.json:mcp",
                registry_record=registered_records[name],
            )
            self._mcp_init_specs[name] = {
                "cfg": cfg,
                "source": "init.json:mcp",
                "client": None,
                "reserved_provenance": reserved_provenance,
            }
            if (
                name == "telegram"
                and registered_records[name].get("source") == "lingtai-curated"
                and reserved_provenance is None
            ):
                logger.warning(
                    "[%s] init.json mcp %r: skipped — curated provenance "
                    "does not match the canonical launch and safe env policy",
                    self.agent_name,
                    name,
                )
                continue
            _spawn(
                name,
                cfg,
                source="init.json:mcp",
                init_spec_name=name,
                reserved_provenance=reserved_provenance,
            )

    def _retry_failed_mcps(self) -> dict:
        """Serialize retry, pending-retirement drain, and replacement."""
        self._ensure_mcp_activation_state()
        with self._mcp_activation_lock:
            if (
                self._mcp_lifecycle_state == "stopping"
                or self._mcp_lifecycle_barrier.is_set()
                or self._mcp_stop_requested.is_set()
            ):
                failed = set(self._mcp_init_specs)
                failed.update(self._mcp_pending_retirements)
                failed.add("lifecycle:stopping")
                return {
                    "retried": [],
                    "recovered": [],
                    "still_failed": sorted(failed),
                    "healthy": [],
                }
            unresolved = self._drain_pending_mcp_retirements_locked()
            if unresolved:
                return {
                    "retried": [],
                    "recovered": [],
                    "still_failed": sorted(unresolved),
                    "healthy": [],
                }
            return self._retry_failed_mcps_locked()

    def _begin_mcp_refresh_ownership(self) -> bool:
        """Hold refresh ownership through preflight, mutation, and handoff."""
        self._ensure_mcp_activation_state()
        self._mcp_activation_lock.acquire()
        if (
            self._mcp_stop_requested.is_set()
            or self._mcp_lifecycle_barrier.is_set()
            or self._mcp_lifecycle_state != "active"
        ):
            self._mcp_activation_lock.release()
            return False
        self._mcp_lifecycle_generation += 1
        self._mcp_lifecycle_state = "refreshing"
        self._mcp_refresh_owner_thread = threading.get_ident()
        self._mcp_refresh_handoff_committed = False
        return True

    def _assert_mcp_refresh_ownership(self) -> None:
        """Fail if stop was requested after refresh acquired ownership."""
        if (
            self._mcp_refresh_owner_thread != threading.get_ident()
            or self._mcp_lifecycle_state != "refreshing"
            or self._mcp_stop_requested.is_set()
            or self._mcp_lifecycle_barrier.is_set()
        ):
            raise RuntimeError("refresh invalidated by pending stop")

    def _commit_mcp_refresh_handoff(self) -> None:
        """Make a successful relaunch handoff terminal in this process."""
        self._assert_mcp_refresh_ownership()
        self._mcp_refresh_handoff_committed = True
        self._mcp_lifecycle_state = "relaunching"
        self._mcp_lifecycle_barrier.set()

    def _end_mcp_refresh_ownership(self) -> None:
        """Release an aborted refresh or preserve a committed handoff barrier."""
        try:
            if self._mcp_refresh_owner_thread != threading.get_ident():
                raise RuntimeError("refresh ownership is not held by this thread")
            self._mcp_refresh_owner_thread = None
            if self._mcp_refresh_handoff_committed:
                self._mcp_lifecycle_state = "relaunching"
                self._mcp_lifecycle_barrier.set()
            else:
                self._mcp_lifecycle_state = "active"
                self._mcp_lifecycle_barrier.clear()
        finally:
            self._mcp_activation_lock.release()

    def _retry_failed_mcps_locked(self) -> dict:
        """Re-spawn any init.json MCP whose subprocess is dead or never started.

        Walks ``self._mcp_init_specs`` (populated by ``_load_mcp_from_workdir``)
        and, for each entry whose tracked client is missing or visibly
        unhealthy, tears it down and re-attempts the spawn with the original
        config. Returns a report dict ``{retried: [...], recovered: [...],
        still_failed: [...], healthy: [...]}``.

        Why this exists: ``system(action="refresh")`` is the documented
        "fix config → refresh" recovery path for curated addons (imap,
        telegram, feishu, wechat). Without this retry, an MCP that exited
        during initial boot stays dead until full process restart — see
        Lingtai-AI/lingtai#34.

        Health check: missing client (boot-time spawn raised) is the
        clearest signal. For clients that registered but whose subprocess
        later died, ``MCPClient.is_connected()`` is the cheapest probe — it
        returns False when the background loop has exited (which happens
        when the stdio transport closes due to subprocess death).
        """
        from lingtai.kernel.logging import get_logger
        logger = get_logger()

        specs = getattr(self, "_mcp_init_specs", None)
        if not specs:
            return {"retried": [], "recovered": [], "still_failed": [],
                    "healthy": []}

        retried: list[str] = []
        recovered: list[str] = []
        still_failed: list[str] = []
        healthy: list[str] = []

        for name, spec in list(specs.items()):
            client = spec.get("client")
            cfg = spec.get("cfg") or {}
            source = spec.get("source", "init.json:mcp")

            # Health: client present AND its session is connected.
            if client is not None and getattr(client, "is_connected", lambda: False)():
                healthy.append(name)
                continue

            retried.append(name)
            self._log("mcp_retry_attempt", name=name, source=source)

            new_client: object | None = None
            try:
                server_type = cfg.get("type", "stdio")
                if server_type == "http":
                    if "url" not in cfg:
                        raise ValueError("http transport requires 'url'")
                    connector = self.connect_mcp_http
                    connector_args = {
                        "url": cfg["url"],
                        "headers": cfg.get("headers"),
                    }
                else:
                    if "command" not in cfg:
                        raise ValueError("stdio transport requires 'command'")
                    merged_env = self._build_mcp_launch_env(name, cfg)
                    connector = self.connect_mcp
                    connector_args = {
                        "command": cfg["command"],
                        "args": cfg.get("args"),
                        "env": merged_env,
                    }

                from .services.mcp_registry import (
                    CuratedMCPProvenance,
                    materialize_curated_provenance,
                )
                stored_provenance = spec.get("reserved_provenance")
                current_provenance = (
                    materialize_curated_provenance(
                        name,
                        cfg,
                        source,
                        expected=stored_provenance,
                    )
                    if isinstance(stored_provenance, CuratedMCPProvenance)
                    else None
                )
                if (
                    isinstance(stored_provenance, CuratedMCPProvenance)
                    and current_provenance is None
                ):
                    raise RuntimeError(
                        "curated MCP provenance changed since initial load"
                    )
                reserved_token = (
                    object() if current_provenance is not None else None
                )
                if reserved_token is not None:
                    self._mcp_reserved_activation_tokens.add(reserved_token)
                try:
                    outcome = connector(
                        **connector_args,
                        _allow_sealed=True,
                        _init_spec_name=name,
                        _predecessor=client,
                        _return_activation_outcome=True,
                        _reserved_activation_token=reserved_token,
                    )
                finally:
                    if reserved_token is not None:
                        self._mcp_reserved_activation_tokens.discard(
                            reserved_token
                        )
                if not isinstance(outcome, MCPActivationOutcome):
                    raise RuntimeError(
                        "production MCP connector returned no activation outcome"
                    )
                self._assert_mcp_activation_outcome(name, outcome)
                new_client = outcome.client
            except Exception as e:
                logger.warning("[%s] MCP %s (%s): retry failed: %s",
                               self.agent_name, name, source, e)
                self._log("mcp_retry_failed", name=name, error=str(e))
                if (
                    client is None
                    and name not in self._mcp_pending_retirements
                ):
                    spec["client"] = None
                still_failed.append(name)
                continue

            spec["client"] = new_client
            if new_client is not None and getattr(
                new_client, "is_connected", lambda: False)():
                logger.info("[%s] MCP %s (%s): retry recovered",
                            self.agent_name, name, source)
                self._log("mcp_retry_recovered", name=name)
                recovered.append(name)
            else:
                # Spawn returned without raising but the client is not
                # connected — treat as still failed.
                self._log("mcp_retry_failed", name=name,
                          error="client not connected after retry")
                still_failed.append(name)

        return {
            "retried": retried,
            "recovered": recovered,
            "still_failed": still_failed,
            "healthy": healthy,
        }

    def _cpr_agent(self, address: str) -> bool | dict | None:
        """Resuscitate a suspended agent by launching it as a detached process.

        Uses the resolved venv Python to run `lingtai run <dir>`.  Success is
        reported only after the target writes a fresh heartbeat; quick child
        exits and startup timeouts are returned as explicit errors.
        """
        import shlex
        import subprocess
        import time
        from lingtai.adapters.posix.agent_presence import PosixAgentPresenceStoreAdapter
        from lingtai.kernel.agent_presence import (
            is_agent as _presence_is_agent,
            observe_alive as _presence_observe_alive,
        )
        from lingtai.kernel.handshake import resolve_address
        from lingtai.venv_resolve import resolve_venv, venv_python

        base_dir = self._working_dir.parent
        target = resolve_address(address, base_dir)
        target_presence = PosixAgentPresenceStoreAdapter(target)
        if not _presence_is_agent(target_presence.observe_manifest()):
            return None

        init_path = target / "init.json"
        if not init_path.is_file():
            self._log("cpr_no_init", path=str(target))
            return None

        # Clean stale signal files so a CPR'd agent boots cleanly.
        for sig in (".suspend", ".sleep", ".interrupt"):
            sig_file = target / sig
            if sig_file.is_file():
                sig_file.unlink(missing_ok=True)

        # Resolve Python: target's init.json venv_path → global runtime
        try:
            import json as _json
            target_data = _json.loads(init_path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            target_data = None
        venv_dir = resolve_venv(target_data)
        python = venv_python(venv_dir)
        cmd = [python, "-m", "lingtai", "run", str(target)]

        def _tail_log(limit: int = 4000) -> str:
            try:
                data = log_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                return ""
            return data[-limit:]

        logs_dir = target / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = logs_dir / "cpr_relaunch.log"
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        quoted_cmd = " ".join(shlex.quote(str(part)) for part in cmd)

        with log_path.open("ab", buffering=0) as log_fh:
            log_fh.write(f"\n--- CPR launch {timestamp}: {quoted_cmd} ---\n".encode("utf-8"))
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                **_detached_spawn_kwargs(),
            )

        self._log("cpr_launched", target=str(target), pid=proc.pid, log=str(log_path))

        deadline = time.time() + 10.0
        while time.time() < deadline:
            if _presence_observe_alive(
                target_presence,
                wall_now=time.time(),
                threshold=3.0,
            ):
                self._log("cpr_alive", target=str(target), pid=proc.pid)
                return True
            code = proc.poll()
            if code is not None:
                tail = _tail_log()
                self._log("cpr_failed", target=str(target), pid=proc.pid, exit_code=code, log=str(log_path))
                message = f"CPR launch exited before heartbeat (exit code {code}); see {log_path}"
                if tail.strip():
                    message += f"\n\nLast log output:\n{tail}"
                return {"error": True, "message": message, "exit_code": code, "log": str(log_path)}
            time.sleep(0.2)

        self._log("cpr_timeout", target=str(target), pid=proc.pid, log=str(log_path))
        return {
            "error": True,
            "message": f"CPR launch did not produce a fresh heartbeat within 10s (pid {proc.pid}); see {log_path}",
            "pid": proc.pid,
            "log": str(log_path),
        }

    def start(self) -> None:
        super().start()
        # LICC poller: watch .mcp_inbox/ for events from out-of-process MCPs.
        from .services.mcp_inbox import MCPInboxPoller
        self._mcp_inbox_poller = MCPInboxPoller(self)
        self._mcp_inbox_poller.start()

    def _expand_agent_placeholders(self, value):
        """Substitute per-agent placeholders in an MCP launch string.

        Lets a single shared MCP registry template scope each agent to its own
        namespace without per-agent hand-editing — e.g. a NoKV workbench root
        ``--workbench-root /agents/{agent_id}/wb``. ``{agent_id}`` and
        ``{agent_address}`` resolve to the agent's stable working-dir name (its
        address); ``{agent_dir}`` resolves to the absolute working directory.
        Non-string values and strings without a placeholder pass through
        unchanged, so ordinary MCP args are never touched.
        """
        if not isinstance(value, str) or "{" not in value:
            return value
        agent_id = self._working_dir.name
        return (
            value.replace("{agent_id}", agent_id)
            .replace("{agent_address}", agent_id)
            .replace("{agent_dir}", str(self._working_dir))
        )

    @staticmethod
    def _mcp_client_is_connected(client: Any) -> bool:
        try:
            probe = getattr(client, "is_connected", None)
            return bool(callable(probe) and probe())
        except Exception:
            return False

    def _ensure_mcp_activation_state(self) -> None:
        """Initialize private activation state for legacy `__new__` callers."""
        if not hasattr(self, "_mcp_activation_lock"):
            self._mcp_activation_lock = threading.RLock()
        if not hasattr(self, "_mcp_lifecycle_generation"):
            self._mcp_lifecycle_generation = 0
        if not hasattr(self, "_mcp_lifecycle_state"):
            self._mcp_lifecycle_state = "active"
        if not hasattr(self, "_mcp_lifecycle_barrier"):
            self._mcp_lifecycle_barrier = threading.Event()
        if not hasattr(self, "_mcp_stop_requested"):
            self._mcp_stop_requested = threading.Event()
        if not hasattr(self, "_mcp_refresh_owner_thread"):
            self._mcp_refresh_owner_thread = None
        if not hasattr(self, "_mcp_refresh_handoff_committed"):
            self._mcp_refresh_handoff_committed = False
        if not hasattr(self, "_mcp_pending_retirements"):
            self._mcp_pending_retirements = {}
        if not hasattr(self, "_mcp_reserved_activation_tokens"):
            self._mcp_reserved_activation_tokens = set()
        if not hasattr(self, "_mcp_clients"):
            self._mcp_clients = []
        if not hasattr(self, "_mcp_clients_by_tool"):
            self._mcp_clients_by_tool = {}
        if not hasattr(self, "_mcp_tool_names"):
            self._mcp_tool_names = set()
        if not hasattr(self, "_mcp_init_specs"):
            self._mcp_init_specs = {}
        if not hasattr(self, "_intrinsics"):
            self._intrinsics = {}
        if not hasattr(self, "_tool_handlers"):
            self._tool_handlers = {}
        if not hasattr(self, "_tool_schemas"):
            self._tool_schemas = []
        if not hasattr(self, "_sealed"):
            self._sealed = False
        if not hasattr(self, "_session"):
            self._session = SimpleNamespace(chat=None)
        if not hasattr(self, "_token_decomp_dirty"):
            self._token_decomp_dirty = False

    def _make_mcp_handler(self, client: Any, tool_name: str):
        def handler(tool_args: dict) -> Any:
            return client.call_tool(tool_name, tool_args)

        handler._lingtai_mcp_client = client  # type: ignore[attr-defined]
        handler._lingtai_mcp_tool_name = tool_name  # type: ignore[attr-defined]
        return handler

    @staticmethod
    def _validate_mcp_schema(name: str, schema: Any) -> dict:
        if schema is None:
            schema = {}
        if not isinstance(schema, dict):
            raise TypeError(f"MCP tool {name!r} schema must be an object")
        prepared = copy.deepcopy(schema)
        prepared.pop("additionalProperties", None)
        properties = prepared.get("properties")
        if properties is not None and not isinstance(properties, dict):
            raise TypeError(f"MCP tool {name!r} schema properties must be an object")
        required = prepared.get("required")
        if required is not None and (
            not isinstance(required, list)
            or not all(isinstance(item, str) for item in required)
        ):
            raise TypeError(
                f"MCP tool {name!r} schema required must be an array of strings"
            )
        try:
            json.dumps(prepared)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"MCP tool {name!r} schema is not JSON serializable"
            ) from exc
        return prepared

    def _prepare_mcp_candidate(self, client: Any) -> list[tuple[str, Any, Any]]:
        """Validate the complete advertised catalog before any publication."""
        from lingtai.kernel.llm import FunctionSchema

        tools = client.list_tools()
        if not isinstance(tools, list):
            raise TypeError("MCP tools/list result must be an array")
        prepared = []
        seen: set[str] = set()
        for index, tool in enumerate(tools):
            if not isinstance(tool, dict):
                raise TypeError(f"MCP tool at index {index} must be an object")
            name = tool.get("name")
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"MCP tool at index {index} has an invalid name")
            if name in seen:
                raise ValueError(f"duplicate MCP candidate tool name: {name!r}")
            seen.add(name)
            description = tool.get("description", "")
            if description is None:
                description = ""
            if not isinstance(description, str):
                raise TypeError(f"MCP tool {name!r} description must be a string")
            schema = self._validate_mcp_schema(name, tool.get("schema", {}))
            prepared.append(
                (
                    name,
                    self._make_mcp_handler(client, name),
                    FunctionSchema(
                        name=name,
                        description=description,
                        parameters=schema,
                    ),
                )
            )
        return prepared

    def _preflight_mcp_candidate(
        self,
        prepared: list[tuple[str, Any, Any]],
        *,
        init_spec_name: str | None = None,
        predecessor: Any | None = None,
        reserved_activation_token: object | None = None,
    ) -> None:
        """Reject built-in, foreign, or inconsistent ownership before commit."""
        collisions = []
        for name, _, _ in prepared:
            # The Telegram composition root is the only explicit reclaim seam
            # for these reserved public names. Generic/direct MCP candidates
            # cannot acquire them merely by advertising the same string.
            if (
                name in {"telegram", "task_card"}
                and reserved_activation_token is not None
                and reserved_activation_token
                in self._mcp_reserved_activation_tokens
            ):
                continue
            if name in self._intrinsics or name in {
                "bash",
                "telegram",
                "task_card",
            }:
                collisions.append((name, "built-in/reserved"))
                continue
            handler = self._tool_handlers.get(name)
            schemas = [schema for schema in self._tool_schemas if schema.name == name]
            owner = self._mcp_clients_by_tool.get(name)
            if handler is None and not schemas and owner is None:
                continue
            if (
                owner is not None
                and handler is not None
                and getattr(handler, "_lingtai_mcp_client", None) is owner
                and len(schemas) == 1
                and name in self._mcp_tool_names
                and self._mcp_clients.count(owner) == 1
                and self._mcp_client_is_connected(owner)
            ):
                collisions.append((name, "healthy foreign MCP"))
            elif handler is not None and owner is None:
                collisions.append((name, "built-in handler"))
            else:
                collisions.append((name, "inconsistent/unowned"))
        if collisions:
            detail = ", ".join(f"{name!r} ({kind})" for name, kind in collisions)
            raise RuntimeError(f"MCP candidate tool collision: {detail}")

    def _validate_mcp_predecessor_projection(
        self, spec_name: str, client: Any
    ) -> _MCPPredecessorProjection:
        """Validate every predecessor projection before irreversible mutation."""
        spec = self._mcp_init_specs.get(spec_name)
        if not isinstance(spec, dict) or spec.get("client") is not client:
            raise RuntimeError("MCP predecessor is not exact for init spec")
        if self._mcp_clients.count(client) != 1:
            raise RuntimeError("MCP predecessor client-list membership is inconsistent")
        if set(self._mcp_clients_by_tool) != set(self._mcp_tool_names):
            raise RuntimeError("MCP predecessor owner/name-set projection is inconsistent")

        owner_names = {
            name
            for name, owner in self._mcp_clients_by_tool.items()
            if owner is client
        }
        handler_names = {
            name
            for name, handler in self._tool_handlers.items()
            if getattr(handler, "_lingtai_mcp_client", None) is client
        }
        if owner_names != handler_names:
            raise RuntimeError("MCP predecessor handler/owner projection is inconsistent")
        for name in owner_names:
            if name not in self._mcp_tool_names:
                raise RuntimeError(
                    f"MCP predecessor tool-name projection is missing {name!r}"
                )
            schemas = [schema for schema in self._tool_schemas if schema.name == name]
            if len(schemas) != 1:
                raise RuntimeError(
                    f"MCP predecessor schema multiplicity is inconsistent for {name!r}"
                )
            handler = self._tool_handlers.get(name)
            if getattr(handler, "_lingtai_mcp_tool_name", None) != name:
                raise RuntimeError(
                    f"MCP predecessor handler capture is inconsistent for {name!r}"
                )
        return _MCPPredecessorProjection(
            spec_name=spec_name,
            client=client,
            tool_names=frozenset(owner_names),
        )

    def _depublish_mcp_client(
        self,
        client: Any,
        *,
        init_spec_name: str | None = None,
        projection: _MCPPredecessorProjection | None = None,
    ) -> None:
        """Irreversibly remove one predecessor from every public projection."""
        names = set(projection.tool_names) if projection is not None else {
            name
            for name, owner in self._mcp_clients_by_tool.items()
            if owner is client
        }
        if projection is None:
            names.update(
                name
                for name, handler in self._tool_handlers.items()
                if getattr(handler, "_lingtai_mcp_client", None) is client
            )
        self._tool_handlers = {
            name: handler
            for name, handler in self._tool_handlers.items()
            if name not in names
        }
        self._tool_schemas = [
            schema for schema in self._tool_schemas if schema.name not in names
        ]
        self._mcp_clients_by_tool = {
            name: owner
            for name, owner in self._mcp_clients_by_tool.items()
            if owner is not client
        }
        self._mcp_tool_names.difference_update(names)
        self._mcp_clients = [
            existing for existing in self._mcp_clients if existing is not client
        ]
        self._token_decomp_dirty = True

    @staticmethod
    def _close_mcp_candidate(client: Any) -> None:
        """Close and verify bounded retirement; repeated close remains safe."""
        client.close()
        thread = getattr(client, "_thread", None)
        if thread is not None and thread.is_alive():
            raise RuntimeError("MCP client thread is still alive after close")
        # Production MCP clients expose `_closed`; verify their public health
        # projection as well. Lightweight host/test clients may implement only
        # close() and an optimistic health probe, so thread retirement is the
        # portable minimum for those legacy adapters.
        if hasattr(client, "_closed"):
            try:
                connected = bool(client.is_connected())
            except Exception:
                connected = False
            if connected:
                raise RuntimeError("MCP client is still connected after close")

    def _record_pending_mcp_retirement(self, key: str, client: Any) -> None:
        existing = self._mcp_pending_retirements.get(key)
        if existing is not None and existing is not client:
            raise RuntimeError(
                f"MCP retirement key {key!r} already owns another client"
            )
        self._mcp_pending_retirements[key] = client
        spec = self._mcp_init_specs.get(key)
        if isinstance(spec, dict):
            spec["client"] = client

    def _drain_pending_mcp_retirements_locked(self) -> dict[str, str]:
        """Retry every keyed close without discarding ownership on failure."""
        unresolved: dict[str, str] = {}
        for key, client in list(self._mcp_pending_retirements.items()):
            try:
                self._close_mcp_candidate(client)
            except Exception as exc:
                unresolved[key] = f"{type(exc).__name__}: {exc}"
                continue
            self._mcp_pending_retirements.pop(key, None)
            spec = self._mcp_init_specs.get(key)
            if isinstance(spec, dict) and spec.get("client") is client:
                spec["client"] = None
        return unresolved

    def _assert_mcp_activation_outcome(
        self, spec_name: str, outcome: MCPActivationOutcome
    ) -> None:
        """Require exact, duplicate-free equality across every projection."""
        client = outcome.client
        spec = self._mcp_init_specs.get(spec_name)
        if not isinstance(spec, dict) or spec.get("client") is not client:
            raise RuntimeError("MCP activation outcome does not own its init spec")
        if self._mcp_clients.count(client) != 1:
            raise RuntimeError("MCP activation outcome client-list mismatch")
        outcome_names = tuple(outcome.tool_names)
        if len(outcome_names) != len(set(outcome_names)):
            raise RuntimeError("MCP activation outcome contains duplicate tool names")
        expected = set(outcome_names)
        global_names = set(self._mcp_tool_names)
        owner_names = set(self._mcp_clients_by_tool)
        tagged_handlers = {
            name: handler
            for name, handler in self._tool_handlers.items()
            if getattr(handler, "_lingtai_mcp_client", None) is not None
        }
        live_client_ids = {id(existing) for existing in self._mcp_clients}
        if len(live_client_ids) != len(self._mcp_clients):
            raise RuntimeError("MCP activation global client-list identity mismatch")
        if not (
            owner_names == global_names == set(tagged_handlers)
        ):
            raise RuntimeError(
                "MCP activation global owner/handler/name-set projection mismatch"
            )
        for name in global_names:
            owner = self._mcp_clients_by_tool[name]
            handler = tagged_handlers[name]
            handler_owner = getattr(handler, "_lingtai_mcp_client", None)
            handler_tool_name = getattr(handler, "_lingtai_mcp_tool_name", None)
            if (
                id(owner) not in live_client_ids
                or handler_owner is not owner
                or handler_tool_name != name
                or sum(schema.name == name for schema in self._tool_schemas) != 1
            ):
                raise RuntimeError(
                    f"MCP activation global schema/client mismatch for {name!r}"
                )
        outcome_owner_names = {
            name
            for name, owner in self._mcp_clients_by_tool.items()
            if owner is client
        }
        handler_names = {
            name
            for name, handler in self._tool_handlers.items()
            if getattr(handler, "_lingtai_mcp_client", None) is client
        }
        name_names = outcome_owner_names & self._mcp_tool_names
        if not (
            expected == outcome_owner_names == handler_names == name_names
        ):
            raise RuntimeError(
                "MCP activation outcome/projection tool-name mismatch"
            )
        for name in expected:
            if sum(schema.name == name for schema in self._tool_schemas) != 1:
                raise RuntimeError(f"MCP activation schema mismatch for {name!r}")

    def _activate_mcp_candidate(
        self,
        client: Any,
        *,
        allow_sealed: bool = False,
        init_spec_name: str | None = None,
        predecessor: Any | None = None,
        reserved_activation_token: object | None = None,
    ) -> MCPActivationOutcome:
        """Retire an exact dead predecessor, then atomically publish a candidate."""
        self._ensure_mcp_activation_state()
        requested_generation = self._mcp_lifecycle_generation
        wait_hook = getattr(self, "_mcp_activation_wait_hook", None)
        if callable(wait_hook):
            wait_hook()
        with self._mcp_activation_lock:
            retired = False
            chat_reconcile_required = False
            candidate_key = init_spec_name or f"candidate:{id(client)}"
            try:
                if requested_generation != self._mcp_lifecycle_generation:
                    raise RuntimeError("MCP activation invalidated by lifecycle transition")
                refresh_owner = (
                    self._mcp_lifecycle_state == "refreshing"
                    and self._mcp_refresh_owner_thread == threading.get_ident()
                )
                if (
                    self._mcp_stop_requested.is_set()
                    or self._mcp_lifecycle_state != "active" and not refresh_owner
                ):
                    raise RuntimeError(
                        f"MCP activation blocked while lifecycle is "
                        f"{self._mcp_lifecycle_state}"
                    )
                init_spec = None
                if init_spec_name is not None:
                    init_spec = self._mcp_init_specs.get(init_spec_name)
                    if (
                        not isinstance(init_spec, dict)
                        or init_spec.get("client") is not predecessor
                    ):
                        raise RuntimeError(
                            "MCP init spec is not exact for activation predecessor"
                        )
                elif predecessor is not None:
                    raise RuntimeError("MCP predecessor requires an init spec")
                unresolved = self._drain_pending_mcp_retirements_locked()
                if unresolved:
                    detail = ", ".join(
                        f"{name}: {error}" for name, error in unresolved.items()
                    )
                    raise RuntimeError(
                        f"MCP pending retirement remains unresolved: {detail}"
                    )
                if self._sealed and not (allow_sealed and init_spec_name):
                    raise RuntimeError("Cannot modify tools after start()")
                if predecessor is not None:
                    assert init_spec_name is not None
                    projection = self._validate_mcp_predecessor_projection(
                        init_spec_name, predecessor
                    )
                    if self._mcp_client_is_connected(predecessor):
                        raise RuntimeError("MCP predecessor is still healthy")
                    self._depublish_mcp_client(
                        predecessor,
                        init_spec_name=init_spec_name,
                        projection=projection,
                    )
                    chat_reconcile_required = True
                    retired = True
                    self._record_pending_mcp_retirement(
                        init_spec_name, predecessor
                    )
                    if self._chat is not None:
                        self._chat.update_tools(self._build_tool_schemas())
                    self._close_mcp_candidate(predecessor)
                    self._mcp_pending_retirements.pop(init_spec_name, None)
                    assert init_spec is not None
                    if init_spec.get("client") is predecessor:
                        init_spec["client"] = None

                client.start()
                prepared = self._prepare_mcp_candidate(client)
                self._preflight_mcp_candidate(
                    prepared,
                    init_spec_name=init_spec_name,
                    predecessor=predecessor,
                    reserved_activation_token=reserved_activation_token,
                )
                if (
                    self._mcp_lifecycle_barrier.is_set()
                    or self._mcp_stop_requested.is_set()
                    or requested_generation != self._mcp_lifecycle_generation
                ):
                    raise RuntimeError(
                        "MCP activation interrupted by lifecycle teardown"
                    )
                names = [name for name, _, _ in prepared]
                name_set = set(names)

                handlers_before = self._tool_handlers
                schemas_before = self._tool_schemas
                owners_before = self._mcp_clients_by_tool
                clients_before = self._mcp_clients
                mcp_names_before = self._mcp_tool_names
                dirty_before = self._token_decomp_dirty

                chat_reconcile_required = True
                self._tool_schemas = [
                    schema
                    for schema in schemas_before
                    if schema.name not in name_set
                ] + [schema for _, _, schema in prepared]
                self._mcp_clients_by_tool = {
                    **owners_before,
                    **{name: client for name in names},
                }
                self._mcp_clients = [*clients_before, client]
                self._mcp_tool_names = set(mcp_names_before) | name_set
                self._tool_handlers = {
                    **handlers_before,
                    **{name: handler for name, handler, _ in prepared},
                }
                self._token_decomp_dirty = True
                if init_spec_name is not None:
                    assert init_spec is not None
                    init_spec["client"] = client
                try:
                    if self._chat is not None:
                        self._chat.update_tools(self._build_tool_schemas())
                    if not self._sealed:
                        self._maybe_setup_task_card_controller()
                        controller = getattr(self, "_task_card_controller", None)
                        task_card = self._tool_handlers.get("task_card")
                        if (
                            controller is not None
                            and getattr(task_card, "__self__", None) is controller
                        ):
                            self._mcp_clients_by_tool.pop("task_card", None)
                            self._mcp_tool_names.discard("task_card")
                    outcome = MCPActivationOutcome(
                        client=client,
                        tool_names=tuple(
                            name
                            for name in names
                            if self._mcp_clients_by_tool.get(name) is client
                        ),
                    )
                    if init_spec_name is not None:
                        self._assert_mcp_activation_outcome(
                            init_spec_name, outcome
                        )
                except Exception:
                    self._tool_handlers = handlers_before
                    self._tool_schemas = schemas_before
                    self._mcp_clients_by_tool = owners_before
                    self._mcp_clients = clients_before
                    self._mcp_tool_names = mcp_names_before
                    self._token_decomp_dirty = dirty_before
                    if init_spec_name is not None:
                        assert init_spec is not None
                        init_spec["client"] = None
                    raise
                return outcome
            except Exception as activation_error:
                compensation_error = None
                if chat_reconcile_required and self._chat is not None:
                    try:
                        self._chat.update_tools(self._build_tool_schemas())
                    except Exception as exc:
                        compensation_error = exc
                cleanup_error = None
                try:
                    self._close_mcp_candidate(client)
                except Exception as exc:
                    cleanup_error = exc
                    if (
                        candidate_key in self._mcp_pending_retirements
                        and self._mcp_pending_retirements[candidate_key] is not client
                    ):
                        candidate_key = (
                            f"{candidate_key}:candidate:{id(client)}"
                        )
                    self._record_pending_mcp_retirement(candidate_key, client)
                if compensation_error is not None or cleanup_error is not None:
                    unresolved = []
                    if compensation_error is not None:
                        unresolved.append(
                            "live-chat compensation unresolved "
                            f"({compensation_error})"
                        )
                    if cleanup_error is not None:
                        unresolved.append(
                            f"candidate cleanup unresolved ({cleanup_error})"
                        )
                    raise RuntimeError(
                        f"MCP activation failed ({activation_error}); "
                        + "; ".join(unresolved)
                    ) from activation_error
                if init_spec_name is not None:
                    spec = self._mcp_init_specs.get(init_spec_name)
                    if (
                        isinstance(spec, dict)
                        and spec.get("client") is client
                    ):
                        spec["client"] = None
                raise

    def connect_mcp(
        self,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        *,
        _allow_sealed: bool = False,
        _init_spec_name: str | None = None,
        _predecessor: Any | None = None,
        _return_activation_outcome: bool = False,
        _reserved_activation_token: object | None = None,
    ) -> list[str] | MCPActivationOutcome:
        """Connect to an MCP server and auto-register all its tools.

        Args:
            command: Executable to run (e.g., "uvx", "xhelio-spice-mcp").
            args: Arguments to the command.
            env: Environment variables for the subprocess.

        Returns:
            List of registered tool names.
        """
        from .services.mcp import MCPClient

        # Expand per-agent placeholders (e.g. {agent_id}) so a shared registry
        # template gives each agent its own scope. See _expand_agent_placeholders.
        command = self._expand_agent_placeholders(command)
        if args:
            args = [self._expand_agent_placeholders(a) for a in args]
        if env:
            env = {k: self._expand_agent_placeholders(v) for k, v in env.items()}

        client_factory = getattr(self, "_mcp_stdio_client_factory", MCPClient)
        client = client_factory(command=command, args=args, env=env)
        outcome = self._activate_mcp_candidate(
            client,
            allow_sealed=_allow_sealed,
            init_spec_name=_init_spec_name,
            predecessor=_predecessor,
            reserved_activation_token=_reserved_activation_token,
        )
        return outcome if _return_activation_outcome else list(outcome.tool_names)

    def _maybe_setup_task_card_controller(self) -> None:
        """Register the Telegram-owned public ``task_card`` controller once a
        Telegram reverse channel exists (Jason #7258/#7259).

        The controller drives the *programmable* slot of the single resident
        Telegram Task Card, sharing the one resident message with the automatic
        slot. It is Telegram MCP-owned (it drives the Telegram-owned reverse
        channel ``_lingtai_telegram_task_card``) and lives under
        ``lingtai.mcp_servers.telegram.task_card``, not in ``lingtai.tools``, so it
        carries no glossary package. This method is only the Composition Root: it
        detects the Telegram route and invokes the Telegram-owned ``setup``.
        Idempotent: a no-op when the current controller owns the public handler
        and its exact schema, and when no Telegram client is present. A full
        refresh clears those public registries but retains active controller
        watches, so reconnect re-registers that same controller rather than
        starting a second manager.
        """
        if "telegram" not in getattr(self, "_mcp_clients_by_tool", {}):
            return
        from .mcp_servers.telegram.task_card import (
            get_description as _get_task_card_description,
            get_schema as _get_task_card_schema,
            setup as _setup_task_card,
        )

        controller = getattr(self, "_task_card_controller", None)
        if controller is not None:
            schemas = [
                schema for schema in self._tool_schemas if schema.name == "task_card"
            ]
            handler = self._tool_handlers.get("task_card")
            has_public_registration = (
                getattr(handler, "__self__", None) is controller
                and len(schemas) == 1
                and schemas[0].description == _get_task_card_description()
                and schemas[0].parameters == _get_task_card_schema()
                and schemas[0].system_prompt == ""
                and schemas[0].glossary_package is None
            )
            if has_public_registration:
                return
        self._task_card_controller = _setup_task_card(self, controller=controller)

    def connect_mcp_http(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        *,
        _allow_sealed: bool = False,
        _init_spec_name: str | None = None,
        _predecessor: Any | None = None,
        _return_activation_outcome: bool = False,
        _reserved_activation_token: object | None = None,
    ) -> list[str] | MCPActivationOutcome:
        """Connect to a remote HTTP MCP server and auto-register all its tools.

        Args:
            url: HTTP endpoint of the MCP server.
            headers: HTTP headers (e.g., {"Authorization": "Bearer ..."}).

        Returns:
            List of registered tool names.
        """
        from .services.mcp import HTTPMCPClient

        client_factory = getattr(self, "_mcp_http_client_factory", HTTPMCPClient)
        client = client_factory(url=url, headers=headers)
        outcome = self._activate_mcp_candidate(
            client,
            allow_sealed=_allow_sealed,
            init_spec_name=_init_spec_name,
            predecessor=_predecessor,
            reserved_activation_token=_reserved_activation_token,
        )
        return outcome if _return_activation_outcome else list(outcome.tool_names)

    def _retire_all_mcp_clients(self, *, context: str) -> dict:
        """Depublish first, then attempt every close and retain unresolved work."""
        self._ensure_mcp_activation_state()
        with self._mcp_activation_lock:
            keyed: dict[str, Any] = dict(self._mcp_pending_retirements)
            for name, spec in self._mcp_init_specs.items():
                if isinstance(spec, dict) and spec.get("client") is not None:
                    keyed.setdefault(name, spec["client"])
            for client in self._mcp_clients:
                if not any(existing is client for existing in keyed.values()):
                    keyed[f"client:{id(client)}"] = client

            # Callable state is removed before the first potentially failing
            # close, while keyed ownership is retained until verified.
            for key, client in keyed.items():
                self._depublish_mcp_client(client)
                self._record_pending_mcp_retirement(key, client)

            unresolved = []
            retired = []
            for key, client in list(keyed.items()):
                try:
                    self._close_mcp_candidate(client)
                except Exception as exc:
                    unresolved.append(
                        {
                            "key": key,
                            "client": type(client).__name__,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                else:
                    retired.append(type(client).__name__)
                    self._mcp_pending_retirements.pop(key, None)
                    spec = self._mcp_init_specs.get(key)
                    if isinstance(spec, dict) and spec.get("client") is client:
                        spec["client"] = None
            report = {
                "context": context,
                "attempted": len(keyed),
                "retired": retired,
                "unresolved": unresolved,
            }
            if unresolved:
                self._log("mcp_cleanup_unresolved", **report)
            return report

    def stop(self, timeout: float = 5.0) -> None:
        self._ensure_mcp_activation_state()
        with self._mcp_activation_lock:
            # Stop and refresh linearize on this lock. A committed refresh may
            # finish completely; a stop that owns the lock blocks all mutation.
            self._mcp_stop_requested.set()
            self._mcp_lifecycle_barrier.set()
            self._mcp_lifecycle_generation += 1
            self._mcp_lifecycle_state = "stopping"
        # Stop LICC poller before closing MCP clients so any in-flight events
        # finish dispatching before subprocess teardown.
        poller = getattr(self, "_mcp_inbox_poller", None)
        if poller is not None:
            try:
                poller.stop()
            except Exception:
                pass

        # Cleanup is best-effort across every client. Failed closes remain in
        # the private retry list but are already depublished, so repeated stop
        # can converge without exposing a dead route.
        self._retire_all_mcp_clients(context="stop")

        super().stop(timeout=timeout)

    def has_capability(self, name: str) -> bool:
        """Check if a capability is registered."""
        from lingtai.tools.registry import canonical_capability_name
        return canonical_capability_name(name) in self._capability_managers

    def get_capability(self, name: str) -> Any:
        """Return a capability manager, accepting the retained ``bash`` alias."""
        from lingtai.tools.registry import canonical_capability_name
        return self._capability_managers.get(canonical_capability_name(name))

    # ------------------------------------------------------------------
    # Deep refresh — full reconstruct from init.json
    # ------------------------------------------------------------------

    @staticmethod
    def _strip_hidden_runtime_manifest_settings(data: dict) -> bool:
        """Remove legacy user-facing runtime knobs that are now kernel-owned.

        `manifest.stamina` used to configure an agent-visible countdown. The
        runtime now keeps only a hidden fixed idle timeout, so persisted init.json
        should not keep presenting stamina as a user-editable setting.
        """
        manifest = data.get("manifest")
        if isinstance(manifest, dict) and "stamina" in manifest:
            manifest.pop("stamina", None)
            return True
        return False

    def _read_init(self) -> dict | None:
        """Read and validate init.json from working directory.

        If ``manifest.preset.active`` is set, materialize the named preset's
        ``llm`` and ``capabilities`` into the manifest before validation. The
        running agent thus always sees a fully resolved manifest.

        On success, the resolved (secret-redacted) manifest is also published
        to ``system/manifest.resolved.json`` via
        ``lingtai.kernel.workdir.write_resolved_manifest`` (issue #259).
        """
        from .init_reader import InitReadStatus, read_init, reader_callbacks

        materialize, prepare = reader_callbacks(
            self._working_dir,
            load_preset=load_preset,
        )
        outcome = read_init(
            self._working_dir,
            materialize=materialize,
            prepare=prepare,
            failure_behavior="KEEP_PREVIOUS_EFFECTIVE",
        )
        self._last_init_read_outcome = outcome
        from lingtai.kernel.workdir import write_resolved_manifest
        if outcome.status is not InitReadStatus.READ_FAILED:
            data = outcome.data
            assert data is not None
            effective_path = write_resolved_manifest(self._working_dir, data)
            if effective_path is not None:
                outcome.effective_config_source = str(effective_path)
            else:
                self._log("resolved_manifest_write_failed")
            self._log("init_read_result", **outcome.log_fields())
            return data

        outcome.fallback_effective = (
            "previous runtime effective config preserved; "
            f"source={outcome.effective_config_source}; freshness=PREVIOUS"
        )
        self._log("init_read_result", **outcome.log_fields())
        return None

    def _activate_preset(self, name: str) -> None:
        """Substitute a preset's llm + capabilities into init.json on disk.

        `name` is the preset's path (absolute, ~-prefixed, or relative to
        working_dir). Substitutes the file's `manifest.llm` and
        `manifest.capabilities` into the agent's init.json, sets
        `manifest.preset.active = name` (storing the path string verbatim —
        no canonicalization), and writes atomically.

        Other manifest fields are preserved, except retired/hidden runtime
        knobs such as ``manifest.stamina``.

        Raises:
            KeyError: the preset file does not exist
            ValueError: the preset file is malformed or the name is invalid
            OSError: the on-disk write failed (init.json untouched)
        """
        import json
        import os

        init_path = self._working_dir / "init.json"
        data = json.loads(init_path.read_text(encoding="utf-8"))
        manifest = data.setdefault("manifest", {})

        # Use the wrapper preset-loader; this explicit activation may write the
        # selected preset, while production preset reads remain migration-free.
        preset = load_preset(name, working_dir=self._working_dir)
        preset_manifest = preset.get("manifest", {})

        preset_llm = dict(preset_manifest.get("llm") or manifest.get("llm") or {})
        # context_limit lives inside manifest.llm in the preset, but stays
        # at manifest root in init.json — strip it from the llm dict before
        # substitution and write it to the root.
        preset_ctx = preset_llm.pop("context_limit", None)
        manifest["llm"] = preset_llm
        manifest["capabilities"] = preset_manifest.get(
            "capabilities", manifest.get("capabilities", {}))
        if preset_ctx is not None:
            manifest["context_limit"] = preset_ctx

        # Set active in the umbrella. Preserve default if already set; otherwise
        # initialize default to the same value as active (first activation).
        # Also ensure `name` appears in `allowed` — _activate_preset is the
        # final gate and the manifest must remain self-consistent. The caller
        # (system._refresh) also validates against `allowed` before invoking
        # us; this is belt-and-braces for direct callers and AED auto-fallback.
        preset_block = manifest.setdefault("preset", {})
        preset_block["active"] = name
        if not preset_block.get("default"):
            preset_block["default"] = name
        allowed = preset_block.get("allowed")
        if not isinstance(allowed, list):
            preset_block["allowed"] = [name]
            self._log("preset_allowed_widened", name=name,
                      reason="allowed_field_initialized")
        elif name not in allowed:
            preset_block["allowed"] = [*allowed, name]
            self._log("preset_allowed_widened", name=name,
                      reason="direct_activate_bypassed_gate")

        self._strip_hidden_runtime_manifest_settings(data)

        # Atomic write
        tmp = init_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False),
                       encoding="utf-8")
        os.replace(str(tmp), str(init_path))

    def _activate_default_preset(self) -> None:
        """Read manifest.preset.default and activate it. Used by AED auto-fallback."""
        import json
        data = json.loads((self._working_dir / "init.json").read_text(encoding="utf-8"))
        preset = data.get("manifest", {}).get("preset") or {}
        default_name = preset.get("default")
        if not default_name:
            raise RuntimeError("no default preset configured")
        self._activate_preset(default_name)

    def _setup_from_init(self) -> None:
        """Serialize the complete deep-refresh reconstruction lifecycle."""
        self._ensure_mcp_activation_state()
        with self._mcp_activation_lock:
            if (
                self._mcp_stop_requested.is_set()
                or self._mcp_lifecycle_barrier.is_set()
                or self._mcp_lifecycle_state != "active"
            ):
                raise RuntimeError("deep refresh blocked by pending stop")
            self._mcp_lifecycle_barrier.set()
            self._mcp_lifecycle_generation += 1
            self._mcp_lifecycle_state = "refreshing"
            self._mcp_refresh_owner_thread = threading.get_ident()
            self._mcp_refresh_handoff_committed = False
            # The generation invalidates waiters. The owning refresh may
            # activate replacement clients while holding the same RLock.
            self._mcp_lifecycle_barrier.clear()
            try:
                self._setup_from_init_locked()
            finally:
                self._mcp_refresh_owner_thread = None
                self._mcp_lifecycle_state = "active"
                self._mcp_lifecycle_barrier.clear()

    def _setup_from_init_locked(self) -> None:
        """Full construct/reconstruct from init.json."""
        self._log("refresh_start")

        data = self._read_init()
        if data is None:
            self._log("refresh_skipped", reason="no valid init.json")
            return

        from lingtai.kernel.config_resolve import (
            load_env_file,
            resolve_env,
            resolve_file,
            _resolve_capabilities,
        )
        env_file = data.get("env_file")
        import os

        overwrite_env_file = os.environ.get("LINGTAI_REFRESH_ENV_OVERWRITE") == "1"
        if env_file:
            load_env_file(env_file, overwrite=overwrite_env_file)
        if overwrite_env_file:
            os.environ.pop("LINGTAI_REFRESH_ENV_OVERWRITE", None)

        # Resolve *_file fields for active top-level text content.
        # The externally changeable prompt surface is exactly `base_prompt`,
        # `covenant`, and `comment` (plus the agent identity/state fields
        # `lingtai` and `pad`). `lingtai` is the agent's configured 灵台 /
        # character value (system/lingtai.md → `character` section), distinct from
        # `base_prompt` (third-party injection point); it was renamed from
        # `prompt` with no legacy alias. Retired prompt-override `_file` fields
        # (principle_file / procedures_file / substrate_file / brief_file) are
        # legacy-known and intentionally not resolved here.
        # Note: "soul" / "soul_file" were retired in v0.7.6 and remain
        # compatibility-known; they are intentionally not resolved here;
        # the shared reader reports them without rewriting init.json.
        for key in ("covenant", "base_prompt",
                    "pad", "lingtai", "comment"):
            file_key = f"{key}_file"
            if file_key in data:
                data[key] = resolve_file(data.get(key), data.pop(file_key))

        m = data["manifest"]

        # Save conversation history
        saved_interface = None
        if self._session.chat is not None:
            saved_interface = self._session.chat.interface

        # Tear down
        # Cancel soul timer to prevent racing on config/service during rebuild
        self._cancel_soul_timer()

        cleanup = self._retire_all_mcp_clients(context="deep_refresh")
        if cleanup["unresolved"]:
            raise RuntimeError(
                "deep refresh blocked by unresolved MCP cleanup: "
                + "; ".join(item["error"] for item in cleanup["unresolved"])
            )

        self._sealed = False
        self._tool_handlers.clear()
        self._tool_schemas.clear()
        self._capabilities.clear()
        self._capability_managers.clear()

        self._intrinsics.clear()
        self._intrinsic_modules.clear()
        self._wire_intrinsics()

        # Reset capability-owned flags (email.boot below resets to "email box"/"email")
        self._mailbox_name = "email box"
        self._mailbox_tool = "email"
        if hasattr(self, "_post_molt_hooks"):
            self._post_molt_hooks.clear()

        # Reset prompt manager
        self._prompt_manager._sections.clear()

        # Reconstruct LLM service if changed
        llm = m["llm"]
        api_key = resolve_env(llm.get("api_key"), llm.get("api_key_env"))
        new_provider = llm["provider"]
        new_model = llm["model"]
        new_base_url = llm.get("base_url")

        # Default 60 matches AgentConfig.max_rpm — existing agents whose
        # init.json predates this field cooperatively share the network-wide
        # 60 RPM cap by default. Set to 0 in init.json to disable gating.
        new_max_rpm = m.get("max_rpm", 60)
        # Pass working_dir so a Codex agent's per-agent session/thread identity
        # (the agent path) is resolved into the provider defaults. The agent
        # path anchor is stable across refresh, but #406 makes start/refresh one
        # of the two cache-affinity rotate triggers: the Codex adapter stamps a
        # fresh epoch at construction, so a live refresh must REBUILD the adapter
        # to rotate the current id (see codex_force_rebuild below).
        new_provider_defaults = build_provider_defaults_from_manifest_llm(
            llm, max_rpm=new_max_rpm, working_dir=self._working_dir
        )

        new_provider_defaults_bucket = (new_provider_defaults or {}).get(
            new_provider.lower(), {}
        )
        cur_provider_defaults_bucket = getattr(
            self.service, "_provider_defaults", {}
        ).get(new_provider.lower(), {})
        # Codex start/refresh is a cache-affinity rotate trigger (Jason's final
        # #406 semantics): the adapter epoch-stamps the *current* affinity id at
        # construction, so the 8-hit stalled-cache shuffle only becomes active
        # once a live refresh REBUILDS the adapter at a fresh epoch. The agent
        # path anchor is stable across refresh, so the provider-defaults bucket
        # is byte-identical and the boot-epoch adapter (with its stale current
        # id) would otherwise survive untouched — exactly the bug where the
        # shuffle never went live. A *live* refresh is one that replays existing
        # history (``saved_interface is not None``); boot has no prior session
        # and already builds a fresh adapter, so it is excluded to avoid a
        # redundant double-build.
        codex_force_rebuild = (
            new_provider.lower() == "codex" and saved_interface is not None
        )
        # Compare the resolved provider-defaults bucket as a whole so explicit
        # init.json changes (codex_session_anchor, default_headers,
        # compact_threshold, max_rpm, api_compat, etc.) rebuild coherently.
        if (
            codex_force_rebuild
            or new_provider != self.service.provider
            or new_model != self.service.model
            or new_base_url != getattr(self.service, "_base_url", None)
            or new_provider_defaults_bucket != cur_provider_defaults_bucket
        ):
            self.service = LLMService(
                provider=new_provider, model=new_model,
                api_key=api_key, base_url=new_base_url,
                provider_defaults=new_provider_defaults,
            )
            self._session._llm_service = self.service

        # Reload admin from init.json (avatars have admin: {}, not inherited from parent)
        self._admin = m.get("admin", {})

        # Reload config by overlaying explicit init.json values onto
        # AgentConfig defaults. Stale max_turns and molt_* manifest values stay
        # deliberately ignored inside build_agent_config.
        self._config = build_agent_config(m, max_rpm=new_max_rpm)
        self._soul_delay = max(1.0, self._config.soul_delay)
        self._session._config = self._config

        # Reload large-result notification threshold from init.json.
        # Default 3000; 0 disables notifications entirely.  An explicit
        # manifest value overrides the default (config override preserved).
        raw_threshold = m.get("summarize_notification_threshold")
        if isinstance(raw_threshold, int) and not isinstance(raw_threshold, bool) and raw_threshold >= 0:
            self._summarize_notification_threshold = raw_threshold
        else:
            self._summarize_notification_threshold = 3000

        # Reload all prompt sections (covenant, character, principle,
        # procedures, brief, rules, pad, comment) from init.json and disk.
        self._reload_prompt_sections(data)

        # Re-boot psyche so the post-molt hook is re-registered on the cleared
        # hook list. `boot` also reloads `character`/`pad` — both `boot` and
        # `_reload_prompt_sections` now route through the same canonical
        # composers (`_lingtai_load`, `_pad_load`), so they produce identical
        # content and the result is independent of which runs last.
        from lingtai.tools import psyche as _psyche
        _psyche.boot(self)

        # Re-boot email so a fresh EmailManager + scheduler thread are wired.
        # ``email.boot`` stops the previous manager's scheduler before
        # starting a new one — without that, the prior daemon thread keeps
        # polling ``mailbox/schedules/*/schedule.json`` and races the new
        # thread, double-sending the same due tick (issue #154).
        from lingtai.tools import email as _email
        _email.boot(self)

        # Decompress addons BEFORE capability setup so the `mcp` capability
        # sees the populated registry on its first reconcile.
        addons = data.get("addons") or []
        if addons:
            try:
                from .services.mcp_registry import decompress_addons
                report = decompress_addons(self._working_dir, addons)
                self._log("mcp_decompress", **report)
            except Exception as e:
                self._log("mcp_decompress_failed", reason=str(e))

        # Re-run capability setup. init.json declares overrides/opt-ins;
        # `apply_core_defaults` ensures the always-on tool floor boots even
        # when the manifest omits it. `manifest.disable` and `"name": null`
        # entries are the opt-out channels.
        raw_caps = m.get("capabilities", {}) or {}
        resolved = _resolve_capabilities(raw_caps)
        # Preserve null sentinels through env-resolution (it converts None to {}).
        null_outs = {n for n, v in raw_caps.items() if v is None}

        from lingtai.tools.registry import (
            _GROUPS,
            apply_core_defaults,
            normalize_capabilities,
        )
        expanded: dict[str, Any] = {}
        for name, cap_kwargs in resolved.items():
            if name in _GROUPS:
                for sub in _GROUPS[name]:
                    expanded[sub] = {}
            elif name in null_outs:
                expanded[name] = None
            elif cap_kwargs is None:
                expanded[name] = None
            else:
                expanded[name] = cap_kwargs
        normalized = normalize_capabilities(
            {n: (v if v is not None else {}) for n, v in expanded.items()}
        )
        for n, v in expanded.items():
            if v is None:
                normalized[n] = None  # type: ignore[assignment]

        disable_list = m.get("disable") or []
        capabilities = apply_core_defaults(normalized, disable=disable_list)

        if capabilities:
            for name, cap_kwargs in capabilities.items():
                try:
                    self._setup_capability(name, **cap_kwargs)
                except (ValueError, ImportError, TypeError) as e:
                    self._log("capability_skipped", capability=name, reason=str(e))

        # Install intrinsic manuals (wipe-and-rewrite .library/intrinsic/)
        # from the bundles shipped with each enabled capability.
        self._install_intrinsic_manuals()

        # Register system prompt reload as post-molt hook — molt should
        # reconstruct the system prompt the same way refresh does.
        if not hasattr(self, "_post_molt_hooks"):
            self._post_molt_hooks = []
        self._post_molt_hooks.append(self._reload_prompt_sections)

        # Reload MCP
        self._load_mcp_from_workdir()

        # Persist LLM config
        self._persist_llm_config()

        # Re-write manifest and identity
        self._update_identity()

        # Re-seal
        self._sealed = True

        # Rebuild session with preserved history
        if saved_interface is not None:
            self._session._rebuild_session(saved_interface)

        self._log(
            "refresh_complete",
            capabilities=[name for name, _ in self._capabilities],
            tools=list(self._tool_handlers.keys()),
        )

    def _reload_prompt_sections(self, data: dict | None = None) -> None:
        """Re-read all prompt sections from init.json and disk.

        Called by _setup_from_init() on refresh (with pre-resolved data) and
        as a post-molt hook (no args — re-reads init.json from scratch).
        Ensures the system prompt after molt is identical to after refresh.
        """
        if data is None:
            data = self._read_init()
            if data is None:
                return
            # Resolve active *_file fields (covenant_file, base_prompt_file,
            # lingtai_file, comment_file). Retired prompt-override `_file` fields
            # are legacy-known and not resolved — see _setup_from_init.
            from lingtai.kernel.config_resolve import resolve_file
            for key in ("covenant", "base_prompt", "pad", "lingtai", "comment"):
                file_key = f"{key}_file"
                if file_key in data:
                    data[key] = resolve_file(data.get(key), data.pop(file_key))

        system_dir = self._working_dir / "system"
        system_dir.mkdir(exist_ok=True)

        # --- Base prompt (third-party prompt injection point) ---
        # `base_prompt` is the init-prompt contract's third-party (application /
        # recipe / preset) system-prompt injection point — one of the three
        # externally changeable prompt surfaces (with `covenant` and `comment`).
        # It is NOT a prompt-manager section: the kernel builder renders it right
        # after the raw kernel-owned `principle` section and before the rest of
        # Batch 1 (see lingtai.kernel.prompt.build_system_prompt_batches), so it
        # is threaded through `self._base_prompt` and passed to the builder by
        # `_build_system_prompt` / `_build_system_prompt_batches`.
        #
        # Resolution precedence:
        #   1. data["base_prompt"]      — inline init.json string (already merged
        #                                 with base_prompt_file by _setup_from_init)
        #   2. system/base_prompt.md    — on-disk mirror (fallback)
        # The on-disk mirror lets the resolved injection survive a post-molt
        # reload that re-reads init.json from scratch and lets operators inspect
        # what is actually injected.
        base_prompt = data.get("base_prompt", "")
        base_prompt_file = system_dir / "base_prompt.md"
        if base_prompt:
            base_prompt_file.write_text(base_prompt)
        elif base_prompt_file.is_file():
            base_prompt = base_prompt_file.read_text(encoding="utf-8")
        self._base_prompt = base_prompt or ""

        # --- Covenant (operator contract — covenant.md alone) ---
        covenant = data.get("covenant", "")
        covenant_file = system_dir / "covenant.md"
        if covenant:
            covenant_file.write_text(covenant)
        elif covenant_file.is_file():
            covenant = covenant_file.read_text(encoding="utf-8")
        if covenant:
            self._prompt_manager.write_section("covenant", covenant, protected=True)

        # --- Character (configured or self-authored identity — system/lingtai.md alone) ---
        # `lingtai` is the configured 灵台 / character value: a value supplied inline or
        # resolved from `lingtai_file` before this point is authoritative when nonempty and replaces
        # system/lingtai.md during boot, refresh, and post-molt reconstruction.
        # An absent or empty resolved value selects self-evolve mode and leaves
        # the existing system/lingtai.md untouched. In either mode the
        # canonical composer loads that file into `character`. This is the
        # agent's OWN voice — distinct from `covenant` above, from the
        # third-party `base_prompt` injection point, and from the mechanical
        # `identity` section written by BaseAgent. (Renamed from `prompt`; no
        # legacy alias.)
        lingtai_seed = data.get("lingtai", "")
        if lingtai_seed:
            (system_dir / "lingtai.md").write_text(lingtai_seed)
        # Delegate to the single canonical composer so boot/refresh/molt all
        # produce byte-identical `character` content and no longer depend on
        # post-molt hook ordering.
        from lingtai.tools.psyche import _lingtai_load
        _lingtai_load(self, {})

        # --- Substrate (kernel-owned, cross-app stable; #39) ---
        # The substrate section sits right after `## tools` and describes
        # the agent's architecture to itself (tool tiers, data-flow
        # topology, life states, channel discipline, attention model).
        #
        # Substrate is kernel-owned: under the init-prompt contract it is NOT an
        # external override. Legacy init.json `substrate` / `substrate_file`
        # values remain compatibility-known, are reported by the shared reader,
        # and are ignored here; the packaged default wins on every boot/refresh.
        #
        # Resolution order:
        #   1. packaged prompts/substrate/substrate.md — kernel default, refreshed on boot
        #   2. system/substrate.md           — fallback only if package missing
        #
        # The packaged default overwrites the on-disk file on every boot so
        # that `pip install -e .` + `system(refresh)` actually propagates
        # kernel updates. The on-disk file is a mirror/debug artifact.
        #
        # The packaged source carries skill-style YAML frontmatter (developer-
        # facing metadata: purpose/summary/audience). The mirror keeps that
        # frontmatter so the on-disk artifact stays self-explanatory, but only
        # the Markdown body is written into the prompt section — the rendered
        # LLM prompt and final system.md must be body-only.
        substrate = ""
        substrate_file = system_dir / "substrate.md"
        try:
            from importlib.resources import files
            packaged = files("lingtai.prompts").joinpath("substrate/substrate.md").read_text(encoding="utf-8")
            substrate_file.write_text(packaged)
            substrate = _strip_frontmatter(packaged)
        except (FileNotFoundError, ModuleNotFoundError, OSError):
            if substrate_file.is_file():
                substrate = _strip_frontmatter(substrate_file.read_text(encoding="utf-8"))
            else:
                substrate = ""
        if substrate:
            self._prompt_manager.write_section("substrate", substrate, protected=True)
        else:
            self._prompt_manager.delete_section("substrate")

        # --- Rules (from system/rules.md, not init.json) ---
        rules_md = system_dir / "rules.md"
        if rules_md.is_file():
            try:
                rules_content = rules_md.read_text(encoding="utf-8").strip()
                if rules_content:
                    self._prompt_manager.write_section("rules", rules_content, protected=True)
                else:
                    self._prompt_manager.delete_section("rules")
            except OSError:
                pass
        else:
            self._prompt_manager.delete_section("rules")

        # --- Pad (pad.md + pinned pad_append.json references) ---
        # Delegate to the single canonical composer rather than re-reading
        # pad.md alone — otherwise the post-molt hook ordering silently drops
        # the pinned append references. `_pad_load` composes both.
        from lingtai.tools.psyche import _pad_load
        _pad_load(self, {})

        # --- Principle (kernel-owned top-level progressive-disclosure contract) ---
        # The principle section is LingTai-owned, not operator-owned: init.json
        # `principle` / `principle_file` values are intentionally ignored here.
        #
        # Resolution order:
        #   1. packaged prompts/principle/principle.md — kernel default, refreshed on boot
        #   2. system/principle.md          — fallback only if package missing
        #
        # The packaged default owns the raison d'être of the resident prompt
        # layers (meta_guidance/procedures/substrate/references) so the rule does
        # not drift across files. The on-disk file is a mirror/debug artifact.
        # As with substrate, the packaged source carries developer-facing YAML
        # frontmatter; the mirror keeps it but the prompt section gets body-only.
        principle = ""
        principle_file = system_dir / "principle.md"
        try:
            from importlib.resources import files
            packaged = files("lingtai.prompts").joinpath("principle/principle.md").read_text(encoding="utf-8")
            principle_file.write_text(packaged)
            principle = _strip_frontmatter(packaged)
        except (FileNotFoundError, ModuleNotFoundError, OSError):
            if principle_file.is_file():
                principle = _strip_frontmatter(principle_file.read_text(encoding="utf-8"))
        if principle:
            self._prompt_manager.write_section("principle", principle, protected=True)
        else:
            self._prompt_manager.delete_section("principle")

        # --- Procedures ---
        # Kernel-owned resident procedures. Legacy init.json procedures values
        # remain compatibility-known, are reported by the shared reader, and
        # are ignored here; the packaged default wins on every boot/refresh.
        # system/procedures.md is only a packaged
        # mirror/debug artifact, and is read as fallback if the package
        # resource is unavailable.
        # Packaged source carries developer-facing YAML frontmatter; the mirror
        # keeps it but the prompt section gets body-only.
        procedures = ""
        procedures_file = system_dir / "procedures.md"
        try:
            from importlib.resources import files
            packaged = files("lingtai.prompts").joinpath("procedures/procedures.md").read_text(encoding="utf-8")
            procedures_file.write_text(packaged)
            procedures = _strip_frontmatter(packaged)
        except (FileNotFoundError, ModuleNotFoundError, OSError):
            if procedures_file.is_file():
                procedures = _strip_frontmatter(procedures_file.read_text(encoding="utf-8"))
            else:
                procedures = ""
        if procedures:
            self._prompt_manager.write_section("procedures", procedures, protected=True)
        else:
            self._prompt_manager.delete_section("procedures")

        # --- Runtime guidance mirror ---
        # `_meta.agent_meta.guidance` is latest-only tool-result metadata, but the TUI
        # also needs a filesystem-visible copy. Runtime guidance is now authored
        # as a skill-style Markdown catalog (lingtai/prompts/meta_guidance/catalog/INDEX.md +
        # <id>.md sections); the kernel assembles it into the same dict shape and
        # we serialize a *derived* `system/guidance.json` here for back-compat
        # with TUI/Portal consumers (schema_version is an int; ids are stable).
        guidance_file = system_dir / "guidance.json"
        try:
            from lingtai.kernel.meta_block import validate_runtime_guidance
            from lingtai.kernel.prompt_catalog import load_guidance_catalog

            guidance_payload = load_guidance_catalog()
            validate_runtime_guidance(guidance_payload)
            guidance_file.write_text(json.dumps(guidance_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        except Exception:
            if not guidance_file.is_file():
                guidance_file.write_text("{}\n", encoding="utf-8")
        # --- Brief (secretary-maintained life context — disk only) ---
        # Under the init-prompt contract `brief` is no longer an external
        # init.json prompt override (the external prompt surface is exactly
        # base_prompt / covenant / comment). Legacy init.json `brief` /
        # `brief_file` values remain compatibility-known, are reported by the
        # shared reader, and are ignored here. The `brief` section is now sourced
        # solely from system/brief.md,
        # which the secretary agent writes directly.
        brief_file = system_dir / "brief.md"
        if brief_file.is_file():
            brief = brief_file.read_text(encoding="utf-8")
            if brief:
                self._prompt_manager.write_section("brief", brief, protected=True)
            else:
                self._prompt_manager.delete_section("brief")
        else:
            self._prompt_manager.delete_section("brief")

        # --- Comment ---
        comment = data.get("comment", "")
        if comment:
            self._prompt_manager.write_section("comment", comment)
        else:
            self._prompt_manager.delete_section("comment")

    def _build_launch_cmd(self) -> list[str] | None:
        """Return the command to relaunch this agent via lingtai-agent run."""
        from .venv_resolve import resolve_venv, venv_python
        data = self._read_init()
        venv_dir = resolve_venv(data)
        python = venv_python(venv_dir)
        return [python, "-m", "lingtai", "run", str(self._working_dir)]
