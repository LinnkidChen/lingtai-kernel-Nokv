# src/lingtai/core/

The always-on agent floor (file I/O, knowledge, skills, daemon, avatar, bash, mcp), the opt-in multimodal capabilities (`vision`, `web_search`), the capability **registry**, and the SDK bundle bridges. Formerly split between `lingtai.core` and a separate `lingtai.capabilities` package; the two were merged here — there is no longer a `lingtai.capabilities` package.

> **Maintenance:** see the `lingtai-kernel-anatomy` skill. **Coding agents** update this file in the same commit as code changes. **LingTai agents** report drift as issues.

## Components

| File / dir | Role |
|---|---|
| `registry.py` | Static registry (`_BUILTIN`, `_GROUPS`, `CORE_DEFAULTS`), normalization (`normalize_capabilities`, `apply_core_defaults`, `expand_groups`), `setup_capability()`, and `get_all_providers()` |
| `_media_host.py` | `resolve_media_host()` — extracts origin from the agent LLM `base_url` |
| `_zhipu_mode.py` | `resolve_z_ai_mode()` — returns `"ZHIPU"` (bigmodel.cn) or `"ZAI"` (international) |
| `read/ write/ edit/ glob/ grep/` | File-I/O capabilities (the `file` group) |
| `knowledge/ skills/ bash/ avatar/ daemon/ mcp/` | The rest of the always-on floor |
| `vision/ web_search/` | Opt-in multimodal capabilities (NOT in `CORE_DEFAULTS`; require provider config) |
| `*_bundle.py` | SDK bundle bridges (see Notes) — `file_bundle`, `system_bundle`, `communication_bundle`, `mcp_bundle`, `knowledge_bundle`, `bash_bundle`, `avatar_bundle`, `psyche_bundle`, `soul_bundle` |
| `__init__.py` | Import-light docstring only; makes the tier visible in the import graph |

## Connections

- **→ `lingtai.core.*`** — `registry._BUILTIN` registers the always-on floor by absolute path (`knowledge`, `skills`, `bash`, `avatar`, `daemon`, `mcp`, `read`, `write`, `edit`, `glob`, `grep`) and the opt-in `vision`/`web_search` by relative path (`.vision`/`.web_search`).
- **← `lingtai.agent.Agent`** — expands groups and calls `normalize_capabilities()` / `apply_core_defaults()` before `setup_capability()` in both construction and refresh.
- **← `lingtai_cli.host`** — imports `CORE_DEFAULTS` (injected via `core_defaults=`) and `get_all_providers` (for `lingtai-agent check-caps`).
- **← `lingtai.core.daemon`** — imports `setup_capability`, `_GROUPS`, `_BUILTIN` from `registry` for preset sandbox instantiation.
- **→ `vision`/`web_search`** — import `_media_host` / `_zhipu_mode` lazily inside their `setup()` for provider-specific kwarg injection.

## Composition

`registry.py` is the dispatch entry point. `_media_host.py` / `_zhipu_mode.py` are private helpers used by `vision`/`web_search`, not by the registry. The capability subpackages and `*_bundle.py` modules sit alongside.

## State

- `_BUILTIN` is static capability name → module path. `knowledge` resolves to `lingtai.core.knowledge`; former durable-memory names `library`/`codex` are not registered.
- `_GROUPS` is static; currently only `"file"` → `[read, write, edit, glob, grep]`.
- `CORE_DEFAULTS` is the static set that boots automatically on every `Agent`: `knowledge`, `skills`, `bash` (`{yolo: true}`), `avatar`, `daemon`, `mcp`, and the file group. `vision`/`web_search` are NOT in this set — they require provider config / API keys and stay opt-in.
- No mutable runtime state is held by the registry.

## Notes

- `setup_capability()` imports the target module (relative names resolve against `lingtai.core` since `__package__` is `lingtai.core`) and calls its `setup()`. Unknown names raise `ValueError` with available capabilities and groups.
- `apply_core_defaults(capabilities, disable)` overlays `CORE_DEFAULTS` with user-supplied kwargs (init.json wins on merge), then strips names listed in `disable`. A `"name": None` entry is an inline opt-out equivalent to `disable`. Called from `Agent.__init__` and `Agent._setup_from_init`.
- `get_all_providers()` returns user-facing capability/provider metadata for `lingtai-agent check-caps`.
- **SDK bundle bridges (`*_bundle.py`).** Each bridge injects a real handler (a capability's `make_handler(agent)` factory, or a kernel intrinsic's `handle(agent, args)` bound to the agent) into the corresponding SDK bundle declaration and returns a `BundleHost`/`NativeBundleHost`, proving the bundle-execution pattern against the *real* tool behavior without changing the live `setup()` / `_wire_intrinsics` registration path. Declared postures (per-action risk tables, bundle-level posture) are declarations only — the bridge hosts and runs the handler; gating is the separate, not-installed `guard_bridge`. Import direction is one-way: the wrapper bridge imports the SDK (lazily) and, for intrinsic-backed bridges, the kernel intrinsic; the SDK declares manifests only and never imports the wrapper, and the kernel never imports the SDK. See `docs/sdk/architecture-foundation.md` (stages 3A–3F).
