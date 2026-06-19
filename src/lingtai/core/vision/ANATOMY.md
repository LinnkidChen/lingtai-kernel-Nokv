# src/lingtai/core/vision/

Vision capability — image understanding via pluggable VisionService backends.

> **Maintenance:** see the `lingtai-kernel-anatomy` skill. **Coding agents** update this file in the same commit as code changes. **LingTai agents** report drift as issues.

## Components

| File | LOC | Role |
|---|---|---|
| `__init__.py` | 153 | `VisionManager`, `setup()`, provider registry, tool schema |

**Key symbols:**
- `PROVIDERS` (L27-31) — supported providers: `minimax`, `zhipu`, `mimo`, `gemini`, `anthropic`, `openai`, `codex`. No static default; OpenAI-compat inherit fallback lives in `setup()`, not the registry.
- `VisionManager` (L53) — handles tool calls; resolves relative image paths via `agent._working_dir` (L73).
- `setup()` (L90) — entry point called by `core.registry.setup_capability()`. Creates `VisionManager`, registers `"vision"` tool on agent (L134).

## Connections

- **→ `lingtai.i18n.t`** (L21) — i18n for tool description and schema strings.
- **→ `lingtai.services.vision.VisionService`** (L22) — abstract service interface + `create_vision_service()` factory.
- **→ `lingtai.core._media_host.resolve_media_host`** (L175) — injected for `minimax` provider.
- **→ `lingtai.core._zhipu_mode.resolve_z_ai_mode`** (L178) — injected for `zhipu` provider.
- **→ `lingtai.kernel.base_agent.BaseAgent`** — type-only (L25).
- **← `lingtai.core.registry`** — registered as `".vision"` in `_BUILTIN`.

## Composition

Single file. No internal state — `VisionManager` instances hold agent + service refs.

## State

- `VisionManager._agent` / `_vision_service` (L61-62) — per-agent instance state. Stateless tool handler otherwise.
- `PROVIDERS` dict is module-level constant.

## Notes

- OpenAI-compat fallback: if the agent's provider isn't in `PROVIDERS` but the main LLM's `_provider_defaults["api_compat"] == "openai"`, vision routes through `OpenAIVisionService` using the LLM's own `base_url`/`model`/`api_key`. Lets `custom`/`openrouter`/`deepseek`/`kimi` users opt into vision via `vision: {"provider": "inherit"}` in their preset. Succeeds only if the relay+model actually support OpenAI-style `image_url` content blocks; otherwise the runtime call surfaces the relay's error.
- Graceful skip: if the agent's provider isn't in `PROVIDERS` AND the LLM is not OpenAI-compatible, setup returns `None` silently. Agent logs `capability_skipped`.
- Codex is exposed through `PROVIDERS` and flows to `create_vision_service("codex", api_key=None)`; the service uses ChatGPT OAuth rather than an API key.
- Provider-specific kwarg injection is opt-in per provider — prevents `TypeError` from passing unsupported kwargs to heterogeneous service constructors.
- Local mlx-vlm provider exists in `services/vision/local.py` but is intentionally hidden from `PROVIDERS` (see docstring L11-14).
