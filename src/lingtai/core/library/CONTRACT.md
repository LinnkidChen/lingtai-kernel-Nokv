# Knowledge capability contract

`knowledge` is the agent-private durable knowledge capability. It stores bounded,
curated entries that survive molts and are summarized into the agent's system
prompt. The implementation currently lives in `src/lingtai/core/library/` for
compatibility; the code remains the source of truth.

## Routing Card

**Use this when:**
- You are editing the private durable knowledge capability.
- You are reviewing tool schema, persistence, prompt-injection, capacity, or rename compatibility changes.
- You need to verify the boundary between private knowledge and portable skills.

**Do not use this for:**
- Skill catalog behavior: read `src/lingtai/core/skills/`.
- Code navigation only: read `src/lingtai/core/library/ANATOMY.md`.
- General procedure authoring: read the `skills-manual` skill.

**Fast paths:** tool schema -> §Tool surface; storage -> §Persistence; rename -> §Scope and compatibility; review -> §Verification matrix.

## Scope and compatibility

- Canonical capability name: `knowledge`.
- Canonical tool name: `knowledge`.
- Compatibility tool/capability name: `library`.
- Deprecated compatibility tool/capability name: `codex`.

`knowledge` means private durable memory: what one agent has learned, decided,
and discovered. `skills` means portable procedure catalog. Knowledge entries may
point to public skills; skills must not depend on private knowledge entry ids,
agent-local paths, mail ids, or other private memory state.

Manifest normalization:

| Manifest shape | Meaning after normalization |
|---|---|
| `"knowledge"` or `knowledge: {...}` | private durable `knowledge` |
| `"codex"` or `codex: {...}` | private durable `knowledge` (deprecated alias) |
| old bare `"library"` / `library: {}` without `skills` | skill catalog `skills` |
| old `library: {paths: [...]}` without `skills` | skill catalog `skills.paths` |
| `library: {library_limit: N}` | explicit durable `knowledge` |
| `library: {}, skills: {...}` | transitional durable `knowledge` + skill catalog `skills` |

## Knowledge / skill directionality

Knowledge entries MAY reference skills by public path/name when an agent has
learned that a skill is useful for a recurring situation.

Skills MUST NOT reference private knowledge entry ids, private agent paths, mail
ids, or agent-local memory state.

Reason: skills are portable shared procedures; knowledge is agent-local
accumulated memory. The dependency direction is knowledge -> skill, never skill
-> private knowledge.

## Tool surface

The schema requires `action` and accepts exactly four actions:

| Action | Required fields | Optional fields | Return on success |
|---|---|---|---|
| `submit` | `title`, `summary` | `content`, `supplementary` | `{status: "ok", id, entries, max}` |
| `view` | `ids` | `include_supplementary` | `{status: "ok", entries: [...]}` |
| `consolidate` | `ids`, `title`, `summary` | `content`, `supplementary` | `{status: "ok", id, removed}` |
| `delete` | `ids` | — | `{status: "ok", removed}` |

Unknown actions return an error and do not mutate state. Removed historical
actions such as `filter` and `export` are intentionally rejected.

`library(...)` and `codex(...)` aliases use the same schema and handler during
the migration window; new callers should use `knowledge(...)`.

## Persistence

The store path is intentionally still `<agent>/codex/codex.json`. The rename is
user-facing; it is not a storage-v2 migration. File shape remains:

```json
{"version": 1, "entries": [ ... ]}
```

Writes are atomic within `codex/`: create a temporary file, write UTF-8 JSON with
`ensure_ascii=False`, close it, then `os.replace()` it over `codex.json`. Reads
are tolerant: missing/invalid/unreadable JSON means an empty store; legacy
entries without `title` are backfilled from old `content`.

## Prompt injection

On setup and after every mutating action, the capability rewrites prompt sections:

- If there are entries, protected prompt section `knowledge` contains a compact
  catalog: total count/max count plus one line per entry with `[id] title:
  summary`, followed by a reminder to call `knowledge(view, ids=[...])`.
- If there are no entries, protected prompt section `knowledge` is cleared.
- Protected prompt sections `library` and `codex` are always cleared so the
  canonical section owns the catalog.

Only ids, titles, and summaries are always injected. Full `content` and
`supplementary` stay out of the prompt until loaded through `view`.

## Capacity configuration

`LibraryManager.DEFAULT_MAX_ENTRIES` is `50`. `knowledge_limit=N` is canonical.
`library_limit=N` and `codex_limit=N` are compatibility kwargs. Precedence:
`knowledge_limit`, then `library_limit`, then `codex_limit`, then default 50.

## Anchored claims

| Claim | Source | Test |
|---|---|---|
| `knowledge`, `library`, and `codex` all resolve to the same implementation | `src/lingtai/capabilities/__init__.py:15-22`, `src/lingtai/core/library/__init__.py:326-346` | `tests/test_library_knowledge.py::test_knowledge_setup_registers_tool_and_aliases` |
| `knowledge` is canonical for manager lookup; compatibility names resolve | `src/lingtai/agent.py:806-812`, `src/lingtai/capabilities/__init__.py:43-54` | `tests/test_library_knowledge.py::test_codex_capability_normalizes_to_knowledge` |
| Old bare/list `library` still normalizes to `skills` | `src/lingtai/capabilities/__init__.py:57-129` | `tests/test_skills.py::test_old_library_empty_config_normalizes_to_skills_only`, `test_old_library_list_config_normalizes_to_skills_only` |
| Prompt catalog lives in `knowledge`; `library`/`codex` sections are cleared | `src/lingtai/core/library/__init__.py:100-122` | `tests/test_library_knowledge.py::test_codex_tool_alias_uses_library_store`, skills rename tests |
| Store remains `codex/codex.json` | `src/lingtai/core/library/__init__.py:93`, `src/lingtai/core/library/__init__.py:128-160` | `tests/test_library_knowledge.py::test_submit_creates_entry` |
| `knowledge_limit` wins over compatibility limit kwargs | `src/lingtai/core/library/__init__.py:76-91`, `src/lingtai/core/library/__init__.py:309-324` | capacity coverage in `tests/test_library_knowledge.py` |

## Verification matrix

| Invariant | Automated test | Manual check | Risk if broken |
|---|---|---|---|
| `knowledge(...)` is the canonical tool | `tests/test_library_knowledge.py` | Boot with `capabilities={"knowledge": {}}` and submit | New agents cannot use the intended name |
| `library(...)` and `codex(...)` aliases still work | `test_codex_tool_alias_uses_library_store` | Call each alias and inspect `knowledge` prompt section | Existing agents break during migration |
| Old `library.paths` remains skill-catalog config | `tests/test_skills.py::*old_library*` | Boot an old manifest with `library.paths` | Old presets lose skill catalog |
| Skills do not depend on private knowledge | documented invariant; enforce by review | Check shared skill docs for private ids/paths | Shared skills become non-portable |
| Full content stays out of prompt catalog | `test_view_returns_content` plus prompt inspection | Submit long content, inspect prompt section | Prompt bloat / private detail leakage |

Run before merging knowledge changes:

```bash
python -m pytest tests/test_library_knowledge.py tests/test_skills.py tests/test_check_caps.py -q
```
