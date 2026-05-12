# core/library

Knowledge capability — private durable knowledge across molts. The implementation
still lives in `core/library` and still reads/writes the legacy
`codex/codex.json` store for compatibility. The canonical user-facing capability
and tool name is `knowledge`; `library` and `codex` remain aliases during the
migration window. Entry id + title + summary are injected into the `knowledge`
system-prompt section; content and supplementary material load on demand.

## Components

- `library/__init__.py` — the capability implementation. `get_description` (`__init__.py:27-28`), `get_schema` (`__init__.py:31-67`), `LibraryManager` (`__init__.py:71-306`), and `setup` (`__init__.py:309-351`). `CodexManager` remains as an import alias (`__init__.py:354-355`).
- `library/CONTRACT.md` — public contract companion for the implementation, including compatibility rules, knowledge/skill directionality, anchored claims, and verification matrix.

## Connections

- `lingtai.capabilities` maps canonical `knowledge` plus compatibility `library`/`codex` capability names here (`../../capabilities/__init__.py:15-22`).
- Capability normalization preserves old skill-catalog `library` manifests while routing explicit durable-knowledge config to `knowledge` (`../../capabilities/__init__.py:57-129`).
- `setup()` registers canonical `knowledge` plus `library` and `codex` tool aliases on the same handler (`__init__.py:326-346`).
- `_inject_catalog()` writes the `knowledge` prompt section and clears old `library`/`codex` prompt sections so the canonical section owns the catalog (`__init__.py:100-122`).

## Public API

The `knowledge` tool exposes four actions:

| Action | Description |
|---|---|
| `submit` | Add a new entry (requires title + summary; content + supplementary optional) |
| `view` | Read full content of entries by ID list; optionally include supplementary material |
| `consolidate` | Merge multiple entries into one new entry (removes originals, creates replacement) |
| `delete` | Remove entries by ID list |

The compatibility `library` and deprecated `codex` tool aliases accept the same
schema and dispatch to the same manager during the migration window.

## State

- Persistent store: `<agent>/codex/codex.json` (`__init__.py:93`). The path intentionally keeps the old directory name in this rename-only change.
- Prompt state: `knowledge` section contains the catalog; `library` and `codex` sections are cleared when this capability injects (`__init__.py:100-122`).
- Capacity: `DEFAULT_MAX_ENTRIES = 50`; `knowledge_limit` is canonical, with `library_limit` and `codex_limit` accepted as compatibility kwargs (`__init__.py:74-91`).

## Notes

- This change is naming-first, not storage-v2: the on-disk JSON shape and `codex/` directory stay unchanged while user-facing concepts move from Library/Codex to Knowledge.
- The old `lingtai.core.codex` module remains a compatibility wrapper; new code should import/use the knowledge capability through the canonical `knowledge` name.
- Knowledge is private, agent-owned memory. Skills are portable procedures. Knowledge may point to public skills; shared skills must not point to private knowledge.
- For the stable behavior contract, read `src/lingtai/core/library/CONTRACT.md` before editing this capability.
