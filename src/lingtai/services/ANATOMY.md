---
related_files:
  - docs/references/licc-notification-wake-runbook.md
  - pyproject.toml
  - setup.py
  - src/lingtai/ANATOMY.md
  - src/lingtai/services/__init__.py
  - src/lingtai/services/file_io.py
  - src/lingtai/services/file_io_sidecar.py
  - src/lingtai/services/mail.py
  - src/lingtai/services/mcp.py
  - src/lingtai/services/mcp_registry.py
  - src/lingtai/services/mcp_inbox.py
  - src/lingtai/services/mcp_licc.py
  - src/lingtai/services/LICC_NOTIFICATION_CONTRACT.md
  - src/lingtai/services/vision/ANATOMY.md
  - src/lingtai/services/websearch/ANATOMY.md
  - src/lingtai/adapters/posix/ANATOMY.md
  - src/lingtai/kernel/mail_transport/ANATOMY.md
  - src/lingtai/kernel/services/ANATOMY.md
  - tests/_mcp_stdio_fixture.py
  - tests/_mcp_activation_stdio_server.py
  - tests/test_mcp_activation_lifecycle.py
  - tests/test_mcp_operation_convergence.py
  - tests/_mcp_structured_stdio_server.py
  - tests/test_mcp_closed_resource_restart.py
  - tests/test_mcp_stdio_fixture.py
  - tests/test_mcp_structured_result.py
  - tests/test_nokv_mcp_structured_error.py
  - src/lingtai/intrinsic_skills/system-manual/reference/environment-variables/SKILL.md
maintenance: |
  Keep related_files as repo-relative paths to real files. Include neighboring
  ANATOMY.md files so the anatomy graph stays connected rather than isolated;
  anatomy links must be bidirectional. If you create a new ANATOMY.md, copy this
  maintenance field. If you notice drift between this anatomy and the code,
  report it. See lingtai-dev-guide for details.
---
# src/lingtai/services/

Root services package — pluggable backends for intrinsic tools and MCP clients.

> **Maintenance:** see the `lingtai-kernel-anatomy` skill. **Coding agents** update this file in the same commit as code changes. **LingTai agents** report drift as issues.

## Components

| File | LOC | Role |
|---|---|---|
| `__init__.py` | 1 | Docstring-only package marker |
| `file_io.py` | 530 | `FileIOService` facade contract + `FileIOBackend`/`LocalFileIOBackend` — backs read/edit/write/glob/grep. `grep` accepts an optional basename `glob_filter` that prunes the candidate set before stat/read |
| `file_io_sidecar.py` | 698 | Rust-backed grep/glob: `RustFileIOBackend`, `SidecarAdapter`, `SidecarError`, plus the `resolve_sidecar_binary` resolver and the `default_file_io_service` factory used by `Agent.__init__`. `grep`'s `glob_filter` is applied as a Python-side basename post-filter (the sidecar wire protocol carries no glob field yet) |
| `mail.py` | 19 | High-level compatibility surface: re-exports the Core `MailTransportPort` as `MailService` and the POSIX adapter as both `PosixFilesystemMailAdapter` and the legacy public name `FilesystemMailService` |
| `mcp.py` | 1219 | `_AsyncOperationBridge` plus `MCPClient` (stdio) and `HTTPMCPClient` (streamable HTTP) — async-to-sync MCP bridges with exact RPC cancellation convergence, shared structured-result decoding, generation-safe stdio restart, and verified lifecycle retirement |
| `mcp_registry.py` | — | MCP registry infrastructure (the non-tool half of the `lingtai/tools/mcp` capability): record schema (`validate_record`), strict optional `template_arg_indices` validation for selective stdio argument expansion, JSONL registry I/O (`read_registry`, `_append_record`), catalog loader (`_load_catalog`, path constant recomputed for this location), secret-safe identity projection (`read_identities`, `IDENTITY_SAFE_ACCOUNT_KEYS`), boot-time addon decompression (`decompress_addons`), and the system-prompt XML renderer (`_build_registry_xml`). Consumed by the `lingtai/tools/mcp` tool slice (lazy import) and `agent.py` |
| `mcp_inbox.py` | — | LICC v1 filesystem inbox poller plus Core projection; in-process publication receives the agent and uses its injected Notification Store while the external inbox path/envelope stays unchanged (`src/lingtai/services/mcp_inbox.py:373-395`). |
| `mcp_licc.py` | — | LICC v1 client producer (`push_inbox_event`); imports contract constants from `mcp_inbox.py` |
| `LICC_NOTIFICATION_CONTRACT.md` | — | The LICC notification two-lane projection contract governing curated IM producers; live diagnosis and recovery are documented in `docs/references/licc-notification-wake-runbook.md` |

**Sub-packages (not covered here):** `vision/` (7 provider files), `websearch/` (6 provider files).
**Sibling crates:** `crates/lingtai-search-sidecar/` (Rust) — opt-in binary that backs `RustFileIOBackend`. Not required for install/tests.

## Connections

- **→ `lingtai.kernel.logging.get_logger`** (`mcp.py:16`) — structured logging.
- **→ `lingtai.kernel.mail_transport`** (`mail.py:9`) — re-exports the Core-owned Port as `MailService`.
- **→ `lingtai.adapters.posix.mail`** (`mail.py:10-13`) — re-exports the production adapter under its canonical name and the legacy public `FilesystemMailService` alias.
- **→ `mcp.client.stdio`**, **`mcp.client.streamable_http`**, **`mcp.client.session`** (`mcp.py:783-784,1120-1124`) — third-party MCP SDK. Imported lazily inside async lifecycle methods; packaging pins the audited SDK boundary to `mcp>=1.10.0,<2` (`pyproject.toml:25`).
- **← `lingtai.tools.vision`** — uses `services.vision.VisionService`.
- **← `lingtai.tools.web_search`** — uses `services.websearch.SearchService`.
- **← `tools.{read,write,edit,glob,grep}`** — the file tools use `FileIOService` (injected as `agent._file_io`).

## Composition

`file_io.py` is a pure stdlib abstraction layer. `LocalFileIOService` is the tool-facing facade while `LocalFileIOBackend` owns the default Python local filesystem implementation. `file_io_sidecar.py` provides `RustFileIOBackend`, an opt-in alternative backend that delegates `read`/`write`/`edit` to a private `LocalFileIOBackend` but routes `grep`/`glob` to the Rust binary under `crates/lingtai-search-sidecar/` via short-lived JSON subprocess calls. `mail.py` is a high-level compatibility re-export across the Core Port and POSIX Adapter; it owns no implementation. `mcp.py` keeps two transport-specific client classes and composes both with one protocol-generic result decoder.

## State

- **`MCPClient` / `HTTPMCPClient`**: each instance manages one background daemon thread, an asyncio event loop (`_loop`), a `ClientSession` (`_session`), and a 50-entry activity log (`MCPClient` starts at `mcp.py:295`; `HTTPMCPClient` at `mcp.py:874`). Their shared `_AsyncOperationBridge` (`mcp.py:83-293`) owns operation tokens/tasks, still-running cancellation-barrier futures, and a fail-closed quarantine reason. Per-client state is guarded by a `threading.RLock` plus readiness events; stdio additionally owns `_restart_lock` and the monotonically increasing `_generation` (`mcp.py:318-338`).
- **`LocalFileIOService`**: facade over a `_backend`; exposes `last_traversal` from the backend for tool metadata.
- **`LocalFileIOBackend`**: default Python local filesystem backend; state is optional `_root` plus `last_traversal`.
- **`RustFileIOBackend`**: holds an embedded `LocalFileIOBackend` (for read/write/edit), a `SidecarAdapter` (subprocess client), and a `last_traversal` rebuilt from each sidecar envelope.
- **`SidecarAdapter`**: stateless apart from the resolved binary path; one subprocess per `call()`.
- **`FileIOService` / `FileIOBackend` ABCs**: pure interfaces, no state.

## Notes

- `MCPClient` uses `stdio_client` transport (subprocess); `HTTPMCPClient` uses `streamablehttp_client` (remote HTTP/SSE). Both expose identical `call_tool()` / `list_tools()` / `close()` API.
- Lazy start: both clients auto-connect on first `call_tool()`.
- **Exact operation convergence:** both transports route `call_tool()` and `list_tools()` through `_run_async_operation` (`mcp.py:229-292`). A synchronous timeout cancels the exact submitted future and then waits a separate ordered event-loop barrier for the actual task's `finally`, bounded to five seconds. If that proof does not converge, the client is quarantined: `is_connected()` becomes false and new calls/catalog reads fail closed. The still-running barrier is retained (never cancelled); a later `close()` must await that exact barrier and clear all operation tokens/tasks before it may clear quarantine and begin transport teardown. Tests: `tests/test_mcp_operation_convergence.py::test_call_timeout_cancels_and_drains_before_sync_return`, `::test_catalog_timeout_cancels_and_drains_before_candidate_cleanup`, and `::test_unconverged_timeout_quarantines_client_until_exact_task_finishes`.
- **Verified lifecycle retirement:** `start()` publishes one lifecycle thread and calls `Thread.start()` inside the state guard (`mcp.py:409-467,917-962`). A `Thread.start()` failure rolls back the not-yet-started thread/lifecycle publication to a retryable, vacuously clean state; an event-loop/bootstrap failure before the lifecycle task is dispatched sets readiness with a verified empty-resource postcondition (`mcp.py:728-753,1083-1103`). Connect, idle wait, and context-manager cleanup otherwise run in one asyncio task to preserve AnyIO cancel-scope ownership. A no-error pair of context-manager exits is cleanup evidence; when an exit raises or re-surfaces cancellation, stdio must explicitly prove child-process exit plus both streams closed, and HTTP must prove the captured httpx client plus both streams closed (`mcp.py:810-871,1145-1219`). `BaseExceptionGroup` leaves are retained in actionable diagnostics while nested `KeyboardInterrupt`/`SystemExit` are re-raised (`mcp.py:361-387`). A started lifecycle with neither error nor terminal proof fails closed. A verified cleanup error is reported once and a repeated `close()` may then converge; an unverified resource state remains unresolved on every retry. Bounded thread-join retries never treat `_closed` as retirement proof. Tests: `tests/test_mcp_operation_convergence.py` and the real transports in `tests/test_mcp_activation_lifecycle.py`.
- **Structured MCP results:** `_decode_tool_result` (`mcp.py:44-80`) is shared by stdio and HTTP. Dictionary `structuredContent` wins over conflicting text, followed by a JSON-object text fallback. When `isError=true`, only top-level `status` is forced to `error`; other fields survive. When `isError` is false or absent, object payloads are returned unchanged, including a missing or explicit data-level `status`. Missing, empty, or non-string structured error messages use the first text block when it is non-empty, then `Unknown MCP error`; non-empty strings, including whitespace-only strings, remain literal. JSON-object text uses its complete source text as that same fallback. Tests: `tests/test_mcp_structured_result.py`; built-NoKV boundary comparison: `tests/test_nokv_mcp_structured_error.py`.
- **Stale-resource recovery (issue #104):** `MCPClient` detects a dead stdio transport in `call_tool` and recovers. `_format_exception` renders `ClassName: message` (class-only when `str(e)` is empty) so an empty `ClosedResourceError` never surfaces as a blank `{"status":"error","message":""}`. `_is_stale_resource_error` flags closed/broken transports by class name + message substrings. `restart(expected_generation=...)` serializes on `_restart_lock` without holding the state lock across operation drain/close; concurrent callers that observed one stale generation reuse the first healthy replacement instead of starting another subprocess (`mcp.py:520-565`). A real restart clears latched startup/session/cleanup/operation state and retries the tool exactly once. Non-stale errors do not churn the subprocess. `HTTPMCPClient` reuses the exception formatter for connect errors only; it has no stale-resource restart. Tests: `tests/test_mcp_closed_resource_restart.py` plus `tests/test_mcp_operation_convergence.py::test_concurrent_stale_calls_share_one_real_lifecycle_replacement_generation` and `::test_restart_does_not_hold_state_lock_across_pending_rpc_drain`.
- The transport lifecycle, `list_tools()`, `_run_loop()`, and `_async_cleanup()` patterns remain duplicated between the two clients; result normalization is deliberately shared.
- `mail.py` is a compatibility-only alias surface. The normative boundary is `lingtai.kernel.mail_transport.MailTransportPort`; the production implementation is `lingtai.adapters.posix.mail.PosixFilesystemMailAdapter`. The legacy public names remain aliases, not a second implementation or a Core shim.
- `file_io_sidecar.py` is the **default native backend** for `Agent`-created file-I/O services. `default_file_io_service` is the factory that `Agent.__init__` calls; it consults `LINGTAI_FILE_IO_BACKEND` (`auto` / `rust` / `python`, default `auto`) and `resolve_sidecar_binary` to pick between Rust and the pure-Python `LocalFileIOBackend`. Resolver priority: explicit `binary_path=` > `LINGTAI_FILE_IO_SIDECAR` env > `LINGTAI_SEARCH_SIDECAR` (legacy) env > packaged `lingtai/bin/` binary (shipped in platform-specific wheels by `setup.py`) > dev-tree `crates/lingtai-search-sidecar/target/{release,debug}/`. The strict `SidecarAdapter()` constructor still ignores packaged / dev-tree sources — opt-in callers see `not_configured` rather than picking up a stale binary. Defaults (`DEFAULT_*` constants) are imported from `file_io.py` so both backends stay in lock-step. Cargo is **not** required for install or the normal test suite — tests use a Python-script "sidecar"; only `test_rust_sidecar_integration_grep_and_glob` is cargo-gated.
