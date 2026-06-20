# intrinsics/notification

Standalone notification surface — the notification-facing verbs carved out of the `system` tool into a dedicated, **mandatory-included** tool. It is a thin façade: every action delegates to the single canonical implementation that `system` already uses, so there is one source of truth and the two tools produce identical results (only the provenance log line differs). The old `system(action="notification"|"dismiss"|"summarize")` verbs remain as compatibility aliases routing into the same functions.

> **Maintenance:** see the `lingtai-kernel-anatomy` skill. **Coding agents** update this file in the same commit as code changes. **LingTai agents** report drift as issues.

## Components

- `__init__.py` — façade + dispatch.
  - `get_description` / `get_schema` (re-exported from `schema.py`) — tool registration.
  - `handle()` (`__init__.py:87-101`) — dispatcher over three actions: `check`, `dismiss`, `summarize`. Unknown actions return a `status="error"` dict.
  - `_check()` (`__init__.py:54-71`) — voluntary read of the notification surface. Returns a placeholder dict (`_notification_placeholder: True` + message), the **same shape** `system(action="notification")` returns. The live payload (`notifications` + `_notification_guidance`) is stamped onto this same result by `meta_block.attach_active_notifications`, which walks backward for the freshest *dict-shaped* tool result (`meta_block.py:244-258`) — tool-name-agnostic, so `notification(action=check)` receives the identical stamp.
  - `_dismiss()` (`__init__.py:74-85`) — delegates to `system.notification._dismiss` after stamping `_invoked_by="notification"`. The dismiss decision logic (allowlist, `post-molt` ack-reason, protected channels, generic-dismiss guard, stale-channel-version refusal, **`large_tool_result` undismissable guard**, atomic `event_id`/`ref_id` removal) all lives in `notifications.dismiss_channel`; `_invoked_by` only affects which provenance log line is emitted, never the result.
  - `_summarize` (imported from `system.summarize`) — the **same** single-source summarize. A successful summarize is the only sanctioned discharge for a `large_tool_result` reminder; it calls `notifications.clear_large_result_reminders` to auto-clear the matching `large_tool_result:{tool_call_id}` event. Failed summarize items do not clear.

- `schema.py` — tool registration. Exposes `action` (`check`/`dismiss`/`summarize`) plus the shared params `channel`, `force`, `event_id`, `ref_id`, `reason`, `items`. Shared-param descriptions **reuse the `system_tool.*` i18n keys** so the two tools describe identical behavior; the tool-level + action strings use new `notification_tool.description` / `notification_tool.action_description` keys (en/zh/wen).

## Connections

- `ALL_INTRINSICS["notification"]` (`intrinsics/__init__.py:8-16`) → `BaseAgent._wire_intrinsics()` (`base_agent/__init__.py:580`) binds `handle()` into every agent's tool surface. **Membership in `ALL_INTRINSICS` is the mandatory-include mechanism** — the wiring loop is unconditional, with no manifest gate, so this tool is always present like `system`.
- Delegates into `intrinsics/system/notification.py` (`_dismiss`) and `intrinsics/system/summarize.py` (`_summarize`); both ultimately call into the kernel-root `notifications.py`. All #424 guards therefore hold through this tool by construction.
- The live-payload stamp is performed by `meta_block.attach_active_notifications`, called from `base_agent/turn.py`; see the kernel-root `ANATOMY.md` "Notifications" section.

## Composition

- **Parent:** `src/lingtai_kernel/intrinsics/` (see `intrinsics/ANATOMY.md`).
- **Siblings:** `system/` (canonical owner of the dismiss/summarize implementations and the producer `publish_notification`/`clear_notification` entry points), `email/`, `soul/`, `psyche/`.

## State

- This intrinsic writes no state of its own. Through delegation it mutates `.notification/system.json` (event removal on dismiss; reminder auto-clear on summarize) and clears `.notification/<channel>.json` files. Producer-owned canonical state (mailbox read-state, etc.) is never touched — mirror operations only clear the notification surface.

## Notes

- **Identical-result invariant:** `notification(action="dismiss"|"summarize", …)` and the corresponding `system(action=…)` alias must return equal result dicts for equal inputs. This is guaranteed by delegating to the same functions; the only intentional difference is the provenance log (`notification_dismiss` with `invoked_by="notification"` vs the extra `system_dismiss` line the system tool emits). Regression-anchored by `tests/test_notification_tool.py`.
- **No `force` backdoor:** `large_tool_result` reminders cannot be cleared by `force`, `event_id`, or `ref_id` through either tool. The only clear path is a successful summarize.
