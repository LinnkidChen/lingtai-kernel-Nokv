---
name: notification-manual
description: >
  Notification filesystem manual for LingTai kernel notifications: channel
  whitelist, `.notification/<channel>.json` files, envelope shape, instructions,
  generic/producer-specific dismiss, protected channels, and per-event system
  dismiss.
version: 0.1.0
tags: [lingtai, notifications, system, channels, dismiss]
---

# Notification Manual

LingTai notifications are a filesystem protocol. Producers write JSON files under
an agent's `.notification/` directory; the kernel reads allowlisted files and
syncs the current notification block into the agent's model context.

## Channel files and whitelist

A notification channel is the filename stem in `.notification/<channel>.json`.
For example:

- `.notification/email.json` → `notifications["email"]`
- `.notification/system.json` → `notifications["system"]`
- `.notification/mcp.telegram.json` → `notifications["mcp.telegram"]`
- `.notification/goal.json` → `notifications["goal"]`

The kernel uses an allowlist: built-in channels such as `email`, `system`,
`soul`, `nudge`, `post-molt`, `tool_loop_guard`, `bash`, `btw`, `cron`, `molt`,
and `goal` are accepted; MCP bridge channels are accepted by the `mcp.` prefix.
Unknown `.json` files are ignored by `collect_notifications()` and kernel helper
publish/dismiss calls reject non-allowlisted channel names.

## Envelope shape

Producer helpers write a standard envelope:

```json
{
  "header": "1 system notification",
  "icon": "🔔",
  "priority": "normal",
  "published_at": "2026-06-10T00:00:00Z",
  "instructions": "Optional agent-facing handling guidance.",
  "data": {"events": []}
}
```

`instructions` is a field inside a channel payload, not a channel name. It should
say what the agent should do with this notification and how to clear it.

## The `notification` tool

There is a dedicated, always-available `notification` tool covering the
notification verbs. It is the preferred surface; the matching `system` actions
remain as compatibility aliases and behave identically.

| notification tool | system alias |
| --- | --- |
| `notification(action="check")` | `system(action="notification")` |
| `notification(action="dismiss", channel=...)` | `system(action="dismiss", channel=...)` |
| `notification(action="summarize", items=[...])` | `system(action="summarize", items=[...])` |

Both tools route into the same kernel implementation, so results are identical
(only internal provenance logging differs). All dismiss semantics below apply to
both tools.

## Model-visible notification block

`notification(action="check")` (or the `system(action="notification")` alias)
returns a placeholder; the kernel stamps the live notification payload onto that
tool result. The same payload is synthesized when notifications arrive while the
agent is IDLE or ASLEEP. The payload contains a global `_notification_guidance`
plus `notifications:{...}` keyed by channel. After handling a notification,
dismiss it and end your turn — do not call `check` voluntarily again.

## Dismiss semantics

Use producer-specific verbs for channels that mirror producer-owned state. For
example, email unread notifications should be cleared by `email(action="read"...)`
or `email(action="dismiss"...)`, not by generic `system.dismiss`.

Generic dismiss clears only the notification surface:

```text
notification(action="dismiss", channel="nudge")
```

For `.notification/system.json`, the old whole-channel behavior remains when no
`event_id` or `ref_id` is supplied:

```text
notification(action="dismiss", channel="system")
```

Atomic per-event dismiss is available for system events:

```text
notification(action="dismiss", channel="system", event_id="evt_...")
notification(action="dismiss", channel="system", ref_id="goal:current")
```

This removes only matching entries from `system.data.events`; if the last event
is removed, `.notification/system.json` is deleted. The `system(action="dismiss",
...)` alias accepts the same arguments and behaves identically.

## Undismissable large-result reminders

System events with `source="large_tool_result"` are **undismissable**. A
whole-channel `system` dismiss (with or without `force=true`), and a targeted
`event_id`/`ref_id` dismiss that matches one, are both refused with
`reason="undismissable_large_result_reminder"` — through both the `notification`
tool and the `system` alias. There is no `force` backdoor.

These reminders represent a large tool result that still costs context budget.
The only way to clear one is to **summarize** the result:

```text
notification(action="summarize", items=[{"tool_call_id": "toolu_...", "summary": "..."}])
```

A successful summarize of that `tool_call_id` auto-clears the matching
`large_tool_result:<tool_call_id>` reminder. A failed summarize item leaves its
reminder in place.

## Protected channels

Some channels are source-of-truth files, not dismissible mirrors. `goal` is
protected: `system(action="dismiss", channel="goal")` refuses even with
`force=true`. To cancel or complete a goal, edit or delete `.notification/goal.json`
as described in the goal manual.

## Cross-reference

For active goal state and goal reminders, read `reference/goal-manual/SKILL.md`.
