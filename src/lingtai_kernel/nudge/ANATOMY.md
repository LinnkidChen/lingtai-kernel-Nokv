# Nudge

Per-agent periodic checks that emit a notification on the shared
`.notification/nudge.json` channel when something needs the agent's
attention. Today: kernel version skew. Designed so additional checks
land as small additions (e.g. MCP version drift, addon updates).

## Entry point

`run_checks(agent)` is called once per heartbeat tick from
`base_agent/lifecycle.py:_heartbeat_loop` (wrapped in try/except so a
bad check never breaks the loop). It dispatches to each check's
`check(agent) -> None` in order.

## File layout

- `__init__.py` — dispatcher (`run_checks`), shared upsert/remove
  helpers (`upsert`, `remove`) that operate on the `nudge.json`
  multi-entry payload under a lazy per-agent lock.
- `kernel_version.py` — the only check today. Compares
  `lingtai.__version__` (frozen at import time) against
  `importlib.metadata.version("lingtai")` (rescans dist-info on each
  call). Emits when they differ; clears when they re-agree.
- `ANATOMY.md` — this file.

## The shared channel

All nudges share `.notification/nudge.json` with this shape:

```json
{
  "header": "1 nudge",
  "icon": "🔔",
  "priority": "low",
  "instructions": "Call system(action='dismiss', channel='nudge') ...",
  "data": {
    "nudges": [
      {"kind": "kernel_version", "title": "...", "detail": "...", ...}
    ]
  }
}
```

Each entry's `kind` is its slot key — `upsert(agent, kind, body)`
replaces by `kind`, `remove(agent, kind)` drops by `kind`. When the
list empties, the channel file is deleted entirely so the wire surface
drops the notification cleanly. The agent dismisses everything at once
with `system(action='dismiss', channel='nudge')`.

## Adding a new nudge

1. Drop `nudge/<name>.py` with a top-level `check(agent) -> None`
   function. Inside:
   - Throttle (compare wall-clock against a state dict you stash on
     the agent as `agent._nudge_<name>_state`).
   - Probe whatever you need to check.
   - On hit: call `upsert(agent, "<unique_kind>", body)` where `body`
     is the per-kind payload dict you want the agent to read.
   - On clear: call `remove(agent, "<unique_kind>")` and reset your
     dedupe state.
2. Add `from . import <name>` and `<name>.check(agent)` to
   `__init__.py:run_checks`.

Keep checks small, side-effect-free except for the upsert/remove call,
and well-throttled. They run inside the heartbeat loop on a 1-second
tick.

## Why not a Check protocol / registry?

Three similar lines is better than a premature abstraction
(CLAUDE.md). At one check today, the per-check throttle boilerplate
(~3 lines) is cheaper than maintaining a `Check` protocol and a
registry. If this grows to ≥3 checks and the duplication starts to
hurt, lift a `_throttled_probe` helper into `__init__.py` — but the
right abstraction shape will be obvious by then.

## Wire surface

The nudge channel flows through the standard `.notification/` sync
machinery (`base_agent/__init__.py:_sync_notifications` →
`meta_block.py` → wire). No special wire path. The agent sees it in
the meta-block alongside any other active notifications.

## Failure isolation

The heartbeat-loop call site wraps `run_checks` in try/except and logs
to the kernel logger on failure. A bug in any individual check is
caught inside `run_checks` (each check call is independent — a failing
check doesn't block subsequent ones, but the dispatcher doesn't wrap
individual calls either, so add per-check try/except inside `check()`
if a check does anything risky in its body).
