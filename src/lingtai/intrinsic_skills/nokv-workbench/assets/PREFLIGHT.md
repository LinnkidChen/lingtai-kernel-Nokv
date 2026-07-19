---
related_files:
- src/lingtai/intrinsic_skills/nokv-workbench/SKILL.md
maintenance: |
  Developer-facing TUI deployment preflight pointed to by nokv-workbench/SKILL.md:49; update it whenever the workbench MCP tool surface, selective argument-template contract, checkpoint-lifecycle fields, or runtime-version compatibility check changes.
---

# TUI runtime preflight (developer-facing)

This is deployment guidance for developers installing a workbench-enabled
LingTai branch into the TUI runtime. It is not agent-facing instruction and
is deliberately kept out of SKILL.md.

Check the runtime package version first:

```bash
~/.lingtai-tui/runtime/venv/bin/python - <<'PY'
import importlib.metadata as md
print(md.version("lingtai"))
PY
```

For a NoKV-generated `template_arg_indices` configuration, verify capability
instead of trusting the package version string alone:

```bash
~/.lingtai-tui/runtime/venv/bin/python - <<'PY'
from inspect import signature
from lingtai.agent import Agent
assert "template_arg_indices" in signature(Agent.connect_mcp).parameters
print("selective MCP argument templates: supported")
PY
```

Do not activate the generated configuration when this check fails. An older
kernel ignores the new field and expands placeholder-looking bytes in every
argument, including opaque workspace and actor identities. Roll back the NoKV
configuration/lock and kernel as one unit; never downgrade only the kernel
while a selective-template configuration remains active.

Do not install a source branch that is older than the runtime package already
used by TUI. Rebase or cherry-pick the workbench skill onto the matching or
newer upstream LingTai release, build/install that branch, then verify that
the runtime can see the skill:

```bash
~/.lingtai-tui/runtime/venv/bin/python - <<'PY'
from pathlib import Path
import lingtai.intrinsic_skills as skills
root = Path(skills.__file__).parent
print((root / "nokv-workbench" / "SKILL.md").exists())
PY
```

Tool-surface note: the supported managed deployment requires the reviewed exact
18-tool Workbench contract, including `workbench_snapshot_retire` and
capability-gated `workbench_restore`. Older 9- or 16-tool servers are not a
degraded supported mode: contract validation fails closed. Upgrade NoKV before
activating or refreshing the registration.

The checkpoint-lifecycle surface — workbench_snapshot_renew and
workbench_snapshot_list, the workbench_snapshot `name`/`ttl_days` parameters
and its `lease_expires_at`/`expiry_warning` output, and the `at_snapshot`
parameter on workbench_read / workbench_list / workbench_stat — needs a NoKV
build that ships Phase 1 snapshot leasing. Against an older build these tools
and parameters are absent, and the "Checkpoints and leases" SKILL section does
not apply; snapshots there fall back to the legacy 1-hour lease with no
renewal path.
