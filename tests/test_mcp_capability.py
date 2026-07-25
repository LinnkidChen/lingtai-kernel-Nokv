"""End-to-end smoke tests for the mcp capability + addons decompression.

Verifies the vertical slice: addons:["imap"] in init.json triggers catalog
decompression into mcp_registry.jsonl, the mcp capability renders the registry
into the system prompt, and the loader gates init.json mcp activation by
registry membership.
"""
from __future__ import annotations

import copy
import importlib
import json
import re
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from lingtai.agent import Agent, MCPActivationOutcome
from lingtai.kernel.base_agent import (
    RefreshHandoffOutcome,
    RefreshHandoffStatus,
)
from lingtai.kernel.llm import FunctionSchema
from lingtai.services.mcp_registry import (
    REGISTRY_FILENAME,
    decompress_addons,
    read_registry,
    validate_record,
)
from tests._service_helpers import make_gemini_mock_service as make_mock_service
from tests._mcp_stdio_fixture import (
    StdioProcessObserver,
    stop_process,
    wait_for_process_exit,
    wait_for_thread_exit,
)
from tests._refresh_watcher_helpers import make_test_refresh_watcher




def _mk_agent(tmp_path: Path, *, addons=None, capabilities=None):
    workdir = tmp_path / "agent"
    return Agent(
        service=make_mock_service(),
        agent_name="test",
        working_dir=workdir,
        capabilities=capabilities or {"mcp": {}},
        addons=addons,
    ), workdir


def _committed_refresh(
    calls: list | None = None,
    *,
    prepare=None,
) -> RefreshHandoffOutcome:
    if prepare is not None:
        preparation_error = prepare()
        if preparation_error is not None:
            return RefreshHandoffOutcome(
                RefreshHandoffStatus.PREPARATION_FAILED,
                preparation_error,
            )
    if calls is not None:
        calls.append(True)
    return RefreshHandoffOutcome(
        RefreshHandoffStatus.COMMITTED,
        "test handoff committed",
    )


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

def test_validator_accepts_valid_stdio_record():
    ok, err = validate_record({
        "name": "imap",
        "summary": "test",
        "transport": "stdio",
        "command": "python",
        "args": ["-m", "lingtai.mcp_servers.imap"],
        "source": "lingtai-curated",
    })
    assert ok, err


def test_validator_accepts_valid_http_record():
    ok, err = validate_record({
        "name": "remote",
        "summary": "test",
        "transport": "http",
        "url": "https://example.com/mcp",
        "source": "user",
    })
    assert ok, err


def test_validator_accepts_optional_homepage():
    ok, err = validate_record({
        "name": "imap",
        "summary": "test",
        "transport": "stdio",
        "command": "python",
        "args": [],
        "source": "lingtai-curated",
        "homepage": "https://github.com/Lingtai-AI/lingtai-imap",
    })
    assert ok, err


def test_validator_accepts_record_without_homepage():
    ok, err = validate_record({
        "name": "imap",
        "summary": "test",
        "transport": "stdio",
        "command": "python",
        "args": [],
        "source": "user",
    })
    assert ok, err


def test_validator_rejects_empty_homepage():
    ok, err = validate_record({
        "name": "imap",
        "summary": "test",
        "transport": "stdio",
        "command": "python",
        "args": [],
        "source": "user",
        "homepage": "",
    })
    assert not ok
    assert "homepage" in err


def test_validator_rejects_bad_name():
    ok, err = validate_record({
        "name": "BAD-NAME",
        "summary": "x",
        "transport": "stdio",
        "command": "a",
        "args": [],
        "source": "u",
    })
    assert not ok
    assert "invalid name" in err


def test_validator_rejects_bad_transport():
    ok, err = validate_record({
        "name": "x",
        "summary": "y",
        "transport": "smtp",
        "source": "u",
    })
    assert not ok
    assert "invalid transport" in err


def test_validator_rejects_long_summary():
    ok, err = validate_record({
        "name": "x",
        "summary": "a" * 500,
        "transport": "stdio",
        "command": "a",
        "args": [],
        "source": "u",
    })
    assert not ok
    assert "summary too long" in err


def test_nokv_workbench_registry_example_is_valid():
    skill_root = Path("src/lingtai/intrinsic_skills/nokv-workbench")
    registry_path = skill_root / "assets" / "mcp_registry.example.jsonl"
    init_path = skill_root / "assets" / "init-snippet.json"

    record = json.loads(registry_path.read_text(encoding="utf-8").strip())
    ok, err = validate_record(record)
    assert ok, err
    assert record["name"] == "nokv-workbench"
    assert record["transport"] == "stdio"
    assert record["args"] == [
        "--server-bind",
        "127.0.0.1:7777",
        "--object-backend",
        "rustfs",
        "--s3-bucket",
        "nokv-lingtai-workbench",
        "mcp",
        "--profile",
        "workbench",
        "--workbench-root",
        "/agents/{agent_id}/wb",
    ]

    init = json.loads(init_path.read_text(encoding="utf-8"))
    spec = init["mcp"]["nokv-workbench"]
    assert spec["type"] == "stdio"
    assert spec["command"] == record["command"]
    assert spec["args"] == record["args"]


def test_nokv_workbench_skill_documents_durable_restore_contract():
    skill_root = Path("src/lingtai/intrinsic_skills/nokv-workbench")
    skill = (skill_root / "SKILL.md").read_text(encoding="utf-8")
    preflight = (skill_root / "assets" / "PREFLIGHT.md").read_text(encoding="utf-8")

    assert "version: 0.5.0" in skill
    assert "workbench_restore" in skill
    assert '"at_snapshot": 417' in skill
    assert "same numeric `snapshot_id`" in skill
    assert "resend the exact request" in skill
    assert "No separate LingTai restore state" in skill
    assert "metadata/restore_manifest.json" in skill
    assert "nokv.workbench.restore_manifest.v1" in skill
    assert "restored_from.snapshot_id" in skill
    assert 'Success means `state="complete"`' in skill
    assert "expired checkpoints cannot be renewed" in skill
    assert "lifecycle `state` (`alive`, `expired`, `retired`, or" in skill
    assert "lifecycle `status`" not in skill
    assert "grace window" not in skill

    for code in (
        "SnapshotNotFound",
        "SnapshotLeaseExpired",
        "SnapshotRootMismatch",
        "SnapshotBindingChanged",
        "SnapshotRenewContended",
        "NotOwner",
        "StaleOwnerEpoch",
        "InvalidOwnerEpoch",
        "LeaseExpired",
        "RestoreTransportUnavailable",
        "RestoreInProgress",
        "RestoreRootChanged",
        "RestoreBindingChanged",
        "RestoreProtocolMismatch",
        "RestoreDestinationConflict",
        "RestoreResourceLimit",
        "RestoreHardlinkUnsupported",
        "RestoreCrossShardUnsupported",
        "StalePreparedArtifactObjectGcEpoch",
        "SyncLogArchiveFailed",
        "CapabilityMismatch",
    ):
        assert code in skill

    assert "complete 18-tool restore-capable workbench surface" in preflight
    assert "The base surface has 17 tools" in preflight
    assert '"workbench_snapshot_retire"' in preflight
    assert '"required": ["id", "manifest", "content_digest_uri"]' in preflight
    assert '"metadata": {' in preflight
    assert '"required": ["id", "at_snapshot", "destination_id"]' in preflight
    assert '"additionalProperties": False' in preflight
    assert '"type": "integer", "minimum": 0' in preflight
    assert '"type": "string", "minLength": 1' in preflight
    assert "restore_to_fork_v1" in preflight
    assert "raw schema mismatch" in preflight
    assert "before Agent registration" in preflight
    assert "--profile full --require-all" in preflight
    assert "two real MCP" in preflight
    assert "hard-coded NoKV gate" in preflight


def test_nokv_workbench_docs_pin_write_read_and_lifecycle_contracts():
    skill_root = Path("src/lingtai/intrinsic_skills/nokv-workbench")
    skill = " ".join(
        (skill_root / "SKILL.md").read_text(encoding="utf-8").split()
    )
    preflight = " ".join(
        (skill_root / "assets" / "PREFLIGHT.md")
        .read_text(encoding="utf-8")
        .split()
    )

    assert (
        "`replace=false` (the default) is create-only and fails when the "
        "target exists; `replace=true` is replace-only and fails when the "
        "target is missing"
    ) in skill
    assert "It is not upsert" in skill
    assert "NoKV does not natively parse `application/x-ndjson`" in skill
    assert "does not promise NDJSON record pagination" in skill
    assert "A `.jsonl` suffix alone selects no parser" in skill
    assert (
        "write it with a `text/*` content type to receive raw `text_lines` "
        "whose `value.text` you parse yourself"
    ) in skill
    assert "use `format=\"bytes\"` for `application/x-ndjson`" in skill

    assert "`nokv.workbench.run_manifest.v1`" in skill
    assert "`content_digest_uri` before the call" in skill
    assert "different content identity conflicts even when" in skill
    assert "`reason` and `metadata` are bounded registry annotations" in skill
    assert "`SnapshotRegistryWritePartial`" in skill
    assert "Use `workbench_snapshot_retire`" in skill
    assert "returns `retired=false` and does not fabricate deletion attribution" in skill

    assert "complete 18-tool restore-capable workbench surface" in preflight
    assert "The base surface has 17 tools" in preflight
    assert '"workbench_snapshot_retire"' in preflight
    assert '"required": ["id", "manifest", "content_digest_uri"]' in preflight
    assert '"reason": {' in preflight
    assert '"metadata": {' in preflight


def _run_nokv_preflight_contract(monkeypatch, tools):
    preflight_path = Path(
        "src/lingtai/intrinsic_skills/nokv-workbench/assets/PREFLIGHT.md"
    )
    preflight = preflight_path.read_text(encoding="utf-8")
    blocks = re.findall(r"<<'PY'\n(.*?)\nPY", preflight, flags=re.DOTALL)
    code = next(block for block in blocks if "expected_restore_schema" in block)

    class FakeMCPClient:
        def __init__(self, command, args):
            self.command = command
            self.args = args

        def list_tools(self):
            return copy.deepcopy(tools)

        def close(self):
            return None

    monkeypatch.setattr("lingtai.services.mcp.MCPClient", FakeMCPClient)
    monkeypatch.setenv("NOKV_BIN", "/tmp/nokv")
    monkeypatch.setenv("NOKV_MCP_ARGS", "[]")
    exec(compile(code, str(preflight_path), "exec"), {})


def _strict_nokv_preflight_tools():
    names = {
        "workbench_create", "workbench_put_file", "workbench_append",
        "workbench_edit", "workbench_stat", "workbench_list", "workbench_read",
        "workbench_grep", "workbench_search", "workbench_aggregate",
        "workbench_catalog", "workbench_find", "workbench_commit",
        "workbench_snapshot", "workbench_snapshot_renew",
        "workbench_snapshot_retire", "workbench_snapshot_list",
        "workbench_restore",
    }
    commit_schema = {
        "type": "object",
        "required": ["id", "manifest", "content_digest_uri"],
        "properties": {
            "id": {"type": "string"},
            "manifest": {"type": "object"},
            "content_digest_uri": {
                "type": "string",
                "pattern": "^sha256:[0-9a-f]{64}$",
                "description": (
                    "Stable caller-computed digest of the committed content. "
                    "It must be known before this call and exactly match "
                    "sha256:<64 lowercase hex>."
                ),
            },
            "replace": {
                "type": "boolean",
                "description": (
                    "Explicitly replace a different or legacy commit. "
                    "Concurrent identity changes still fail closed. Defaults false."
                ),
            },
        },
        "additionalProperties": False,
    }
    snapshot_schema = {
        "type": "object",
        "required": ["id"],
        "properties": {
            "id": {"type": "string"},
            "name": {
                "type": ["string", "null"],
                "description": (
                    "Checkpoint alias matching [A-Za-z0-9_-]{1,64}. Resolves "
                    "to this snapshot in workbench_snapshot_renew, "
                    "workbench_snapshot_list, and at_snapshot reads."
                ),
            },
            "ttl_days": {
                "type": "integer",
                "minimum": 1,
                "maximum": 90,
                "description": (
                    "Lease length in days. Defaults to 7; values above 90 are "
                    "rejected."
                ),
            },
            "reason": {
                "type": ["string", "null"],
                "minLength": 1,
                "maxLength": 256,
                "description": (
                    "Optional human-readable checkpoint reason. At most 256 "
                    "Unicode characters and 1024 UTF-8 bytes."
                ),
            },
            "metadata": {
                "type": ["object", "null"],
                "maxProperties": 64,
                "description": (
                    "Optional JSON annotation object. Canonical encoded size "
                    "is at most 4096 bytes, with at most 8 container levels "
                    "and 64 object keys across the complete value."
                ),
            },
        },
        "additionalProperties": False,
    }
    retire_schema = {
        "type": "object",
        "required": ["id"],
        "properties": {
            "id": {"type": "string", "minLength": 1},
            "snapshot_id": {
                "type": "integer",
                "minimum": 0,
                "description": (
                    "Snapshot id to retire. Provide exactly one of snapshot_id "
                    "or name."
                ),
            },
            "name": {
                "type": "string",
                "minLength": 1,
                "description": (
                    "Checkpoint name to retire. Provide exactly one of "
                    "snapshot_id or name."
                ),
            },
            "reason": {
                "type": ["string", "null"],
                "minLength": 1,
                "maxLength": 256,
                "description": (
                    "Optional human-readable retirement reason. At most 256 "
                    "Unicode characters and 1024 UTF-8 bytes."
                ),
            },
        },
        "oneOf": [
            {"required": ["snapshot_id"]},
            {"required": ["name"]},
        ],
        "additionalProperties": False,
    }
    restore_schema = {
        "type": "object",
        "required": ["id", "at_snapshot", "destination_id"],
        "properties": {
            "id": {"type": "string", "minLength": 1},
            "at_snapshot": {
                "anyOf": [
                    {"type": "integer", "minimum": 0},
                    {"type": "string", "minLength": 1},
                ]
            },
            "destination_id": {"type": "string", "minLength": 1},
        },
        "additionalProperties": False,
    }
    schemas = {
        "workbench_commit": commit_schema,
        "workbench_snapshot": snapshot_schema,
        "workbench_snapshot_retire": retire_schema,
        "workbench_restore": restore_schema,
    }
    return [
        {"name": name, "schema": schemas.get(name, {})}
        for name in sorted(names)
    ]


def test_nokv_preflight_executes_strict_raw_schema_gate(monkeypatch):
    _run_nokv_preflight_contract(monkeypatch, _strict_nokv_preflight_tools())


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing-surface-tool", "NoKV workbench tools missing"),
        ("capability-tool-absent", "NoKV workbench tools missing"),
        ("commit-content-identity", "workbench_commit raw schema mismatch"),
        ("snapshot-annotation", "workbench_snapshot raw schema mismatch"),
        ("retire-target", "workbench_snapshot_retire raw schema mismatch"),
        ("nullable-snapshot", "workbench_restore raw schema mismatch"),
        ("additional-properties", "workbench_restore raw schema mismatch"),
    ],
)
def test_nokv_preflight_rejects_contract_drift(monkeypatch, mutation, message):
    tools = _strict_nokv_preflight_tools()
    if mutation == "missing-surface-tool":
        tools = [tool for tool in tools if tool["name"] != "workbench_read"]
    elif mutation == "capability-tool-absent":
        tools = [tool for tool in tools if tool["name"] != "workbench_restore"]
    elif mutation == "commit-content-identity":
        commit = next(tool for tool in tools if tool["name"] == "workbench_commit")
        commit["schema"]["required"].remove("content_digest_uri")
    elif mutation == "snapshot-annotation":
        snapshot = next(
            tool for tool in tools if tool["name"] == "workbench_snapshot"
        )
        del snapshot["schema"]["properties"]["metadata"]
    elif mutation == "retire-target":
        retire = next(
            tool for tool in tools if tool["name"] == "workbench_snapshot_retire"
        )
        del retire["schema"]["oneOf"]
    else:
        restore = next(tool for tool in tools if tool["name"] == "workbench_restore")
        if mutation == "nullable-snapshot":
            restore["schema"]["properties"]["at_snapshot"]["anyOf"].append(
                {"type": "null"}
            )
        else:
            restore["schema"]["additionalProperties"] = True

    with pytest.raises(SystemExit, match=message):
        _run_nokv_preflight_contract(monkeypatch, tools)


def test_expand_agent_placeholders_scopes_workbench_root(tmp_path):
    # Per-agent root injection: a shared registry template resolves to a root
    # unique to each agent, so agents cannot address each other's workbenches.
    agent, workdir = _mk_agent(tmp_path)  # workdir.name == "agent"
    assert agent._expand_agent_placeholders("/agents/{agent_id}/wb") == "/agents/agent/wb"
    # {agent_address} is an alias for the stable working-dir name.
    assert agent._expand_agent_placeholders("/agents/{agent_address}/wb") == "/agents/agent/wb"
    # {agent_dir} expands to the absolute working directory.
    assert agent._expand_agent_placeholders("{agent_dir}/x") == f"{workdir}/x"
    # Strings without a placeholder and non-strings pass through untouched.
    assert agent._expand_agent_placeholders("--profile") == "--profile"
    assert agent._expand_agent_placeholders(None) is None


# ---------------------------------------------------------------------------
# Decompression
# ---------------------------------------------------------------------------

def test_decompress_appends_known_addon(tmp_path):
    rep = decompress_addons(tmp_path, ["imap"])
    assert rep["appended"] == ["imap"]
    assert rep["skipped"] == []
    records, problems = read_registry(tmp_path)
    assert [r["name"] for r in records] == ["imap"]
    assert problems == []


def test_decompress_is_idempotent(tmp_path):
    decompress_addons(tmp_path, ["imap"])
    rep2 = decompress_addons(tmp_path, ["imap"])
    assert rep2["appended"] == []
    assert rep2["skipped"] == ["imap"]
    records, _ = read_registry(tmp_path)
    assert len(records) == 1  # no duplicate


def test_decompress_unknown_addon_logged_not_raised(tmp_path):
    rep = decompress_addons(tmp_path, ["nonexistent"])
    assert rep["unknown"] == ["nonexistent"]
    assert rep["appended"] == []
    # Registry file may or may not exist — either is fine for unknown-only input.


def test_registry_drops_duplicates_by_name(tmp_path):
    registry = tmp_path / REGISTRY_FILENAME
    rec = {
        "name": "imap",
        "summary": "x",
        "transport": "stdio",
        "command": "a",
        "args": [],
        "source": "u",
    }
    registry.write_text(json.dumps(rec) + "\n" + json.dumps(rec) + "\n")
    records, problems = read_registry(tmp_path)
    assert len(records) == 1
    assert any("duplicate" in p["error"] for p in problems)


def test_registry_drops_invalid_lines(tmp_path):
    registry = tmp_path / REGISTRY_FILENAME
    valid = json.dumps({
        "name": "imap",
        "summary": "x",
        "transport": "stdio",
        "command": "a",
        "args": [],
        "source": "u",
    })
    registry.write_text(valid + "\n" + "not-json\n" + "{}\n")
    records, problems = read_registry(tmp_path)
    assert len(records) == 1
    assert len(problems) == 2


# ---------------------------------------------------------------------------
# Capability integration
# ---------------------------------------------------------------------------

def test_addons_list_triggers_decompression(tmp_path):
    agent, workdir = _mk_agent(tmp_path, addons=["imap"])
    registry_path = workdir / REGISTRY_FILENAME
    assert registry_path.is_file()
    records, problems = read_registry(workdir)
    assert [r["name"] for r in records] == ["imap"]
    assert problems == []


def test_mcp_capability_renders_registry_into_prompt(tmp_path):
    agent, workdir = _mk_agent(tmp_path, addons=["imap"])
    section = agent._prompt_manager._sections.get("mcp")
    assert section is not None
    body = section.body if hasattr(section, "body") else str(section)
    assert "<registered_mcp>" in body
    assert "imap" in body
    # Catalog ships the imap homepage; render should surface it.
    assert "<homepage>" in body
    assert "github.com/Lingtai-AI/lingtai-imap" in body


def test_addons_dict_still_works_for_legacy(tmp_path):
    """Legacy dict shape should not break — addon load may fail without
    config but the agent must not raise."""
    # Don't actually load IMAP (no config); just ensure the dict path is taken.
    agent, workdir = _mk_agent(tmp_path, addons={})
    # Should construct fine; no decompression should have happened.
    registry_path = workdir / REGISTRY_FILENAME
    assert not registry_path.exists()


def test_mcp_show_action_returns_health_snapshot(tmp_path):
    agent, workdir = _mk_agent(tmp_path, addons=["imap"])
    handler = agent._tool_handlers.get("mcp")
    assert handler is not None
    result = handler({"action": "info"})
    assert result["status"] == "ok"
    assert result["registered_count"] == 1
    assert result["registered"][0]["name"] == "imap"
    assert "mcp_manual" not in result
    manual = handler({"action": "manual"})
    assert manual["status"] == "ok"
    assert "mcp_manual" in manual and manual["mcp_manual"]  # umbrella SKILL.md body


def test_mcp_manual_preserves_tui_command_boundary():
    """The shipped MCP router must not route setup to a retired TUI surface."""
    manual_root = Path(__file__).resolve().parents[1] / "src/lingtai/tools/mcp/manual"
    router = (manual_root / "SKILL.md").read_text(encoding="utf-8")
    curated = (manual_root / "reference/curated-addons.md").read_text(encoding="utf-8")

    assert re.search(r'/addon.{0,120}(?:retired|never recommended)', router, re.I | re.S)
    assert not re.search(r'(?:use|open|run|launch|recommend)[ \t]+`?/addon', router, re.I)
    assert re.search(r'/mcp.{0,140}only current TUI command', router, re.I | re.S)
    assert re.search(r'/mcp.{0,180}read[- ]only', router, re.I | re.S)
    assert re.search(
        r"/mcp.{0,240}(?:not|isn't).{0,90}(?:guided[ \t]+)?(?:setup|configuration)",
        router,
        re.I | re.S,
    )
    assert re.search(
        r'curated addon setup.{0,220}(?:curated-addons.*contract|provider docs)',
        router,
        re.I | re.S,
    )
    assert re.search(r'explicit human authorization', router, re.I)

    # Keep the existing four-step mechanism in the owning reference while the
    # router adds only the TUI boundary and authorization rule.
    assert re.search(r'## The four-step setup', curated)
    for step in (
        r'1[.].*read.*setup docs',
        r'2[.].*init[.]json',
        r'3[.].*config file',
        r"4[.].*system[(]action=.*refresh.*[)]",
    ):
        assert re.search(step, curated, re.I | re.S)


def test_mcp_show_unknown_action_returns_error(tmp_path):
    agent, workdir = _mk_agent(tmp_path, addons=["imap"])
    handler = agent._tool_handlers.get("mcp")
    result = handler({"action": "register"})  # not supported in slice
    assert result["status"] == "error"
    # Exact model-visible envelope must survive the dispatch-helper migration
    # (issue #513).
    assert result == {
        "status": "error",
        "message": "unknown action: 'register', only 'info' or 'manual' is supported",
    }
    # Missing action key renders the empty-string default, not None.
    assert handler({}) == {
        "status": "error",
        "message": "unknown action: '', only 'info' or 'manual' is supported",
    }
    # Invalid JSON can make `action` unhashable (issue #513 blocker): the router
    # must render the unknown-action envelope, not raise TypeError.
    assert handler({"action": []}) == {
        "status": "error",
        "message": "unknown action: [], only 'info' or 'manual' is supported",
    }
    assert handler({"action": {}}) == {
        "status": "error",
        "message": "unknown action: {}, only 'info' or 'manual' is supported",
    }


# ---------------------------------------------------------------------------
# Loader gating
# ---------------------------------------------------------------------------

def test_loader_skips_unregistered_init_mcp(tmp_path, caplog):
    """init.json mcp entry not in registry should be skipped with a warning."""
    workdir = tmp_path / "agent"
    workdir.mkdir(parents=True)
    # Pre-create init.json with an unregistered mcp entry.
    init = {
        "mcp": {
            "rogue": {"type": "stdio", "command": "false", "args": []},
        },
    }
    (workdir / "init.json").write_text(json.dumps(init))

    Agent(
        service=make_mock_service(),
        agent_name="test",
        working_dir=workdir,
        capabilities={"mcp": {}},
        # No addons → registry is empty → rogue should be skipped.
    )

    # We can't easily intercept the kernel logger here, but the registry stays empty
    # and no MCP client should have been added.
    # (The legacy mcp/servers.json path is also untouched.)


# ---------------------------------------------------------------------------
# Failed-MCP retry on refresh — regression for Lingtai-AI/lingtai#34
# ---------------------------------------------------------------------------

class _FakeMCPClient:
    """Minimal stand-in for MCPClient/HTTPMCPClient.

    `is_connected_value` controls health probes; tool list is empty so the
    Agent's tool registration loop is a no-op (no need to fake schemas).
    """

    def __init__(self, is_connected_value: bool):
        self._connected = is_connected_value
        self.closed = False

    def start(self):
        return None

    def is_connected(self) -> bool:
        return self._connected and not self.closed

    def list_tools(self, timeout: float = 10):
        return []

    def close(self):
        self.closed = True


def test_retry_failed_mcps_records_dead_then_recovers(tmp_path, monkeypatch):
    """A registered init.json MCP that boots dead should be retried on
    `_retry_failed_mcps()` and reported as recovered when the second attempt
    succeeds. Regression for Lingtai-AI/lingtai#34."""
    workdir = tmp_path / "agent"
    workdir.mkdir(parents=True)
    # Pre-stage registry so the init.json mcp entry passes the gate.
    (workdir / "mcp_registry.jsonl").write_text(json.dumps({
        "name": "telegram",
        "summary": "test",
        "transport": "stdio",
        "command": "/bin/true",
        "args": [],
        "source": "user",
    }) + "\n")
    init = {
        "mcp": {
            "telegram": {"type": "stdio", "command": "/bin/true", "args": []},
        },
    }
    (workdir / "init.json").write_text(json.dumps(init))

    # Inject transport clients while retaining the production activation
    # transaction: first launch is dead, the retry is live.
    call_count = {"n": 0}

    def fake_client_factory(**kwargs):
        call_count["n"] += 1
        return _FakeMCPClient(is_connected_value=(call_count["n"] >= 2))

    monkeypatch.setattr(
        "lingtai.services.mcp.MCPClient", fake_client_factory
    )

    agent = Agent(
        service=make_mock_service(),
        agent_name="test",
        working_dir=workdir,
        capabilities={"mcp": {}},
    )

    # Boot recorded the spec, but the tracked client is dead.
    assert "telegram" in agent._mcp_init_specs
    boot_client = agent._mcp_init_specs["telegram"]["client"]
    assert boot_client is not None
    assert not boot_client.is_connected()

    # Retry: should detect death, close+remove, respawn — second spawn
    # returns a live client → reported as recovered.
    report = agent._retry_failed_mcps()
    assert "telegram" in report["retried"]
    assert "telegram" in report["recovered"]
    assert report["still_failed"] == []
    # The dead client should have been closed and dropped.
    assert boot_client.closed
    assert boot_client not in agent._mcp_clients
    # New client tracked.
    new_client = agent._mcp_init_specs["telegram"]["client"]
    assert new_client is not None and new_client.is_connected()


def test_retry_failed_mcps_skips_healthy(tmp_path, monkeypatch):
    """A live MCP should be reported as `healthy`, not retried."""
    workdir = tmp_path / "agent"
    workdir.mkdir(parents=True)
    (workdir / "mcp_registry.jsonl").write_text(json.dumps({
        "name": "telegram",
        "summary": "test",
        "transport": "stdio",
        "command": "/bin/true",
        "args": [],
        "source": "user",
    }) + "\n")
    (workdir / "init.json").write_text(json.dumps({
        "mcp": {"telegram": {"type": "stdio", "command": "/bin/true"}},
    }))

    monkeypatch.setattr(
        "lingtai.services.mcp.MCPClient",
        lambda **kwargs: _FakeMCPClient(is_connected_value=True),
    )

    agent = Agent(
        service=make_mock_service(),
        agent_name="test",
        working_dir=workdir,
        capabilities={"mcp": {}},
    )

    report = agent._retry_failed_mcps()
    assert report["retried"] == []
    assert report["recovered"] == []
    assert report["still_failed"] == []
    assert "telegram" in report["healthy"]


def test_retry_failed_mcps_no_specs_is_noop(tmp_path):
    """An agent with no init.json mcp entries should return an empty
    report — never raise, never assume `_mcp_init_specs` exists."""
    workdir = tmp_path / "agent"
    workdir.mkdir(parents=True)
    agent = Agent(
        service=make_mock_service(),
        agent_name="test",
        working_dir=workdir,
        capabilities={"mcp": {}},
    )
    report = agent._retry_failed_mcps()
    assert report == {"retried": [], "recovered": [],
                      "still_failed": [], "healthy": []}


def test_retry_failed_mcps_no_specs_reports_unresolved_pending_retirement(tmp_path):
    agent, _ = _mk_agent(tmp_path)

    class _Unclosable(_FakeMCPClient):
        def close(self):
            raise RuntimeError("close blocked")

    pending = _Unclosable(is_connected_value=False)
    agent._mcp_pending_retirements["candidate:pending"] = pending

    report = agent._retry_failed_mcps()

    assert report["still_failed"] == ["candidate:pending"]
    assert agent._mcp_pending_retirements["candidate:pending"] is pending


def test_retry_failed_mcps_stopping_is_fail_closed_without_init_specs(tmp_path):
    agent, _ = _mk_agent(tmp_path)
    agent._mcp_lifecycle_state = "stopping"
    agent._mcp_lifecycle_barrier.set()

    report = agent._retry_failed_mcps()

    assert report["retried"] == []
    assert report["still_failed"] == ["lifecycle:stopping"]


def test_public_refresh_that_owns_lifecycle_lock_wins_race_with_stop(
    tmp_path, monkeypatch
):
    agent, _ = _mk_agent(tmp_path)
    agent._refresh_watcher = make_test_refresh_watcher()
    agent._mcp_activation_lock.acquire()
    stopping = threading.Thread(target=agent.stop)
    stopping.start()
    try:
        result = agent._intrinsics["system"]({"action": "refresh"})
        assert result["status"] == "ok"
        assert len(agent._refresh_watcher.calls) == 1
        assert agent._mcp_lifecycle_state == "relaunching"
        assert agent._mcp_lifecycle_barrier.is_set()
    finally:
        agent._mcp_activation_lock.release()
    stopping.join(timeout=2.0)

    assert not stopping.is_alive()
    assert agent._mcp_lifecycle_state == "stopping"
    assert agent._mcp_lifecycle_barrier.is_set()


def test_refresh_stop_after_precondition_completes_refresh_before_stop(
    tmp_path, monkeypatch
):
    agent, workdir = _mk_agent(tmp_path)
    (workdir / "init.json").write_text(
        json.dumps({
            "manifest": {"preset": {"allowed": ["next"]}},
        })
    )
    monkeypatch.setattr(
        agent,
        "load_preset",
        lambda name: {"manifest": {}},
    )
    activated = []
    performed = []
    monkeypatch.setattr(agent, "_activate_preset", activated.append)
    monkeypatch.setattr(
        agent,
        "_perform_refresh",
        lambda **kwargs: _committed_refresh(performed, **kwargs),
    )
    stop_thread = None

    def retry_then_request_stop():
        nonlocal stop_thread
        stop_thread = threading.Thread(target=agent.stop)
        stop_thread.start()
        return {
            "retried": [],
            "recovered": [],
            "still_failed": [],
            "healthy": [],
        }

    monkeypatch.setattr(agent, "_retry_failed_mcps", retry_then_request_stop)

    result = agent._intrinsics["system"]({
        "action": "refresh",
        "preset": "next",
    })
    assert stop_thread is not None
    stop_thread.join(timeout=2.0)

    assert not stop_thread.is_alive()
    assert result["status"] == "ok"
    assert activated == ["next"]
    assert performed == [True]
    assert agent._mcp_lifecycle_state == "stopping"
    assert agent._mcp_stop_requested.is_set()
    assert agent._mcp_lifecycle_barrier.is_set()


def test_deep_refresh_never_clears_pending_stop(tmp_path):
    agent, _ = _mk_agent(tmp_path)
    agent.stop()
    with pytest.raises(RuntimeError, match="pending stop"):
        agent._setup_from_init()
    assert agent._mcp_lifecycle_state == "stopping"
    assert agent._mcp_lifecycle_barrier.is_set()


def test_stop_after_deep_refresh_final_check_keeps_terminal_barrier(
    tmp_path, monkeypatch
):
    agent, _ = _mk_agent(tmp_path)
    stopping = None
    stop_attempted = threading.Event()

    def request_stop():
        nonlocal stopping
        def stop():
            stop_attempted.set()
            agent.stop()

        stopping = threading.Thread(target=stop)
        stopping.start()
        assert stop_attempted.wait(timeout=1.0)

    monkeypatch.setattr(agent, "_setup_from_init_locked", request_stop)
    agent._setup_from_init()
    assert stopping is not None
    stopping.join(timeout=2.0)

    assert not stopping.is_alive()
    assert agent._mcp_lifecycle_state == "stopping"
    assert agent._mcp_lifecycle_barrier.is_set()


def test_successful_refresh_handoff_remains_terminal_in_old_process(
    tmp_path, monkeypatch
):
    agent, _ = _mk_agent(tmp_path)
    agent._refresh_watcher = make_test_refresh_watcher()

    result = agent._intrinsics["system"]({"action": "refresh"})

    assert result["status"] == "ok"
    assert agent._refresh_watcher.spawned
    assert agent._shutdown.is_set()
    assert agent._mcp_lifecycle_state == "relaunching"
    assert agent._mcp_lifecycle_barrier.is_set()
    candidate = _ActivationClient([_tool("late")])
    with pytest.raises(RuntimeError, match="relaunching"):
        agent._activate_mcp_candidate(candidate)
    assert candidate.closed


def test_system_refresh_ack_failure_does_not_commit_terminal_handoff(
    tmp_path, monkeypatch
):
    agent, workdir = _mk_agent(tmp_path)
    agent._refresh_watcher = make_test_refresh_watcher()
    real_touch = Path.touch

    def fail_refresh_ack(self, *args, **kwargs):
        if self.name == ".refresh.taken":
            raise OSError("simulated ack write failure")
        return real_touch(self, *args, **kwargs)

    monkeypatch.setattr(Path, "touch", fail_refresh_ack)
    result = agent._intrinsics["system"]({"action": "refresh"})

    assert result["status"] == "error"
    assert ".refresh.taken" in result["message"]
    assert not agent._refresh_watcher.spawned
    assert not agent._shutdown.is_set()
    assert agent._mcp_lifecycle_state == "active"
    assert not agent._mcp_lifecycle_barrier.is_set()
    assert not (workdir / ".refresh.taken").exists()


def test_system_refresh_no_launch_command_restores_active(
    tmp_path, monkeypatch
):
    agent, _ = _mk_agent(tmp_path)
    monkeypatch.setattr(agent, "_build_launch_cmd", lambda: None)

    result = agent._intrinsics["system"]({"action": "refresh"})

    assert result["status"] == "error"
    assert "no relaunch command" in result["message"]
    assert not agent._shutdown.is_set()
    assert agent._mcp_lifecycle_state == "active"
    assert not agent._mcp_lifecycle_barrier.is_set()


def test_system_refresh_watcher_failure_restores_active(
    tmp_path, monkeypatch
):
    agent, _ = _mk_agent(tmp_path)

    def fail_spawn(request):
        raise OSError("simulated watcher failure")

    monkeypatch.setattr(agent._refresh_watcher, "spawn_detached", fail_spawn)
    result = agent._intrinsics["system"]({"action": "refresh"})

    assert result["status"] == "error"
    assert "watcher failed to start" in result["message"]
    assert not agent._shutdown.is_set()
    assert agent._mcp_lifecycle_state == "active"
    assert not agent._mcp_lifecycle_barrier.is_set()


def test_system_refresh_shutdown_signal_failure_is_terminal_and_not_respawned(tmp_path):
    agent, _ = _mk_agent(tmp_path)
    agent._refresh_watcher = make_test_refresh_watcher()

    class _BrokenShutdown:
        def set(self):
            raise RuntimeError("simulated shutdown signal failure")

        def is_set(self):
            return False

    agent._shutdown = _BrokenShutdown()
    first = agent._intrinsics["system"]({"action": "refresh"})
    second = agent._intrinsics["system"]({"action": "refresh"})

    assert first["status"] == "error"
    assert "committed with degraded post-spawn completion" in first["message"]
    assert "shutdown signaling failed" in first["message"]
    assert second["status"] == "error"
    assert "terminal lifecycle transition is pending" in second["message"]
    assert len(agent._refresh_watcher.calls) == 1
    assert agent._mcp_lifecycle_state == "relaunching"
    assert agent._mcp_lifecycle_barrier.is_set()


def test_system_refresh_post_spawn_log_failure_is_terminal_and_not_respawned(
    tmp_path,
):
    agent, _ = _mk_agent(tmp_path)
    agent._refresh_watcher = make_test_refresh_watcher()
    real_log = agent._log

    def fail_deferred_relaunch_log(event, **fields):
        if event == "refresh_deferred_relaunch":
            raise RuntimeError("simulated post-spawn telemetry failure")
        return real_log(event, **fields)

    agent._log = fail_deferred_relaunch_log

    first = agent._intrinsics["system"]({"action": "refresh"})
    second = agent._intrinsics["system"]({"action": "refresh"})

    assert first["status"] == "error"
    assert "committed with degraded post-spawn completion" in first["message"]
    assert "post-spawn deferred-relaunch telemetry failed" in first["message"]
    assert second["status"] == "error"
    assert "terminal lifecycle transition is pending" in second["message"]
    assert len(agent._refresh_watcher.calls) == 1
    assert agent._mcp_lifecycle_state == "relaunching"
    assert agent._mcp_lifecycle_barrier.is_set()


def test_stop_winning_before_preset_refresh_preserves_exact_init_bytes(
    tmp_path, monkeypatch
):
    agent, workdir = _mk_agent(tmp_path)
    init_path = workdir / "init.json"
    init_path.write_text(
        '{\n  "manifest": {"preset": {"allowed": ["next"]}},\n'
        '  "sentinel": "exact bytes"\n}\n'
    )
    before = init_path.read_bytes()
    performed = []
    monkeypatch.setattr(
        agent,
        "load_preset",
        lambda name: {"manifest": {"llm": {}, "capabilities": {}}},
    )
    monkeypatch.setattr(
        agent,
        "_perform_refresh",
        lambda **kwargs: _committed_refresh(performed, **kwargs),
    )

    agent.stop()
    result = agent._intrinsics["system"](
        {"action": "refresh", "preset": "next"}
    )

    assert result["status"] == "error"
    assert "lifecycle:stopping" in result["message"]
    assert init_path.read_bytes() == before
    assert performed == []
    assert agent._mcp_lifecycle_barrier.is_set()


@pytest.mark.parametrize("window", ["activation", "default_update"])
def test_stop_during_preset_mutation_linearizes_after_complete_refresh(
    tmp_path, monkeypatch, window
):
    agent, workdir = _mk_agent(tmp_path)
    init_path = workdir / "init.json"
    init_path.write_text(
        json.dumps(
            {"manifest": {"preset": {"allowed": ["next.json"]}}},
            indent=2,
        )
    )
    (workdir / "next.json").write_text(
        json.dumps(
            {
                "name": "next",
                "description": {"summary": "next"},
                "manifest": {
                    "llm": {"provider": "gemini", "model": "gemini-test"},
                    "capabilities": {},
                },
            }
        )
    )
    before = init_path.read_bytes()
    monkeypatch.setattr(
        agent,
        "load_preset",
        lambda name: {
            "manifest": {
                "llm": {"provider": "gemini", "model": "gemini-test"},
                "capabilities": {},
            }
        },
    )
    performed = []
    monkeypatch.setattr(
        agent,
        "_perform_refresh",
        lambda **kwargs: _committed_refresh(performed, **kwargs),
    )
    stopping = None
    stop_attempted = threading.Event()

    def start_stop():
        nonlocal stopping
        def stop():
            stop_attempted.set()
            agent.stop()

        stopping = threading.Thread(target=stop)
        stopping.start()
        assert stop_attempted.wait(timeout=1.0)

    if window == "activation":
        activate_preset = agent._activate_preset

        def activate_while_stopping(name):
            start_stop()
            activate_preset(name)

        monkeypatch.setattr(agent, "_activate_preset", activate_while_stopping)
    else:
        monkeypatch.setattr(agent, "_activate_preset", lambda name: None)
        preset_module = importlib.import_module("lingtai.tools.system.preset")
        update_default = preset_module._update_default_preset

        def update_while_stopping(target_agent, name):
            start_stop()
            update_default(target_agent, name)

        monkeypatch.setattr(
            preset_module, "_update_default_preset", update_while_stopping
        )

    result = agent._intrinsics["system"](
        {"action": "refresh", "preset": "next.json"}
    )
    assert stopping is not None
    stopping.join(timeout=2.0)

    assert not stopping.is_alive()
    assert result["status"] == "ok", result
    assert init_path.read_bytes() != before
    assert performed == [True]
    assert agent._mcp_lifecycle_state == "stopping"
    assert agent._mcp_lifecycle_barrier.is_set()


class _ActivationClient:
    def __init__(self, tools, *, list_error=None, on_start=None):
        self.tools = tools
        self.list_error = list_error
        self.on_start = on_start
        self.started = False
        self.closed = False

    def start(self):
        if self.on_start:
            self.on_start()
        self.started = True

    def list_tools(self, timeout=10):
        if self.list_error:
            raise self.list_error
        return copy.deepcopy(self.tools)

    def is_connected(self):
        return self.started and not self.closed

    def close(self):
        self.closed = True

    def call_tool(self, name, args):
        return {"status": "success", "name": name, "args": args}


def _tool(name):
    return {
        "name": name,
        "description": f"{name} description",
        "schema": {"type": "object", "properties": {}},
    }


def test_mcp_activation_listing_failure_cleans_unpublished_candidate(
    tmp_path, monkeypatch
):
    agent, _ = _mk_agent(tmp_path)
    candidate = _ActivationClient([], list_error=RuntimeError("list failed"))
    monkeypatch.setattr(
        "lingtai.services.mcp.MCPClient",
        lambda **kwargs: candidate,
    )
    handlers = agent._tool_handlers
    schemas = agent._tool_schemas

    with pytest.raises(RuntimeError, match="list failed"):
        agent.connect_mcp("/bin/false")

    assert candidate.closed
    assert agent._mcp_clients == []
    assert agent._mcp_clients_by_tool == {}
    assert agent._mcp_tool_names == set()
    assert agent._tool_handlers is handlers
    assert agent._tool_schemas is schemas


def test_mcp_activation_collision_rejects_whole_candidate_before_mutation(
    tmp_path, monkeypatch
):
    agent, _ = _mk_agent(tmp_path)
    candidate = _ActivationClient([_tool("system"), _tool("free_name")])
    monkeypatch.setattr(
        "lingtai.services.mcp.MCPClient",
        lambda **kwargs: candidate,
    )
    handlers = agent._tool_handlers
    schemas = agent._tool_schemas

    with pytest.raises(RuntimeError, match="built-in/reserved"):
        agent.connect_mcp("/bin/false")

    assert candidate.closed
    assert agent._tool_handlers is handlers
    assert agent._tool_schemas is schemas
    assert "free_name" not in agent._tool_handlers


def test_mcp_activation_duplicate_names_fail_closed(tmp_path, monkeypatch):
    agent, _ = _mk_agent(tmp_path)
    candidate = _ActivationClient([_tool("dup"), _tool("dup")])
    monkeypatch.setattr(
        "lingtai.services.mcp.MCPClient",
        lambda **kwargs: candidate,
    )

    with pytest.raises(ValueError, match="duplicate"):
        agent.connect_mcp("/bin/false")

    assert candidate.closed
    assert "dup" not in agent._tool_handlers


@pytest.mark.parametrize(
    ("catalog", "error"),
    [
        ({}, "array"),
        ([None], "object"),
        ([{"name": None}], "invalid name"),
        ([{"name": "  "}], "invalid name"),
        ([{"name": "x", "description": 7}], "description"),
        ([{"name": "x", "schema": []}], "schema must be an object"),
        (
            [{"name": "x", "schema": {"properties": []}}],
            "properties must be an object",
        ),
        (
            [{"name": "x", "schema": {"required": "x"}}],
            "required must be an array",
        ),
        (
            [{"name": "x", "schema": {"required": [1]}}],
            "required must be an array",
        ),
        (
            [{"name": "x", "schema": {"properties": {"x": {1}}}}],
            "not JSON serializable",
        ),
    ],
)
def test_mcp_candidate_catalog_malformed_matrix_is_atomic(
    tmp_path, catalog, error
):
    agent, _ = _mk_agent(tmp_path)
    candidate = _ActivationClient(catalog)
    spec = {"cfg": {}, "source": "init.json:mcp", "client": None}
    agent._mcp_init_specs = {"candidate": spec}
    handlers = agent._tool_handlers
    schemas = agent._tool_schemas
    owners = agent._mcp_clients_by_tool
    clients = agent._mcp_clients
    names = agent._mcp_tool_names

    with pytest.raises((TypeError, ValueError), match=error):
        agent._activate_mcp_candidate(
            candidate,
            allow_sealed=True,
            init_spec_name="candidate",
            predecessor=None,
        )

    assert candidate.closed
    assert agent._tool_handlers is handlers
    assert agent._tool_schemas is schemas
    assert agent._mcp_clients_by_tool is owners
    assert agent._mcp_clients is clients
    assert agent._mcp_tool_names is names
    assert agent._mcp_init_specs["candidate"] is spec
    assert spec["client"] is None


@pytest.mark.parametrize("case", ["missing", "wrong", "unnamed_predecessor"])
def test_mcp_init_spec_identity_is_validated_before_candidate_start(tmp_path, case):
    agent, _ = _mk_agent(tmp_path)
    candidate = _ActivationClient([_tool("candidate")])
    predecessor = None
    init_spec_name = "example"
    if case == "wrong":
        agent._mcp_init_specs = {
            "example": {"client": _ActivationClient([]), "cfg": {}}
        }
    elif case == "unnamed_predecessor":
        init_spec_name = None
        predecessor = _ActivationClient([])
    handlers = agent._tool_handlers
    schemas = agent._tool_schemas
    owners = agent._mcp_clients_by_tool
    clients = agent._mcp_clients
    names = agent._mcp_tool_names

    with pytest.raises(RuntimeError):
        agent._activate_mcp_candidate(
            candidate,
            allow_sealed=True,
            init_spec_name=init_spec_name,
            predecessor=predecessor,
        )

    assert not candidate.started
    assert candidate.closed
    assert agent._tool_handlers is handlers
    assert agent._tool_schemas is schemas
    assert agent._mcp_clients_by_tool is owners
    assert agent._mcp_clients is clients
    assert agent._mcp_tool_names is names


class _RecordingChat:
    def __init__(self, fail_calls=()):
        self.calls = []
        self.fail_calls = set(fail_calls)

    def update_tools(self, schemas):
        self.calls.append(tuple(schema.name for schema in schemas))
        if len(self.calls) in self.fail_calls:
            raise RuntimeError(f"chat update {len(self.calls)} failed")


def test_initial_activation_failure_compensates_live_chat(tmp_path, monkeypatch):
    agent, _ = _mk_agent(tmp_path)
    chat = _RecordingChat()
    agent._chat = chat
    candidate = _ActivationClient([_tool("candidate")])
    agent._mcp_init_specs = {"candidate": {"cfg": {}, "client": None}}
    monkeypatch.setattr(
        agent,
        "_maybe_setup_task_card_controller",
        lambda: (_ for _ in ()).throw(RuntimeError("task-card setup failed")),
    )

    with pytest.raises(RuntimeError, match="task-card setup failed"):
        agent._activate_mcp_candidate(
            candidate, allow_sealed=True, init_spec_name="candidate"
        )

    assert chat.calls[-1].count("candidate") == 0
    assert "candidate" not in agent._tool_handlers
    assert candidate.closed


def test_replacement_failure_compensates_chat_after_predecessor_retirement(
    tmp_path, monkeypatch
):
    agent, _ = _mk_agent(tmp_path)
    predecessor = _ActivationClient([])
    _install_dead_projection(agent, "example", predecessor)
    chat = _RecordingChat()
    agent._chat = chat
    candidate = _ActivationClient([_tool("candidate")])
    monkeypatch.setattr(
        agent,
        "_maybe_setup_task_card_controller",
        lambda: (_ for _ in ()).throw(RuntimeError("task-card setup failed")),
    )

    with pytest.raises(RuntimeError, match="task-card setup failed"):
        agent._activate_mcp_candidate(
            candidate,
            allow_sealed=True,
            init_spec_name="example",
            predecessor=predecessor,
        )

    assert predecessor.closed
    assert chat.calls[-1].count("old") == 0
    assert chat.calls[-1].count("candidate") == 0
    assert "old" not in agent._tool_handlers
    assert "candidate" not in agent._tool_handlers


def test_chat_compensation_failure_is_composite_evidence(tmp_path, monkeypatch):
    agent, _ = _mk_agent(tmp_path)
    agent._chat = _RecordingChat(fail_calls={2})
    candidate = _ActivationClient([_tool("candidate")])
    agent._mcp_init_specs = {"candidate": {"cfg": {}, "client": None}}
    monkeypatch.setattr(
        agent,
        "_maybe_setup_task_card_controller",
        lambda: (_ for _ in ()).throw(RuntimeError("task-card setup failed")),
    )

    with pytest.raises(
        RuntimeError,
        match="task-card setup failed.*live-chat compensation unresolved",
    ):
        agent._activate_mcp_candidate(
            candidate, allow_sealed=True, init_spec_name="candidate"
        )

    assert candidate.closed


def test_mcp_activation_publication_failure_exact_restores_initial_state(
    tmp_path, monkeypatch
):
    agent, _ = _mk_agent(tmp_path)
    candidate = _ActivationClient([_tool("candidate_tool")])
    monkeypatch.setattr(
        "lingtai.services.mcp.MCPClient",
        lambda **kwargs: candidate,
    )
    handlers = agent._tool_handlers
    schemas = agent._tool_schemas
    agent._mcp_init_specs = {
        "candidate": {"cfg": {}, "source": "init.json:mcp", "client": None}
    }
    monkeypatch.setattr(
        agent,
        "_maybe_setup_task_card_controller",
        lambda: (_ for _ in ()).throw(RuntimeError("publish hook failed")),
    )

    with pytest.raises(RuntimeError, match="publish hook failed"):
        agent._activate_mcp_candidate(
            candidate,
            allow_sealed=True,
            init_spec_name="candidate",
        )

    assert candidate.closed
    assert agent._tool_handlers is handlers
    assert agent._tool_schemas is schemas
    assert agent._mcp_clients == []
    assert agent._mcp_clients_by_tool == {}
    assert agent._mcp_tool_names == set()
    assert agent._mcp_init_specs["candidate"]["client"] is None


@pytest.mark.parametrize(
    "damage",
    [
        "omitted",
        "extra",
        "duplicate",
        "handler_only",
        "owner_only",
        "name_set_only",
        "foreign_owner_name",
        "handler_schema_client_mismatch",
        "duplicate_client_identity",
        "owner_absent_live",
        "zero_schema_foreign",
        "duplicate_schema_foreign",
        "handler_capture_mismatch",
    ],
)
def test_activation_outcome_requires_exact_complete_projection(tmp_path, damage):
    agent, _ = _mk_agent(tmp_path)
    client = _ActivationClient([_tool("one"), _tool("two")])
    agent._mcp_init_specs = {
        "candidate": {"cfg": {}, "source": "init.json:mcp", "client": None}
    }
    outcome = agent._activate_mcp_candidate(
        client, allow_sealed=True, init_spec_name="candidate"
    )
    names = outcome.tool_names
    if damage == "omitted":
        names = ("one",)
    elif damage == "extra":
        names = (*names, "extra")
    elif damage == "duplicate":
        names = (*names, "one")
    elif damage == "handler_only":
        agent._tool_handlers["handler_only"] = agent._make_mcp_handler(
            client, "handler_only"
        )
    elif damage == "name_set_only":
        agent._mcp_tool_names.add("name_set_only")
    elif damage == "foreign_owner_name":
        foreign = _ActivationClient([])
        agent._mcp_clients.append(foreign)
        agent._mcp_clients_by_tool["foreign"] = foreign
        agent._mcp_tool_names.add("foreign")
    elif damage == "handler_schema_client_mismatch":
        foreign = _ActivationClient([])
        agent._mcp_clients.append(foreign)
        agent._mcp_clients_by_tool["mismatch"] = foreign
        agent._mcp_tool_names.add("mismatch")
        agent._tool_handlers["mismatch"] = agent._make_mcp_handler(
            client, "mismatch"
        )
        agent._tool_schemas.append(
            FunctionSchema(
                name="mismatch",
                description="mismatch",
                parameters={"type": "object", "properties": {}},
            )
        )
    elif damage == "duplicate_client_identity":
        foreign = _ActivationClient([])
        agent._mcp_clients.extend([foreign, foreign])
    elif damage in {
        "owner_absent_live",
        "zero_schema_foreign",
        "duplicate_schema_foreign",
    }:
        foreign = _ActivationClient([])
        if damage != "owner_absent_live":
            agent._mcp_clients.append(foreign)
        agent._mcp_clients_by_tool["foreign"] = foreign
        agent._mcp_tool_names.add("foreign")
        agent._tool_handlers["foreign"] = agent._make_mcp_handler(
            foreign, "foreign"
        )
        if damage != "zero_schema_foreign":
            schema = FunctionSchema(
                name="foreign",
                description="foreign",
                parameters={"type": "object", "properties": {}},
            )
            agent._tool_schemas.append(schema)
            if damage == "duplicate_schema_foreign":
                agent._tool_schemas.append(schema)
    elif damage == "handler_capture_mismatch":
        agent._tool_handlers["one"]._lingtai_mcp_tool_name = "wrong"
    else:
        agent._mcp_clients_by_tool["owner_only"] = client
        agent._mcp_tool_names.add("owner_only")

    expected_error = (
        "global client-list identity mismatch"
        if damage == "duplicate_client_identity"
        else "duplicate|projection|mismatch"
    )
    with pytest.raises(RuntimeError, match=expected_error):
        agent._assert_mcp_activation_outcome(
            "candidate",
            MCPActivationOutcome(client=client, tool_names=tuple(names)),
        )


def test_mcp_activation_healthy_foreign_collision_preserves_identity(tmp_path):
    agent, _ = _mk_agent(tmp_path)
    owner = _ActivationClient([_tool("claimed")])
    owner.started = True
    owner_handler = agent._make_mcp_handler(owner, "claimed")
    owner_schema = FunctionSchema(
        name="claimed",
        description="claimed",
        parameters={"type": "object", "properties": {}},
    )
    agent._mcp_clients.append(owner)
    agent._mcp_clients_by_tool["claimed"] = owner
    agent._mcp_tool_names.add("claimed")
    agent._tool_handlers["claimed"] = owner_handler
    agent._tool_schemas.append(owner_schema)
    candidate = _ActivationClient([_tool("claimed")])

    with pytest.raises(RuntimeError, match="healthy foreign MCP"):
        agent._activate_mcp_candidate(candidate)

    assert candidate.closed
    assert agent._tool_handlers["claimed"] is owner_handler
    assert agent._mcp_clients_by_tool["claimed"] is owner
    assert [s for s in agent._tool_schemas if s.name == "claimed"] == [owner_schema]
    assert agent._mcp_clients == [owner]


def test_mcp_activation_inconsistent_collision_preserves_identity(tmp_path):
    agent, _ = _mk_agent(tmp_path)
    existing = lambda args: args
    agent._tool_handlers["orphan"] = existing
    schemas = agent._tool_schemas
    candidate = _ActivationClient([_tool("orphan")])

    with pytest.raises(RuntimeError, match="built-in handler"):
        agent._activate_mcp_candidate(candidate)

    assert candidate.closed
    assert agent._tool_handlers["orphan"] is existing
    assert agent._tool_schemas is schemas
    assert agent._mcp_clients == []


def test_mcp_predecessor_must_be_exact_init_spec_identity(tmp_path):
    agent, _ = _mk_agent(tmp_path)
    recorded = _ActivationClient([])
    candidate = _ActivationClient([_tool("new")])
    other = _ActivationClient([])
    agent._mcp_init_specs = {
        "example": {"cfg": {}, "source": "init.json:mcp", "client": recorded}
    }

    with pytest.raises(RuntimeError, match="not exact"):
        agent._activate_mcp_candidate(
            candidate,
            allow_sealed=True,
            init_spec_name="example",
            predecessor=other,
        )

    assert not candidate.started
    assert candidate.closed
    assert agent._mcp_init_specs["example"]["client"] is recorded


def test_mcp_predecessor_close_failure_never_starts_replacement(tmp_path):
    agent, _ = _mk_agent(tmp_path)

    class _Unclosable(_ActivationClient):
        def close(self):
            raise RuntimeError("close failed")

    predecessor = _Unclosable([])
    predecessor.started = False
    agent._mcp_clients.append(predecessor)
    agent._mcp_init_specs = {
        "example": {"cfg": {}, "source": "init.json:mcp", "client": predecessor}
    }
    candidate = _ActivationClient([_tool("new")])

    with pytest.raises(RuntimeError, match="close failed"):
        agent._activate_mcp_candidate(
            candidate,
            allow_sealed=True,
            init_spec_name="example",
            predecessor=predecessor,
        )

    assert not candidate.started
    assert candidate.closed
    assert agent._mcp_init_specs["example"]["client"] is predecessor
    assert predecessor not in agent._mcp_clients
    assert agent._mcp_pending_retirements["example"] is predecessor


def test_mcp_predecessor_retires_before_replacement_and_reconciles_names(
    tmp_path,
):
    agent, _ = _mk_agent(tmp_path)
    predecessor = _ActivationClient([])
    predecessor.started = True
    predecessor.closed = False
    for name in ("shared", "old_only"):
        agent._tool_handlers[name] = agent._make_mcp_handler(predecessor, name)
        agent._tool_schemas.append(
            FunctionSchema(
                name=name,
                description=name,
                parameters={"type": "object", "properties": {}},
            )
        )
        agent._mcp_clients_by_tool[name] = predecessor
    agent._mcp_clients.append(predecessor)
    agent._mcp_tool_names.update({"shared", "old_only"})
    agent._mcp_init_specs = {
        "example": {"cfg": {}, "source": "init.json:mcp", "client": predecessor}
    }
    # Model a transport whose loop died before retirement.
    predecessor.started = False

    candidate = _ActivationClient(
        [_tool("shared"), _tool("new_only")],
        on_start=lambda: (
            None
            if predecessor.closed
            else (_ for _ in ()).throw(
                AssertionError("replacement overlapped predecessor")
            )
        ),
    )

    outcome = agent._activate_mcp_candidate(
        candidate,
        allow_sealed=True,
        init_spec_name="example",
        predecessor=predecessor,
    )

    assert outcome.client is candidate
    assert outcome.tool_names == ("shared", "new_only")
    assert predecessor.closed
    assert agent._mcp_init_specs["example"]["client"] is candidate
    assert set(agent._mcp_clients_by_tool) >= {"shared", "new_only"}
    assert "old_only" not in agent._tool_handlers
    assert "old_only" not in agent._mcp_tool_names


def test_post_retirement_publication_fault_never_restores_closed_predecessor(
    tmp_path, monkeypatch
):
    agent, _ = _mk_agent(tmp_path)
    predecessor = _ActivationClient([])
    _install_dead_projection(agent, "example", predecessor)
    candidate = _ActivationClient([_tool("new")])
    monkeypatch.setattr(
        agent,
        "_maybe_setup_task_card_controller",
        lambda: (_ for _ in ()).throw(RuntimeError("post-retirement fault")),
    )

    with pytest.raises(RuntimeError, match="post-retirement fault"):
        agent._activate_mcp_candidate(
            candidate,
            allow_sealed=True,
            init_spec_name="example",
            predecessor=predecessor,
        )

    assert predecessor.closed
    assert candidate.closed
    assert agent._mcp_init_specs["example"]["client"] is None
    assert predecessor not in agent._mcp_clients
    assert "old" not in agent._tool_handlers
    assert "new" not in agent._tool_handlers
    assert agent._mcp_pending_retirements == {}


def test_mcp_teardown_attempts_every_client_and_later_converges(tmp_path):
    agent, _ = _mk_agent(tmp_path)

    class _FlakyClose(_ActivationClient):
        def __init__(self, name, failures):
            super().__init__([_tool(name)])
            self.name = name
            self.failures = failures
            self.started = True
            self.close_calls = 0

        def close(self):
            self.close_calls += 1
            if self.close_calls <= self.failures:
                raise RuntimeError(f"{self.name} close failed")
            self.closed = True

    flaky = _FlakyClose("flaky", failures=1)
    healthy = _FlakyClose("healthy", failures=0)
    for client in (flaky, healthy):
        name = client.name
        agent._mcp_clients.append(client)
        agent._mcp_clients_by_tool[name] = client
        agent._mcp_tool_names.add(name)
        agent._tool_handlers[name] = agent._make_mcp_handler(client, name)
        agent._tool_schemas.append(
            FunctionSchema(
                name=name,
                description=name,
                parameters={"type": "object", "properties": {}},
            )
        )
    agent._mcp_init_specs = {
        client.name: {"cfg": {}, "source": "init.json:mcp", "client": client}
        for client in (flaky, healthy)
    }

    first = agent._retire_all_mcp_clients(context="test")

    assert first["attempted"] == 2
    assert len(first["unresolved"]) == 1
    assert healthy.closed
    assert healthy.close_calls == 1
    assert flaky.close_calls == 1
    assert agent._mcp_clients == []
    assert agent._mcp_pending_retirements == {"flaky": flaky}
    assert agent._mcp_clients_by_tool == {}
    assert "flaky" not in agent._tool_handlers
    assert "healthy" not in agent._tool_handlers
    assert agent._mcp_init_specs["flaky"]["client"] is flaky
    assert agent._mcp_init_specs["healthy"]["client"] is None

    second = agent._retire_all_mcp_clients(context="test_retry")

    assert second["unresolved"] == []
    assert flaky.closed
    assert flaky.close_calls == 2
    assert agent._mcp_clients == []
    assert agent._mcp_pending_retirements == {}
    assert agent._mcp_init_specs["flaky"]["client"] is None


def test_mcp_real_stdio_listing_failure_retires_child_and_thread(
    tmp_path, monkeypatch
):
    from lingtai.services import mcp as mcp_module

    observer = StdioProcessObserver()
    observer.install(monkeypatch)
    instances = []
    real_client = mcp_module.MCPClient

    def capture_client(**kwargs):
        client = real_client(**kwargs)
        instances.append(client)
        return client

    monkeypatch.setattr(mcp_module, "MCPClient", capture_client)
    agent, _ = _mk_agent(tmp_path)
    try:
        with pytest.raises(Exception):
            agent.connect_mcp(
                sys.executable,
                ["-m", "tests._mcp_activation_stdio_server"],
            )
        child = observer.wait_for_records(1)[0]
        assert instances
        assert wait_for_thread_exit(instances[0]._thread)
        assert wait_for_process_exit(child)
        assert agent._mcp_clients == []
        assert agent._mcp_clients_by_tool == {}
    finally:
        for child in observer.records():
            if not wait_for_process_exit(child, timeout=0):
                stop_process(child)


@pytest.mark.parametrize(
    "failing_module",
    [
        "tests._mcp_startup_timeout_stdio_server",
        "tests._mcp_activation_stdio_server",
    ],
)
def test_agent_retry_real_stdio_failure_then_single_live_replacement(
    tmp_path, monkeypatch, failing_module
):
    from lingtai.services.mcp import MCPClient

    observer = StdioProcessObserver()
    observer.install(monkeypatch)
    agent, _ = _mk_agent(tmp_path)
    instances = []

    def factory(**kwargs):
        startup_timeout = (
            2.0
            if "tests._mcp_structured_stdio_server" in kwargs.get("args", [])
            else 0.15
        )
        client = MCPClient(
            **kwargs,
            startup_timeout=startup_timeout,
            close_timeout=4.0,
        )
        instances.append(client)
        return client

    agent._mcp_stdio_client_factory = factory
    agent._mcp_init_specs = {
        "example": {
            "cfg": {
                "type": "stdio",
                "command": sys.executable,
                "args": ["-m", failing_module],
            },
            "source": "init.json:mcp",
            "client": None,
        }
    }
    try:
        first = agent._retry_failed_mcps()
        first_child = observer.wait_for_records(1)[0]
        assert first["still_failed"] == ["example"]
        assert wait_for_thread_exit(instances[0]._thread, timeout=2.0)
        assert wait_for_process_exit(first_child, timeout=2.0)

        agent._mcp_init_specs["example"]["cfg"]["args"] = [
            "-m",
            "tests._mcp_structured_stdio_server",
        ]
        second = agent._retry_failed_mcps()
        children = observer.wait_for_records(2)

        assert second["recovered"] == ["example"]
        replacement = agent._mcp_init_specs["example"]["client"]
        assert replacement is instances[1]
        assert replacement.is_connected()
        assert agent._mcp_clients == [replacement]
        assert len([client for client in instances if client.is_connected()]) == 1
        assert wait_for_process_exit(children[0], timeout=0)
    finally:
        agent._retire_all_mcp_clients(context="test_cleanup")
        for child in observer.records():
            if not wait_for_process_exit(child, timeout=0):
                stop_process(child)


def _free_local_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _wait_for_local_port(port, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                return
        except OSError:
            time.sleep(0.02)
    raise AssertionError(f"local HTTP MCP did not listen on {port}")


def test_agent_retry_local_http_failure_then_single_live_replacement(tmp_path):
    from lingtai.services.mcp import HTTPMCPClient

    agent, _ = _mk_agent(tmp_path)
    instances = []

    def factory(**kwargs):
        client = HTTPMCPClient(
            **kwargs,
            startup_timeout=0.5,
            close_timeout=1.0,
        )
        instances.append(client)
        return client

    agent._mcp_http_client_factory = factory
    unavailable_port = _free_local_port()
    agent._mcp_init_specs = {
        "example": {
            "cfg": {
                "type": "http",
                "url": f"http://127.0.0.1:{unavailable_port}/mcp",
            },
            "source": "init.json:mcp",
            "client": None,
        }
    }
    process = None
    try:
        first = agent._retry_failed_mcps()
        assert first["still_failed"] == ["example"]
        assert wait_for_thread_exit(instances[0]._thread, timeout=2.0)

        live_port = _free_local_port()
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "tests._mcp_http_server",
                "--port",
                str(live_port),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        _wait_for_local_port(live_port)
        agent._mcp_init_specs["example"]["cfg"]["url"] = (
            f"http://127.0.0.1:{live_port}/mcp"
        )
        second = agent._retry_failed_mcps()

        assert second["recovered"] == ["example"]
        replacement = agent._mcp_init_specs["example"]["client"]
        assert replacement is instances[1]
        assert replacement.is_connected()
        assert agent._mcp_clients == [replacement]
        assert len([client for client in instances if client.is_connected()]) == 1
    finally:
        agent._retire_all_mcp_clients(context="test_cleanup")
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait(timeout=3.0)


def test_agent_retry_http_established_list_failure_retires_before_replacement(
    tmp_path
):
    from lingtai.services.mcp import HTTPMCPClient

    agent, _ = _mk_agent(tmp_path)
    port = _free_local_port()
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "tests._mcp_http_server",
            "--port",
            str(port),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    instances = []

    class _EstablishedListFailure(HTTPMCPClient):
        def list_tools(self, timeout=10):
            tools = super().list_tools(timeout=timeout)
            assert tools and self.is_connected()
            raise RuntimeError("intentional established HTTP tools/list failure")

    def factory(**kwargs):
        if instances:
            assert not instances[0]._thread.is_alive()
            client = HTTPMCPClient(
                **kwargs, startup_timeout=2.0, close_timeout=1.0
            )
        else:
            client = _EstablishedListFailure(
                **kwargs, startup_timeout=2.0, close_timeout=1.0
            )
        instances.append(client)
        return client

    agent._mcp_http_client_factory = factory
    agent._mcp_init_specs = {
        "example": {
            "cfg": {
                "type": "http",
                "url": f"http://127.0.0.1:{port}/mcp",
            },
            "source": "init.json:mcp",
            "client": None,
        }
    }
    try:
        _wait_for_local_port(port)
        first = agent._retry_failed_mcps()
        assert first["still_failed"] == ["example"]
        assert not instances[0]._thread.is_alive()

        second = agent._retry_failed_mcps()
        assert second["recovered"] == ["example"]
        replacement = agent._mcp_init_specs["example"]["client"]
        assert replacement is instances[1]
        assert replacement.is_connected()
        assert agent._mcp_clients == [replacement]
        assert len([client for client in instances if client.is_connected()]) == 1
    finally:
        agent._retire_all_mcp_clients(context="test_cleanup")
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=3.0)


def _install_dead_projection(agent, spec_name, client, names=("old",)):
    client.started = False
    client.closed = False
    agent._mcp_clients.append(client)
    for name in names:
        agent._mcp_clients_by_tool[name] = client
        agent._mcp_tool_names.add(name)
        agent._tool_handlers[name] = agent._make_mcp_handler(client, name)
        agent._tool_schemas.append(
            FunctionSchema(
                name=name,
                description=name,
                parameters={"type": "object", "properties": {}},
            )
        )
    agent._mcp_init_specs = {
        spec_name: {
            "cfg": {"type": "stdio", "command": "/bin/false"},
            "source": "init.json:mcp",
            "client": client,
        }
    }


def test_retry_failed_mcps_transactional_replacement_identity_production_signature(
    tmp_path, monkeypatch
):
    agent, _ = _mk_agent(tmp_path)
    predecessor = _ActivationClient([])
    _install_dead_projection(agent, "example", predecessor)
    replacement = _ActivationClient([_tool("new")])
    monkeypatch.setattr(
        "lingtai.services.mcp.MCPClient", lambda **kwargs: replacement
    )

    report = agent._retry_failed_mcps()

    assert report == {
        "retried": ["example"],
        "recovered": ["example"],
        "still_failed": [],
        "healthy": [],
    }
    assert agent._mcp_init_specs["example"]["client"] is replacement
    assert agent._mcp_clients == [replacement]
    assert agent._mcp_clients_by_tool["new"] is replacement
    assert agent._tool_handlers["new"]._lingtai_mcp_client is replacement
    assert "new" in agent._mcp_tool_names
    assert sum(schema.name == "new" for schema in agent._tool_schemas) == 1


def test_retry_failed_mcps_retirement_close_failure_converges_second_attempt(
    tmp_path, monkeypatch
):
    agent, _ = _mk_agent(tmp_path)

    class _FlakyPredecessor(_ActivationClient):
        def __init__(self):
            super().__init__([])
            self.close_calls = 0

        def close(self):
            self.close_calls += 1
            if self.close_calls == 1:
                raise RuntimeError("predecessor close blocked")
            self.closed = True

    predecessor = _FlakyPredecessor()
    _install_dead_projection(agent, "example", predecessor)
    unused = _ActivationClient([_tool("unused")])
    replacement = _ActivationClient([_tool("new")])
    replacements = [unused, replacement]
    monkeypatch.setattr(
        "lingtai.services.mcp.MCPClient", lambda **kwargs: replacements.pop(0)
    )

    first = agent._retry_failed_mcps()
    assert first["still_failed"] == ["example"]
    assert agent._mcp_init_specs["example"]["client"] is predecessor
    assert agent._mcp_pending_retirements["example"] is predecessor

    second = agent._retry_failed_mcps()
    assert second["recovered"] == ["example"]
    assert second["still_failed"] == []
    assert predecessor.close_calls == 2
    assert agent._mcp_pending_retirements == {}
    assert agent._mcp_init_specs["example"]["client"] is replacement


def test_retry_failed_mcps_candidate_cleanup_failure_is_keyed_and_drained(
    tmp_path, monkeypatch
):
    agent, _ = _mk_agent(tmp_path)
    agent._mcp_init_specs = {
        "example": {
            "cfg": {"type": "stdio", "command": "/bin/false"},
            "source": "init.json:mcp",
            "client": None,
        }
    }

    class _CleanupFlaky(_ActivationClient):
        def __init__(self):
            super().__init__([], on_start=lambda: (_ for _ in ()).throw(
                RuntimeError("start failed")
            ))
            self.close_calls = 0

        def close(self):
            self.close_calls += 1
            if self.close_calls == 1:
                raise RuntimeError("candidate close blocked")
            self.closed = True

    failed = _CleanupFlaky()
    recovered = _ActivationClient([_tool("new")])
    candidates = [failed, recovered]
    monkeypatch.setattr(
        "lingtai.services.mcp.MCPClient", lambda **kwargs: candidates.pop(0)
    )

    first = agent._retry_failed_mcps()
    assert first["still_failed"] == ["example"]
    assert agent._mcp_pending_retirements["example"] is failed
    assert agent._mcp_init_specs["example"]["client"] is failed

    second = agent._retry_failed_mcps()
    assert second["recovered"] == ["example"]
    assert failed.close_calls == 2
    assert agent._mcp_pending_retirements == {}
    assert agent._mcp_init_specs["example"]["client"] is recovered


@pytest.mark.parametrize(
    "damage",
    [
        "handler",
        "handler_tool_name",
        "owner",
        "schema",
        "name_set",
        "client_list",
        "init_spec",
    ],
)
def test_predecessor_projection_matrix_rejects_object_identically(tmp_path, damage):
    agent, _ = _mk_agent(tmp_path)
    predecessor = _ActivationClient([])
    _install_dead_projection(agent, "example", predecessor)
    handler = agent._tool_handlers["old"]
    schema = next(schema for schema in agent._tool_schemas if schema.name == "old")
    clients = agent._mcp_clients
    owners = agent._mcp_clients_by_tool
    names = agent._mcp_tool_names
    spec = agent._mcp_init_specs["example"]
    handlers_container = agent._tool_handlers
    schemas_container = agent._tool_schemas

    if damage == "handler":
        agent._tool_handlers["old"] = lambda args: args
    elif damage == "handler_tool_name":
        agent._tool_handlers["old"]._lingtai_mcp_tool_name = "wrong"
    elif damage == "owner":
        agent._mcp_clients_by_tool.pop("old")
    elif damage == "schema":
        agent._tool_schemas.append(copy.deepcopy(schema))
    elif damage == "name_set":
        agent._mcp_tool_names.remove("old")
    elif damage == "client_list":
        agent._mcp_clients.append(predecessor)
    else:
        spec["client"] = _ActivationClient([])

    candidate = _ActivationClient([_tool("new")])
    before_handlers = dict(agent._tool_handlers)
    before_schemas = list(agent._tool_schemas)
    before_owners = dict(agent._mcp_clients_by_tool)
    before_clients = list(agent._mcp_clients)
    before_names = set(agent._mcp_tool_names)
    before_spec_client = spec["client"]

    with pytest.raises(RuntimeError):
        agent._activate_mcp_candidate(
            candidate,
            allow_sealed=True,
            init_spec_name="example",
            predecessor=predecessor,
        )

    assert agent._tool_handlers == before_handlers
    assert agent._tool_schemas == before_schemas
    assert agent._mcp_clients_by_tool == before_owners
    assert agent._mcp_clients == before_clients
    assert agent._mcp_tool_names == before_names
    assert spec["client"] is before_spec_client
    if damage not in {"handler", "owner", "init_spec"}:
        assert agent._tool_handlers["old"] is handler
    assert candidate.closed
    assert not predecessor.closed
    assert handlers_container is agent._tool_handlers
    assert schemas_container is agent._tool_schemas
    assert clients is agent._mcp_clients
    assert owners is agent._mcp_clients_by_tool
    assert names is agent._mcp_tool_names


@pytest.mark.parametrize("reserved", ["telegram", "task_card"])
def test_generic_foreign_candidate_cannot_claim_reserved_names(tmp_path, reserved):
    agent, _ = _mk_agent(tmp_path)
    candidate = _ActivationClient([_tool(reserved)])

    with pytest.raises(RuntimeError, match="built-in/reserved"):
        agent._activate_mcp_candidate(candidate)

    assert candidate.closed
    assert reserved not in agent._tool_handlers
    assert reserved not in agent._mcp_clients_by_tool

    forged = _ActivationClient([_tool(reserved)])
    with pytest.raises(RuntimeError, match="built-in/reserved"):
        agent._activate_mcp_candidate(
            forged,
            reserved_activation_token=object(),
        )
    assert forged.closed


def _write_telegram_registration(workdir, *, source="lingtai-curated"):
    cfg = {
        "type": "stdio",
        "command": sys.executable,
        "args": ["-m", "lingtai.mcp_servers.telegram"],
    }
    (workdir / "init.json").write_text(json.dumps({"mcp": {"telegram": cfg}}))
    (workdir / "mcp_registry.jsonl").write_text(
        json.dumps(
            {
                "name": "telegram",
                "summary": "telegram",
                "transport": "stdio",
                "command": sys.executable,
                "args": ["-m", "lingtai.mcp_servers.telegram"],
                "source": source,
            }
        )
        + "\n"
    )


def _mutate_telegram_registration(workdir, damage):
    init_path = workdir / "init.json"
    registry_path = workdir / "mcp_registry.jsonl"
    init = json.loads(init_path.read_text())
    record = json.loads(registry_path.read_text())
    cfg = init["mcp"]["telegram"]
    if damage == "transport":
        cfg.clear()
        cfg.update({"type": "http", "url": "http://127.0.0.1/forged"})
        record["transport"] = "http"
        record.pop("command", None)
        record.pop("args", None)
        record["url"] = cfg["url"]
    elif damage == "command":
        cfg["command"] = "/forged/python"
        record["command"] = "/forged/python"
    elif damage == "args":
        cfg["args"] = ["-m", "forged.telegram"]
        record["args"] = ["-m", "forged.telegram"]
    else:
        cfg["env"] = {damage: "forged"}
    init_path.write_text(json.dumps(init))
    registry_path.write_text(json.dumps(record) + "\n")


@pytest.mark.parametrize(
    "damage",
    [
        "transport",
        "command",
        "args",
        "LINGTAI_AGENT_DIR",
        "LINGTAI_MCP_NAME",
        "PYTHONPATH",
        "PYTHONHOME",
        "LD_PRELOAD",
        "DYLD_INSERT_LIBRARIES",
    ],
)
def test_reserved_provenance_initial_negative_matrix_never_starts(
    tmp_path, monkeypatch, damage
):
    workdir = tmp_path / "agent"
    workdir.mkdir()
    _write_telegram_registration(workdir)
    _mutate_telegram_registration(workdir, damage)
    starts = []
    monkeypatch.setattr(
        "lingtai.services.mcp.MCPClient",
        lambda **kwargs: starts.append(kwargs) or _ActivationClient([]),
    )

    agent = Agent(
        service=make_mock_service(),
        agent_name="test",
        working_dir=workdir,
        capabilities={"mcp": {}},
    )

    assert starts == []
    spec = agent._mcp_init_specs["telegram"]
    assert spec["reserved_provenance"] is None
    assert spec["client"] is None


def test_reserved_provenance_safe_env_initial_and_retry_injection(
    tmp_path, monkeypatch
):
    workdir = tmp_path / "agent"
    workdir.mkdir()
    _write_telegram_registration(workdir)
    init = json.loads((workdir / "init.json").read_text())
    init["mcp"]["telegram"]["env"] = {
        "LINGTAI_TELEGRAM_CONFIG": "telegram/config.json"
    }
    (workdir / "init.json").write_text(json.dumps(init))
    clients = [
        _ActivationClient([_tool("telegram")]),
        _ActivationClient([_tool("telegram")]),
    ]
    launches = []

    def factory(**kwargs):
        launches.append(kwargs)
        return clients.pop(0)

    monkeypatch.setattr("lingtai.services.mcp.MCPClient", factory)
    agent = Agent(
        service=make_mock_service(),
        agent_name="test",
        working_dir=workdir,
        capabilities={"mcp": {}},
    )
    predecessor = agent._mcp_init_specs["telegram"]["client"]
    predecessor.started = False
    report = agent._retry_failed_mcps()

    assert report["recovered"] == ["telegram"]
    assert len(launches) == 2
    for launch in launches:
        assert launch["env"]["LINGTAI_TELEGRAM_CONFIG"] == "telegram/config.json"
        assert launch["env"]["LINGTAI_AGENT_DIR"] == str(workdir)
        assert launch["env"]["LINGTAI_MCP_NAME"] == "telegram"


def test_mcp_runtime_owned_env_cannot_be_overridden(tmp_path, monkeypatch):
    workdir = tmp_path / "agent"
    workdir.mkdir()
    cfg = {
        "command": "/bin/true",
        "env": {
            "LINGTAI_AGENT_DIR": "/forged",
            "LINGTAI_MCP_NAME": "forged",
            "lingtai_agent_dir": "/case-forged",
            "lingtai_mcp_name": "case-forged",
        },
    }
    (workdir / "init.json").write_text(
        json.dumps({"mcp": {"example": cfg}})
    )
    (workdir / "mcp_registry.jsonl").write_text(
        json.dumps({
            "name": "example",
            "summary": "example",
            "transport": "stdio",
            "command": "/bin/true",
            "args": [],
            "source": "user",
        }) + "\n"
    )
    launches = []
    monkeypatch.setattr(
        "lingtai.services.mcp.MCPClient",
        lambda **kwargs: launches.append(kwargs) or _ActivationClient([]),
    )

    Agent(
        service=make_mock_service(),
        agent_name="test",
        working_dir=workdir,
        capabilities={"mcp": {}},
    )

    assert launches[0]["env"]["LINGTAI_AGENT_DIR"] == str(workdir)
    assert launches[0]["env"]["LINGTAI_MCP_NAME"] == "example"
    assert "lingtai_agent_dir" not in launches[0]["env"]
    assert "lingtai_mcp_name" not in launches[0]["env"]


@pytest.mark.parametrize(
    "damage",
    [
        "transport",
        "command",
        "args",
        "source",
        "LINGTAI_AGENT_DIR",
        "PYTHONPATH",
        "DYLD_LIBRARY_PATH",
    ],
)
def test_reserved_provenance_retry_drift_never_starts_replacement(
    tmp_path, monkeypatch, damage
):
    workdir = tmp_path / "agent"
    workdir.mkdir()
    _write_telegram_registration(workdir)
    launches = []
    predecessor = _ActivationClient([_tool("telegram")])

    def factory(**kwargs):
        launches.append(kwargs)
        return predecessor

    monkeypatch.setattr("lingtai.services.mcp.MCPClient", factory)
    agent = Agent(
        service=make_mock_service(),
        agent_name="test",
        working_dir=workdir,
        capabilities={"mcp": {}},
    )
    predecessor.started = False
    cfg = agent._mcp_init_specs["telegram"]["cfg"]
    if damage == "transport":
        cfg.clear()
        cfg.update({"type": "http", "url": "http://127.0.0.1/forged"})
    elif damage == "command":
        cfg["command"] = "/forged/python"
    elif damage == "args":
        cfg["args"] = ["-m", "forged.telegram"]
    elif damage == "source":
        agent._mcp_init_specs["telegram"]["source"] = "user"
    else:
        cfg["env"] = {damage: "forged"}

    report = agent._retry_failed_mcps()

    assert report["still_failed"] == ["telegram"]
    assert len(launches) == 1
    assert agent._mcp_init_specs["telegram"]["client"] is predecessor
    assert not predecessor.closed


def test_curated_telegram_provenance_survives_retry(tmp_path, monkeypatch):
    workdir = tmp_path / "agent"
    workdir.mkdir()
    _write_telegram_registration(workdir)
    clients = [
        _ActivationClient([_tool("telegram")]),
        _ActivationClient([_tool("telegram")]),
    ]
    monkeypatch.setattr(
        "lingtai.services.mcp.MCPClient", lambda **kwargs: clients.pop(0)
    )
    agent = Agent(
        service=make_mock_service(),
        agent_name="test",
        working_dir=workdir,
        capabilities={"mcp": {}},
    )
    predecessor = agent._mcp_init_specs["telegram"]["client"]
    predecessor.started = False

    report = agent._retry_failed_mcps()

    assert report["recovered"] == ["telegram"]
    replacement = agent._mcp_init_specs["telegram"]["client"]
    assert replacement is agent._mcp_clients_by_tool["telegram"]
    assert predecessor.closed


@pytest.mark.parametrize("registration", ["user", "legacy"])
def test_unverified_telegram_registration_has_no_reserved_authority(
    tmp_path, monkeypatch, registration
):
    workdir = tmp_path / "agent"
    workdir.mkdir()
    if registration == "user":
        _write_telegram_registration(workdir, source="user")
    else:
        (workdir / "init.json").write_text("{}")
        mcp_dir = workdir / "mcp"
        mcp_dir.mkdir()
        (mcp_dir / "servers.json").write_text(
            json.dumps(
                {
                    "telegram": {
                        "command": sys.executable,
                        "args": ["-m", "lingtai.mcp_servers.telegram"],
                    }
                }
            )
        )
    candidate = _ActivationClient([_tool("telegram")])
    monkeypatch.setattr(
        "lingtai.services.mcp.MCPClient", lambda **kwargs: candidate
    )

    agent = Agent(
        service=make_mock_service(),
        agent_name="test",
        working_dir=workdir,
        capabilities={"mcp": {}},
    )

    assert candidate.closed
    assert "telegram" not in agent._tool_handlers
    if registration == "user":
        assert agent._mcp_init_specs["telegram"]["reserved_provenance"] is None
        assert agent._mcp_init_specs["telegram"]["client"] is None


def test_legacy_connector_override_cannot_bypass_activation_outcome(
    tmp_path, monkeypatch
):
    workdir = tmp_path / "agent"
    workdir.mkdir()
    (workdir / "init.json").write_text(
        json.dumps({"mcp": {"example": {"command": "/bin/true"}}})
    )
    (workdir / "mcp_registry.jsonl").write_text(
        json.dumps(
            {
                "name": "example",
                "summary": "example",
                "transport": "stdio",
                "command": "/bin/true",
                "args": [],
                "source": "user",
            }
        )
        + "\n"
    )
    called = []

    def legacy_override(self, command, args=None, env=None):
        called.append(command)
        return ["forged"]

    monkeypatch.setattr(Agent, "connect_mcp", legacy_override)
    agent = Agent(
        service=make_mock_service(),
        agent_name="test",
        working_dir=workdir,
        capabilities={"mcp": {}},
    )

    assert called == []
    assert agent._mcp_init_specs["example"]["client"] is None
    assert "forged" not in agent._tool_handlers


def test_stop_during_candidate_start_retires_complete_publication(tmp_path):
    agent, _ = _mk_agent(tmp_path)
    started = threading.Event()
    release = threading.Event()
    candidate = _ActivationClient(
        [_tool("late")],
        on_start=lambda: (started.set(), release.wait(timeout=2.0)),
    )
    activation_errors = []
    stop_errors = []

    def activate():
        try:
            agent._activate_mcp_candidate(candidate)
        except Exception as exc:
            activation_errors.append(exc)

    activation = threading.Thread(target=activate)
    activation.start()
    assert started.wait(timeout=1.0)

    def stop():
        try:
            agent.stop()
        except Exception as exc:
            stop_errors.append(exc)

    stopping = threading.Thread(target=stop)
    stopping.start()
    release.set()
    activation.join(timeout=2.0)
    stopping.join(timeout=2.0)

    assert not activation.is_alive()
    assert not stopping.is_alive()
    assert activation_errors == []
    assert stop_errors == []
    assert "late" not in agent._tool_handlers
    assert candidate.closed


def test_waiting_activation_is_invalidated_by_public_deep_refresh(
    tmp_path, monkeypatch
):
    agent, _ = _mk_agent(tmp_path)
    candidate = _ActivationClient([_tool("late")])
    errors = []
    waiting = threading.Event()
    release = threading.Event()
    agent._mcp_activation_wait_hook = lambda: (
        waiting.set(),
        release.wait(timeout=2.0),
    )

    def activate():
        try:
            agent._activate_mcp_candidate(candidate)
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=activate)
    thread.start()
    assert waiting.wait(timeout=1.0)
    monkeypatch.setattr(agent, "_setup_from_init_locked", lambda: None)
    agent._setup_from_init()
    release.set()
    thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert errors
    assert "invalidated" in str(errors[0])
    assert candidate.closed
    assert "late" not in agent._tool_handlers


def test_public_stop_partial_retirement_then_repeat_converges(tmp_path):
    agent, _ = _mk_agent(tmp_path)

    class _Flaky(_ActivationClient):
        def __init__(self):
            super().__init__([_tool("flaky")])
            self.started = True
            self.close_calls = 0

        def close(self):
            self.close_calls += 1
            if self.close_calls == 1:
                raise RuntimeError("close failed")
            self.closed = True

    client = _Flaky()
    _install_dead_projection(agent, "example", client, names=("flaky",))

    agent.stop()
    assert agent._mcp_pending_retirements["example"] is client
    assert "flaky" not in agent._tool_handlers
    agent.stop()
    assert agent._mcp_pending_retirements == {}
    assert client.close_calls == 2


def test_curated_catalog_includes_whatsapp(tmp_path: Path):
    rep = decompress_addons(tmp_path, ["whatsapp"])
    assert rep["appended"] == ["whatsapp"]
    records, problems = read_registry(tmp_path)
    assert problems == []
    assert records[0]["name"] == "whatsapp"
    assert records[0]["args"] == ["-m", "lingtai.mcp_servers.whatsapp"]
    assert records[0]["homepage"] == "https://github.com/Lingtai-AI/lingtai-whatsapp"


def test_curated_mcp_modules_ship_inside_lingtai_distribution():
    """Curated MCPs ship from the canonical kernel distribution package."""
    import importlib
    from importlib import resources

    modules = {
        "imap": "lingtai.mcp_servers.imap",
        "telegram": "lingtai.mcp_servers.telegram",
        "feishu": "lingtai.mcp_servers.feishu",
        "wechat": "lingtai.mcp_servers.wechat",
        "whatsapp": "lingtai.mcp_servers.whatsapp",
        "cloud_mail": "lingtai.mcp_servers.cloud_mail",
    }
    for module in modules.values():
        imported = importlib.import_module(module)
        assert imported is not None

    for module in (
        "lingtai.mcp_servers.telegram",
        "lingtai.mcp_servers.feishu",
        "lingtai.mcp_servers.wechat",
        "lingtai.mcp_servers.whatsapp",
    ):
        header = resources.files(module).joinpath("notification_header.md")
        assert header.is_file()
        assert header.read_text(encoding="utf-8").strip()


def test_curated_mcp_catalog_launches_embedded_modules(tmp_path: Path):
    modules = {
        "imap": "lingtai.mcp_servers.imap",
        "telegram": "lingtai.mcp_servers.telegram",
        "feishu": "lingtai.mcp_servers.feishu",
        "wechat": "lingtai.mcp_servers.wechat",
        "whatsapp": "lingtai.mcp_servers.whatsapp",
    }
    rep = decompress_addons(tmp_path, list(modules))
    assert rep["appended"] == list(modules)
    records, problems = read_registry(tmp_path)
    assert problems == []
    by_name = {r["name"]: r for r in records}
    for name, module in modules.items():
        assert by_name[name]["command"] == sys.executable
        assert by_name[name]["args"] == ["-m", module]
        assert by_name[name]["source"] == "lingtai-curated"
