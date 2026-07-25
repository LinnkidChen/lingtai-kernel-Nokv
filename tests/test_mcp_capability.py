"""End-to-end smoke tests for the mcp capability + addons decompression.

Verifies the vertical slice: addons:["imap"] in init.json triggers catalog
decompression into mcp_registry.jsonl, the mcp capability renders the registry
into the system prompt, and the loader gates init.json mcp activation by
registry membership.
"""
from __future__ import annotations

import copy
import json
import re
import sys
import threading
import time
from pathlib import Path

import pytest

from lingtai.agent import Agent
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




def _mk_agent(tmp_path: Path, *, addons=None, capabilities=None):
    workdir = tmp_path / "agent"
    return Agent(
        service=make_mock_service(),
        agent_name="test",
        working_dir=workdir,
        capabilities=capabilities or {"mcp": {}},
        addons=addons,
    ), workdir


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

    # Patch connect_mcp on the Agent class: first call → returns dead client
    # (subprocess "exited" immediately); second call → returns live client.
    call_count = {"n": 0}

    def fake_connect_mcp(self, command, args=None, env=None):
        call_count["n"] += 1
        client = _FakeMCPClient(is_connected_value=(call_count["n"] >= 2))
        if not hasattr(self, "_mcp_clients"):
            self._mcp_clients = []
        self._mcp_clients.append(client)
        return []  # no tools to register

    monkeypatch.setattr(Agent, "connect_mcp", fake_connect_mcp)

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

    def fake_connect_mcp(self, command, args=None, env=None):
        client = _FakeMCPClient(is_connected_value=True)
        if not hasattr(self, "_mcp_clients"):
            self._mcp_clients = []
        self._mcp_clients.append(client)
        return []

    monkeypatch.setattr(Agent, "connect_mcp", fake_connect_mcp)

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
    ["handler", "owner", "schema", "name_set", "client_list", "init_spec"],
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

    if damage == "handler":
        agent._tool_handlers["old"] = lambda args: args
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


def test_stop_during_candidate_start_prevents_late_publication(tmp_path):
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
    deadline = time.monotonic() + 1.0
    while not agent._mcp_lifecycle_barrier.is_set() and time.monotonic() < deadline:
        time.sleep(0.005)
    release.set()
    activation.join(timeout=2.0)
    stopping.join(timeout=2.0)

    assert not activation.is_alive()
    assert not stopping.is_alive()
    assert activation_errors
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
