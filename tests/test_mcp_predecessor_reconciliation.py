"""Dead-predecessor reconciliation tests for MCP retry activation.

These tests intentionally exercise the in-memory projections that already form
the MCP runtime contract.  A predecessor is replaceable only when the exact
unhealthy client recorded by one init spec appears exactly once in the active
client list and its handler, schema, owner, and MCP-name projections agree.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from lingtai.agent import Agent
from lingtai.kernel.llm.base import FunctionSchema
from lingtai.services import mcp as mcp_module
from tests._service_helpers import make_gemini_mock_service


SPEC_NAME = "predecessor-reconciliation"
OLD_ONLY = "predecessor_old_only"
SHARED = "predecessor_shared"
NEW_ONLY = "predecessor_new_only"
PREDECESSOR_NAMES = {OLD_ONLY, SHARED}
CANDIDATE_NAMES = {SHARED, NEW_ONLY}


def _agent(tmp_path: Path) -> Agent:
    return Agent(
        service=make_gemini_mock_service(),
        agent_name="mcp-predecessor-test",
        working_dir=tmp_path / "agent",
        capabilities={"mcp": {}},
    )


def _cfg(transport: str) -> dict[str, Any]:
    if transport == "stdio":
        return {"type": "stdio", "command": "fake", "args": []}
    return {"type": "http", "url": "http://fake"}


def _catalog(*names: str, generation: str = "candidate") -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "description": f"{generation} schema for {name}",
            "schema": {
                "type": "object",
                "properties": {
                    "generation": {"type": "string", "const": generation},
                },
            },
        }
        for name in names
    ]


class _Predecessor:
    def __init__(self, events: list[str], *, healthy: bool = False):
        self.events = events
        self.healthy = healthy
        self.closed = False
        # The control/session health probe can be false while a process or
        # transport thread still exists.  close() is the event that retires it.
        self.runtime_open = True
        self.close_calls = 0

    def is_connected(self) -> bool:
        return self.healthy and not self.closed

    def close(self) -> None:
        self.events.append("predecessor.close")
        self.close_calls += 1
        self.closed = True
        self.runtime_open = False

    def call_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if self.closed:
            raise RuntimeError("closed predecessor handler was resurrected")
        return {"client": "predecessor", "name": name, "args": args}


class _ForeignClient:
    def __init__(self):
        self.closed = False

    def is_connected(self) -> bool:
        return not self.closed

    def close(self) -> None:
        self.closed = True

    def call_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        return {"client": "foreign", "name": name, "args": args}


def _install_predecessor(
    agent: Agent,
    predecessor: _Predecessor,
    transport: str,
) -> dict[str, Any]:
    handlers = dict(agent._tool_handlers)
    owners = dict(agent._mcp_clients_by_tool)
    schemas = list(agent._tool_schemas)
    names = set(agent._mcp_tool_names)
    for name in (OLD_ONLY, SHARED):
        handlers[name] = agent._make_mcp_handler(predecessor, name)
        owners[name] = predecessor
        schemas.append(
            FunctionSchema(
                name=name,
                description=f"predecessor schema for {name}",
                parameters={
                    "type": "object",
                    "properties": {
                        "generation": {
                            "type": "string",
                            "const": "predecessor",
                        }
                    },
                },
            )
        )
        names.add(name)

    agent._tool_handlers = handlers
    agent._tool_schemas = schemas
    agent._mcp_clients_by_tool = owners
    agent._mcp_tool_names = names
    agent._mcp_clients = [*agent._mcp_clients, predecessor]
    spec = {
        "cfg": _cfg(transport),
        "source": "init.json:mcp",
        "client": predecessor,
    }
    agent._mcp_init_specs = {SPEC_NAME: spec}
    cfg = spec["cfg"]
    (agent._working_dir / "init.json").write_text(
        json.dumps({"mcp": {SPEC_NAME: cfg}}),
        encoding="utf-8",
    )
    record = {
        "name": SPEC_NAME,
        "summary": "predecessor reconciliation test MCP",
        "transport": transport,
        "source": "test",
    }
    if transport == "http":
        record["url"] = cfg["url"]
    else:
        record["command"] = cfg["command"]
        record["args"] = cfg["args"]
    (agent._working_dir / "mcp_registry.jsonl").write_text(
        json.dumps(record) + "\n",
        encoding="utf-8",
    )
    return spec


def _snapshot(agent: Agent, spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "handlers_ref": agent._tool_handlers,
        "handlers_items": list(agent._tool_handlers.items()),
        "schemas_ref": agent._tool_schemas,
        "schemas_items": list(agent._tool_schemas),
        "owners_ref": agent._mcp_clients_by_tool,
        "owners_items": list(agent._mcp_clients_by_tool.items()),
        "names_ref": agent._mcp_tool_names,
        "names_items": set(agent._mcp_tool_names),
        "clients_ref": agent._mcp_clients,
        "clients_items": list(agent._mcp_clients),
        "specs_ref": agent._mcp_init_specs,
        "specs_items": list(agent._mcp_init_specs.items()),
        "spec_clients": {
            name: item.get("client") for name, item in agent._mcp_init_specs.items()
        },
        "spec_ref": spec,
        "spec_client": spec["client"],
        "retiring_ref": agent._mcp_retiring_clients,
        "retiring_items": list(agent._mcp_retiring_clients),
        "token_dirty": agent._token_decomp_dirty,
    }


def _assert_snapshot_identity(agent: Agent, snapshot: dict[str, Any]) -> None:
    assert agent._tool_handlers is snapshot["handlers_ref"]
    assert list(agent._tool_handlers.items()) == snapshot["handlers_items"]
    assert agent._tool_schemas is snapshot["schemas_ref"]
    assert len(agent._tool_schemas) == len(snapshot["schemas_items"])
    assert all(
        actual is expected
        for actual, expected in zip(agent._tool_schemas, snapshot["schemas_items"])
    )
    assert agent._mcp_clients_by_tool is snapshot["owners_ref"]
    assert list(agent._mcp_clients_by_tool.items()) == snapshot["owners_items"]
    assert agent._mcp_tool_names is snapshot["names_ref"]
    assert agent._mcp_tool_names == snapshot["names_items"]
    assert agent._mcp_clients is snapshot["clients_ref"]
    assert len(agent._mcp_clients) == len(snapshot["clients_items"])
    assert all(
        actual is expected
        for actual, expected in zip(agent._mcp_clients, snapshot["clients_items"])
    )
    assert agent._mcp_init_specs is snapshot["specs_ref"]
    assert list(agent._mcp_init_specs.items()) == snapshot["specs_items"]
    for name, expected in snapshot["spec_clients"].items():
        assert agent._mcp_init_specs[name]["client"] is expected
    assert agent._mcp_init_specs[SPEC_NAME] is snapshot["spec_ref"]
    assert agent._mcp_init_specs[SPEC_NAME]["client"] is snapshot["spec_client"]
    assert agent._mcp_retiring_clients is snapshot["retiring_ref"]
    assert len(agent._mcp_retiring_clients) == len(snapshot["retiring_items"])
    assert all(
        actual is expected
        for actual, expected in zip(
            agent._mcp_retiring_clients, snapshot["retiring_items"]
        )
    )
    assert agent._token_decomp_dirty is snapshot["token_dirty"]


def _patch_candidate(
    monkeypatch,
    transport: str,
    predecessor: _Predecessor,
    events: list[str],
    *,
    failure_phase: str | None = None,
):
    class Candidate:
        instances: list["Candidate"] = []

        def __init__(self, *args, **kwargs):
            self.started = False
            self.closed = False
            self.close_calls = 0
            type(self).instances.append(self)

        def start(self) -> None:
            events.append("candidate.start")
            # This is the no-overlap assertion: even an unhealthy predecessor
            # may still own a process/session until verified close completes.
            assert predecessor.closed is True
            assert predecessor.runtime_open is False
            self.started = True
            if failure_phase == "startup":
                raise RuntimeError("candidate startup failed")

        def list_tools(self) -> list[dict[str, Any]]:
            events.append("candidate.list")
            if failure_phase == "list":
                raise RuntimeError("candidate tools/list failed")
            return _catalog(SHARED, NEW_ONLY)

        def is_connected(self) -> bool:
            return self.started and not self.closed

        def close(self) -> None:
            events.append("candidate.close")
            self.close_calls += 1
            self.closed = True

        def call_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
            if self.closed:
                raise RuntimeError("closed candidate called")
            return {"client": "candidate", "name": name, "args": args}

    class_name = "MCPClient" if transport == "stdio" else "HTTPMCPClient"
    monkeypatch.setattr(mcp_module, class_name, Candidate)
    return Candidate


@pytest.mark.parametrize("transport", ["stdio", "http"])
def test_healthy_exact_predecessor_is_skipped_without_state_change(
    tmp_path, monkeypatch, transport
):
    events: list[str] = []
    predecessor = _Predecessor(events, healthy=True)
    agent = _agent(tmp_path)
    spec = _install_predecessor(agent, predecessor, transport)
    Candidate = _patch_candidate(monkeypatch, transport, predecessor, events)
    snapshot = _snapshot(agent, spec)
    try:
        report = agent._retry_failed_mcps()

        assert report["healthy"] == [SPEC_NAME]
        assert report["retried"] == []
        assert Candidate.instances == []
        assert predecessor.close_calls == 0
        assert events == []
        _assert_snapshot_identity(agent, snapshot)
    finally:
        agent._workdir_lease.release()


@pytest.mark.parametrize("transport", ["stdio", "http"])
@pytest.mark.parametrize(
    "invalid_projection",
    [
        "foreign",
        "wrong_spec",
        "missing_schema",
        "missing_owner",
        "missing_mcp_name",
        "duplicate_active_membership",
    ],
)
def test_non_exact_predecessor_fails_before_retirement_and_exact_restores_state(
    tmp_path, monkeypatch, transport, invalid_projection
):
    events: list[str] = []
    predecessor = _Predecessor(events)
    agent = _agent(tmp_path)
    spec = _install_predecessor(agent, predecessor, transport)
    foreign = _ForeignClient()

    if invalid_projection == "foreign":
        # The owner projection claims predecessor while the callable handler
        # captures a foreign client.  That foreign handler must never be
        # treated as replaceable predecessor state.
        agent._tool_handlers[SHARED] = agent._make_mcp_handler(foreign, SHARED)
        agent._mcp_clients.append(foreign)
    elif invalid_projection == "wrong_spec":
        # One runtime client cannot be the predecessor of two init specs.
        agent._mcp_init_specs["other-spec"] = {
            "cfg": _cfg(transport),
            "source": "init.json:mcp",
            "client": predecessor,
        }
        init_data = json.loads(
            (agent._working_dir / "init.json").read_text(encoding="utf-8")
        )
        init_data["mcp"]["other-spec"] = _cfg(transport)
        (agent._working_dir / "init.json").write_text(
            json.dumps(init_data), encoding="utf-8"
        )
        other_record = {
            "name": "other-spec",
            "summary": "second predecessor owner probe",
            "transport": transport,
            "source": "test",
        }
        if transport == "http":
            other_record["url"] = _cfg(transport)["url"]
        else:
            other_record["command"] = _cfg(transport)["command"]
            other_record["args"] = _cfg(transport)["args"]
        with (agent._working_dir / "mcp_registry.jsonl").open(
            "a", encoding="utf-8"
        ) as registry:
            registry.write(json.dumps(other_record) + "\n")
    elif invalid_projection == "missing_schema":
        # Owner and handler claim the predecessor, but the schema projection
        # is missing.  Partial ownership is not eligible for reconciliation.
        agent._tool_schemas = [
            schema for schema in agent._tool_schemas if schema.name != SHARED
        ]
    elif invalid_projection == "missing_owner":
        del agent._mcp_clients_by_tool[SHARED]
    elif invalid_projection == "missing_mcp_name":
        agent._mcp_tool_names.remove(SHARED)
    else:
        agent._mcp_clients.append(predecessor)

    Candidate = _patch_candidate(monkeypatch, transport, predecessor, events)
    snapshot = _snapshot(agent, spec)
    agent._sealed = True
    try:
        report = agent._retry_failed_mcps()

        assert SPEC_NAME in report["still_failed"]
        assert SPEC_NAME not in report["recovered"]
        assert predecessor.close_calls == 0
        assert "predecessor.close" not in events
        assert all(not candidate.started for candidate in Candidate.instances)
        _assert_snapshot_identity(agent, snapshot)
    finally:
        agent._workdir_lease.release()


@pytest.mark.parametrize("transport", ["stdio", "http"])
def test_exact_unhealthy_predecessor_reconciles_old_union_new_after_retirement(
    tmp_path, monkeypatch, transport
):
    events: list[str] = []
    predecessor = _Predecessor(events)
    agent = _agent(tmp_path)
    spec = _install_predecessor(agent, predecessor, transport)
    old_handlers = {name: agent._tool_handlers[name] for name in PREDECESSOR_NAMES}
    Candidate = _patch_candidate(monkeypatch, transport, predecessor, events)
    agent._sealed = True
    try:
        report = agent._retry_failed_mcps()

        assert report["retried"] == [SPEC_NAME]
        assert report["recovered"] == [SPEC_NAME]
        assert report["still_failed"] == []
        assert len(Candidate.instances) == 1
        candidate = Candidate.instances[0]

        assert events.index("predecessor.close") < events.index("candidate.start")
        assert events.index("candidate.start") < events.index("candidate.list")
        assert predecessor.close_calls == 1
        assert predecessor.closed is True
        assert candidate.closed is False

        assert OLD_ONLY not in agent._tool_handlers
        assert OLD_ONLY not in agent._mcp_clients_by_tool
        assert OLD_ONLY not in agent._mcp_tool_names
        assert all(schema.name != OLD_ONLY for schema in agent._tool_schemas)
        assert all(
            handler is not old_handlers[OLD_ONLY]
            for handler in agent._tool_handlers.values()
        )

        assert set(agent._tool_handlers).issuperset(CANDIDATE_NAMES)
        assert {
            name: agent._mcp_clients_by_tool[name] for name in CANDIDATE_NAMES
        } == {name: candidate for name in CANDIDATE_NAMES}
        assert CANDIDATE_NAMES.issubset(agent._mcp_tool_names)
        for name in CANDIDATE_NAMES:
            handler = agent._tool_handlers[name]
            assert handler is not old_handlers.get(name)
            assert handler._lingtai_mcp_client is candidate
            assert handler._lingtai_mcp_tool_name == name
            schemas = [schema for schema in agent._tool_schemas if schema.name == name]
            assert len(schemas) == 1
            assert schemas[0].description == f"candidate schema for {name}"
            assert schemas[0].parameters["properties"]["generation"]["const"] == (
                "candidate"
            )

        assert sum(client is predecessor for client in agent._mcp_clients) == 0
        assert sum(client is candidate for client in agent._mcp_clients) == 1
        assert spec["client"] is candidate
        assert not any(
            getattr(handler, "_lingtai_mcp_client", None) is predecessor
            for handler in agent._tool_handlers.values()
        )
    finally:
        for candidate in Candidate.instances:
            if not candidate.closed:
                candidate.close()
        agent._workdir_lease.release()


@pytest.mark.parametrize("transport", ["stdio", "http"])
@pytest.mark.parametrize("failure_phase", ["startup", "list", "publication"])
def test_failure_after_predecessor_retirement_is_irreversible_and_retryable(
    tmp_path, monkeypatch, transport, failure_phase
):
    events: list[str] = []
    predecessor = _Predecessor(events)
    agent = _agent(tmp_path)
    spec = _install_predecessor(agent, predecessor, transport)
    old_handlers = {name: agent._tool_handlers[name] for name in PREDECESSOR_NAMES}
    Candidate = _patch_candidate(
        monkeypatch,
        transport,
        predecessor,
        events,
        failure_phase=failure_phase,
    )

    if failure_phase == "publication":
        class FailCandidatePublicationUpdate:
            def __init__(self):
                self.calls = 0

            def update_tools(self, tools) -> None:
                self.calls += 1
                # Retirement first removes the predecessor from the live
                # inventory. Fail the subsequent candidate publication update.
                if self.calls == 2:
                    raise RuntimeError("candidate publication failed")

        agent._chat = FailCandidatePublicationUpdate()

    # Runtime retries happen after start() seals the public tool surface.  The
    # same-spec predecessor path is an explicitly authorised internal mutation.
    agent._sealed = True
    try:
        report = agent._retry_failed_mcps()

        assert report["retried"] == [SPEC_NAME]
        assert report["recovered"] == []
        assert report["still_failed"] == [SPEC_NAME]
        assert predecessor.close_calls == 1
        assert predecessor.closed is True
        assert events.index("predecessor.close") < events.index("candidate.start")

        assert len(Candidate.instances) == 1
        candidate = Candidate.instances[0]
        assert candidate.closed is True
        assert candidate.close_calls == 1
        assert spec["client"] is None
        assert predecessor not in agent._mcp_clients
        assert candidate not in agent._mcp_clients

        for name in PREDECESSOR_NAMES | CANDIDATE_NAMES:
            assert name not in agent._tool_handlers
            assert name not in agent._mcp_clients_by_tool
            assert name not in agent._mcp_tool_names
            assert all(schema.name != name for schema in agent._tool_schemas)
        assert all(
            handler not in old_handlers.values()
            for handler in agent._tool_handlers.values()
        )
        assert not any(
            getattr(handler, "_lingtai_mcp_client", None) in {predecessor, candidate}
            for handler in agent._tool_handlers.values()
        )
    finally:
        agent._workdir_lease.release()


@pytest.mark.parametrize("transport", ["stdio", "http"])
def test_predecessor_depublication_inventory_failure_stays_pending_until_resynced(
    tmp_path, monkeypatch, transport
):
    """A failed live depublish cannot be reported as converged cleanup."""
    events: list[str] = []
    predecessor = _Predecessor(events)
    agent = _agent(tmp_path)
    spec = _install_predecessor(agent, predecessor, transport)
    Candidate = _patch_candidate(monkeypatch, transport, predecessor, events)

    class FailFirstInventoryUpdate:
        def __init__(self):
            self.snapshots: list[set[str]] = []

        def update_tools(self, tools) -> None:
            self.snapshots.append({tool.name for tool in tools})
            if len(self.snapshots) == 1:
                raise RuntimeError("predecessor depublish inventory failed")

    chat = FailFirstInventoryUpdate()
    agent._chat = chat
    agent._sealed = True
    try:
        report = agent._retry_failed_mcps()

        assert report["retried"] == [SPEC_NAME]
        assert report["recovered"] == []
        assert report["still_failed"] == [SPEC_NAME]
        assert report["converged"] is False
        assert predecessor.closed is True
        assert spec["client"] is None
        assert agent._mcp_inventory_sync_pending is True
        assert len(Candidate.instances) == 1
        candidate = Candidate.instances[0]
        assert candidate.started is False
        assert candidate.closed is True
        assert "candidate.start" not in events
        assert PREDECESSOR_NAMES.isdisjoint(chat.snapshots[0])

        cleanup = agent._retire_all_mcp_clients(context="inventory-resync")

        assert cleanup == {
            "context": "inventory-resync",
            "attempted": [],
                "retired": [],
                "unresolved": [],
                "transport_converged": True,
                "inventory_sync_deferred": False,
                "converged": True,
            }
        assert agent._mcp_inventory_sync_pending is False
        assert len(chat.snapshots) == 2
        assert PREDECESSOR_NAMES.isdisjoint(chat.snapshots[1])
    finally:
        agent._workdir_lease.release()


@pytest.mark.parametrize("transport", ["stdio", "http"])
def test_retry_health_probe_exception_is_unhealthy_and_reconciles_exact_owner(
    tmp_path, monkeypatch, transport
):
    """A broken old health probe cannot preserve a callable stale route."""
    events: list[str] = []
    predecessor = _Predecessor(events)
    agent = _agent(tmp_path)
    spec = _install_predecessor(agent, predecessor, transport)

    def _raising_health_probe() -> bool:
        if not predecessor.closed:
            raise RuntimeError("old health probe exploded")
        return False

    predecessor.is_connected = _raising_health_probe  # type: ignore[method-assign]
    Candidate = _patch_candidate(monkeypatch, transport, predecessor, events)
    agent._sealed = True
    try:
        report = agent._retry_failed_mcps()

        assert report["recovered"] == [SPEC_NAME]
        assert report["still_failed"] == []
        assert predecessor.closed is True
        assert len(Candidate.instances) == 1
        replacement = Candidate.instances[0]
        assert spec["client"] is replacement
        assert all(
            owner is replacement
            for owner in agent._mcp_clients_by_tool.values()
        )
    finally:
        agent._retire_all_mcp_clients(context="health-probe-test")
        agent._workdir_lease.release()
