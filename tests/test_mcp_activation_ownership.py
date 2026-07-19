"""Contract tests for MCP ownership and collision preflight."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from lingtai.agent import Agent
from lingtai.kernel.llm.base import FunctionSchema
from lingtai.services import mcp as mcp_module
from tests._service_helpers import make_gemini_mock_service


_FREE_NAME = "ownership_candidate_free"
_BASELINE_MCP_NAME = "ownership_existing_mcp"


class _ExistingClient:
    def __init__(self, label: str, *, healthy: bool = True):
        self.label = label
        self.healthy = healthy
        self.closed = False

    def call_tool(self, name: str, args: dict) -> dict:
        return {"status": "ok", "name": name, "args": args}

    def is_connected(self) -> bool:
        return self.healthy and not self.closed

    def close(self) -> None:
        self.closed = True


class _PublicationSpy:
    def __init__(self):
        self.calls = 0

    def update_tools(self, tools) -> None:
        self.calls += 1


@dataclass(frozen=True)
class _SurfaceSnapshot:
    handlers: dict[str, Any]
    handler_items: tuple[tuple[str, Any], ...]
    schemas: list[FunctionSchema]
    schema_items: tuple[FunctionSchema, ...]
    owners: dict[str, Any]
    owner_items: tuple[tuple[str, Any], ...]
    names: set[str]
    name_values: frozenset[str]
    clients: list[Any]
    client_items: tuple[Any, ...]
    init_specs: dict[str, dict]
    init_spec_items: tuple[tuple[str, dict], ...]
    init_spec_contents: dict[str, tuple[tuple[str, Any], ...]]
    token_dirty: bool


def _agent(tmp_path: Path) -> Agent:
    return Agent(
        service=make_gemini_mock_service(),
        agent_name="mcp-ownership-test",
        working_dir=tmp_path / "agent",
        capabilities={"mcp": {}},
    )


def _schema(name: str, *, description: str = "existing registration") -> FunctionSchema:
    return FunctionSchema(
        name=name,
        description=description,
        parameters={"type": "object", "properties": {}},
    )


def _tool(name: str, *, schema: Any = None) -> dict:
    if schema is None:
        schema = {"type": "object", "properties": {}}
    return {
        "name": name,
        "description": f"candidate tool {name}",
        "schema": schema,
    }


def _seed_consistent_existing_mcp(agent: Agent) -> _ExistingClient:
    client = _ExistingClient("healthy-foreign")
    agent._tool_handlers[_BASELINE_MCP_NAME] = agent._make_mcp_handler(
        client, _BASELINE_MCP_NAME
    )
    agent._tool_schemas.append(_schema(_BASELINE_MCP_NAME))
    agent._mcp_clients_by_tool[_BASELINE_MCP_NAME] = client
    agent._mcp_tool_names.add(_BASELINE_MCP_NAME)
    agent._mcp_clients.append(client)
    cfg = {"type": "stdio", "command": "existing-server", "args": []}
    agent._mcp_init_specs = {
        "existing-spec": {
            "cfg": cfg,
            "source": "init.json:mcp",
            "client": client,
        }
    }
    return client


def _snapshot(agent: Agent) -> _SurfaceSnapshot:
    return _SurfaceSnapshot(
        handlers=agent._tool_handlers,
        handler_items=tuple(agent._tool_handlers.items()),
        schemas=agent._tool_schemas,
        schema_items=tuple(agent._tool_schemas),
        owners=agent._mcp_clients_by_tool,
        owner_items=tuple(agent._mcp_clients_by_tool.items()),
        names=agent._mcp_tool_names,
        name_values=frozenset(agent._mcp_tool_names),
        clients=agent._mcp_clients,
        client_items=tuple(agent._mcp_clients),
        init_specs=agent._mcp_init_specs,
        init_spec_items=tuple(agent._mcp_init_specs.items()),
        init_spec_contents={
            name: tuple(spec.items()) for name, spec in agent._mcp_init_specs.items()
        },
        token_dirty=agent._token_decomp_dirty,
    )


def _assert_identity_mapping_unchanged(
    actual: dict[str, Any],
    expected_object: dict[str, Any],
    expected_items: tuple[tuple[str, Any], ...],
) -> None:
    assert actual is expected_object
    assert tuple(actual) == tuple(name for name, _ in expected_items)
    for name, value in expected_items:
        assert actual[name] is value


def _assert_surface_unchanged(agent: Agent, before: _SurfaceSnapshot) -> None:
    _assert_identity_mapping_unchanged(
        agent._tool_handlers, before.handlers, before.handler_items
    )
    assert agent._tool_schemas is before.schemas
    assert len(agent._tool_schemas) == len(before.schema_items)
    assert all(
        actual is expected
        for actual, expected in zip(agent._tool_schemas, before.schema_items)
    )
    _assert_identity_mapping_unchanged(
        agent._mcp_clients_by_tool, before.owners, before.owner_items
    )
    assert agent._mcp_tool_names is before.names
    assert agent._mcp_tool_names == before.name_values
    assert agent._mcp_clients is before.clients
    assert len(agent._mcp_clients) == len(before.client_items)
    assert all(
        actual is expected
        for actual, expected in zip(agent._mcp_clients, before.client_items)
    )
    _assert_identity_mapping_unchanged(
        agent._mcp_init_specs, before.init_specs, before.init_spec_items
    )
    for name, spec in before.init_spec_items:
        assert tuple(agent._mcp_init_specs[name]) == tuple(
            key for key, _ in before.init_spec_contents[name]
        )
        for key, value in before.init_spec_contents[name]:
            assert agent._mcp_init_specs[name][key] is value
    assert agent._token_decomp_dirty is before.token_dirty


def _install_candidate(monkeypatch, transport: str, catalog: list[dict]):
    class Candidate:
        instances: list["Candidate"] = []

        def __init__(self, *args, **kwargs):
            self.started = False
            self.list_calls = 0
            self.closed = False
            type(self).instances.append(self)

        def start(self) -> None:
            self.started = True

        def list_tools(self) -> list[dict]:
            self.list_calls += 1
            return catalog

        def is_connected(self) -> bool:
            return self.started and not self.closed

        def call_tool(self, name: str, args: dict) -> dict:
            return {"status": "ok", "name": name, "args": args}

        def close(self) -> None:
            self.closed = True

    class_name = "MCPClient" if transport == "stdio" else "HTTPMCPClient"
    monkeypatch.setattr(mcp_module, class_name, Candidate)
    return Candidate


def _connect(agent: Agent, transport: str) -> list[str]:
    if transport == "stdio":
        return agent.connect_mcp(command="candidate-server")
    return agent.connect_mcp_http(url="http://candidate.invalid/mcp")


def _arrange_collision(agent: Agent, collision_kind: str) -> str:
    if collision_kind == "intrinsic":
        assert "system" in agent._intrinsics
        return "system"

    if collision_kind == "builtin_handler":
        name = "ownership_builtin_handler"
        agent.add_tool(
            name,
            schema={"type": "object", "properties": {}},
            handler=lambda args: {"status": "ok", "args": args},
            description="non-MCP built-in registration",
        )
        return name

    if collision_kind == "reserved_task_card":
        assert "task_card" not in agent._tool_handlers
        assert all(item.name != "task_card" for item in agent._tool_schemas)
        return "task_card"

    if collision_kind == "reserved_bash_alias":
        # `bash` remains an accepted wire alias for the intrinsic `shell`
        # route even though it has no separately published handler/schema.
        agent.add_tool(
            "shell",
            schema={"type": "object", "properties": {}},
            handler=lambda args: {"status": "ok", "args": args},
            description="local shell compatibility target",
        )
        assert "shell" in agent._tool_handlers
        assert "bash" not in agent._tool_handlers
        return "bash"

    if collision_kind == "healthy_foreign_mcp":
        return _BASELINE_MCP_NAME

    if collision_kind == "unowned_projection":
        name = "ownership_unowned_projection"
        agent._tool_schemas.append(_schema(name, description="orphaned schema"))
        agent._mcp_tool_names.add(name)
        assert name not in agent._tool_handlers
        assert name not in agent._mcp_clients_by_tool
        return name

    if collision_kind == "inconsistent_projection":
        name = "ownership_inconsistent_projection"
        mapped_client = _ExistingClient("mapped-owner")
        handler_client = _ExistingClient("handler-owner")
        agent._mcp_clients.extend([mapped_client, handler_client])
        agent._mcp_clients_by_tool[name] = mapped_client
        agent._mcp_tool_names.add(name)
        agent._tool_handlers[name] = agent._make_mcp_handler(handler_client, name)
        agent._tool_schemas.append(_schema(name, description="split ownership"))
        return name

    raise AssertionError(f"unknown collision kind: {collision_kind}")


@pytest.mark.parametrize("transport", ["stdio", "http"])
@pytest.mark.parametrize(
    "collision_kind",
    [
        "intrinsic",
        "builtin_handler",
        "reserved_task_card",
        "reserved_bash_alias",
        "healthy_foreign_mcp",
        "unowned_projection",
        "inconsistent_projection",
    ],
)
def test_complete_ownership_preflight_rejects_collision_without_mutation(
    tmp_path, monkeypatch, transport, collision_kind
):
    """Every advertised name is classified before any candidate publication."""
    agent = _agent(tmp_path)
    _seed_consistent_existing_mcp(agent)
    collision_name = _arrange_collision(agent, collision_kind)
    publication = _PublicationSpy()
    agent._chat = publication
    before = _snapshot(agent)
    Candidate = _install_candidate(
        monkeypatch,
        transport,
        [_tool(_FREE_NAME), _tool(collision_name)],
    )

    try:
        with pytest.raises(Exception) as exc_info:
            _connect(agent, transport)

        candidate = Candidate.instances[-1]
        assert collision_name in str(exc_info.value)
        assert candidate.started is True
        assert candidate.list_calls == 1
        assert candidate.closed is True
        assert publication.calls == 0
        _assert_surface_unchanged(agent, before)
        assert candidate not in agent._mcp_clients
        assert candidate not in agent._mcp_clients_by_tool.values()
        assert candidate not in agent._mcp_retiring_clients
        assert _FREE_NAME not in agent._tool_handlers
        assert _FREE_NAME not in agent._mcp_clients_by_tool
    finally:
        agent._workdir_lease.release()


@pytest.mark.parametrize("transport", ["stdio", "http"])
@pytest.mark.parametrize("invalid_catalog", ["duplicate_names", "malformed_schema"])
def test_invalid_complete_catalog_fails_closed_without_partial_publication(
    tmp_path, monkeypatch, transport, invalid_catalog
):
    """Duplicate names and malformed schemas reject the whole candidate batch."""
    agent = _agent(tmp_path)
    _seed_consistent_existing_mcp(agent)
    publication = _PublicationSpy()
    agent._chat = publication
    before = _snapshot(agent)
    if invalid_catalog == "duplicate_names":
        catalog = [_tool(_FREE_NAME), _tool(_FREE_NAME)]
        error_fragment = "duplicate"
    else:
        catalog = [
            _tool(_FREE_NAME),
            _tool(
                "ownership_candidate_bad_schema",
                schema={"type": "object", "properties": ["not", "an", "object"]},
            ),
        ]
        error_fragment = "schema"
    Candidate = _install_candidate(monkeypatch, transport, catalog)

    try:
        with pytest.raises((TypeError, ValueError), match=error_fragment):
            _connect(agent, transport)

        candidate = Candidate.instances[-1]
        assert candidate.started is True
        assert candidate.list_calls == 1
        assert candidate.closed is True
        assert publication.calls == 0
        _assert_surface_unchanged(agent, before)
        assert candidate not in agent._mcp_retiring_clients
        assert _FREE_NAME not in agent._tool_handlers
        assert _FREE_NAME not in agent._mcp_clients_by_tool
    finally:
        agent._workdir_lease.release()
