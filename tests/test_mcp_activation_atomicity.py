"""Atomic publication and retirement tests for MCP activation."""
from __future__ import annotations

import threading
import time
import json
from pathlib import Path

import pytest

from lingtai.agent import Agent
from lingtai.kernel.llm.base import ToolCall
from lingtai.kernel.types import UnknownToolError
from lingtai.services import mcp as mcp_module
from lingtai.tools.system.preset import _refresh
from tests._service_helpers import make_gemini_mock_service
from tests._workdir_lease_helpers import RecordingWorkdirLease


TOOL_NAME = "activation_atomic_probe"


def _agent(tmp_path: Path, *, workdir_lease=None) -> Agent:
    lease_kwargs = (
        {"workdir_lease": workdir_lease}
        if workdir_lease is not None
        else {}
    )
    return Agent(
        service=make_gemini_mock_service(),
        agent_name="activation-atomicity-test",
        working_dir=tmp_path / "agent",
        capabilities={"mcp": {}},
        **lease_kwargs,
    )


def _catalog() -> list[dict]:
    return [
        {
            "name": TOOL_NAME,
            "description": "MCP activation atomicity probe",
            "schema": {
                "type": "object",
                "properties": {"value": {"type": "string"}},
            },
        }
    ]


def _write_current_mcp_specs(agent: Agent, specs: dict[str, dict]) -> None:
    """Materialize retry specs through the same init/registry gate as boot."""
    (agent._working_dir / "init.json").write_text(
        json.dumps({"mcp": specs}),
        encoding="utf-8",
    )
    records = []
    for name, cfg in specs.items():
        record = {
            "name": name,
            "summary": f"{name} test MCP",
            "transport": cfg.get("type", "stdio"),
            "source": "test",
        }
        if record["transport"] == "http":
            record["url"] = cfg["url"]
        else:
            record["command"] = cfg["command"]
            record["args"] = cfg.get("args", [])
        records.append(json.dumps(record))
    (agent._working_dir / "mcp_registry.jsonl").write_text(
        "\n".join(records) + "\n",
        encoding="utf-8",
    )


def _schema_names(tools) -> list[str]:
    return [tool.name for tool in tools]


@pytest.mark.parametrize("transport", ["stdio", "http"])
def test_post_publication_chat_failure_exactly_rolls_back_and_closes_candidate(
    tmp_path, monkeypatch, transport
):
    """A live-chat update failure restores every prior state object exactly."""

    class Candidate:
        instances: list["Candidate"] = []

        def __init__(self, *args, **kwargs):
            self.started = False
            self.closed = False
            type(self).instances.append(self)

        def start(self):
            self.started = True

        def list_tools(self):
            return _catalog()

        def is_connected(self):
            return self.started and not self.closed

        def close(self):
            self.closed = True

    class FailFirstChatUpdate:
        def __init__(self):
            self.tool_snapshots: list[list[str]] = []

        def update_tools(self, tools):
            self.tool_snapshots.append(_schema_names(tools))
            if len(self.tool_snapshots) == 1:
                raise RuntimeError("live chat rejected candidate tools")

    class_name = "MCPClient" if transport == "stdio" else "HTTPMCPClient"
    monkeypatch.setattr(mcp_module, class_name, Candidate)
    agent = _agent(tmp_path)
    unrelated_handler = lambda args: {"kept": args}
    agent.add_tool(
        "unrelated",
        schema={"type": "object", "properties": {}},
        handler=unrelated_handler,
        description="pre-existing tool",
    )
    chat = FailFirstChatUpdate()
    agent._chat = chat

    handlers_before = agent._tool_handlers
    schemas_before = agent._tool_schemas
    clients_before = agent._mcp_clients
    owners_before = agent._mcp_clients_by_tool
    names_before = agent._mcp_tool_names
    token_dirty_before = agent._token_decomp_dirty

    connect = agent.connect_mcp if transport == "stdio" else agent.connect_mcp_http
    kwargs = {"command": "fake"} if transport == "stdio" else {"url": "http://fake"}
    try:
        with pytest.raises(RuntimeError, match="live chat rejected candidate tools"):
            connect(**kwargs)

        candidate = Candidate.instances[-1]
        assert candidate.started is True
        assert candidate.closed is True
        assert agent._tool_handlers is handlers_before
        assert agent._tool_schemas is schemas_before
        assert agent._mcp_clients is clients_before
        assert agent._mcp_clients_by_tool is owners_before
        assert agent._mcp_tool_names is names_before
        assert agent._token_decomp_dirty is token_dirty_before
        assert agent._tool_handlers["unrelated"] is unrelated_handler
        assert candidate not in agent._mcp_retiring_clients

        assert len(chat.tool_snapshots) == 2
        assert TOOL_NAME in chat.tool_snapshots[0]
        assert TOOL_NAME not in chat.tool_snapshots[1]
        assert "unrelated" in chat.tool_snapshots[0]
        assert "unrelated" in chat.tool_snapshots[1]
    finally:
        agent._workdir_lease.release()


def test_dispatch_waits_for_activation_rollback_and_never_calls_candidate(tmp_path):
    """Concurrent dispatch cannot observe a candidate whose publication fails."""
    published_to_chat = threading.Event()
    allow_chat_failure = threading.Event()
    dispatch_lock_attempted = threading.Event()
    dispatch_done = threading.Event()

    class Candidate:
        def __init__(self):
            self.closed = False
            self.calls: list[tuple[str, dict]] = []

        def start(self):
            pass

        def list_tools(self):
            return _catalog()

        def call_tool(self, name, args):
            self.calls.append((name, dict(args)))
            return {"status": "ok"}

        def is_connected(self):
            return not self.closed

        def close(self):
            self.closed = True

    class BlockingFailFirstChatUpdate:
        def __init__(self):
            self.calls = 0

        def update_tools(self, tools):
            self.calls += 1
            if self.calls == 1:
                assert TOOL_NAME in _schema_names(tools)
                published_to_chat.set()
                assert allow_chat_failure.wait(5), "test did not release chat update"
                raise RuntimeError("publish hook failed")
            assert TOOL_NAME not in _schema_names(tools)

    agent = _agent(tmp_path)
    candidate = Candidate()
    agent._chat = BlockingFailFirstChatUpdate()

    class InstrumentedLifecycleLock:
        """Expose the dispatch thread's actual registry-lock attempt."""

        def __init__(self, inner):
            self.inner = inner

        def acquire(self, *args, **kwargs):
            if threading.current_thread().name == "mcp-dispatch-test":
                dispatch_lock_attempted.set()
            return self.inner.acquire(*args, **kwargs)

        def release(self):
            return self.inner.release()

        def __enter__(self):
            self.acquire()
            return self

        def __exit__(self, *_args):
            self.release()

    agent._mcp_lifecycle_lock = InstrumentedLifecycleLock(
        agent._mcp_lifecycle_lock
    )
    activation_errors: list[BaseException] = []
    dispatch_errors: list[BaseException] = []

    def activate():
        try:
            agent._activate_mcp_candidate(candidate)
        except BaseException as exc:  # captured for assertions in the test thread
            activation_errors.append(exc)

    def dispatch():
        try:
            agent._dispatch_tool(ToolCall(name=TOOL_NAME, args={}, id="call-1"))
        except BaseException as exc:  # captured for assertions in the test thread
            dispatch_errors.append(exc)
        finally:
            dispatch_done.set()

    activation_thread = threading.Thread(target=activate, name="mcp-activation-test")
    dispatch_thread = threading.Thread(target=dispatch, name="mcp-dispatch-test")
    try:
        activation_thread.start()
        assert published_to_chat.wait(5), "candidate did not reach live-chat publication"
        dispatch_thread.start()
        assert dispatch_lock_attempted.wait(1), "dispatch never attempted registry lock"
        assert not dispatch_done.wait(0.2), "dispatch escaped the lifecycle lock"

        allow_chat_failure.set()
        activation_thread.join(timeout=5)
        dispatch_thread.join(timeout=5)
        assert not activation_thread.is_alive()
        assert not dispatch_thread.is_alive()

        assert len(activation_errors) == 1
        assert isinstance(activation_errors[0], RuntimeError)
        assert "publish hook failed" in str(activation_errors[0])
        assert len(dispatch_errors) == 1
        assert isinstance(dispatch_errors[0], UnknownToolError)
        assert candidate.calls == []
        assert candidate.closed is True
        assert TOOL_NAME not in agent._tool_handlers
        assert TOOL_NAME not in agent._mcp_clients_by_tool
    finally:
        allow_chat_failure.set()
        activation_thread.join(timeout=5)
        if dispatch_thread.ident is not None:
            dispatch_thread.join(timeout=5)
        agent._workdir_lease.release()


@pytest.mark.parametrize("retry_surface", ["stop", "deep_refresh"])
def test_unpublished_close_failure_is_retried_to_convergence(
    tmp_path, monkeypatch, retry_surface
):
    """Stop and deep refresh both retry an unpublished pending retirement."""

    class CloseFailsOnceCandidate:
        def __init__(self):
            self.close_calls = 0
            self.closed = False

        def start(self):
            pass

        def list_tools(self):
            raise RuntimeError("catalog unavailable")

        def is_connected(self):
            return False

        def close(self):
            self.close_calls += 1
            if self.close_calls == 1:
                raise RuntimeError("first close failed")
            self.closed = True

    agent = _agent(tmp_path)
    candidate = CloseFailsOnceCandidate()
    stopped = False
    try:
        with pytest.raises(RuntimeError, match="candidate cleanup is unresolved"):
            agent._activate_mcp_candidate(candidate)

        assert candidate.close_calls == 1
        assert agent._mcp_retiring_clients == [candidate]
        assert candidate not in agent._mcp_clients
        assert TOOL_NAME not in agent._tool_handlers

        if retry_surface == "stop":
            agent.stop()
            stopped = True
            assert agent._last_mcp_cleanup_report["transport_converged"] is True
            assert agent._last_mcp_cleanup_report["inventory_sync_deferred"] is True
            assert agent._last_mcp_cleanup_report["converged"] is False
        else:
            monkeypatch.setattr(agent, "_read_init", lambda: None)
            agent._setup_from_init()

        assert candidate.close_calls == 2
        assert candidate.closed is True
        assert agent._mcp_retiring_clients == []
    finally:
        if not stopped:
            agent._workdir_lease.release()


def test_stop_boundedly_blocks_on_staging_candidate_and_keeps_lease(tmp_path):
    """Stop cannot release ownership while startup/catalog staging holds the gate."""

    class StagingCandidate:
        def __init__(self):
            self.started = threading.Event()
            self.allow_start = threading.Event()
            self.closed = False

        def start(self):
            self.started.set()
            assert self.allow_start.wait(2.0)

        def list_tools(self):
            return _catalog()

        def is_connected(self):
            return not self.closed

        def close(self):
            self.closed = True

    lease = RecordingWorkdirLease()
    agent = _agent(tmp_path, workdir_lease=lease)
    candidate = StagingCandidate()
    activation_errors: list[BaseException] = []

    def activate():
        try:
            agent._activate_mcp_candidate(candidate)
        except BaseException as exc:
            activation_errors.append(exc)

    activation_thread = threading.Thread(target=activate, daemon=True)
    activation_thread.start()
    assert candidate.started.wait(1.0)
    stopped = False
    try:
        with pytest.raises(RuntimeError, match="MCP cleanup remained unresolved"):
            agent.stop(timeout=0.01)

        report = agent._last_mcp_cleanup_report
        assert report["transport_converged"] is False
        assert report["unresolved"][0]["client"] == "activation-transaction"
        assert lease.held is True
        assert lease.releases == 0

        candidate.allow_start.set()
        activation_thread.join(timeout=2.0)
        assert not activation_thread.is_alive()
        assert len(activation_errors) == 1
        assert "shutting down" in str(activation_errors[0])
        assert candidate.closed is True
        assert TOOL_NAME not in agent._tool_handlers

        agent.stop(timeout=1.0)
        stopped = True
        assert lease.held is False
        assert lease.releases == 1
    finally:
        candidate.allow_start.set()
        activation_thread.join(timeout=2.0)
        if not stopped:
            agent._workdir_lease.release()


@pytest.mark.parametrize("transport", ["stdio", "http"])
def test_sealed_failed_initial_spec_retry_recovers_and_publishes_once(
    tmp_path, monkeypatch, transport
):
    """An internal failed-initial retry may publish once on a sealed agent."""

    class Candidate:
        instances: list["Candidate"] = []

        def __init__(self, *args, **kwargs):
            self.started = False
            self.closed = False
            type(self).instances.append(self)

        def start(self):
            self.started = True

        def list_tools(self):
            return _catalog()

        def is_connected(self):
            return self.started and not self.closed

        def call_tool(self, name, args):
            return {"status": "ok", "name": name, "args": args}

        def close(self):
            self.closed = True

    class_name = "MCPClient" if transport == "stdio" else "HTTPMCPClient"
    monkeypatch.setattr(mcp_module, class_name, Candidate)
    agent = _agent(tmp_path)
    cfg = (
        {"type": "stdio", "command": "fake", "args": []}
        if transport == "stdio"
        else {"type": "http", "url": "http://fake"}
    )
    agent._mcp_init_specs = {
        "failed-initial": {
            "cfg": cfg,
            "source": "init.json:mcp",
            "client": None,
        }
    }
    _write_current_mcp_specs(agent, {"failed-initial": cfg})
    agent._sealed = True
    stopped = False
    try:
        first_report = agent._retry_failed_mcps()
        assert first_report == {
            "retried": ["failed-initial"],
            "recovered": ["failed-initial"],
            "still_failed": [],
            "healthy": [],
            "unresolved": [],
            "converged": True,
        }
        assert len(Candidate.instances) == 1
        candidate = Candidate.instances[0]
        assert agent._mcp_init_specs["failed-initial"]["client"] is candidate
        assert agent._mcp_clients == [candidate]
        assert agent._mcp_clients_by_tool == {TOOL_NAME: candidate}
        assert [schema.name for schema in agent._tool_schemas].count(TOOL_NAME) == 1
        assert agent._tool_handlers[TOOL_NAME]._lingtai_mcp_client is candidate

        second_report = agent._retry_failed_mcps()
        assert second_report == {
            "retried": [],
            "recovered": [],
            "still_failed": [],
            "healthy": ["failed-initial"],
            "unresolved": [],
            "converged": True,
        }
        assert len(Candidate.instances) == 1
        assert agent._mcp_clients == [candidate]
        assert [schema.name for schema in agent._tool_schemas].count(TOOL_NAME) == 1

        agent.stop()
        stopped = True
        assert candidate.closed is True
    finally:
        if not stopped:
            agent._workdir_lease.release()


def test_sealed_failed_initial_telegram_retry_registers_one_task_card(tmp_path, monkeypatch):
    """Telegram recovery composes its reserved controller once while sealed."""

    class TelegramCandidate:
        instances: list["TelegramCandidate"] = []

        def __init__(self, *args, **kwargs):
            self.started = False
            self.closed = False
            type(self).instances.append(self)

        def start(self):
            self.started = True

        def list_tools(self):
            return [
                {
                    "name": "telegram",
                    "description": "fake Telegram reverse route",
                    "schema": {
                        "type": "object",
                        "properties": {"action": {"type": "string"}},
                        "required": ["action"],
                    },
                }
            ]

        def is_connected(self):
            return self.started and not self.closed

        def call_tool(self, name, args, timeout=None):
            return {"status": "ok", "name": name, "args": args}

        def close(self):
            self.closed = True

    monkeypatch.setattr(mcp_module, "MCPClient", TelegramCandidate)
    agent = _agent(tmp_path)
    cfg = {"type": "stdio", "command": "fake-telegram", "args": []}
    agent._mcp_init_specs = {
        "telegram": {
            "cfg": cfg,
            "source": "init.json:mcp",
            "client": None,
        }
    }
    _write_current_mcp_specs(agent, {"telegram": cfg})
    agent._sealed = True
    stopped = False
    try:
        first_report = agent._retry_failed_mcps()
        assert first_report == {
            "retried": ["telegram"],
            "recovered": ["telegram"],
            "still_failed": [],
            "healthy": [],
            "unresolved": [],
            "converged": True,
        }
        assert len(TelegramCandidate.instances) == 1
        candidate = TelegramCandidate.instances[0]
        controller = agent._task_card_controller
        assert agent._mcp_init_specs["telegram"]["client"] is candidate
        assert agent._mcp_clients == [candidate]
        assert agent._mcp_clients_by_tool == {"telegram": candidate}
        assert agent._mcp_tool_names == {"telegram"}
        assert agent._tool_handlers["telegram"]._lingtai_mcp_client is candidate
        assert getattr(agent._tool_handlers["task_card"], "__self__", None) is controller
        assert [schema.name for schema in agent._tool_schemas].count("telegram") == 1
        assert [schema.name for schema in agent._tool_schemas].count("task_card") == 1

        second_report = agent._retry_failed_mcps()
        assert second_report == {
            "retried": [],
            "recovered": [],
            "still_failed": [],
            "healthy": ["telegram"],
            "unresolved": [],
            "converged": True,
        }
        assert len(TelegramCandidate.instances) == 1
        assert agent._task_card_controller is controller
        assert agent._mcp_clients == [candidate]
        assert getattr(agent._tool_handlers["task_card"], "__self__", None) is controller
        assert [schema.name for schema in agent._tool_schemas].count("telegram") == 1
        assert [schema.name for schema in agent._tool_schemas].count("task_card") == 1

        agent.stop()
        stopped = True
        assert candidate.closed is True
    finally:
        if not stopped:
            agent._workdir_lease.release()


@pytest.mark.parametrize("transport", ["stdio", "http"])
def test_shutdown_blocks_sealed_retry_after_pending_cleanup_converges(
    tmp_path, monkeypatch, transport
):
    """Stop's shutdown gate cannot be crossed by a later internal retry."""

    class Candidate:
        instances: list["Candidate"] = []

        def __init__(self, *args, **kwargs):
            self.started = False
            self.closed = False
            type(self).instances.append(self)

        def start(self):
            self.started = True

        def list_tools(self):
            return _catalog()

        def is_connected(self):
            return self.started and not self.closed

        def close(self):
            self.closed = True

    class PendingCleanup:
        def __init__(self):
            self.close_calls = 0
            self.closed = False

        def is_connected(self):
            return False

        def close(self):
            self.close_calls += 1
            self.closed = True

    class_name = "MCPClient" if transport == "stdio" else "HTTPMCPClient"
    monkeypatch.setattr(mcp_module, class_name, Candidate)
    agent = _agent(tmp_path)
    pending = PendingCleanup()
    agent._mcp_retiring_clients = [pending]
    cfg = (
        {"type": "stdio", "command": "fake", "args": []}
        if transport == "stdio"
        else {"type": "http", "url": "http://fake"}
    )
    agent._mcp_init_specs = {
        "shutdown-retry": {
            "cfg": cfg,
            "source": "init.json:mcp",
            "client": None,
        }
    }
    _write_current_mcp_specs(agent, {"shutdown-retry": cfg})
    agent._sealed = True
    agent._shutdown.set()
    try:
        report = agent._retry_failed_mcps()

        assert report == {
            "retried": ["shutdown-retry"],
            "recovered": [],
            "still_failed": ["shutdown-retry"],
            "healthy": [],
            "unresolved": [],
            "converged": False,
        }
        assert pending.close_calls == 1
        assert pending.closed is True
        assert agent._mcp_retiring_clients == []
        assert len(Candidate.instances) == 1
        candidate = Candidate.instances[0]
        assert candidate.started is False
        assert candidate.closed is True
        assert agent._mcp_init_specs["shutdown-retry"]["client"] is None
        assert candidate not in agent._mcp_clients
        assert TOOL_NAME not in agent._tool_handlers
        assert TOOL_NAME not in agent._mcp_clients_by_tool
    finally:
        agent._workdir_lease.release()


def test_refresh_retries_failed_mcp_with_current_gated_init_config(
    tmp_path, monkeypatch
):
    """Editing a bad boot command before refresh must use the repaired command."""

    class Candidate:
        instances: list["Candidate"] = []

        def __init__(self, command, args=None, env=None):
            self.command = command
            self.started = False
            self.closed = False
            type(self).instances.append(self)

        def start(self):
            self.started = True

        def list_tools(self):
            return _catalog()

        def is_connected(self):
            return self.started and not self.closed

        def close(self):
            self.closed = True

    monkeypatch.setattr(mcp_module, "MCPClient", Candidate)
    agent = _agent(tmp_path)
    agent._sealed = True
    agent._mcp_init_specs = {
        "repaired": {
            "cfg": {"type": "stdio", "command": "broken-command"},
            "source": "init.json:mcp",
            "client": None,
        }
    }
    (agent._working_dir / "init.json").write_text(
        json.dumps(
            {
                "mcp": {
                    "repaired": {
                        "type": "stdio",
                        "command": "fixed-command",
                        "args": ["--fixed"],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    (agent._working_dir / "mcp_registry.jsonl").write_text(
        json.dumps(
            {
                "name": "repaired",
                "summary": "repaired MCP",
                "transport": "stdio",
                "command": "fixed-command",
                "args": ["--fixed"],
                "source": "test",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    handoffs: list[bool] = []

    def perform_refresh(**kwargs):
        before_spawn = kwargs.get("_before_spawn")
        if callable(before_spawn) and before_spawn() is not True:
            return False
        handoffs.append(True)
        return True

    agent._perform_refresh = perform_refresh
    try:
        result = _refresh(agent, {"reason": "use repaired MCP config"})

        assert result["status"] == "ok"
        assert handoffs == [True]
        assert len(Candidate.instances) == 1
        candidate = Candidate.instances[0]
        assert candidate.command == "fixed-command"
        assert candidate.started is True
        assert candidate.closed is True
        assert agent._mcp_clients == []
        assert agent._mcp_retiring_clients == []
        assert agent._mcp_init_specs["repaired"]["cfg"]["command"] == (
            "fixed-command"
        )
        assert agent._mcp_init_specs["repaired"]["client"] is None
    finally:
        agent._workdir_lease.release()


@pytest.mark.parametrize(
    "source_change",
    [
        "init_file_removed",
        "init_entry_removed",
        "init_entry_malformed",
        "registry_removed",
    ],
)
def test_retry_never_constructs_candidate_after_source_authorization_revoked(
    tmp_path, monkeypatch, source_change
):
    """A stale boot snapshot cannot outlive its current config/registry grant."""

    class Candidate:
        constructed = 0
        started = 0

        def __init__(self, *args, **kwargs):
            type(self).constructed += 1

        def start(self):
            type(self).started += 1

    monkeypatch.setattr(mcp_module, "MCPClient", Candidate)
    agent = _agent(tmp_path)
    cfg = {"type": "stdio", "command": "stale-command", "args": []}
    agent._mcp_init_specs = {
        "revoked": {
            "cfg": cfg,
            "source": "init.json:mcp",
            "client": None,
        }
    }
    _write_current_mcp_specs(agent, {"revoked": cfg})
    init_path = agent._working_dir / "init.json"
    registry_path = agent._working_dir / "mcp_registry.jsonl"
    if source_change == "init_file_removed":
        init_path.unlink()
    elif source_change == "init_entry_removed":
        init_path.write_text(json.dumps({"mcp": {}}), encoding="utf-8")
    elif source_change == "init_entry_malformed":
        init_path.write_text(
            json.dumps({"mcp": {"revoked": "not-an-object"}}),
            encoding="utf-8",
        )
    else:
        registry_path.write_text("", encoding="utf-8")

    agent._sealed = True
    try:
        report = agent._retry_failed_mcps()

        assert report == {
            "retried": [],
            "recovered": [],
            "still_failed": [],
            "healthy": [],
            "unresolved": [],
            "converged": True,
        }
        assert "revoked" not in agent._mcp_init_specs
        assert Candidate.constructed == 0
        assert Candidate.started == 0
        assert agent._mcp_clients == []
        assert TOOL_NAME not in agent._tool_handlers
    finally:
        agent._workdir_lease.release()


@pytest.mark.parametrize("transport", ["stdio", "http"])
def test_post_publish_health_probe_exception_depublishes_and_closes_candidate(
    tmp_path, monkeypatch, transport
):
    """A throwing post-publish probe is unhealthy, never an escaping route."""

    class Candidate:
        instances: list["Candidate"] = []

        def __init__(self, *args, **kwargs):
            self.started = False
            self.closed = False
            type(self).instances.append(self)

        def start(self):
            self.started = True

        def list_tools(self):
            return _catalog()

        def is_connected(self):
            if self.started and not self.closed:
                raise RuntimeError("candidate health probe exploded")
            return False

        def close(self):
            self.closed = True

    class_name = "MCPClient" if transport == "stdio" else "HTTPMCPClient"
    monkeypatch.setattr(mcp_module, class_name, Candidate)
    agent = _agent(tmp_path)
    cfg = (
        {"type": "stdio", "command": "fake", "args": []}
        if transport == "stdio"
        else {"type": "http", "url": "http://fake"}
    )
    agent._mcp_init_specs = {
        "throwing-health": {
            "cfg": cfg,
            "source": "init.json:mcp",
            "client": None,
        }
    }
    _write_current_mcp_specs(agent, {"throwing-health": cfg})
    agent._sealed = True
    try:
        report = agent._retry_failed_mcps()

        assert report["recovered"] == []
        assert report["still_failed"] == ["throwing-health"]
        assert report["converged"] is False
        candidate = Candidate.instances[0]
        assert candidate.closed is True
        assert agent._mcp_init_specs["throwing-health"]["client"] is None
        assert TOOL_NAME not in agent._tool_handlers
        assert TOOL_NAME not in agent._mcp_clients_by_tool
        assert candidate not in agent._mcp_clients
    finally:
        agent._workdir_lease.release()


@pytest.mark.parametrize("transport", ["stdio", "http"])
def test_retirement_winning_post_publish_health_race_closes_exactly_once(
    tmp_path, monkeypatch, transport
):
    """A stop-owned retirement cannot be repeated by the retry health path."""
    health_entered = threading.Event()
    release_health = threading.Event()

    class Candidate:
        instances: list["Candidate"] = []

        def __init__(self, *args, **kwargs):
            self.started = False
            self.closed = False
            self.close_calls = 0
            self.health_calls = 0
            type(self).instances.append(self)

        def start(self):
            self.started = True

        def list_tools(self):
            return _catalog()

        def is_connected(self):
            self.health_calls += 1
            if self.health_calls == 1:
                health_entered.set()
                assert release_health.wait(timeout=2)
            return self.started and not self.closed

        def close(self):
            self.close_calls += 1
            self.closed = True

    monkeypatch.setattr(
        mcp_module,
        "MCPClient" if transport == "stdio" else "HTTPMCPClient",
        Candidate,
    )
    agent = _agent(tmp_path)
    cfg = (
        {"type": "stdio", "command": "fake", "args": []}
        if transport == "stdio"
        else {"type": "http", "url": "http://fake"}
    )
    agent._mcp_init_specs = {
        "health-race": {
            "cfg": cfg,
            "source": "init.json:mcp",
            "client": None,
        }
    }
    _write_current_mcp_specs(agent, {"health-race": cfg})
    agent._sealed = True
    retry_reports: list[dict] = []
    retry_errors: list[BaseException] = []

    def retry() -> None:
        try:
            retry_reports.append(agent._retry_failed_mcps())
        except BaseException as exc:
            retry_errors.append(exc)

    retry_thread = threading.Thread(target=retry, daemon=True)
    retry_thread.start()
    assert health_entered.wait(timeout=1)
    try:
        cleanup = agent._retire_all_mcp_clients(context="concurrent-stop")
        candidate = Candidate.instances[0]
        assert cleanup["transport_converged"] is True
        assert candidate.close_calls == 1
        assert agent._mcp_init_specs["health-race"]["client"] is None

        release_health.set()
        retry_thread.join(timeout=2)
        assert not retry_thread.is_alive()
        assert retry_errors == []
        assert retry_reports[0]["still_failed"] == ["health-race"]
        assert candidate.close_calls == 1
        assert candidate not in agent._mcp_retiring_clients
    finally:
        release_health.set()
        retry_thread.join(timeout=2)
        agent._retire_all_mcp_clients(context="health-race-test")
        agent._workdir_lease.release()


@pytest.mark.parametrize("transport", ["stdio", "http"])
def test_legacy_delegating_connector_retry_keeps_internal_transaction(
    tmp_path, monkeypatch, transport
):
    """Old override signatures can retry when they delegate to Agent."""

    class LegacyDelegatingAgent(Agent):
        def connect_mcp(self, command, args=None, env=None):
            return super().connect_mcp(command=command, args=args, env=env)

        def connect_mcp_http(self, url, headers=None):
            return super().connect_mcp_http(url=url, headers=headers)

    class Candidate:
        instances: list["Candidate"] = []

        def __init__(self, *args, **kwargs):
            self.started = False
            self.closed = False
            type(self).instances.append(self)

        def start(self):
            self.started = True

        def list_tools(self):
            return _catalog()

        def is_connected(self):
            return self.started and not self.closed

        def close(self):
            self.closed = True

    class_name = "MCPClient" if transport == "stdio" else "HTTPMCPClient"
    monkeypatch.setattr(mcp_module, class_name, Candidate)
    agent = LegacyDelegatingAgent(
        service=make_gemini_mock_service(),
        agent_name="legacy-mcp-connector-test",
        working_dir=tmp_path / "agent",
        capabilities={"mcp": {}},
    )
    cfg = (
        {"type": "stdio", "command": "legacy-fake", "args": []}
        if transport == "stdio"
        else {"type": "http", "url": "http://legacy-fake"}
    )
    agent._mcp_init_specs = {
        "legacy-delegating": {
            "cfg": cfg,
            "source": "init.json:mcp",
            "client": None,
        }
    }
    _write_current_mcp_specs(agent, {"legacy-delegating": cfg})
    agent._sealed = True
    try:
        report = agent._retry_failed_mcps()

        assert report["recovered"] == ["legacy-delegating"]
        assert report["still_failed"] == []
        replacement = Candidate.instances[0]
        assert agent._mcp_init_specs["legacy-delegating"]["client"] is replacement
        assert agent._mcp_clients == [replacement]
    finally:
        agent._retire_all_mcp_clients(context="legacy-connector-test")
        agent._workdir_lease.release()


def test_shutdown_rejects_new_mcp_lease_before_remote_dispatch(tmp_path):
    """Once stop signals shutdown, no published route may gain a fresh lease."""

    class Client:
        def __init__(self):
            self.calls = 0

        def call_tool(self, name, args):
            self.calls += 1
            return {"status": "ok"}

    client = Client()
    agent = _agent(tmp_path)
    handler = agent._make_mcp_handler(client, TOOL_NAME)
    agent._mcp_clients = [client]
    agent._mcp_clients_by_tool = {TOOL_NAME: client}
    agent._tool_handlers[TOOL_NAME] = handler
    agent._shutdown.set()
    try:
        with pytest.raises(RuntimeError, match="shutting down"):
            handler({"value": "must not dispatch"})

        assert client.calls == 0
        assert agent._mcp_inflight_calls == {}
    finally:
        agent._workdir_lease.release()


@pytest.mark.parametrize("transport", ["stdio", "http"])
def test_remote_catalog_wait_does_not_hold_lifecycle_registry_lock(
    tmp_path, monkeypatch, transport
):
    """Unpublished remote latency cannot block unrelated registry snapshots."""
    catalog_entered = threading.Event()
    release_catalog = threading.Event()

    class Candidate:
        def __init__(self, *args, **kwargs):
            self.closed = False

        def start(self):
            pass

        def list_tools(self):
            catalog_entered.set()
            assert release_catalog.wait(timeout=2)
            return _catalog()

        def is_connected(self):
            return not self.closed

        def close(self):
            self.closed = True

    monkeypatch.setattr(
        mcp_module,
        "MCPClient" if transport == "stdio" else "HTTPMCPClient",
        Candidate,
    )
    agent = _agent(tmp_path)
    errors: list[BaseException] = []
    activation = threading.Thread(
        target=lambda: _capture_connect_error(agent, transport, errors)
    )
    activation.start()
    assert catalog_entered.wait(timeout=1)
    try:
        assert agent._mcp_lifecycle_lock.acquire(timeout=0.2)
        agent._mcp_lifecycle_lock.release()
    finally:
        release_catalog.set()
        activation.join(timeout=2)
        assert not activation.is_alive()
        agent._retire_all_mcp_clients(context="catalog-lock-test")
        agent._workdir_lease.release()
    assert errors == []


def _capture_connect_error(agent: Agent, transport: str, errors: list) -> None:
    try:
        if transport == "stdio":
            agent.connect_mcp(command="blocking-catalog")
        else:
            agent.connect_mcp_http(url="http://blocking-catalog")
    except BaseException as exc:  # test thread must retain assertion evidence
        errors.append(exc)


@pytest.mark.parametrize("transport", ["stdio", "http"])
def test_shutdown_during_remote_catalog_prevents_commit_and_closes_candidate(
    tmp_path, monkeypatch, transport
):
    """A stop signal observed after startup can never cross publication commit."""
    catalog_entered = threading.Event()
    release_catalog = threading.Event()

    class Candidate:
        instances: list["Candidate"] = []

        def __init__(self, *args, **kwargs):
            self.closed = False
            type(self).instances.append(self)

        def start(self):
            pass

        def list_tools(self):
            catalog_entered.set()
            assert release_catalog.wait(timeout=2)
            return _catalog()

        def is_connected(self):
            return not self.closed

        def close(self):
            self.closed = True

    monkeypatch.setattr(
        mcp_module,
        "MCPClient" if transport == "stdio" else "HTTPMCPClient",
        Candidate,
    )
    agent = _agent(tmp_path)
    errors: list[BaseException] = []
    activation = threading.Thread(
        target=lambda: _capture_connect_error(agent, transport, errors)
    )
    activation.start()
    assert catalog_entered.wait(timeout=1)
    agent._shutdown.set()
    release_catalog.set()
    activation.join(timeout=2)
    try:
        assert not activation.is_alive()
        assert len(errors) == 1
        assert "shutting down" in str(errors[0])
        candidate = Candidate.instances[0]
        assert candidate.closed is True
        assert candidate not in agent._mcp_clients
        assert TOOL_NAME not in agent._tool_handlers
        assert TOOL_NAME not in agent._mcp_clients_by_tool
    finally:
        agent._workdir_lease.release()


def test_inflight_mcp_call_lease_depublishes_before_close_and_does_not_hold_registry_lock(
    tmp_path,
):
    """Retirement waits for a leased call without locking across remote latency."""
    call_entered = threading.Event()
    release_call = threading.Event()

    class BlockingCandidate:
        def __init__(self):
            self.closed = False
            self.close_calls = 0

        def start(self):
            pass

        def list_tools(self):
            return _catalog()

        def is_connected(self):
            return not self.closed

        def call_tool(self, name, args):
            call_entered.set()
            assert release_call.wait(5), "test did not release the MCP call"
            assert self.closed is False, "retirement closed a leased MCP call"
            return {"status": "ok", "name": name, "args": args}

        def close(self):
            self.close_calls += 1
            self.closed = True

    agent = _agent(tmp_path)
    candidate = BlockingCandidate()
    agent._activate_mcp_candidate(candidate)
    call_results: list[dict] = []
    call_errors: list[BaseException] = []
    retirement_reports: list[dict] = []
    retirement_errors: list[BaseException] = []

    def call_tool():
        try:
            call_results.append(
                agent._dispatch_tool(ToolCall(name=TOOL_NAME, args={}, id="call-lease"))
            )
        except BaseException as exc:
            call_errors.append(exc)

    def retire():
        try:
            retirement_reports.append(
                agent._retire_all_mcp_clients(context="inflight-lease-test")
            )
        except BaseException as exc:
            retirement_errors.append(exc)

    call_thread = threading.Thread(target=call_tool, name="mcp-leased-call-test")
    retirement_thread = threading.Thread(target=retire, name="mcp-retirement-test")
    try:
        call_thread.start()
        assert call_entered.wait(5), "MCP call did not acquire its lease"

        # The handler is blocked in client.call_tool(), but the registry lock is
        # already released and remains available for publication/retirement.
        assert agent._mcp_lifecycle_lock.acquire(timeout=0.5)
        agent._mcp_lifecycle_lock.release()

        retirement_thread.start()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            with agent._mcp_lifecycle_lock:
                depublished = (
                    TOOL_NAME not in agent._tool_handlers
                    and TOOL_NAME not in agent._mcp_clients_by_tool
                    and candidate in agent._mcp_retiring_clients
                )
            if depublished:
                break
            threading.Event().wait(0.005)
        assert depublished, "retirement did not depublish the leased client"
        assert candidate.close_calls == 0
        assert candidate.closed is False
        assert retirement_thread.is_alive(), "retirement did not wait for the lease"

        release_call.set()
        call_thread.join(timeout=5)
        retirement_thread.join(timeout=5)
        assert not call_thread.is_alive()
        assert not retirement_thread.is_alive()
        assert call_errors == []
        assert retirement_errors == []
        assert call_results == [{"status": "ok", "name": TOOL_NAME, "args": {}}]
        assert retirement_reports[0]["converged"] is True
        assert retirement_reports[0]["unresolved"] == []
        assert candidate.close_calls == 1
        assert candidate.closed is True
        assert agent._mcp_retiring_clients == []
        assert agent._mcp_inflight_calls == {}
    finally:
        release_call.set()
        call_thread.join(timeout=5)
        if retirement_thread.ident is not None:
            retirement_thread.join(timeout=5)
        agent._workdir_lease.release()


def test_inflight_mcp_call_drain_timeout_remains_pending_then_converges(
    tmp_path, monkeypatch
):
    """A bounded lease timeout is unresolved until a later retirement pass."""
    import lingtai.agent as agent_module

    monkeypatch.setattr(agent_module, "_MCP_CALL_DRAIN_TIMEOUT_SECONDS", 0.02)
    call_entered = threading.Event()
    release_call = threading.Event()

    class BlockingCandidate:
        def __init__(self):
            self.closed = False
            self.close_calls = 0

        def start(self):
            pass

        def list_tools(self):
            return _catalog()

        def is_connected(self):
            return not self.closed

        def call_tool(self, name, args):
            call_entered.set()
            assert release_call.wait(5), "test did not release the MCP call"
            return {"status": "ok"}

        def close(self):
            self.close_calls += 1
            self.closed = True

    agent = _agent(tmp_path)
    candidate = BlockingCandidate()
    agent._activate_mcp_candidate(candidate)
    call_errors: list[BaseException] = []

    def call_tool():
        try:
            agent._dispatch_tool(ToolCall(name=TOOL_NAME, args={}, id="call-timeout"))
        except BaseException as exc:
            call_errors.append(exc)

    call_thread = threading.Thread(target=call_tool, name="mcp-timeout-call-test")
    try:
        call_thread.start()
        assert call_entered.wait(5), "MCP call did not acquire its lease"

        first_report = agent._retire_all_mcp_clients(context="lease-timeout-first")
        assert first_report["converged"] is False
        assert len(first_report["unresolved"]) == 1
        assert first_report["unresolved"][0]["phase"] == "lease-timeout-first"
        assert "TimeoutError" in first_report["unresolved"][0]["error"]
        assert candidate.close_calls == 0
        assert candidate.closed is False
        assert agent._mcp_retiring_clients == [candidate]
        assert TOOL_NAME not in agent._tool_handlers
        assert TOOL_NAME not in agent._mcp_clients_by_tool

        release_call.set()
        call_thread.join(timeout=5)
        assert not call_thread.is_alive()
        assert call_errors == []
        assert agent._mcp_inflight_calls == {}

        second_report = agent._retire_all_mcp_clients(context="lease-timeout-second")
        assert second_report["converged"] is True
        assert second_report["unresolved"] == []
        assert candidate.close_calls == 1
        assert candidate.closed is True
        assert agent._mcp_retiring_clients == []
    finally:
        release_call.set()
        call_thread.join(timeout=5)
        agent._workdir_lease.release()
