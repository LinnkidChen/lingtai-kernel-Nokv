"""Real transport evidence for fail-closed MCP candidate activation."""
from __future__ import annotations

import json
import socket
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import pytest

from lingtai.agent import Agent
from lingtai.services import mcp as mcp_module
from tests._mcp_activation_stdio_server import TOOL_NAME
from tests._mcp_stdio_fixture import (
    StdioProcessObserver,
    assert_stdio_client_retired,
    stop_process,
)
from tests._service_helpers import make_gemini_mock_service


def _agent(tmp_path: Path) -> Agent:
    return Agent(
        service=make_gemini_mock_service(),
        agent_name="activation-test",
        working_dir=tmp_path / "agent",
        capabilities={"mcp": {}},
    )


def _materialize_retry_spec(
    agent: Agent,
    *,
    name: str,
    cfg: dict,
    client,
) -> dict:
    """Associate one live client with the current registry-gated init source."""
    spec = {
        "cfg": cfg,
        "source": "init.json:mcp",
        "client": client,
    }
    agent._mcp_init_specs = {name: spec}
    (agent._working_dir / "init.json").write_text(
        json.dumps({"mcp": {name: cfg}}),
        encoding="utf-8",
    )
    record = {
        "name": name,
        "summary": "real predecessor lifecycle test MCP",
        "transport": cfg.get("type", "stdio"),
        "source": "test",
    }
    if record["transport"] == "http":
        record["url"] = cfg["url"]
    else:
        record["command"] = cfg["command"]
        record["args"] = cfg.get("args", [])
    (agent._working_dir / "mcp_registry.jsonl").write_text(
        json.dumps(record) + "\n",
        encoding="utf-8",
    )
    return spec


def _assert_http_client_retired(
    client,
    *,
    timeout: float = 3.0,
    expected_cleanup_errors: tuple[tuple[str, ...], ...] = ((),),
) -> None:
    """Prove the real HTTP session, transport, and lifecycle are terminal."""
    assert client.cleanup_complete.wait(timeout)
    assert client._thread is not None and not client._thread.is_alive()
    assert client._session is None
    assert client._cleanup_postcondition_verified is True
    actual_cleanup_errors = tuple(client._cleanup_errors)
    assert actual_cleanup_errors in expected_cleanup_errors
    if actual_cleanup_errors:
        assert client._cleanup_errors_reported is True
    assert client._http_client is not None
    assert client._http_client.is_closed is True


@pytest.mark.parametrize("failure_mode", ["hang_start", "list_fail"])
def test_real_stdio_failed_candidate_retires_before_replacement(
    tmp_path, monkeypatch, failure_mode
):
    observer = StdioProcessObserver()
    observer.install(monkeypatch)
    original = mcp_module.MCPClient

    class ObservedMCPClient(original):
        instances: list["ObservedMCPClient"] = []
        _START_TIMEOUT_SECONDS = 5.0
        _CLOSE_TIMEOUT_SECONDS = 3.0

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if self._args[-1:] == ["hang_start"]:
                self._START_TIMEOUT_SECONDS = 0.25
            type(self).instances.append(self)

    monkeypatch.setattr(mcp_module, "MCPClient", ObservedMCPClient)
    agent = _agent(tmp_path)
    module = "tests._mcp_activation_stdio_server"
    try:
        with pytest.raises(Exception):
            agent.connect_mcp(
                command=sys.executable,
                args=["-m", module, failure_mode],
            )
        failed = ObservedMCPClient.instances[0]
        first_child = observer.wait_for_records(1, timeout=3)[0]
        assert_stdio_client_retired(failed, first_child, timeout=3)
        assert agent._mcp_clients == []
        assert TOOL_NAME not in agent._tool_handlers
        assert TOOL_NAME not in agent._mcp_clients_by_tool

        # Retirement is proven before a distinct replacement process starts.
        ObservedMCPClient._START_TIMEOUT_SECONDS = 5.0
        assert agent.connect_mcp(
            command=sys.executable,
            args=["-m", module, "valid"],
        ) == [TOOL_NAME]
        children = observer.wait_for_records(2, timeout=3)
        replacement = ObservedMCPClient.instances[1]
        assert children[0].pid != children[1].pid
        assert agent._mcp_clients == [replacement]
        assert agent._mcp_clients_by_tool == {TOOL_NAME: replacement}
        assert [s.name for s in agent._tool_schemas].count(TOOL_NAME) == 1
    finally:
        for client in ObservedMCPClient.instances:
            try:
                client.close()
            except Exception:
                pass
        for index, record in enumerate(observer.records()):
            try:
                assert_stdio_client_retired(
                    ObservedMCPClient.instances[index], record, timeout=3
                )
            except Exception:
                stop_process(record, timeout=3)
        agent._workdir_lease.release()


@contextmanager
def _local_streamable_http_server(*, fail_list_once: bool = False):
    from mcp import types as mcp_types
    from mcp.server.fastmcp import FastMCP
    import uvicorn

    server_mcp = FastMCP(
        "lingtai-http-activation-fixture",
        stateless_http=True,
        json_response=True,
    )

    @server_mcp.tool(name=TOOL_NAME, description="HTTP activation probe")
    def _probe() -> dict:
        return {"status": "ok"}

    if fail_list_once:
        list_handler = server_mcp._mcp_server.request_handlers[
            mcp_types.ListToolsRequest
        ]
        failure_state = {"pending": True}

        async def _fail_first_protocol_list(request):
            if failure_state["pending"]:
                failure_state["pending"] = False
                raise RuntimeError("fixture HTTP tools/list failure")
            return await list_handler(request)

        server_mcp._mcp_server.request_handlers[
            mcp_types.ListToolsRequest
        ] = _fail_first_protocol_list

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    sock.listen(128)
    host, port = sock.getsockname()
    server = uvicorn.Server(
        uvicorn.Config(
            server_mcp.streamable_http_app(),
            log_level="error",
            lifespan="on",
        )
    )
    thread = threading.Thread(
        target=lambda: server.run(sockets=[sock]),
        daemon=True,
        name="mcp-http-activation-server",
    )
    thread.start()
    deadline = time.monotonic() + 5
    while not server.started and thread.is_alive() and time.monotonic() < deadline:
        threading.Event().wait(0.01)
    if not server.started:
        server.should_exit = True
        thread.join(timeout=3)
        sock.close()
        raise RuntimeError("local MCP HTTP server did not start")
    try:
        yield f"http://{host}:{port}/mcp"
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        sock.close()
        assert not thread.is_alive(), "local MCP HTTP server thread did not retire"


def test_real_http_failed_candidate_retires_session_and_thread_before_replacement(
    tmp_path, monkeypatch
):
    original = mcp_module.HTTPMCPClient

    class ObservedHTTPMCPClient(original):
        instances: list["ObservedHTTPMCPClient"] = []
        _START_TIMEOUT_SECONDS = 5.0

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.cleanup_complete = threading.Event()
            type(self).instances.append(self)

        async def _async_cleanup(self):
            try:
                await super()._async_cleanup()
            finally:
                self.cleanup_complete.set()

    monkeypatch.setattr(mcp_module, "HTTPMCPClient", ObservedHTTPMCPClient)
    agent = _agent(tmp_path)
    try:
        with _local_streamable_http_server(fail_list_once=True) as url:
            with pytest.raises(Exception, match="tools/list failure|Internal error"):
                agent.connect_mcp_http(url=url)
            failed = ObservedHTTPMCPClient.instances[0]
            _assert_http_client_retired(failed)
            assert failed not in agent._mcp_clients
            assert TOOL_NAME not in agent._tool_handlers

            ObservedHTTPMCPClient._START_TIMEOUT_SECONDS = 5.0
            assert agent.connect_mcp_http(url=url) == [TOOL_NAME]
            replacement = ObservedHTTPMCPClient.instances[1]
            assert agent._mcp_clients == [replacement]
            assert replacement is not failed
            replacement.close()
            _assert_http_client_retired(replacement)
    finally:
        for client in ObservedHTTPMCPClient.instances:
            try:
                client.close()
            except Exception:
                pass
        for client in ObservedHTTPMCPClient.instances:
            _assert_http_client_retired(client)
        agent._workdir_lease.release()


def test_real_http_startup_failure_retires_thread_before_replacement(
    tmp_path, monkeypatch
):
    original = mcp_module.HTTPMCPClient

    class ObservedHTTPMCPClient(original):
        instances: list["ObservedHTTPMCPClient"] = []
        _START_TIMEOUT_SECONDS = 0.25

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.cleanup_complete = threading.Event()
            type(self).instances.append(self)

        async def _async_cleanup(self):
            try:
                await super()._async_cleanup()
            finally:
                self.cleanup_complete.set()

    monkeypatch.setattr(mcp_module, "HTTPMCPClient", ObservedHTTPMCPClient)
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    host, port = probe.getsockname()
    probe.close()

    agent = _agent(tmp_path)
    try:
        with pytest.raises(RuntimeError, match="failed to connect"):
            agent.connect_mcp_http(url=f"http://{host}:{port}/mcp")
        failed = ObservedHTTPMCPClient.instances[0]
        _assert_http_client_retired(
            failed,
            expected_cleanup_errors=((), ("session: CancelledError",)),
        )
        assert failed not in agent._mcp_clients

        with _local_streamable_http_server() as url:
            ObservedHTTPMCPClient._START_TIMEOUT_SECONDS = 5.0
            assert agent.connect_mcp_http(url=url) == [TOOL_NAME]
            replacement = ObservedHTTPMCPClient.instances[1]
            assert replacement is not failed
            assert agent._mcp_clients == [replacement]
            replacement.close()
            _assert_http_client_retired(replacement)
    finally:
        for client in ObservedHTTPMCPClient.instances:
            try:
                client.close()
            except Exception:
                pass
        for index, client in enumerate(ObservedHTTPMCPClient.instances):
            expected_errors = (
                ((), ("session: CancelledError",)) if index == 0 else ((),)
            )
            _assert_http_client_retired(
                client,
                expected_cleanup_errors=expected_errors,
            )
        agent._workdir_lease.release()


def test_real_stdio_unhealthy_predecessor_retry_retires_child_before_replacement_start(
    tmp_path, monkeypatch
):
    observer = StdioProcessObserver()
    observer.install(monkeypatch)
    original = mcp_module.MCPClient
    predecessor_retired_before_start = threading.Event()

    class ObservedMCPClient(original):
        instances: list["ObservedMCPClient"] = []
        _START_TIMEOUT_SECONDS = 5.0
        _CLOSE_TIMEOUT_SECONDS = 3.0

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            type(self).instances.append(self)

        def start(self):
            if len(type(self).instances) == 2 and self is type(self).instances[1]:
                # No replacement process exists yet. The exact predecessor's
                # client thread and real child must already be terminal before
                # production start() is allowed to create the second process.
                records = observer.records()
                assert len(records) == 1
                assert_stdio_client_retired(
                    type(self).instances[0], records[0], timeout=0
                )
                predecessor_retired_before_start.set()
            return super().start()

    monkeypatch.setattr(mcp_module, "MCPClient", ObservedMCPClient)
    agent = _agent(tmp_path)
    module = "tests._mcp_activation_stdio_server"
    cfg = {
        "type": "stdio",
        "command": sys.executable,
        "args": ["-m", module, "valid"],
    }
    spec_name = "real-stdio-predecessor"
    try:
        assert agent.connect_mcp(
            command=cfg["command"],
            args=cfg["args"],
        ) == [TOOL_NAME]
        predecessor = ObservedMCPClient.instances[0]
        first_child = observer.wait_for_records(1, timeout=3)[0]
        assert predecessor._session is not None
        assert predecessor._thread is not None and predecessor._thread.is_alive()
        assert first_child.process.returncode is None

        spec = _materialize_retry_spec(
            agent,
            name=spec_name,
            cfg=cfg,
            client=predecessor,
        )
        agent._sealed = True
        monkeypatch.setattr(predecessor, "is_connected", lambda: False)

        # Only the exact public health probe is false; its real process/session
        # remains live until the retry transaction retires it.
        assert predecessor.is_connected() is False
        assert predecessor._session is not None
        assert predecessor._thread.is_alive()
        assert first_child.process.returncode is None

        report = agent._retry_failed_mcps()

        assert report["retried"] == [spec_name]
        assert report["recovered"] == [spec_name]
        assert report["still_failed"] == []
        assert report["unresolved"] == []
        assert report["converged"] is True
        assert predecessor_retired_before_start.is_set()
        assert_stdio_client_retired(predecessor, first_child, timeout=0)

        children = observer.wait_for_records(2, timeout=3)
        assert children[0].pid != children[1].pid
        assert len(ObservedMCPClient.instances) == 2
        replacement = ObservedMCPClient.instances[1]
        assert agent._mcp_clients == [replacement]
        assert agent._mcp_clients_by_tool == {TOOL_NAME: replacement}
        assert agent._mcp_tool_names == {TOOL_NAME}
        assert spec["client"] is replacement
        assert agent._tool_handlers[TOOL_NAME]._lingtai_mcp_client is replacement
        assert [s.name for s in agent._tool_schemas].count(TOOL_NAME) == 1
        assert replacement.is_connected() is True
    finally:
        for client in ObservedMCPClient.instances:
            try:
                client.close()
            except Exception:
                pass
        for index, record in enumerate(observer.records()):
            try:
                assert_stdio_client_retired(
                    ObservedMCPClient.instances[index], record, timeout=3
                )
            except Exception:
                stop_process(record, timeout=3)
        agent._workdir_lease.release()


def test_real_stdio_predecessor_close_failure_blocks_then_next_retry_converges(
    tmp_path, monkeypatch
):
    observer = StdioProcessObserver()
    observer.install(monkeypatch)
    original = mcp_module.MCPClient
    predecessor_retired_before_start = threading.Event()
    replacement_start_calls: list[int] = []

    class ObservedMCPClient(original):
        instances: list["ObservedMCPClient"] = []
        _START_TIMEOUT_SECONDS = 5.0
        _CLOSE_TIMEOUT_SECONDS = 3.0

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            type(self).instances.append(self)

        def start(self):
            if self is not type(self).instances[0]:
                replacement_start_calls.append(id(self))
                records = observer.records()
                assert len(records) == 1
                assert_stdio_client_retired(
                    type(self).instances[0], records[0], timeout=0
                )
                predecessor_retired_before_start.set()
            return super().start()

    monkeypatch.setattr(mcp_module, "MCPClient", ObservedMCPClient)
    agent = _agent(tmp_path)
    module = "tests._mcp_activation_stdio_server"
    cfg = {
        "type": "stdio",
        "command": sys.executable,
        "args": ["-m", module, "valid"],
    }
    spec_name = "real-stdio-close-retry"
    try:
        assert agent.connect_mcp(
            command=cfg["command"],
            args=cfg["args"],
        ) == [TOOL_NAME]
        predecessor = ObservedMCPClient.instances[0]
        predecessor_child = observer.wait_for_records(1, timeout=3)[0]
        assert predecessor._thread is not None and predecessor._thread.is_alive()
        assert predecessor_child.process.returncode is None
        spec = _materialize_retry_spec(
            agent,
            name=spec_name,
            cfg=cfg,
            client=predecessor,
        )
        agent._sealed = True
        monkeypatch.setattr(predecessor, "is_connected", lambda: False)
        real_close = predecessor.close
        close_attempts: list[int] = []

        def _fail_first_close():
            close_attempts.append(len(close_attempts) + 1)
            if len(close_attempts) == 1:
                raise RuntimeError("injected predecessor close/join failure")
            return real_close()

        monkeypatch.setattr(predecessor, "close", _fail_first_close)

        first_report = agent._retry_failed_mcps()

        assert first_report["retried"] == [spec_name]
        assert first_report["recovered"] == []
        assert first_report["still_failed"] == [spec_name]
        assert first_report["converged"] is False
        assert len(first_report["unresolved"]) == 1
        assert first_report["unresolved"][0]["client_id"] == str(id(predecessor))
        assert first_report["unresolved"][0]["phase"] == "retry_cleanup"
        assert close_attempts == [1]
        assert replacement_start_calls == []
        assert len(observer.records()) == 1
        assert predecessor._thread.is_alive()
        assert predecessor_child.process.returncode is None
        assert agent._mcp_retiring_clients == [predecessor]
        assert agent._mcp_clients == []
        assert spec["client"] is None
        assert TOOL_NAME not in agent._tool_handlers
        assert TOOL_NAME not in agent._mcp_clients_by_tool
        assert TOOL_NAME not in agent._mcp_tool_names
        assert [s.name for s in agent._tool_schemas].count(TOOL_NAME) == 0

        # The first never-started candidate was closed by activation rollback.
        assert len(ObservedMCPClient.instances) == 2
        first_candidate = ObservedMCPClient.instances[1]
        assert first_candidate._closed is True
        assert first_candidate._thread is None
        assert first_candidate not in agent._mcp_retiring_clients

        second_report = agent._retry_failed_mcps()

        assert second_report["retried"] == [spec_name]
        assert second_report["recovered"] == [spec_name]
        assert second_report["still_failed"] == []
        assert second_report["unresolved"] == []
        assert second_report["converged"] is True
        assert close_attempts == [1, 2]
        assert predecessor_retired_before_start.is_set()
        assert len(replacement_start_calls) == 1
        assert_stdio_client_retired(predecessor, predecessor_child, timeout=0)

        children = observer.wait_for_records(2, timeout=3)
        assert children[0].pid != children[1].pid
        assert len(ObservedMCPClient.instances) == 3
        replacement = ObservedMCPClient.instances[2]
        assert replacement_start_calls == [id(replacement)]
        assert agent._mcp_retiring_clients == []
        assert agent._mcp_clients == [replacement]
        assert agent._mcp_clients_by_tool == {TOOL_NAME: replacement}
        assert agent._mcp_tool_names == {TOOL_NAME}
        assert spec["client"] is replacement
        assert agent._tool_handlers[TOOL_NAME]._lingtai_mcp_client is replacement
        assert [s.name for s in agent._tool_schemas].count(TOOL_NAME) == 1
        assert replacement.is_connected() is True
    finally:
        for client in ObservedMCPClient.instances:
            try:
                client.close()
            except Exception:
                pass
        started_clients = [
            client
            for client in ObservedMCPClient.instances
            if client._thread is not None
        ]
        for client, record in zip(started_clients, observer.records()):
            try:
                assert_stdio_client_retired(client, record, timeout=3)
            except Exception:
                stop_process(record, timeout=3)
        agent._workdir_lease.release()


def test_real_http_unhealthy_predecessor_retry_retires_session_before_replacement_start(
    tmp_path, monkeypatch
):
    original = mcp_module.HTTPMCPClient
    predecessor_retired_before_start = threading.Event()

    class ObservedHTTPMCPClient(original):
        instances: list["ObservedHTTPMCPClient"] = []
        _START_TIMEOUT_SECONDS = 5.0
        _CLOSE_TIMEOUT_SECONDS = 3.0

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.cleanup_complete = threading.Event()
            type(self).instances.append(self)

        def start(self):
            if len(type(self).instances) == 2 and self is type(self).instances[1]:
                predecessor = type(self).instances[0]
                # This override runs before production start() creates the new
                # lifecycle thread, so every assertion is a no-overlap proof.
                _assert_http_client_retired(predecessor, timeout=0)
                predecessor_retired_before_start.set()
            return super().start()

        async def _async_cleanup(self):
            try:
                await super()._async_cleanup()
            finally:
                self.cleanup_complete.set()

    monkeypatch.setattr(mcp_module, "HTTPMCPClient", ObservedHTTPMCPClient)
    agent = _agent(tmp_path)
    spec_name = "real-http-predecessor"
    try:
        with _local_streamable_http_server() as url:
            cfg = {"type": "http", "url": url}
            assert agent.connect_mcp_http(url=url) == [TOOL_NAME]
            predecessor = ObservedHTTPMCPClient.instances[0]
            predecessor_session = predecessor._session
            predecessor_thread = predecessor._thread
            assert predecessor_session is not None
            assert predecessor_thread is not None and predecessor_thread.is_alive()
            assert predecessor._http_client is not None
            assert predecessor._http_client.is_closed is False

            spec = _materialize_retry_spec(
                agent,
                name=spec_name,
                cfg=cfg,
                client=predecessor,
            )
            agent._sealed = True
            monkeypatch.setattr(predecessor, "is_connected", lambda: False)

            # Force only Agent's exact predecessor health probe to fail while
            # the production HTTP session and lifecycle thread remain live.
            assert predecessor.is_connected() is False
            assert predecessor._session is predecessor_session
            assert predecessor_thread.is_alive()
            assert predecessor._http_client.is_closed is False

            report = agent._retry_failed_mcps()

            assert report["retried"] == [spec_name]
            assert report["recovered"] == [spec_name]
            assert report["still_failed"] == []
            assert report["unresolved"] == []
            assert report["converged"] is True
            assert predecessor_retired_before_start.is_set()
            _assert_http_client_retired(predecessor, timeout=0)
            assert not predecessor_thread.is_alive()

            assert len(ObservedHTTPMCPClient.instances) == 2
            replacement = ObservedHTTPMCPClient.instances[1]
            assert agent._mcp_clients == [replacement]
            assert agent._mcp_clients_by_tool == {TOOL_NAME: replacement}
            assert agent._mcp_tool_names == {TOOL_NAME}
            assert spec["client"] is replacement
            assert agent._tool_handlers[TOOL_NAME]._lingtai_mcp_client is replacement
            assert [s.name for s in agent._tool_schemas].count(TOOL_NAME) == 1
            assert replacement._session is not None
            assert replacement._thread is not predecessor_thread
            assert replacement._thread.is_alive()
            assert replacement.is_connected() is True
            replacement.close()
            _assert_http_client_retired(replacement)
    finally:
        for client in ObservedHTTPMCPClient.instances:
            try:
                client.close()
            except Exception:
                pass
        for client in ObservedHTTPMCPClient.instances:
            _assert_http_client_retired(client)
        agent._workdir_lease.release()
