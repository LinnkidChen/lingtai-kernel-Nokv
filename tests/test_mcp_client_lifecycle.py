"""Bounded lifecycle tests for stdio and streamable-HTTP MCP clients."""
from __future__ import annotations

import socket
import subprocess
import sys
import time

import pytest

from lingtai.services.mcp import HTTPMCPClient, MCPClient
from tests._mcp_stdio_fixture import (
    StdioProcessObserver,
    stop_process,
    wait_for_process_exit,
    wait_for_thread_exit,
)


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _wait_for_port(port: int, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                return
        except OSError:
            time.sleep(0.02)
    raise AssertionError(f"local HTTP MCP did not listen on {port}")


def test_real_stdio_startup_timeout_retires_process_and_thread(
    monkeypatch,
):
    observer = StdioProcessObserver()
    observer.install(monkeypatch)
    client = MCPClient(
        command=sys.executable,
        args=["-m", "tests._mcp_startup_timeout_stdio_server"],
        startup_timeout=0.1,
        close_timeout=1.0,
    )
    try:
        with pytest.raises(RuntimeError, match="startup timed out"):
            client.start()
        child = observer.wait_for_records(1)[0]
        assert wait_for_thread_exit(client._thread, timeout=2.0)
        assert wait_for_process_exit(child, timeout=2.0)
        client.close()
    finally:
        for child in observer.records():
            if not wait_for_process_exit(child, timeout=0):
                stop_process(child)


def test_close_rejects_still_alive_stdio_and_http_threads():
    class _Alive:
        def is_alive(self):
            return True

        def join(self, timeout):
            return None

    for client in (
        MCPClient("/bin/true", close_timeout=0.01),
        HTTPMCPClient("http://127.0.0.1:1/mcp", close_timeout=0.01),
    ):
        client._thread = _Alive()
        with pytest.raises(RuntimeError, match="still alive"):
            client.close()


def test_local_http_lifecycle_and_list_failure_after_server_exit():
    port = _free_port()
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
    client = HTTPMCPClient(
        f"http://127.0.0.1:{port}/mcp",
        startup_timeout=2.0,
        close_timeout=1.0,
    )
    try:
        _wait_for_port(port)
        client.start()
        assert [tool["name"] for tool in client.list_tools(timeout=2.0)] == ["ping"]
        process.terminate()
        process.wait(timeout=3.0)
        with pytest.raises(Exception):
            client.list_tools(timeout=1.0)
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=3.0)
        client.close()
    assert wait_for_thread_exit(client._thread, timeout=1.0)


def test_local_http_startup_failure_is_bounded_and_retired():
    port = _free_port()
    client = HTTPMCPClient(
        f"http://127.0.0.1:{port}/mcp",
        startup_timeout=0.5,
        close_timeout=1.0,
    )
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="failed to connect|startup timed out"):
        client.start()
    assert time.monotonic() - started < 2.0
    assert wait_for_thread_exit(client._thread, timeout=1.0)
    client.close()
