"""Reusable real-stdio MCP process observation and bounded teardown helpers.

The MCP client owns stdio process creation inside the third-party transport.
This fixture wraps that one process factory without replacing it, so consuming
tests still launch the real command through the production client path while
gaining deterministic PID, command, thread, and teardown evidence.
"""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
import threading
import time
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StdioProcessRecord:
    """One command launched by MCP's real stdio transport."""

    command: str
    args: tuple[str, ...]
    pid: int
    process: Any


def _mcp_stdio_process_factory(stdio: Any) -> Any:
    """Return the narrow, version-gated MCP seam used only for observation."""

    try:
        installed_version = version("mcp")
        major = int(installed_version.split(".", 1)[0])
    except (PackageNotFoundError, ValueError):
        installed_version = "unknown"
        major = None
    if major != 1:
        raise RuntimeError(
            "real-stdio observer supports the tested MCP 1.x transport API; "
            f"installed mcp={installed_version}. Update the adapter before running this smoke."
        )
    original = getattr(stdio, "_create_platform_compatible_process", None)
    if not callable(original):
        raise RuntimeError(
            "installed MCP stdio transport no longer exposes "
            "_create_platform_compatible_process; update the real-stdio observer adapter"
        )
    return original


class StdioProcessObserver:
    """Observe arbitrary commands launched by ``mcp.client.stdio``.

    ``install`` delegates to the original factory. It does not fake a process,
    session, stream, or protocol response, so callers continue through the
    production stdio transport.
    """

    def __init__(self) -> None:
        self._records: list[StdioProcessRecord] = []
        self._condition = threading.Condition()

    def install(self, monkeypatch: Any) -> None:
        import mcp.client.stdio as stdio

        original = _mcp_stdio_process_factory(stdio)

        async def record_process(
            command: str,
            args: list[str],
            *factory_args: Any,
            **factory_kwargs: Any,
        ) -> Any:
            process = await original(command, args, *factory_args, **factory_kwargs)
            record = StdioProcessRecord(
                command=command,
                args=tuple(args),
                pid=process.pid,
                process=process,
            )
            with self._condition:
                self._records.append(record)
                self._condition.notify_all()
            return process

        monkeypatch.setattr(stdio, "_create_platform_compatible_process", record_process)

    def wait_for_records(
        self,
        count: int,
        *,
        timeout: float = 10.0,
    ) -> tuple[StdioProcessRecord, ...]:
        deadline = time.monotonic() + timeout
        with self._condition:
            while len(self._records) < count:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise AssertionError(
                        f"expected {count} stdio process launches, observed {len(self._records)}"
                    )
                self._condition.wait(remaining)
            return tuple(self._records)

    def records(self) -> tuple[StdioProcessRecord, ...]:
        """Return every real stdio child observed so far."""

        with self._condition:
            return tuple(self._records)


def wait_for_thread_exit(thread: threading.Thread | None, *, timeout: float = 10.0) -> bool:
    """Wait for a client-owned thread to retire, using its observable state."""

    if thread is None:
        return True
    deadline = time.monotonic() + timeout
    while thread.is_alive():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        thread.join(min(remaining, 0.1))
    return True


def wait_for_process_exit(record: StdioProcessRecord, *, timeout: float = 10.0) -> bool:
    """Wait for the transport-owned process to be reaped after client close."""

    deadline = time.monotonic() + timeout
    while True:
        # AnyIO's process handle owns the portable exit status.  Do not use
        # ``os.kill(pid, 0)`` as a liveness probe here: on Windows Python maps
        # signal 0 to TerminateProcess, which can turn a leaked child into a
        # false successful teardown.
        returncode = getattr(record.process, "returncode", None)
        fallback_popen = getattr(record.process, "popen", None)
        if returncode is None and fallback_popen is not None:
            returncode = fallback_popen.poll()
        if returncode is not None:
            return True
        if time.monotonic() >= deadline:
            return False
        threading.Event().wait(0.05)


def stop_process(record: StdioProcessRecord, *, timeout: float = 10.0) -> bool:
    """Best-effort portable terminate, then kill, for a leaked stdio child."""

    process = record.process
    fallback_popen = getattr(process, "popen", None)
    for action in ("terminate", "kill"):
        operation = getattr(process, action, None)
        if not callable(operation):
            operation = getattr(fallback_popen, action, None)
        if not callable(operation):
            continue
        operation()
        if wait_for_process_exit(record, timeout=timeout):
            return True
    return wait_for_process_exit(record, timeout=0)
