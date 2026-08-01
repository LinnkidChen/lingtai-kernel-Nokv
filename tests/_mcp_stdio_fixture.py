"""Reusable observation and bounded teardown for real MCP stdio children."""
from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
import threading
import time
from typing import Any


@dataclass(frozen=True)
class StdioProcessRecord:
    command: str
    args: tuple[str, ...]
    pid: int
    process: Any


def _process_factory(stdio: Any) -> Any:
    try:
        installed = version("mcp")
        major = int(installed.split(".", 1)[0])
    except (PackageNotFoundError, ValueError):
        installed = "unknown"
        major = None
    if major != 1:
        raise RuntimeError(
            "real-stdio observer supports MCP 1.x; "
            f"installed mcp={installed}"
        )
    factory = getattr(stdio, "_create_platform_compatible_process", None)
    if not callable(factory):
        raise RuntimeError(
            "MCP stdio transport has no supported process-observation seam"
        )
    return factory


class StdioProcessObserver:
    """Observe the production transport without replacing its process."""

    def __init__(self) -> None:
        self._records: list[StdioProcessRecord] = []
        self._condition = threading.Condition()

    def install(self, monkeypatch: Any) -> None:
        import mcp.client.stdio as stdio

        original = _process_factory(stdio)

        async def record(command: str, args: list[str], *pos: Any, **kw: Any) -> Any:
            process = await original(command, args, *pos, **kw)
            item = StdioProcessRecord(command, tuple(args), process.pid, process)
            with self._condition:
                self._records.append(item)
                self._condition.notify_all()
            return process

        monkeypatch.setattr(stdio, "_create_platform_compatible_process", record)

    def records(self) -> tuple[StdioProcessRecord, ...]:
        with self._condition:
            return tuple(self._records)

    def wait_for_records(
        self, count: int, *, timeout: float = 10.0
    ) -> tuple[StdioProcessRecord, ...]:
        deadline = time.monotonic() + timeout
        with self._condition:
            while len(self._records) < count:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise AssertionError(
                        f"expected {count} stdio launches, got {len(self._records)}"
                    )
                self._condition.wait(remaining)
            return tuple(self._records)


def wait_for_thread_exit(
    thread: threading.Thread | None, *, timeout: float = 10.0
) -> bool:
    if thread is None:
        return True
    deadline = time.monotonic() + timeout
    while thread.is_alive():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        thread.join(min(remaining, 0.1))
    return True


def wait_for_process_exit(
    record: StdioProcessRecord, *, timeout: float = 10.0
) -> bool:
    deadline = time.monotonic() + timeout
    while True:
        returncode = getattr(record.process, "returncode", None)
        fallback = getattr(record.process, "popen", None)
        if returncode is None and fallback is not None:
            returncode = fallback.poll()
        if returncode is not None:
            return True
        if time.monotonic() >= deadline:
            return False
        threading.Event().wait(0.05)


def stop_process(record: StdioProcessRecord, *, timeout: float = 10.0) -> bool:
    process = record.process
    fallback = getattr(process, "popen", None)
    for action in ("terminate", "kill"):
        operation = getattr(process, action, None)
        if not callable(operation):
            operation = getattr(fallback, action, None)
        if not callable(operation):
            continue
        operation()
        if wait_for_process_exit(record, timeout=timeout):
            return True
    return wait_for_process_exit(record, timeout=0)
