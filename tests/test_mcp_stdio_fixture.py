"""Unit coverage for portable real-stdio process observation helpers."""
from __future__ import annotations

import pytest

from tests._mcp_stdio_fixture import (
    StdioProcessRecord,
    _mcp_stdio_process_factory,
    stop_process,
    wait_for_process_exit,
)


class _ExitedProcess:
    returncode = 0


class _FallbackPopen:
    def poll(self) -> int:
        return 0


class _WindowsFallbackProcess:
    returncode = None
    popen = _FallbackPopen()


class _ResistantProcess:
    def __init__(self) -> None:
        self.returncode: int | None = None
        self.actions: list[str] = []

    def terminate(self) -> None:
        self.actions.append("terminate")

    def kill(self) -> None:
        self.actions.append("kill")
        self.returncode = -9


def _record(process: object) -> StdioProcessRecord:
    return StdioProcessRecord(
        command="fixture-command",
        args=(),
        pid=12345,
        process=process,
    )


def test_wait_for_process_exit_reads_transport_returncode() -> None:
    assert wait_for_process_exit(_record(_ExitedProcess()), timeout=0)


def test_wait_for_process_exit_supports_windows_fallback_process() -> None:
    assert wait_for_process_exit(_record(_WindowsFallbackProcess()), timeout=0)


def test_stop_process_escalates_from_terminate_to_kill() -> None:
    process = _ResistantProcess()
    assert stop_process(_record(process), timeout=0)
    assert process.actions == ["terminate", "kill"]


def test_observer_reports_missing_private_mcp_factory() -> None:
    with pytest.raises(RuntimeError, match="observer adapter"):
        _mcp_stdio_process_factory(object())
