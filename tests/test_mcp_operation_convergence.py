"""Cancellation and retry convergence for MCP async-to-sync bridges."""
from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable
from typing import Any

import pytest

from lingtai.services.mcp import HTTPMCPClient, MCPClient


_PUBLIC_CALL_TIMEOUT_SECONDS = 0.5
_CALLER_JOIN_TIMEOUT_SECONDS = 2.0


def _start_sync_call(call: Callable[[], Any]):
    outcome: dict[str, Any] = {}

    def _run() -> None:
        try:
            outcome["result"] = call()
        except BaseException as exc:
            outcome["error"] = exc

    caller = threading.Thread(target=_run, daemon=True)
    caller.start()
    return caller, outcome


def _join_sync_call(
    caller: threading.Thread,
    outcome: dict[str, Any],
) -> dict[str, Any]:
    caller.join(timeout=_CALLER_JOIN_TIMEOUT_SECONDS)
    assert not caller.is_alive(), "synchronous MCP caller did not return"
    assert ("result" in outcome) != ("error" in outcome)
    return outcome


def _assert_sync_error(
    outcome: dict[str, Any],
    error_type: type[BaseException],
    message: str,
) -> None:
    error = outcome.get("error")
    assert isinstance(error, error_type)
    assert message in str(error)


class _BlockingSession:
    def __init__(self) -> None:
        self.call_started = threading.Event()
        self.call_cancelled = threading.Event()
        self.list_started = threading.Event()
        self.list_cancelled = threading.Event()

    async def call_tool(self, **_kwargs: Any) -> Any:
        self.call_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            self.call_cancelled.set()

    async def list_tools(self) -> Any:
        self.list_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            self.list_cancelled.set()


def _connected_client(kind: str, session=None):
    client = (
        MCPClient(command="fake")
        if kind == "stdio"
        else HTTPMCPClient(url="http://fake")
    )
    session = session or _BlockingSession()
    loop = asyncio.new_event_loop()
    ready = threading.Event()

    def _run() -> None:
        asyncio.set_event_loop(loop)
        ready.set()
        loop.run_forever()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    assert ready.wait(timeout=1)
    client._session = session
    client._loop = loop
    client._closed = False
    return client, session, loop, thread


def _stop_loop(loop: asyncio.AbstractEventLoop, thread: threading.Thread) -> None:
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=1)
    assert not thread.is_alive()
    loop.close()


@pytest.mark.parametrize("kind", ["stdio", "http"])
def test_call_timeout_cancels_and_drains_before_sync_return(kind: str) -> None:
    client, session, loop, thread = _connected_client(kind)
    try:
        caller, outcome = _start_sync_call(
            lambda: client.call_tool(
                "blocked", {}, timeout=_PUBLIC_CALL_TIMEOUT_SECONDS
            )
        )
        assert session.call_started.wait(timeout=1)
        _join_sync_call(caller, outcome)

        if kind == "stdio":
            result = outcome["result"]
            assert result["status"] == "error"
            assert "timed out" in result["message"]
        else:
            _assert_sync_error(outcome, TimeoutError, "cancelled and drained")

        assert session.call_cancelled.is_set()
        assert client._operation_tokens == set()
        assert client._operation_tasks == {}
    finally:
        _stop_loop(loop, thread)


@pytest.mark.parametrize("kind", ["stdio", "http"])
def test_catalog_timeout_cancels_and_drains_before_candidate_cleanup(
    kind: str,
) -> None:
    client, session, loop, thread = _connected_client(kind)
    try:
        caller, outcome = _start_sync_call(
            lambda: client.list_tools(timeout=_PUBLIC_CALL_TIMEOUT_SECONDS)
        )
        assert session.list_started.wait(timeout=1)
        _join_sync_call(caller, outcome)
        _assert_sync_error(outcome, TimeoutError, "cancelled and drained")

        assert session.list_cancelled.is_set()
        assert client._operation_tokens == set()
        assert client._operation_tasks == {}
    finally:
        _stop_loop(loop, thread)


class _StubbornSession:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.cancel_ignored = threading.Event()
        self.finalized = threading.Event()
        self.release: asyncio.Event | None = None

    async def call_tool(self, **_kwargs: Any) -> Any:
        self.release = asyncio.Event()
        self.started.set()
        try:
            while not self.release.is_set():
                try:
                    await self.release.wait()
                except asyncio.CancelledError:
                    self.cancel_ignored.set()
                    continue
            return {"status": "success"}
        finally:
            self.finalized.set()


class _LifecycleCancellationProbe:
    def __init__(self) -> None:
        self.cancelled = threading.Event()

    def done(self) -> bool:
        return False

    def cancel(self) -> None:
        self.cancelled.set()


@pytest.mark.parametrize("kind", ["stdio", "http"])
def test_unconverged_timeout_quarantines_client_until_exact_task_finishes(
    kind: str,
) -> None:
    session = _StubbornSession()
    client, _, loop, thread = _connected_client(kind, session=session)
    client._OPERATION_CANCEL_TIMEOUT_SECONDS = 0.05
    lifecycle = _LifecycleCancellationProbe()
    client._lifecycle_task = lifecycle
    try:
        caller, outcome = _start_sync_call(
            lambda: client.call_tool(
                "stubborn", {}, timeout=_PUBLIC_CALL_TIMEOUT_SECONDS
            )
        )
        assert session.started.wait(timeout=1)
        _join_sync_call(caller, outcome)

        if kind == "stdio":
            result = outcome["result"]
            assert result["status"] == "error"
            assert "did not converge" in result["message"]
        else:
            _assert_sync_error(outcome, RuntimeError, "did not converge")

        assert session.cancel_ignored.is_set()
        assert client._operation_quarantine_error is not None
        assert client.is_connected() is False
        with pytest.raises(RuntimeError, match="quarantined"):
            client.call_tool("must-not-overlap", {}, timeout=0.01)

        pending_barriers = set(client._operation_cancellation_barriers)
        pending_tokens = set(client._operation_tokens)
        pending_tasks = dict(client._operation_tasks)
        assert len(pending_barriers) == 1
        assert len(pending_tokens) == 1
        assert len(pending_tasks) == 1

        close_started = time.monotonic()
        with pytest.raises(RuntimeError, match="prior async cancellation barrier"):
            client.close()
        assert time.monotonic() - close_started < 1
        assert client._operation_cancellation_barriers == pending_barriers
        assert client._operation_tokens == pending_tokens
        assert client._operation_tasks == pending_tasks
        assert client._operation_quarantine_error is not None
        assert not lifecycle.cancelled.is_set()

        assert session.release is not None
        loop.call_soon_threadsafe(session.release.set)
        assert session.finalized.wait(timeout=1)

        # The later exact cleanup attempt waits the retained barrier, observes no
        # live task, clears quarantine, and only then cancels lifecycle work.
        client.close()
        assert lifecycle.cancelled.wait(timeout=1)
        assert client._operation_quarantine_error is None
        assert client._operation_cancellation_barriers == set()
        assert client._operation_tokens == set()
        assert client._operation_tasks == {}
    finally:
        _stop_loop(loop, thread)


class _DeadThread:
    def join(self, timeout=None) -> None:
        pass

    def is_alive(self) -> bool:
        return False


class _ClosedStream:
    _closed = True


class _FailingExit:
    async def __aexit__(self, *_args) -> None:
        raise RuntimeError("terminal cleanup probe")


class _MixedFailingExit:
    async def __aexit__(self, *_args) -> None:
        raise BaseExceptionGroup(
            "mixed cleanup failure",
            [
                asyncio.CancelledError(),
                RuntimeError("terminal cleanup probe"),
            ],
        )


class _FatalGroupExit:
    def __init__(self, fatal_type: type[BaseException]) -> None:
        self._fatal_type = fatal_type

    async def __aexit__(self, *_args) -> None:
        raise BaseExceptionGroup(
            "fatal cleanup failure",
            [
                self._fatal_type("must propagate"),
                RuntimeError("ordinary cleanup evidence"),
            ],
        )


def _cleanup_client(kind: str):
    return (
        MCPClient(command="fake")
        if kind == "stdio"
        else HTTPMCPClient(url="http://fake")
    )


def _set_cleanup_resource_probe(client, kind: str, *, verified: bool) -> None:
    client._read_stream = _ClosedStream()
    client._write_stream = _ClosedStream()
    if kind == "stdio":
        client._stdio_transport_entered = True
        client._stdio_process = type(
            "_Process", (), {"returncode": 0 if verified else None}
        )()
    else:
        client._http_transport_entered = True
        client._http_client = type(
            "_HTTPClient", (), {"is_closed": verified}
        )()


async def _materialize_cleanup_error(kind: str, *, verified: bool):
    client = _cleanup_client(kind)
    client._session_cm = _FailingExit()
    _set_cleanup_resource_probe(client, kind, verified=verified)
    if kind == "stdio":
        client._stdio_cm = _FailingExit()
    else:
        client._transport_cm = _FailingExit()
    await client._async_cleanup()
    client._thread = _DeadThread()
    return client


async def _materialize_mixed_cleanup_error(kind: str, *, verified: bool):
    client = _cleanup_client(kind)
    client._session_cm = _MixedFailingExit()
    _set_cleanup_resource_probe(client, kind, verified=verified)
    await client._async_cleanup()
    return client


def _exception_leaves(exc: BaseException) -> list[BaseException]:
    if isinstance(exc, BaseExceptionGroup):
        return [
            leaf
            for child in exc.exceptions
            for leaf in _exception_leaves(child)
        ]
    return [exc]


@pytest.mark.parametrize("kind", ["stdio", "http"])
def test_verified_cleanup_error_is_reported_once_then_close_converges(
    kind: str,
) -> None:
    client = asyncio.run(_materialize_cleanup_error(kind, verified=True))

    assert client._cleanup_postcondition_verified is True
    assert client._cleanup_errors

    with pytest.raises(RuntimeError, match="postcondition verified"):
        client.close()
    client.close()

    # Evidence remains available even though the resource postcondition allows
    # a later Agent retirement attempt to converge.
    assert client._cleanup_errors


@pytest.mark.parametrize("kind", ["stdio", "http"])
def test_unverified_cleanup_error_remains_permanently_unresolved(kind: str) -> None:
    client = asyncio.run(_materialize_cleanup_error(kind, verified=False))

    assert client._cleanup_postcondition_verified is False
    for _ in range(2):
        with pytest.raises(RuntimeError, match="resource retirement is unverified"):
            client.close()

    assert client._cleanup_errors


@pytest.mark.parametrize("kind", ["stdio", "http"])
@pytest.mark.parametrize("verified", [True, False])
def test_mixed_cleanup_exception_group_records_all_leaves_and_resource_probe(
    kind: str,
    verified: bool,
) -> None:
    client = asyncio.run(
        _materialize_mixed_cleanup_error(kind, verified=verified)
    )

    evidence = "; ".join(client._cleanup_errors)
    assert "CancelledError" in evidence
    assert "RuntimeError: terminal cleanup probe" in evidence
    assert client._cleanup_postcondition_verified is verified


@pytest.mark.parametrize("kind", ["stdio", "http"])
@pytest.mark.parametrize("fatal_type", [SystemExit, KeyboardInterrupt])
def test_cleanup_exception_group_never_swallows_process_control_exceptions(
    kind: str,
    fatal_type: type[BaseException],
) -> None:
    client = _cleanup_client(kind)
    client._session_cm = _FatalGroupExit(fatal_type)

    with pytest.raises(BaseException) as raised:
        asyncio.run(client._async_cleanup())

    leaves = _exception_leaves(raised.value)
    assert any(isinstance(leaf, fatal_type) for leaf in leaves)


@pytest.mark.parametrize("kind", ["stdio", "http"])
def test_started_lifecycle_without_cleanup_proof_fails_closed_even_without_error(
    kind: str,
) -> None:
    client = (
        MCPClient(command="fake")
        if kind == "stdio"
        else HTTPMCPClient(url="http://fake")
    )
    client._thread = _DeadThread()
    client._lifecycle_started = True
    client._cleanup_errors = []
    client._cleanup_postcondition_verified = False

    for _ in range(2):
        with pytest.raises(RuntimeError, match="retirement is unverified"):
            client.close()


def test_close_join_timeout_is_independently_retryable() -> None:
    """Thread join retry remains separate from async cleanup evidence."""
    client = MCPClient(command="fake")

    class _DeadThread:
        def join(self, timeout=None) -> None:
            pass

        def is_alive(self) -> bool:
            return False

    client._thread = _DeadThread()
    client.close()


@pytest.mark.parametrize("kind", ["stdio", "http"])
def test_thread_start_failure_rolls_back_lifecycle_publication_and_close_retries(
    kind: str,
    monkeypatch,
) -> None:
    client = _cleanup_client(kind)
    injected = RuntimeError("thread start injection")

    def _fail_thread_start(_thread) -> None:
        raise injected

    monkeypatch.setattr(threading.Thread, "start", _fail_thread_start)

    with pytest.raises(RuntimeError) as raised:
        client.start()

    assert raised.value is injected
    assert client._thread is None
    assert client._loop is None
    assert client._lifecycle_task is None
    assert client._lifecycle_started is False
    if kind == "stdio":
        assert client._generation == 0

    client.close()
    client.close()


@pytest.mark.parametrize("kind", ["stdio", "http"])
def test_event_loop_bootstrap_failure_is_ready_bounded_and_cleanup_verified(
    kind: str,
    monkeypatch,
) -> None:
    client = _cleanup_client(kind)
    client._START_TIMEOUT_SECONDS = 0.5

    def _fail_new_event_loop():
        raise RuntimeError("event-loop bootstrap injection")

    monkeypatch.setattr(asyncio, "new_event_loop", _fail_new_event_loop)

    started = time.monotonic()
    with pytest.raises(RuntimeError) as raised:
        client.start()
    elapsed = time.monotonic() - started

    assert elapsed < 1
    assert "event-loop bootstrap injection" in str(raised.value)
    assert "timed out" not in str(raised.value)
    assert client._ready.is_set()
    assert client._error is not None
    assert "event-loop bootstrap injection" in client._error
    assert client._cleanup_postcondition_verified is True
    assert client._thread is not None
    assert not client._thread.is_alive()
    assert client._loop is None
    assert client._lifecycle_task is None

    client.close()
    client.close()


class ClosedResourceError(Exception):
    pass


class _StaleGenerationSession:
    def __init__(self) -> None:
        self.both_started = threading.Event()
        self._arrivals = 0
        self._release: asyncio.Event | None = None

    async def call_tool(self, **_kwargs: Any) -> Any:
        if self._release is None:
            self._release = asyncio.Event()
        self._arrivals += 1
        if self._arrivals == 2:
            self.both_started.set()
            self._release.set()
        await self._release.wait()
        raise ClosedResourceError()


class _SuccessfulToolResult:
    structuredContent = {"status": "success", "text": "replacement"}
    content: list[Any] = []
    isError = False


class _ReplacementGenerationSession:
    def __init__(self) -> None:
        self.calls = 0

    async def call_tool(self, **_kwargs: Any) -> Any:
        self.calls += 1
        return _SuccessfulToolResult()


class _RecordingSessionContext:
    def __init__(
        self,
        generation: int,
        events: list[tuple[str, int]],
    ) -> None:
        self._generation = generation
        self._events = events

    async def __aexit__(self, *_args) -> None:
        self._events.append(("retire", self._generation))


def test_concurrent_stale_calls_share_one_real_lifecycle_replacement_generation(
    monkeypatch,
) -> None:
    client = MCPClient(command="fake")
    client._START_TIMEOUT_SECONDS = 1
    client._CLOSE_TIMEOUT_SECONDS = 1
    stale_session = _StaleGenerationSession()
    replacement_session = _ReplacementGenerationSession()
    lifecycle_events: list[tuple[str, int]] = []

    async def _fake_connect() -> None:
        generation = client._generation
        session = stale_session if generation == 1 else replacement_session
        client._session_cm = _RecordingSessionContext(
            generation, lifecycle_events
        )
        client._session = session
        lifecycle_events.append(("start", generation))
        client._ready.set()

    monkeypatch.setattr(client, "_async_connect", _fake_connect)

    client.start()
    assert client._generation == 1
    old_thread = client._thread
    assert old_thread is not None and old_thread.is_alive()

    callers_and_outcomes = [
        _start_sync_call(lambda: client.call_tool("probe", {}, timeout=2))
        for _ in range(2)
    ]
    try:
        assert stale_session.both_started.wait(timeout=1)
        outcomes = [
            _join_sync_call(caller, outcome)
            for caller, outcome in callers_and_outcomes
        ]

        assert [outcome["result"] for outcome in outcomes] == [
            {"status": "success", "text": "replacement"},
            {"status": "success", "text": "replacement"},
        ]
        assert replacement_session.calls == 2
        assert client._generation == 2
        replacement_thread = client._thread
        assert replacement_thread is not None
        assert replacement_thread is not old_thread
        assert replacement_thread.is_alive()
        assert not old_thread.is_alive()

        assert lifecycle_events.count(("start", 1)) == 1
        assert lifecycle_events.count(("retire", 1)) == 1
        assert lifecycle_events.count(("start", 2)) == 1
        assert lifecycle_events.index(("retire", 1)) < lifecycle_events.index(
            ("start", 2)
        )
    finally:
        for caller, _outcome in callers_and_outcomes:
            caller.join(timeout=3)
        if client._thread is not None and client._thread.is_alive():
            client.close()


def test_restart_does_not_hold_state_lock_across_pending_rpc_drain(
    monkeypatch,
) -> None:
    client, session, old_loop, old_thread = _connected_client("stdio")
    client._generation = 1

    class _ReplacementLoop:
        def is_running(self) -> bool:
            return True

    starts: list[int] = []

    def _start_replacement() -> None:
        starts.append(1)
        client._generation += 1
        client._session = object()
        client._loop = _ReplacementLoop()
        client._closed = False

    monkeypatch.setattr(client, "start", _start_replacement)
    results: list[dict[str, Any]] = []
    caller = threading.Thread(
        target=lambda: results.append(
            client.call_tool("blocked-during-restart", {}, timeout=10)
        )
    )
    caller.start()
    assert session.call_started.wait(timeout=1)
    try:
        client.restart(expected_generation=1)
        caller.join(timeout=1)

        assert not caller.is_alive()
        assert session.call_cancelled.is_set()
        assert starts == [1]
        assert client._generation == 2
        assert results[0]["status"] == "error"
        assert "CancelledError" in results[0]["message"]
    finally:
        _stop_loop(old_loop, old_thread)
