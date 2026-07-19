"""MCP clients — async-to-sync bridges for MCP servers.

MCPClient: stdio subprocess servers (e.g., uvx minimax-coding-plan-mcp).
HTTPMCPClient: remote HTTP/SSE servers (e.g., api.z.ai/api/mcp/...).

Both provide the same synchronous call_tool() interface. A background daemon
thread runs the async event loop; the public API is thread-safe.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta, timezone
from typing import Any

from lingtai.kernel.logging import get_logger

logger = get_logger()

_JSON_PARSE_FAILED = object()


def _first_text_content(result: Any) -> str | None:
    """Return the first text block carried by an MCP tool result."""
    for block in getattr(result, "content", None) or []:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            return text
    return None


def _error_payload(payload: dict[str, Any], fallback_message: str) -> dict[str, Any]:
    """Keep a structured MCP error while retaining the legacy envelope."""
    error = dict(payload)
    # The protocol's isError bit is authoritative. Do not let a malformed or
    # hostile payload disguise a failed tool call as success.
    error["status"] = "error"
    message = error.get("message")
    if not isinstance(message, str) or not message:
        error["message"] = fallback_message
    return error


def _decode_tool_result(result: Any) -> Any:
    """Decode an MCP result without flattening structured tool errors.

    MCP transports may return the same object in ``structuredContent`` and in
    a JSON text block. Prefer the structured form, then a JSON object in text.
    Plain-text errors retain the historical ``status``/``message`` envelope.
    This decoder is intentionally protocol-generic and is shared by stdio and
    HTTP clients.
    """
    text = _first_text_content(result)
    structured = getattr(result, "structuredContent", None)

    if isinstance(structured, dict):
        if getattr(result, "isError", False):
            return _error_payload(
                structured,
                text if text else "Unknown MCP error",
            )
        return dict(structured)

    if text is not None:
        try:
            decoded = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            decoded = _JSON_PARSE_FAILED
        if isinstance(decoded, dict):
            if getattr(result, "isError", False):
                return _error_payload(decoded, text)
            return decoded
        if not getattr(result, "isError", False) and decoded is not _JSON_PARSE_FAILED:
            # Preserve the pre-existing behavior for successful JSON values,
            # even though MCP tool handlers conventionally return objects.
            return decoded

    if getattr(result, "isError", False):
        return {"status": "error", "message": text or "Unknown MCP error"}
    return {"status": "success", "text": text or ""}


class _AsyncOperationBridge:
    """Bound synchronous waits to async RPC cancellation convergence.

    A synchronous timeout is not enough evidence that the coroutine stopped:
    ``concurrent.futures.Future.result(timeout=...)`` leaves the event-loop
    task running.  Track every submitted RPC, cancel the exact timed-out task,
    and await its ``finally`` before the caller may release its Agent lease or
    candidate activation may start transport cleanup.
    """

    _OPERATION_CANCEL_TIMEOUT_SECONDS = 5.0

    def _initialize_async_operations(self) -> None:
        self._operation_tokens: set[object] = set()
        self._operation_tasks: dict[object, Any] = {}
        self._operation_cancellation_barriers: set[Any] = set()
        self._operation_quarantine_error: str | None = None

    @staticmethod
    def _stream_is_closed(stream: Any) -> bool:
        """Return an explicit stream-close postcondition, never an assumption."""
        return stream is None or getattr(stream, "_closed", None) is True

    async def _cancel_async_operation_tasks(
        self, tokens: set[object] | None = None
    ) -> None:
        """Cancel and await selected RPC tasks from their owning loop."""
        import asyncio

        if not self._operation_tasks:
            return
        # Yield once so callbacks queued before this barrier can create (or
        # cancel-before-start) their tasks. run_coroutine_threadsafe preserves
        # that loop queue ordering for submissions from the same caller.
        await asyncio.sleep(0)
        selected = [
            task
            for token, task in list(self._operation_tasks.items())
            if tokens is None or token in tokens
        ]
        current = asyncio.current_task()
        selected = [task for task in selected if task is not current]
        for task in selected:
            task.cancel()
        if selected:
            await asyncio.gather(*selected, return_exceptions=True)
        await asyncio.sleep(0)
        remaining = {
            token
            for token, task in self._operation_tasks.items()
            if (tokens is None or token in tokens) and not task.done()
        }
        if remaining:
            raise RuntimeError(
                f"{len(remaining)} MCP async operation(s) did not cancel"
            )

    def _cancel_and_drain_async_operations(
        self,
        tokens: set[object] | None = None,
        *,
        reason: str,
        clear_quarantine: bool = False,
    ) -> None:
        """Run the event-loop cancellation barrier with a bounded sync wait."""
        import asyncio
        import concurrent.futures

        if clear_quarantine:
            with self._lock:
                pending_barriers = list(self._operation_cancellation_barriers)
            for pending in pending_barriers:
                try:
                    pending.result(timeout=self._OPERATION_CANCEL_TIMEOUT_SECONDS)
                except concurrent.futures.TimeoutError as exc:
                    message = (
                        f"{reason}: prior async cancellation barrier did not "
                        "converge"
                    )
                    with self._lock:
                        self._operation_quarantine_error = message
                    raise RuntimeError(message) from exc
                except Exception as exc:
                    message = (
                        f"{reason}: prior async cancellation barrier failed: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    with self._lock:
                        self._operation_cancellation_barriers.discard(pending)
                        self._operation_quarantine_error = message
                    raise RuntimeError(message) from exc
                else:
                    with self._lock:
                        self._operation_cancellation_barriers.discard(pending)

        with self._lock:
            selected = set(self._operation_tokens)
            if tokens is not None:
                selected.intersection_update(tokens)
            loop = self._loop
        if not selected:
            if clear_quarantine:
                with self._lock:
                    if not self._operation_tokens and not self._operation_tasks:
                        self._operation_quarantine_error = None
            return
        if loop is None or not loop.is_running():
            message = f"{reason}: MCP event loop stopped with unresolved operations"
            with self._lock:
                self._operation_quarantine_error = message
            raise RuntimeError(message)

        barrier = asyncio.run_coroutine_threadsafe(
            self._cancel_async_operation_tasks(selected), loop
        )
        try:
            barrier.result(timeout=self._OPERATION_CANCEL_TIMEOUT_SECONDS)
        except concurrent.futures.TimeoutError as exc:
            message = (
                f"{reason}: async operation cancellation did not converge "
                f"within {self._OPERATION_CANCEL_TIMEOUT_SECONDS:g}s"
            )
            with self._lock:
                # Do not cancel this barrier: its gather owns proof that the
                # stubborn task eventually reached finally. A later close waits
                # this exact future before clearing quarantine/closing the loop.
                self._operation_cancellation_barriers.add(barrier)
                self._operation_quarantine_error = message
            raise RuntimeError(message) from exc
        except Exception as exc:
            message = (
                f"{reason}: async operation cancellation failed: "
                f"{type(exc).__name__}: {exc}"
            )
            with self._lock:
                self._operation_quarantine_error = message
            raise RuntimeError(message) from exc
        else:
            # A task cancelled before its coroutine body first ran cannot
            # execute the wrapper's finally. The ordered loop barrier above
            # proves that such a token has no live task, so it is safe to drop.
            with self._lock:
                self._operation_tokens.difference_update(selected)
                if clear_quarantine and not self._operation_tokens:
                    self._operation_quarantine_error = None

    def _run_async_operation(
        self,
        operation_factory,
        *,
        timeout: float,
        label: str,
    ) -> Any:
        """Submit one RPC and prove cancellation before surfacing a timeout."""
        import asyncio
        import concurrent.futures

        token = object()

        async def _tracked() -> Any:
            task = asyncio.current_task()
            if task is None:  # pragma: no cover - asyncio always supplies one
                raise RuntimeError("MCP async operation has no owning task")
            self._operation_tasks[token] = task
            try:
                return await operation_factory()
            finally:
                self._operation_tasks.pop(token, None)
                with self._lock:
                    self._operation_tokens.discard(token)

        with self._lock:
            if self._closed:
                raise RuntimeError("MCP client has been closed")
            if self._operation_quarantine_error is not None:
                raise RuntimeError(
                    "MCP client is quarantined after unresolved async "
                    f"operation cancellation: {self._operation_quarantine_error}"
                )
            loop = self._loop
            if loop is None or not loop.is_running():
                raise RuntimeError("MCP client event loop is not running")
            self._operation_tokens.add(token)
            try:
                future = asyncio.run_coroutine_threadsafe(_tracked(), loop)
            except Exception:
                self._operation_tokens.discard(token)
                raise

        timed_out = False
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError as exc:
            timed_out = True
            # Mark the submitted future cancelled, then use a separate ordered
            # barrier to await the actual asyncio task's finally block.
            future.cancel()
            self._cancel_and_drain_async_operations(
                {token}, reason=f"{label} timeout"
            )
            raise TimeoutError(
                f"{label} timed out after {timeout:g}s; async operation "
                "cancelled and drained"
            ) from exc
        finally:
            if not timed_out:
                # Real tasks remove this in their finally. This defensive drop
                # also keeps test/fallback Future adapters from leaking tokens.
                with self._lock:
                    self._operation_tokens.discard(token)


class MCPClient(_AsyncOperationBridge):
    """Async-to-sync bridge for any MCP stdio server.

    Args:
        command: Executable to run (e.g., "uvx").
        args: Arguments to the command (e.g., ["minimax-coding-plan-mcp", "-y"]).
        env: Environment variables for the subprocess. If None, inherits
            the current process environment.
    """

    _START_TIMEOUT_SECONDS = 30.0
    _CLOSE_TIMEOUT_SECONDS = 5.0

    def __init__(
        self,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ):
        self._command = command
        self._args = args or []
        self._env = env

        self._session: Any = None
        self._read_stream: Any = None
        self._write_stream: Any = None
        self._loop: Any = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._error: str | None = None
        self._lock = threading.RLock()
        self._restart_lock = threading.Lock()
        self._closed = False
        self._stdio_cm: Any = None
        self._session_cm: Any = None
        self._cleanup_errors: list[str] = []
        self._cleanup_errors_reported = False
        self._cleanup_postcondition_verified = False
        self._lifecycle_started = False
        self._lifecycle_task: Any = None
        self._generation = 0
        self._stdio_process: Any = None
        self._stdio_transport_entered = False
        self._initialize_async_operations()

        # Activity log for debugging — last 50 calls
        self._activity_log: list[dict[str, Any]] = []
        self._activity_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Exception helpers — never surface a blank error (issue #104)
    # ------------------------------------------------------------------

    @staticmethod
    def _format_exception(exc: BaseException) -> str:
        """Render an exception as ``ClassName: message``.

        Some MCP/anyio exceptions (notably ``ClosedResourceError``) have an
        empty ``str()``. Falling through to that empty string is what produced
        the blank ``{"status": "error", "message": ""}`` in issue #104. When the
        message is empty we fall back to the class name alone.
        """
        cls = type(exc).__name__
        msg = str(exc).strip()
        return f"{cls}: {msg}" if msg else cls

    @classmethod
    def _format_exception_tree(cls, exc: BaseException) -> str:
        """Render ExceptionGroup leaves so cleanup evidence stays actionable."""
        if isinstance(exc, BaseExceptionGroup):
            leaves = "; ".join(
                cls._format_exception_tree(child) for child in exc.exceptions
            )
            return f"{type(exc).__name__}: {leaves}"
        return cls._format_exception(exc)

    @classmethod
    def _exception_tree_has_only(
        cls, exc: BaseException, allowed_names: set[str]
    ) -> bool:
        if isinstance(exc, BaseExceptionGroup):
            return bool(exc.exceptions) and all(
                cls._exception_tree_has_only(child, allowed_names)
                for child in exc.exceptions
            )
        return type(exc).__name__ in allowed_names

    @staticmethod
    def _reraise_fatal_exception_group(exc: BaseExceptionGroup) -> None:
        """Never turn process-control exceptions into cleanup diagnostics."""
        fatal, _nonfatal = exc.split((KeyboardInterrupt, SystemExit))
        if fatal is not None:
            raise fatal

    @staticmethod
    def _is_stale_resource_error(exc: BaseException) -> bool:
        """Detect a dead/closed MCP transport that warrants a restart.

        Primary signal is the exception class name ``ClosedResourceError``
        (anyio) — matched by name so we need not import anyio. As a secondary
        signal we look for closed-stream substrings in the message.
        """
        if type(exc).__name__ == "ClosedResourceError":
            return True
        text = str(exc).lower()
        return any(
            marker in text
            for marker in ("closed", "broken pipe", "stream", "transport")
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Spawn the background thread and connect to the MCP server.

        Called automatically by call_tool() if not yet connected.
        """
        if self.is_connected():
            return
        with self._lock:
            if self._closed:
                raise RuntimeError("MCP client has been closed")
            # One client owns at most one lifecycle thread. A readiness timeout
            # must never let list_tools() start a second thread over the first.
            if self._thread is None:
                generation_before = self._generation
                self._cleanup_errors = []
                self._cleanup_errors_reported = False
                self._cleanup_postcondition_verified = False
                self._lifecycle_started = True
                self._generation += 1
                thread = threading.Thread(target=self._run_loop, daemon=True)
                self._thread = thread
                try:
                    thread.start()
                except Exception:
                    # No lifecycle owns a transport when Python could not
                    # start the fresh thread. Restore a retryable, vacuously
                    # clean state so Agent candidate cleanup cannot wedge on
                    # join-before-start.
                    if thread.ident is None and not thread.is_alive():
                        self._thread = None
                        self._lifecycle_started = False
                        self._cleanup_postcondition_verified = True
                        self._generation = generation_before
                    raise

        if not self._ready.wait(timeout=self._START_TIMEOUT_SECONDS):
            cleanup_error = None
            try:
                self.close()
            except Exception as exc:  # preserve both startup and cleanup evidence
                cleanup_error = self._format_exception(exc)
            message = (
                "MCP server startup timed out after "
                f"{self._START_TIMEOUT_SECONDS:g}s"
            )
            if cleanup_error:
                message += f"; candidate cleanup failed: {cleanup_error}"
            raise TimeoutError(message)
        if self._error:
            startup_error = self._error
            cleanup_error = None
            try:
                self.close()
            except Exception as exc:
                cleanup_error = self._format_exception(exc)
            message = f"MCP server failed to start: {startup_error}"
            if cleanup_error:
                message += f"; candidate cleanup failed: {cleanup_error}"
            raise RuntimeError(message)

    def close(self) -> None:
        """Shut down the session and prove the lifecycle thread retired.

        Repeated calls deliberately re-issue the bounded join.  A first close
        may time out while the async transport is still unwinding; treating the
        latched ``_closed`` flag as success would make later cleanup unable to
        converge.
        """
        with self._lock:
            self._closed = True
        # Never enter session/transport cleanup while a timed-out or direct
        # RPC is still unwinding. A failed barrier is deliberately retryable:
        # leave the lifecycle task alive and let Agent retain this client in
        # its pending-retirement set.
        self._cancel_and_drain_async_operations(
            reason="MCP client close", clear_quarantine=True
        )
        if self._loop and self._loop.is_running():
            task = self._lifecycle_task
            if task is not None and not task.done():
                self._loop.call_soon_threadsafe(task.cancel)
        if self._thread:
            self._thread.join(timeout=self._CLOSE_TIMEOUT_SECONDS)
            if self._thread.is_alive():
                raise RuntimeError(
                    "MCP client thread did not retire within "
                    f"{self._CLOSE_TIMEOUT_SECONDS:g}s"
                )
        if self._lifecycle_started and not self._cleanup_postcondition_verified:
            evidence = (
                "; ".join(self._cleanup_errors)
                if self._cleanup_errors
                else "async cleanup did not establish a terminal postcondition"
            )
            raise RuntimeError(
                "MCP client resource retirement is unverified: " + evidence
            )
        if self._cleanup_errors:
            message = "MCP client async cleanup failed: " + "; ".join(
                self._cleanup_errors
            )
            if not self._cleanup_postcondition_verified:
                raise RuntimeError(message + "; resource retirement is unverified")
            if not self._cleanup_errors_reported:
                # Verified resource state permits a later convergence attempt,
                # but the first caller still receives the original evidence.
                self._cleanup_errors_reported = True
                raise RuntimeError(
                    message + "; resource retirement postcondition verified"
                )

    def restart(self, *, expected_generation: int | None = None) -> None:
        """Tear down a (possibly stale) session and reconnect from scratch.

        ``start()`` early-returns when it believes it is connected and never
        clears latched startup state, so a stale ``_ready``/``_error`` or a
        ``_closed`` flag from a prior ``close()`` would make a fresh ``start()``
        lie (return immediately, or raise on the *old* error). This resets all
        startup/session fields so the subsequent ``start()`` is a real reconnect.
        Used by ``call_tool`` to recover from a closed stdio resource (issue #104).
        """
        # Serialize generations without holding the state lock across close's
        # event-loop cancellation barrier. Operation-task finally blocks need
        # that state lock to discard their tokens.
        with self._restart_lock:
            # Two calls can observe the same stale generation concurrently.
            # Once one caller installed a healthy replacement, the later
            # caller must reuse it instead of spawning a second subprocess.
            with self._lock:
                replacement_is_live = (
                    expected_generation is not None
                    and self._generation != expected_generation
                    and self.is_connected()
                )
            if replacement_is_live:
                return
            self.close()
            with self._lock:
                self._ready.clear()
                self._error = None
                self._closed = False
                self._session = None
                self._read_stream = None
                self._write_stream = None
                self._loop = None
                self._thread = None
                self._stdio_cm = None
                self._session_cm = None
                self._cleanup_errors = []
                self._cleanup_errors_reported = False
                self._cleanup_postcondition_verified = False
                self._lifecycle_started = False
                self._lifecycle_task = None
                self._stdio_process = None
                self._stdio_transport_entered = False
                self._initialize_async_operations()
            self.start()

    def is_connected(self) -> bool:
        """Check if the client has an active session."""
        return (
            self._session is not None
            and self._loop is not None
            and self._loop.is_running()
            and not self._closed
            and self._operation_quarantine_error is None
        )

    # ------------------------------------------------------------------
    # Tool calls
    # ------------------------------------------------------------------

    def call_tool(self, name: str, args: dict, timeout: float = 120) -> Any:
        """Call an MCP tool synchronously.

        Starts the connection lazily if not yet connected.

        Args:
            name: Tool name (e.g., "web_search").
            args: Tool arguments dict.
            timeout: Timeout in seconds.

        Returns:
            Decoded MCP result. JSON scalars, arrays, and ``null`` retain
            their protocol value; structured objects remain dictionaries.

        Raises:
            RuntimeError: If the client is closed or connection fails.
        """
        if self._closed:
            raise RuntimeError("MCP client has been closed")
        if self._operation_quarantine_error is not None:
            raise RuntimeError(
                "MCP client is quarantined after unresolved async operation "
                f"cancellation: {self._operation_quarantine_error}"
            )

        # Lazy start
        if not self.is_connected():
            self.start()

        if self._session is None or self._loop is None:
            raise RuntimeError("MCP client not connected")

        def _attempt() -> Any:
            observed_generation = self._generation

            async def _call():
                result = await self._session.call_tool(
                    name=name,
                    arguments=args,
                    read_timeout_seconds=timedelta(seconds=timeout),
                )
                return _decode_tool_result(result)

            return (
                self._run_async_operation(
                    _call,
                    timeout=timeout,
                    label=f"MCP tool {name!r}",
                ),
                observed_generation,
            )

        attempt_generation = self._generation
        try:
            result, attempt_generation = _attempt()
        except Exception as exc:
            formatted = self._format_exception(exc)
            if not self._is_stale_resource_error(exc):
                # Non-stale failure: surface the class name so the error is
                # never blank (issue #104), but don't churn the subprocess.
                result = {"status": "error", "message": formatted}
            else:
                # Stale/closed resource: tear down and reconnect, retry once.
                logger.warning(
                    "MCP tool %s hit stale resource (%s); restarting and "
                    "retrying once", name, formatted,
                )
                try:
                    self.restart(expected_generation=attempt_generation)
                except Exception as restart_exc:
                    result = {
                        "status": "error",
                        "message": (
                            f"{formatted}: MCP session closed; restart failed: "
                            f"{self._format_exception(restart_exc)}"
                        ),
                    }
                else:
                    try:
                        result, _retry_generation = _attempt()
                    except Exception as retry_exc:
                        result = {
                            "status": "error",
                            "message": (
                                f"{self._format_exception(retry_exc)}: MCP "
                                "session closed; restarted once but retry failed"
                            ),
                        }

        # Log activity
        with self._activity_lock:
            self._activity_log.append({
                "tool": name,
                "args": args,
                "result": result,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            if len(self._activity_log) > 50:
                self._activity_log[:] = self._activity_log[-50:]

        return result

    def list_tools(self, timeout: float = 10) -> list[dict]:
        """List available tools from the MCP server.

        Returns a list of dicts with 'name', 'description', and 'schema' keys.
        """
        if self._operation_quarantine_error is not None:
            raise RuntimeError(
                "MCP client is quarantined after unresolved async operation "
                f"cancellation: {self._operation_quarantine_error}"
            )
        if not self.is_connected():
            self.start()

        if self._session is None or self._loop is None:
            raise RuntimeError("MCP client not connected")

        async def _list():
            result = await self._session.list_tools()
            tools = []
            for tool in result.tools:
                schema = {}
                if tool.inputSchema:
                    schema = tool.inputSchema if isinstance(tool.inputSchema, dict) else {}
                tools.append({
                    "name": tool.name,
                    "description": tool.description or "",
                    "schema": schema,
                })
            return tools

        return self._run_async_operation(
            _list,
            timeout=timeout,
            label="MCP tool catalog",
        )

    def get_activity_log(self) -> list[dict[str, Any]]:
        """Get recent MCP tool calls for debugging."""
        with self._activity_lock:
            return list(self._activity_log)

    # ------------------------------------------------------------------
    # Internal — background thread
    # ------------------------------------------------------------------

    def _run_loop(self) -> None:
        """Background thread: run the async event loop with the MCP session."""
        import asyncio

        loop = None
        lifecycle_dispatched = False
        try:
            loop = asyncio.new_event_loop()
            self._loop = loop
            asyncio.set_event_loop(loop)
            self._lifecycle_task = loop.create_task(self._async_lifecycle())
            lifecycle_dispatched = True
            loop.run_until_complete(self._lifecycle_task)
        except Exception as e:
            # Preserve the class name so a blank str(e) (e.g. ClosedResourceError)
            # does not produce "MCP server failed to start: " (issue #104).
            self._error = self._format_exception(e)
            if not lifecycle_dispatched:
                # Event-loop/bootstrap failure occurred before transport code
                # could run. This is a verified empty-resource postcondition,
                # not a permanently unresolved candidate cleanup.
                self._cleanup_postcondition_verified = True
            self._ready.set()
        finally:
            if loop is not None:
                loop.close()

    async def _async_lifecycle(self) -> None:
        """Own connect, wait, and cleanup inside one asyncio task.

        AnyIO transport context managers bind cancel scopes to the task that
        entered them. Keeping teardown in this same task makes cleanup both
        verifiable and portable instead of relying on swallowed cross-task
        ``__aexit__`` errors.
        """
        import asyncio

        try:
            await self._async_connect()
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            # ``close()`` uses Task.cancel() to interrupt a connection that may
            # still be initializing. Clear the task's pending cancellation
            # count before awaiting AnyIO context-manager exits; cleanup must
            # remain in this same task (cancel-scope ownership) and must not be
            # immediately cancelled a second time.
            task = asyncio.current_task()
            if task is not None:
                while task.cancelling():
                    task.uncancel()
        finally:
            await self._async_cleanup()

    async def _async_connect(self) -> None:
        """Establish the MCP stdio connection (runs in background thread)."""
        from mcp.client.stdio import stdio_client, StdioServerParameters
        from mcp.client.session import ClientSession

        server_params = StdioServerParameters(
            command=self._command,
            args=self._args,
            env=self._env,
        )

        self._stdio_cm = stdio_client(server_params)
        self._read_stream, self._write_stream = await self._stdio_cm.__aenter__()
        self._stdio_transport_entered = True
        generator = getattr(self._stdio_cm, "gen", None)
        frame = getattr(generator, "ag_frame", None)
        if frame is not None:
            # The SDK context owns the child process but does not expose it in
            # the yielded stream tuple. Retain the handle solely to verify the
            # mandatory no-live-child cleanup postcondition.
            self._stdio_process = frame.f_locals.get("process")

        self._session_cm = ClientSession(self._read_stream, self._write_stream)
        self._session = await self._session_cm.__aenter__()

        await self._session.initialize()

        self._ready.set()

    async def _async_cleanup(self) -> None:
        """Clean up MCP session and stdio transport."""
        import asyncio

        # Lifecycle cleanup is the final safety net for unexpected transport
        # exits. Normal close already ran the same barrier synchronously.
        await self._cancel_async_operation_tasks()
        errors: list[str] = []
        if self._session_cm:
            try:
                await self._session_cm.__aexit__(None, None, None)
            except BaseExceptionGroup as exc:
                self._reraise_fatal_exception_group(exc)
                errors.append(
                    "session: " + self._format_exception_tree(exc)
                )
            except asyncio.CancelledError:
                # AnyIO task-group exits may re-surface the close cancellation
                # after completing their own cleanup. Continue to the outer
                # transport exit in this same owning task, but do not treat the
                # exit as a normal-return proof. Explicit process/stream probes
                # below must establish retirement.
                errors.append("session: CancelledError")
            except Exception as exc:
                errors.append(
                    "session: " + self._format_exception_tree(exc)
                )
        if self._stdio_cm:
            try:
                await self._stdio_cm.__aexit__(None, None, None)
            except BaseExceptionGroup as exc:
                self._reraise_fatal_exception_group(exc)
                errors.append(
                    "stdio transport: " + self._format_exception_tree(exc)
                )
            except asyncio.CancelledError:
                errors.append("stdio transport: CancelledError")
            except Exception as exc:
                errors.append(
                    "stdio transport: " + self._format_exception_tree(exc)
                )
        process_retired = (
            self._stdio_process is not None
            and getattr(self._stdio_process, "returncode", None) is not None
        )
        streams_closed = self._stream_is_closed(
            self._read_stream
        ) and self._stream_is_closed(self._write_stream)
        self._cleanup_postcondition_verified = (
            not errors
            or (
                self._stdio_transport_entered
                and process_retired
                and streams_closed
            )
        )
        self._cleanup_errors = errors
        self._session = None
        self._read_stream = None
        self._write_stream = None
        self._session_cm = None
        self._stdio_cm = None


class HTTPMCPClient(_AsyncOperationBridge):
    """Async-to-sync bridge for remote HTTP MCP servers.

    Connects to a remote MCP server via streamable HTTP transport.
    Same call_tool() interface as MCPClient.

    Args:
        url: HTTP endpoint of the MCP server (e.g., "https://api.z.ai/api/mcp/web_search_prime/mcp").
        headers: HTTP headers (e.g., {"Authorization": "Bearer ..."}).
    """

    _START_TIMEOUT_SECONDS = 30.0
    _CLOSE_TIMEOUT_SECONDS = 5.0

    def __init__(
        self,
        url: str,
        headers: dict[str, str] | None = None,
    ):
        self._url = url
        self._headers = headers or {}

        self._session: Any = None
        self._loop: Any = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._error: str | None = None
        self._lock = threading.RLock()
        self._closed = False
        self._transport_cm: Any = None
        self._session_cm: Any = None
        self._cleanup_errors: list[str] = []
        self._cleanup_errors_reported = False
        self._cleanup_postcondition_verified = False
        self._lifecycle_started = False
        self._lifecycle_task: Any = None
        self._http_client: Any = None
        self._http_transport_entered = False
        self._initialize_async_operations()

        self._activity_log: list[dict[str, Any]] = []
        self._activity_lock = threading.Lock()

    def start(self) -> None:
        if self.is_connected():
            return
        with self._lock:
            if self._closed:
                raise RuntimeError("HTTP MCP client has been closed")
            if self._thread is None:
                self._cleanup_errors = []
                self._cleanup_errors_reported = False
                self._cleanup_postcondition_verified = False
                self._lifecycle_started = True
                thread = threading.Thread(target=self._run_loop, daemon=True)
                self._thread = thread
                try:
                    thread.start()
                except Exception:
                    if thread.ident is None and not thread.is_alive():
                        self._thread = None
                        self._lifecycle_started = False
                        self._cleanup_postcondition_verified = True
                    raise

        if not self._ready.wait(timeout=self._START_TIMEOUT_SECONDS):
            cleanup_error = None
            try:
                self.close()
            except Exception as exc:
                cleanup_error = MCPClient._format_exception(exc)
            message = (
                "HTTP MCP server startup timed out after "
                f"{self._START_TIMEOUT_SECONDS:g}s"
            )
            if cleanup_error:
                message += f"; candidate cleanup failed: {cleanup_error}"
            raise TimeoutError(message)
        if self._error:
            startup_error = self._error
            cleanup_error = None
            try:
                self.close()
            except Exception as exc:
                cleanup_error = MCPClient._format_exception(exc)
            message = f"HTTP MCP server failed to connect: {startup_error}"
            if cleanup_error:
                message += f"; candidate cleanup failed: {cleanup_error}"
            raise RuntimeError(message)

    def close(self) -> None:
        with self._lock:
            self._closed = True
        self._cancel_and_drain_async_operations(
            reason="HTTP MCP client close", clear_quarantine=True
        )
        if self._loop and self._loop.is_running():
            task = self._lifecycle_task
            if task is not None and not task.done():
                self._loop.call_soon_threadsafe(task.cancel)
        if self._thread:
            self._thread.join(timeout=self._CLOSE_TIMEOUT_SECONDS)
            if self._thread.is_alive():
                raise RuntimeError(
                    "HTTP MCP client thread did not retire within "
                    f"{self._CLOSE_TIMEOUT_SECONDS:g}s"
                )
        if self._lifecycle_started and not self._cleanup_postcondition_verified:
            evidence = (
                "; ".join(self._cleanup_errors)
                if self._cleanup_errors
                else "async cleanup did not establish a terminal postcondition"
            )
            raise RuntimeError(
                "HTTP MCP client resource retirement is unverified: " + evidence
            )
        if self._cleanup_errors:
            message = "HTTP MCP client async cleanup failed: " + "; ".join(
                self._cleanup_errors
            )
            if not self._cleanup_postcondition_verified:
                raise RuntimeError(message + "; resource retirement is unverified")
            if not self._cleanup_errors_reported:
                self._cleanup_errors_reported = True
                raise RuntimeError(
                    message + "; resource retirement postcondition verified"
                )

    def is_connected(self) -> bool:
        return (
            self._session is not None
            and self._loop is not None
            and self._loop.is_running()
            and not self._closed
            and self._operation_quarantine_error is None
        )

    def call_tool(self, name: str, args: dict, timeout: float = 120) -> Any:
        """Call an MCP tool synchronously. Same interface as MCPClient."""
        if self._closed:
            raise RuntimeError("HTTP MCP client has been closed")
        if self._operation_quarantine_error is not None:
            raise RuntimeError(
                "HTTP MCP client is quarantined after unresolved async "
                f"operation cancellation: {self._operation_quarantine_error}"
            )
        if not self.is_connected():
            self.start()
        if self._session is None or self._loop is None:
            raise RuntimeError("HTTP MCP client not connected")

        async def _call():
            result = await self._session.call_tool(
                name=name,
                arguments=args,
                read_timeout_seconds=timedelta(seconds=timeout),
            )
            return _decode_tool_result(result)

        result = self._run_async_operation(
            _call,
            timeout=timeout,
            label=f"HTTP MCP tool {name!r}",
        )

        with self._activity_lock:
            self._activity_log.append({
                "tool": name,
                "args": args,
                "result": result,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            if len(self._activity_log) > 50:
                self._activity_log[:] = self._activity_log[-50:]

        return result

    def list_tools(self, timeout: float = 10) -> list[dict]:
        """List available tools from the MCP server."""
        if self._operation_quarantine_error is not None:
            raise RuntimeError(
                "HTTP MCP client is quarantined after unresolved async "
                f"operation cancellation: {self._operation_quarantine_error}"
            )
        if not self.is_connected():
            self.start()
        if self._session is None or self._loop is None:
            raise RuntimeError("HTTP MCP client not connected")

        async def _list():
            result = await self._session.list_tools()
            tools = []
            for tool in result.tools:
                schema = {}
                if tool.inputSchema:
                    schema = tool.inputSchema if isinstance(tool.inputSchema, dict) else {}
                tools.append({
                    "name": tool.name,
                    "description": tool.description or "",
                    "schema": schema,
                })
            return tools

        return self._run_async_operation(
            _list,
            timeout=timeout,
            label="HTTP MCP tool catalog",
        )

    def _run_loop(self) -> None:
        import asyncio
        loop = None
        lifecycle_dispatched = False
        try:
            loop = asyncio.new_event_loop()
            self._loop = loop
            asyncio.set_event_loop(loop)
            self._lifecycle_task = loop.create_task(self._async_lifecycle())
            lifecycle_dispatched = True
            loop.run_until_complete(self._lifecycle_task)
        except Exception as e:
            # Preserve the class name so a blank str(e) is not surfaced as an
            # empty connect error (issue #104).
            self._error = MCPClient._format_exception(e)
            if not lifecycle_dispatched:
                self._cleanup_postcondition_verified = True
            self._ready.set()
        finally:
            if loop is not None:
                loop.close()

    async def _async_lifecycle(self) -> None:
        import asyncio

        try:
            await self._async_connect()
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            task = asyncio.current_task()
            if task is not None:
                while task.cancelling():
                    task.uncancel()
        finally:
            await self._async_cleanup()

    async def _async_connect(self) -> None:
        from mcp.client.streamable_http import (
            create_mcp_http_client,
            streamablehttp_client,
        )
        from mcp.client.session import ClientSession

        def _capturing_http_client_factory(**kwargs):
            client = create_mcp_http_client(**kwargs)
            self._http_client = client
            return client

        self._transport_cm = streamablehttp_client(
            url=self._url,
            headers=self._headers,
            httpx_client_factory=_capturing_http_client_factory,
        )
        self._read_stream, self._write_stream, _ = await self._transport_cm.__aenter__()
        self._http_transport_entered = True

        self._session_cm = ClientSession(self._read_stream, self._write_stream)
        self._session = await self._session_cm.__aenter__()

        await self._session.initialize()
        self._ready.set()

    async def _async_cleanup(self) -> None:
        import asyncio

        await self._cancel_async_operation_tasks()
        errors: list[str] = []
        if self._session_cm:
            try:
                await self._session_cm.__aexit__(None, None, None)
            except BaseExceptionGroup as exc:
                MCPClient._reraise_fatal_exception_group(exc)
                errors.append(
                    "session: " + MCPClient._format_exception_tree(exc)
                )
            except asyncio.CancelledError:
                errors.append("session: CancelledError")
            except Exception as exc:
                errors.append(
                    "session: " + MCPClient._format_exception_tree(exc)
                )
        if self._transport_cm:
            try:
                await self._transport_cm.__aexit__(None, None, None)
            except BaseExceptionGroup as exc:
                MCPClient._reraise_fatal_exception_group(exc)
                formatted = MCPClient._format_exception_tree(exc)
                if (
                    not self._ready.is_set()
                    and MCPClient._exception_tree_has_only(
                        exc, {"ConnectError", "ConnectTimeout"}
                    )
                ):
                    self._error = formatted
                    self._ready.set()
                else:
                    errors.append("HTTP transport: " + formatted)
            except asyncio.CancelledError:
                errors.append("HTTP transport: CancelledError")
            except Exception as exc:
                formatted = MCPClient._format_exception_tree(exc)
                if (
                    not self._ready.is_set()
                    and MCPClient._exception_tree_has_only(
                        exc, {"ConnectError", "ConnectTimeout"}
                    )
                ):
                    # The SDK's streamable transport can defer a refused
                    # connection into its TaskGroup exit when startup is
                    # cancelled. This is the primary startup failure, not a
                    # second cleanup failure: the owning thread still completes
                    # both context exits and close() verifies its bounded join.
                    self._error = formatted
                    self._ready.set()
                else:
                    errors.append("HTTP transport: " + formatted)
        http_closed = (
            self._http_client is not None
            and getattr(self._http_client, "is_closed", None) is True
        )
        streams_closed = self._stream_is_closed(
            self._read_stream
        ) and self._stream_is_closed(self._write_stream)
        self._cleanup_postcondition_verified = (
            not errors
            or (
                self._http_transport_entered
                and http_closed
                and streams_closed
            )
        )
        self._cleanup_errors = errors
        self._session = None
        self._session_cm = None
        self._transport_cm = None
        self._read_stream = None
        self._write_stream = None
