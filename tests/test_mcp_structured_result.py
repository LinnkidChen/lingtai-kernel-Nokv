"""MCP structured-result decoding shared by stdio and HTTP transports."""

from __future__ import annotations

import asyncio
import json
import sys
from types import SimpleNamespace
from typing import Any

import pytest
from mcp.types import CallToolResult, ImageContent, TextContent

from lingtai.services.mcp import HTTPMCPClient, MCPClient
from tests._mcp_stdio_fixture import (
    StdioProcessObserver,
    stop_process,
    wait_for_process_exit,
    wait_for_thread_exit,
)
from tests._mcp_structured_stdio_server import STRUCTURED_ERROR, TOOL_NAME


_MISSING = object()


class _ImmediateFuture:
    def __init__(self, value: Any) -> None:
        self._value = value

    def result(self, timeout: float | None = None) -> Any:
        del timeout
        return self._value


class _RunningLoop:
    def is_running(self) -> bool:
        return True


class _Session:
    def __init__(self, result: Any) -> None:
        self._result = result

    async def call_tool(self, **kwargs: Any) -> Any:
        del kwargs
        return self._result


def _result(
    *,
    is_error: bool | object = False,
    structured: dict[str, Any] | None = None,
    text: str | None = None,
) -> Any:
    content = [] if text is None else [TextContent(type="text", text=text)]
    if is_error is _MISSING:
        return SimpleNamespace(content=content, structuredContent=structured)
    return CallToolResult(
        isError=is_error,
        structuredContent=structured,
        content=content,
    )


@pytest.fixture(params=["stdio", "http"])
def client(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Any:
    if request.param == "stdio":
        instance = MCPClient(command="/bin/true")
    else:
        instance = HTTPMCPClient(url="https://example.invalid/mcp")

    instance._loop = _RunningLoop()
    instance._closed = False

    real_asyncio_run = asyncio.run

    def run_now(coro: Any, loop: Any) -> _ImmediateFuture:
        del loop
        return _ImmediateFuture(real_asyncio_run(coro))

    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", run_now)
    return instance


def _call(client: Any, result: Any) -> Any:
    client._session = _Session(result)
    return client.call_tool("example", {"value": 1})


def test_structured_error_has_priority_and_keeps_fields(client: Any) -> None:
    structured = {
        "status": "success",
        "code": "RestoreInProgress",
        "message": "restore is still preparing",
        "retryable": True,
        "details": {"operation_id": "restore-123"},
        "extra": {"kept": True},
    }
    result = _call(
        client,
        _result(
            is_error=True,
            structured=structured,
            text='{"code":"WrongTextFallback","message":"wrong"}',
        ),
    )

    assert result == {**structured, "status": "error"}


@pytest.mark.parametrize(
    ("structured", "text", "expected_message"),
    [
        (
            {"code": "RestoreProtocolMismatch", "retryable": True},
            "metadata owner returned an inconsistent outcome",
            "metadata owner returned an inconsistent outcome",
        ),
        (
            {
                "code": "RestoreProtocolMismatch",
                "message": "",
                "retryable": True,
            },
            None,
            "Unknown MCP error",
        ),
        (
            {
                "code": "RestoreProtocolMismatch",
                "message": 17,
                "retryable": True,
            },
            "typed fallback",
            "typed fallback",
        ),
        (
            {
                "code": "RestoreProtocolMismatch",
                "message": "  ",
                "retryable": True,
            },
            "must not replace whitespace",
            "  ",
        ),
    ],
    ids=["missing", "empty", "non-string", "whitespace-is-literal"],
)
def test_structured_error_message_policy(
    client: Any,
    structured: dict[str, Any],
    text: str | None,
    expected_message: str,
) -> None:
    result = _call(
        client,
        _result(is_error=True, structured=structured, text=text),
    )

    assert result == {
        **structured,
        "status": "error",
        "message": expected_message,
    }


@pytest.mark.parametrize(
    ("text", "expected_message"),
    [
        ('{"code":"E","retryable":false}', None),
        ('{"code":"E","message":"","retryable":false}', None),
        ('{"code":"E","message":17,"retryable":false}', None),
        ('{"code":"E","message":"  ","retryable":false}', "  "),
    ],
    ids=["missing", "empty", "non-string", "whitespace-is-literal"],
)
def test_json_object_error_message_policy(
    client: Any,
    text: str,
    expected_message: str | None,
) -> None:
    expected = json.loads(text)
    expected["status"] = "error"
    expected["message"] = text if expected_message is None else expected_message

    assert _call(client, _result(is_error=True, text=text)) == expected


def test_json_object_error_is_promoted_to_top_level(client: Any) -> None:
    result = _call(
        client,
        _result(
            is_error=True,
            text=(
                '{"status":"success","code":"SnapshotLeaseExpired",'
                '"message":"checkpoint expired","retryable":false,'
                '"details":{"snapshot_id":42},'
                '"error":{"kind":"checkpoint"}}'
            ),
        ),
    )

    assert result == {
        "status": "error",
        "code": "SnapshotLeaseExpired",
        "message": "checkpoint expired",
        "retryable": False,
        "details": {"snapshot_id": 42},
        "error": {"kind": "checkpoint"},
    }


@pytest.mark.parametrize(
    "text",
    ["true", "[1,2]", "null"],
    ids=["bool", "list", "null"],
)
def test_json_non_object_error_keeps_legacy_envelope(
    client: Any,
    text: str,
) -> None:
    assert _call(client, _result(is_error=True, text=text)) == {
        "status": "error",
        "message": text,
    }


def test_plain_text_error_keeps_legacy_envelope(client: Any) -> None:
    result = _call(
        client,
        _result(is_error=True, text="tool rejected the request"),
    )

    assert result == {
        "status": "error",
        "message": "tool rejected the request",
    }


def test_empty_error_keeps_nonempty_fallback(client: Any) -> None:
    result = _call(client, _result(is_error=True))

    assert result == {"status": "error", "message": "Unknown MCP error"}


@pytest.mark.parametrize(
    "is_error",
    [False, _MISSING],
    ids=["explicit-false", "transport-bit-missing"],
)
@pytest.mark.parametrize(
    "payload",
    [
        {"value": 7},
        {"status": "error", "value": 7},
    ],
    ids=["status-absent", "explicit-status-is-data"],
)
@pytest.mark.parametrize("source", ["structured", "json-text"])
def test_success_objects_are_not_rewritten(
    client: Any,
    source: str,
    payload: dict[str, Any],
    is_error: bool | object,
) -> None:
    if source == "structured":
        result = _result(
            is_error=is_error,
            structured=payload,
            text='{"status":"success","value":8}',
        )
    else:
        result = _result(
            is_error=is_error,
            text=json.dumps(payload, separators=(",", ":")),
        )

    assert _call(client, result) == payload


def test_structured_success_is_preferred_over_text(client: Any) -> None:
    result = _call(
        client,
        _result(
            is_error=False,
            structured={"status": "success", "value": 7},
            text='{"status":"success","value":8}',
        ),
    )

    assert result == {"status": "success", "value": 7}


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("true", True),
        ("42", 42),
        ('[1,"two",null]', [1, "two", None]),
        ("null", None),
    ],
    ids=["bool", "number", "list", "null"],
)
def test_json_non_object_success_preserves_legacy_result(
    client: Any,
    text: str,
    expected: Any,
) -> None:
    result = _call(client, _result(is_error=False, text=text))

    assert result == expected


def test_empty_success_text_keeps_legacy_envelope(client: Any) -> None:
    result = _call(client, _result(is_error=False, text=""))

    assert result == {"status": "success", "text": ""}


@pytest.mark.parametrize(
    "text",
    ["ordinary success text", "{malformed-json"],
    ids=["plain-text", "malformed-json"],
)
def test_non_json_success_keeps_legacy_envelope(client: Any, text: str) -> None:
    result = _call(client, _result(is_error=False, text=text))

    assert result == {"status": "success", "text": text}


def test_first_text_block_wins_after_non_text_content(client: Any) -> None:
    result = _call(
        client,
        CallToolResult(
            isError=False,
            content=[
                ImageContent(type="image", data="AA==", mimeType="image/png"),
                TextContent(type="text", text='{"value":"first"}'),
                TextContent(type="text", text='{"value":"second"}'),
            ],
        ),
    )

    assert result == {"value": "first"}


def test_real_stdio_preserves_typed_error_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Production stdio client/SDK/process path preserves the typed object."""

    observer = StdioProcessObserver()
    observer.install(monkeypatch)
    module = "tests._mcp_structured_stdio_server"
    client = MCPClient(command=sys.executable, args=["-m", module])
    try:
        result = client.call_tool(TOOL_NAME, {}, timeout=10)
        children = observer.wait_for_records(1)
        assert len(children) == 1
        child = children[0]
        assert child.command == sys.executable
        assert child.args == ("-m", module)
        assert child.pid > 0
        assert observer.records() == (child,)
        assert client.is_connected()

        assert result == {**STRUCTURED_ERROR, "status": "error"}
        assert result["status"] == "error"
        assert result["code"] == "HermeticTypedFailure"
        assert result["message"] == "real stdio preserved this error"
        assert result["retryable"] is True
        assert result["details"] == {
            "attempt": 2,
            "operation_id": "stdio-structured-1",
        }
    finally:
        cleanup_failures: list[str] = []
        client_thread = client._thread
        try:
            client.close()
        except Exception as error:  # pragma: no cover - failure-path evidence
            cleanup_failures.append(f"MCP client close failed: {error!r}")
        if not wait_for_thread_exit(client_thread):
            cleanup_failures.append("MCP client thread did not retire")
        for child in observer.records():
            if wait_for_process_exit(child):
                continue
            cleanup_failures.append(f"MCP stdio child did not retire: pid={child.pid}")
            try:
                stopped = stop_process(child)
            except Exception as error:  # pragma: no cover - failure-path evidence
                cleanup_failures.append(
                    f"MCP stdio child cleanup failed for {child.pid}: {error!r}"
                )
            else:
                if not stopped:
                    cleanup_failures.append(
                        f"MCP stdio child remains after terminate/kill: pid={child.pid}"
                    )
        assert not cleanup_failures, "; ".join(cleanup_failures)
