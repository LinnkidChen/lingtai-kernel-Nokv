"""Opt-in built-NoKV broad-rollout gate for structured MCP errors.

This gate is deliberately separate from the core NoKV x LingTai smoke. It
builds the supplied NoKV checkout, captures the typed Workbench error at the
MCP SDK boundary, then compares it with LingTai's public ``MCPClient`` result.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import subprocess
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from lingtai.services.mcp import MCPClient
from tests._mcp_stdio_fixture import (
    StdioProcessObserver,
    stop_process,
    wait_for_process_exit,
    wait_for_thread_exit,
)


_GATE_OPT_IN = "NOKV_LINGTAI_MCP_STRUCTURED_ERROR"
_SOURCE_ENV = "NOKV_LINGTAI_MCP_SOURCE"
_CARGO_TOOLCHAIN_ENV = "NOKV_LINGTAI_CARGO_TOOLCHAIN"
_BUILD_TIMEOUT_SECONDS = 900.0
_CALL_TIMEOUT_SECONDS = 10.0
_TEARDOWN_TIMEOUT_SECONDS = 10.0
_TOOL_NAME = "workbench_commit"
_TOOL_ARGUMENTS = {
    "id": "structured-error-probe",
    "manifest": {},
    "content_digest_uri": "sha256:ABC",
}
_EXPECTED_ERROR = {
    "status": "error",
    "code": "InvalidContentDigestUri",
    "message": "content_digest_uri must exactly match sha256:<64 lowercase hex>",
    "retryable": False,
    "details": {
        "field": "content_digest_uri",
        "expected_pattern": "^sha256:[0-9a-f]{64}$",
        "actual": "sha256:ABC",
    },
}
_PRESERVED_FIELDS = ("status", "code", "message", "retryable", "details")


def _require_source() -> Path:
    if os.environ.get(_GATE_OPT_IN) != "1":
        pytest.skip(
            f"set {_GATE_OPT_IN}=1 with {_SOURCE_ENV} to run the built-NoKV "
            "structured-error gate"
        )
    if not os.environ.get(_SOURCE_ENV):
        pytest.fail(
            "built-NoKV structured-error gate is enabled but missing required "
            f"environment: {_SOURCE_ENV}"
        )

    source = Path(os.environ[_SOURCE_ENV]).expanduser().resolve()
    required_files = (
        source / "Cargo.toml",
        source / "Cargo.lock",
        source / "crates" / "nokv" / "src" / "bin" / "nokv.rs",
    )
    if not source.joinpath(".git").exists() or not all(
        path.is_file() for path in required_files
    ):
        pytest.fail(f"built-NoKV structured-error source is incomplete: {source}")
    return source


def _checkout_fingerprint(root: Path) -> tuple[str, str, bool]:
    """Return HEAD plus a content-sensitive fingerprint of a Git worktree."""

    head = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    status = subprocess.check_output(
        [
            "git",
            "-C",
            str(root),
            "status",
            "--porcelain=v1",
            "--untracked-files=normal",
        ]
    )
    diff = subprocess.check_output(
        ["git", "-C", str(root), "diff", "--binary", "HEAD"]
    )
    untracked = subprocess.check_output(
        ["git", "-C", str(root), "ls-files", "--others", "--exclude-standard", "-z"]
    )
    digest = hashlib.sha256()
    for value in (head.encode("ascii"), status, diff):
        digest.update(value)
        digest.update(b"\0")
    for encoded_name in filter(None, untracked.split(b"\0")):
        path = root / os.fsdecode(encoded_name)
        digest.update(encoded_name)
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return head, digest.hexdigest(), bool(status)


def _cargo_command() -> list[str]:
    """Return Cargo plus an optional explicit rustup toolchain selector."""

    command = ["cargo"]
    toolchain = os.environ.get(_CARGO_TOOLCHAIN_ENV)
    if toolchain:
        command.append(f"+{toolchain}")
    return command


def _build_binary(
    source: Path,
    fixture_root: Path,
) -> tuple[Path, tuple[str, str, bool]]:
    """Build and stage the exact binary from one unchanged NoKV checkout."""

    before = _checkout_fingerprint(source)
    if before[2]:
        pytest.fail(f"built-NoKV structured-error source must be clean: {source}")
    target_dir = source / "target"
    binary_name = "nokv.exe" if os.name == "nt" else "nokv"
    try:
        completed = subprocess.run(
            [
                *_cargo_command(),
                "build",
                "--locked",
                "--target-dir",
                str(target_dir),
                "--manifest-path",
                str(source / "Cargo.toml"),
                "-p",
                "nokv",
                "--bin",
                "nokv",
            ],
            cwd=source,
            check=False,
            capture_output=True,
            text=True,
            timeout=_BUILD_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as error:
        pytest.fail(
            f"NoKV source build exceeded {_BUILD_TIMEOUT_SECONDS:.0f}s: {error}"
        )
    if completed.returncode != 0:
        pytest.fail(
            "NoKV source build failed before structured-error activation:\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )

    after = _checkout_fingerprint(source)
    if after != before:
        pytest.fail("NoKV source checkout changed while the gate binary was building")
    compiled_binary = target_dir / "debug" / binary_name
    if not compiled_binary.is_file() or not os.access(compiled_binary, os.X_OK):
        pytest.fail(f"NoKV source build did not produce an executable: {compiled_binary}")
    binary = fixture_root / binary_name
    shutil.copy2(compiled_binary, binary)
    if hashlib.sha256(binary.read_bytes()).digest() != hashlib.sha256(
        compiled_binary.read_bytes()
    ).digest():
        pytest.fail("test-owned NoKV binary copy does not match the source build")
    return binary.resolve(), after


async def _raw_boundary_result(
    binary: Path,
    activation_args: list[str],
) -> Any:
    params = StdioServerParameters(command=str(binary), args=activation_args)
    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            return await session.call_tool(
                _TOOL_NAME,
                _TOOL_ARGUMENTS,
                read_timeout_seconds=timedelta(seconds=_CALL_TIMEOUT_SECONDS),
            )


def _activation_args(object_root: Path) -> list[str]:
    # Both endpoints are intentionally unreachable. The selected validation
    # error is produced before the first metadata or object-store operation.
    return [
        "--server-bind",
        "127.0.0.1:9",
        "--object-backend",
        "rustfs",
        "--s3-bucket",
        "nokv-lingtai-structured-error",
        "--s3-endpoint",
        "http://127.0.0.1:9",
        "--hot-object-root",
        str(object_root),
        "mcp",
        "--profile",
        "lingtai",
        "--workbench-root",
        "/agents/structured-error-probe/wb",
        "--workspace-id",
        "structured-rollout",
        "--workspace-actor-id",
        "structured-client",
        "--workspace-dev-membership",
        "reader",
    ]


def test_built_nokv_typed_error_reaches_lingtai_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    record_property: pytest.RecordProperty,
) -> None:
    """Compare NoKV's raw typed error with LingTai's public client result."""

    source = _require_source()
    fixture_root = tmp_path / "nokv-lingtai-structured-error"
    object_root = fixture_root / "objects"
    fixture_root.mkdir()
    object_root.mkdir()

    observer = StdioProcessObserver()
    observer.install(monkeypatch)
    client: MCPClient | None = None
    try:
        binary, (source_head, source_fingerprint, source_dirty) = _build_binary(
            source,
            fixture_root,
        )
        binary_sha256 = hashlib.sha256(binary.read_bytes()).hexdigest()
        record_property("nokv_source_head", source_head)
        record_property("nokv_source_dirty", str(source_dirty).lower())
        record_property("nokv_source_fingerprint", source_fingerprint)
        record_property("nokv_binary", str(binary))
        record_property("nokv_binary_sha256", binary_sha256)
        cargo_command = _cargo_command()
        record_property(
            "nokv_cargo_version",
            subprocess.check_output(
                [*cargo_command, "--version"], cwd=source, text=True
            ).strip(),
        )
        cargo_toolchain = os.environ.get(_CARGO_TOOLCHAIN_ENV)
        rustc_command = (
            ["rustup", "run", cargo_toolchain, "rustc", "--version"]
            if cargo_toolchain
            else ["rustc", "--version"]
        )
        record_property(
            "nokv_rustc_version",
            subprocess.check_output(rustc_command, cwd=source, text=True).strip(),
        )

        activation_args = _activation_args(object_root)
        raw = asyncio.run(_raw_boundary_result(binary, activation_args))
        raw_children = observer.wait_for_records(1)
        assert len(raw_children) == 1
        assert wait_for_process_exit(raw_children[0], timeout=_TEARDOWN_TIMEOUT_SECONDS)
        assert raw.isError is True
        assert raw.structuredContent == _EXPECTED_ERROR
        assert json.loads(raw.content[0].text) == _EXPECTED_ERROR

        client = MCPClient(command=str(binary), args=activation_args)
        observed = client.call_tool(
            _TOOL_NAME,
            _TOOL_ARGUMENTS,
            timeout=_CALL_TIMEOUT_SECONDS,
        )
        children = observer.wait_for_records(2)
        assert len(children) == 2
        assert all(child.command == str(binary) for child in children)
        assert all(child.args == tuple(activation_args) for child in children)
        assert len({child.pid for child in children}) == 2
        assert observer.records() == children
        assert client.is_connected()

        assert observed == _EXPECTED_ERROR
        assert {
            field: observed[field] for field in _PRESERVED_FIELDS
        } == {
            field: raw.structuredContent[field] for field in _PRESERVED_FIELDS
        }
        assert client.get_activity_log()[-1]["result"] == _EXPECTED_ERROR
    finally:
        cleanup_failures: list[str] = []
        client_thread = None if client is None else client._thread
        if client is not None:
            try:
                client.close()
            except Exception as error:  # pragma: no cover - failure-path evidence
                cleanup_failures.append(f"MCP client close failed: {error!r}")
        if not wait_for_thread_exit(client_thread, timeout=_TEARDOWN_TIMEOUT_SECONDS):
            cleanup_failures.append("MCP client thread did not retire")
        for child in observer.records():
            if wait_for_process_exit(child, timeout=_TEARDOWN_TIMEOUT_SECONDS):
                continue
            cleanup_failures.append(f"NoKV MCP child did not retire: pid={child.pid}")
            try:
                stopped = stop_process(child, timeout=_TEARDOWN_TIMEOUT_SECONDS)
            except Exception as error:  # pragma: no cover - failure-path evidence
                cleanup_failures.append(
                    f"NoKV MCP child cleanup failed for {child.pid}: {error!r}"
                )
            else:
                if not stopped:
                    cleanup_failures.append(
                        f"NoKV MCP child remains after terminate/kill: pid={child.pid}"
                    )
        assert not cleanup_failures, "; ".join(cleanup_failures)
