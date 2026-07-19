"""Real NoKV x LingTai MCP stdio smoke for the hidden experimental profile.

This is intentionally opt-in because it starts a real NoKV metadata server and
freshly builds NoKV from the supplied checkout before copying the exact output
into test-owned state. When opted in, it follows LingTai's actual
``mcp_registry.jsonl`` allow-list and
``init.json:mcp`` activation path; it does not mock an MCP client, PID, or
protocol response.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import socket
import subprocess
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from lingtai.agent import Agent
from tests._mcp_stdio_fixture import (
    StdioProcessObserver,
    stop_process,
    wait_for_process_exit,
    wait_for_thread_exit,
)
from tests._service_helpers import make_gemini_mock_service


_SMOKE_OPT_IN = "NOKV_LINGTAI_MCP_SMOKE"
_SOURCE_ENV = "NOKV_LINGTAI_MCP_SOURCE"
# A cold NoKV source build is intentionally allowed more time than the
# process-lifecycle assertions below; the latter remain tightly bounded.
_BUILD_TIMEOUT_SECONDS = 900.0
_SERVER_TIMEOUT_SECONDS = 10.0
_SERVER_START_ATTEMPTS = 3
_TEARDOWN_TIMEOUT_SECONDS = 10.0


def _require_smoke_source() -> Path:
    if os.environ.get(_SMOKE_OPT_IN) != "1":
        pytest.skip(
            f"set {_SMOKE_OPT_IN}=1 with {_SOURCE_ENV} to run the real NoKV "
            "MCP smoke"
        )

    if not os.environ.get(_SOURCE_ENV):
        pytest.fail(
            "real NoKV MCP smoke is enabled but missing required environment: "
            + _SOURCE_ENV
        )

    source = Path(os.environ[_SOURCE_ENV]).expanduser().resolve()
    contract = source / "scripts" / "lingtai-workbench" / "workbench_contract.py"
    required_files = (source / "Cargo.toml", source / "Cargo.lock", contract)
    if not source.joinpath(".git").exists() or not all(
        path.is_file() for path in required_files
    ):
        pytest.fail(f"real NoKV MCP smoke source is incomplete: {source}")
    return source


def _checkout_fingerprint(root: Path) -> tuple[str, str, bool]:
    """Return HEAD plus a content-sensitive fingerprint of a Git worktree."""

    head = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
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


def _build_smoke_binary(source: Path, fixture_root: Path) -> tuple[Path, tuple[str, str, bool]]:
    """Build then stage the exercised binary from this exact NoKV worktree state."""

    before = _checkout_fingerprint(source)
    target_dir = source / "target"
    binary_name = "nokv.exe" if os.name == "nt" else "nokv"
    try:
        completed = subprocess.run(
            [
                "cargo",
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
            "NoKV source build failed before real MCP activation:\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    after = _checkout_fingerprint(source)
    if after != before:
        pytest.fail("NoKV source checkout changed while the smoke binary was building")
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


def _load_workbench_contract(source: Path) -> ModuleType:
    path = source / "scripts" / "lingtai-workbench" / "workbench_contract.py"
    spec = importlib.util.spec_from_file_location("nokv_workbench_contract", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load NoKV frozen workbench contract from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_server(server: subprocess.Popen[bytes], endpoint: tuple[str, int], log: Path) -> None:
    deadline = time.monotonic() + _SERVER_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if server.poll() is not None:
            output = log.read_text(encoding="utf-8", errors="replace")
            raise AssertionError(
                f"NoKV metadata server exited early ({server.returncode}): {output}"
            )
        try:
            with socket.create_connection(endpoint, timeout=0.2):
                return
        except OSError:
            pass
    output = log.read_text(encoding="utf-8", errors="replace")
    raise AssertionError(f"NoKV metadata server did not listen on {endpoint}: {output}")


def _stop_server(server: subprocess.Popen[bytes], *, timeout: float = _TEARDOWN_TIMEOUT_SECONDS) -> None:
    if server.poll() is not None:
        return
    server.terminate()
    try:
        server.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        server.kill()
        server.wait(timeout=timeout)


def _start_server(
    *,
    binary: Path,
    meta_dir: Path,
    object_root: Path,
    server_log: Path,
) -> tuple[subprocess.Popen[bytes], str, str]:
    """Start an isolated server, retrying the released-port bind race."""

    last_bind_error: AssertionError | None = None
    for attempt in range(_SERVER_START_ATTEMPTS):
        port = _free_loopback_port()
        endpoint = f"127.0.0.1:{port}"
        bucket = f"nokv-lingtai-smoke-{port}"
        with server_log.open("ab" if attempt else "wb") as output:
            server = subprocess.Popen(
                [
                    str(binary),
                    "--meta",
                    str(meta_dir),
                    "--server-bind",
                    endpoint,
                    "--object-backend",
                    "rustfs",
                    "--s3-bucket",
                    bucket,
                    "--s3-endpoint",
                    "http://127.0.0.1:9",
                    "--hot-object-root",
                    str(object_root),
                    "--no-metadata-checkpoint-archive",
                    "--object-gc-interval-ms",
                    "600000",
                    "serve",
                ],
                stdout=output,
                stderr=subprocess.STDOUT,
            )
        try:
            _wait_for_server(server, ("127.0.0.1", port), server_log)
        except AssertionError as error:
            _stop_server(server)
            output_text = server_log.read_text(encoding="utf-8", errors="replace")
            if "address already in use" not in output_text.lower() and "eaddrinuse" not in output_text.lower():
                raise error
            last_bind_error = error
            continue
        return server, endpoint, bucket
    raise AssertionError(
        f"NoKV metadata server lost {_SERVER_START_ATTEMPTS} loopback port races: "
        f"{last_bind_error}"
    )


def _registered_activation(
    *,
    binary: Path,
    endpoint: str,
    bucket: str,
    object_root: Path,
) -> tuple[dict[str, Any], list[str]]:
    args = [
        "--server-bind",
        endpoint,
        "--object-backend",
        "rustfs",
        "--s3-bucket",
        bucket,
        "--s3-endpoint",
        "http://127.0.0.1:9",
        "--hot-object-root",
        str(object_root),
        "mcp",
        "--profile",
        "lingtai",
        "--workbench-root",
        "/agents/{agent_id}/wb",
    ]
    record = {
        "name": "nokv-smoke",
        "summary": "Real NoKV MCP smoke fixture",
        "transport": "stdio",
        "command": str(binary),
        "args": args,
        "source": "test",
    }
    return record, args


def test_registered_lingtai_profile_runs_frozen_workbench_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    record_property: pytest.RecordProperty,
) -> None:
    """Integration/lifecycle: real registered stdio activation and teardown."""

    source = _require_smoke_source()
    contract = _load_workbench_contract(source)
    kernel_root = Path(__file__).resolve().parents[1]
    kernel_head, kernel_fingerprint, kernel_dirty = _checkout_fingerprint(kernel_root)
    record_property("lingtai_kernel_head", kernel_head)
    record_property("lingtai_kernel_dirty", str(kernel_dirty).lower())
    record_property("lingtai_kernel_fingerprint", kernel_fingerprint)

    fixture_root = tmp_path / "nokv-lingtai-mcp-smoke"
    meta_dir = fixture_root / "metadata"
    object_root = fixture_root / "objects"
    server_log = fixture_root / "nokv-server.log"
    fixture_root.mkdir()

    observer = StdioProcessObserver()
    observer.install(monkeypatch)
    server: subprocess.Popen[bytes] | None = None
    agent: Agent | None = None
    client: Any = None
    try:
        binary, (source_head, source_fingerprint, source_dirty) = _build_smoke_binary(
            source, fixture_root
        )
        record_property(
            "nokv_cargo_version",
            subprocess.check_output(
                ["cargo", "--version"], cwd=source, text=True
            ).strip(),
        )
        record_property(
            "nokv_rustc_version",
            subprocess.check_output(
                ["rustc", "--version"], cwd=source, text=True
            ).strip(),
        )
        record_property("nokv_source_head", source_head)
        record_property("nokv_source_dirty", str(source_dirty).lower())
        record_property("nokv_source_fingerprint", source_fingerprint)
        record_property("nokv_binary", str(binary))
        record_property("nokv_binary_sha256", hashlib.sha256(binary.read_bytes()).hexdigest())

        agent_dir = fixture_root / ".lingtai" / "fixture-agent"
        agent_dir.mkdir(parents=True)
        server, endpoint, bucket = _start_server(
            binary=binary,
            meta_dir=meta_dir,
            object_root=object_root,
            server_log=server_log,
        )
        registry_record, activation_args = _registered_activation(
            binary=binary,
            endpoint=endpoint,
            bucket=bucket,
            object_root=object_root,
        )
        registry_path = agent_dir / "mcp_registry.jsonl"
        init_path = agent_dir / "init.json"
        registry_path.write_text(json.dumps(registry_record) + "\n", encoding="utf-8")
        init_path.write_text(
            json.dumps(
                {
                    "mcp": {
                        registry_record["name"]: {
                            "type": "stdio",
                            "command": str(binary),
                            "args": activation_args,
                        }
                    }
                }
            ),
            encoding="utf-8",
        )

        # Exactly one allow-list record and one matching init.json launch; the
        # legacy source must be absent so no second activation can occur.
        stored_registry = [
            json.loads(line) for line in registry_path.read_text(encoding="utf-8").splitlines()
        ]
        stored_init = json.loads(init_path.read_text(encoding="utf-8"))
        assert stored_registry == [registry_record]
        assert list(stored_init["mcp"]) == ["nokv-smoke"]
        assert stored_init["mcp"]["nokv-smoke"]["command"] == registry_record["command"]
        assert stored_init["mcp"]["nokv-smoke"]["args"] == registry_record["args"]
        assert not (agent_dir / "mcp" / "servers.json").exists()

        agent = Agent(
            service=make_gemini_mock_service(),
            agent_name="fixture-agent",
            working_dir=agent_dir,
            capabilities={"mcp": {}},
        )
        client = agent._mcp_init_specs["nokv-smoke"]["client"]
        assert client is not None and client.is_connected()

        children = observer.wait_for_records(1)
        assert len(children) == 1
        child = children[0]
        assert child.command == str(binary)
        expected_child_args = tuple(
            value.replace("{agent_id}", agent_dir.name)
            if isinstance(value, str)
            else value
            for value in activation_args
        )
        assert child.args == expected_child_args
        assert "/agents/fixture-agent/wb" in child.args

        # Compare the raw MCP list before LingTai strips top-level
        # additionalProperties while registering FunctionSchema instances.
        raw_tools = client.list_tools()
        contract.validate_tool_contract(
            [
                {"name": tool["name"], "inputSchema": tool["schema"]}
                for tool in raw_tools
            ]
        )
        contract.validate_tool_order(raw_tools)

        created = client.call_tool("workbench_create", {"id": "smoke"})
        expected_root = "/agents/fixture-agent/wb"
        assert created["status"] == "success"
        assert created["path"] == f"{expected_root}/smoke"

        listed = client.call_tool("workbench_list", {"id": "smoke"})
        assert listed["status"] == "success"
        assert listed["path"] == f"{expected_root}/smoke"

        outside = client.call_tool(
            "workbench_list",
            {"id": "smoke", "section": "outputs", "path": "../../outside"},
        )
        assert outside["status"] == "error"
        assert outside["message"]
        assert client.is_connected(), "ordinary tool error must not kill the MCP client"
        assert observer.records() == (child,), "one activation must remain one child"
    finally:
        cleanup_failures: list[str] = []
        client_thread = getattr(client, "_thread", None)
        agent_stopped = agent is None
        if agent is not None:
            try:
                agent.stop()
                agent_stopped = True
            except Exception as error:  # pragma: no cover - failure-path evidence
                cleanup_failures.append(f"Agent.stop failed: {error!r}")
        if client is not None and not agent_stopped:
            try:
                client.close()
            except Exception as error:  # pragma: no cover - failure-path evidence
                cleanup_failures.append(f"MCP client close failed: {error!r}")
        if not wait_for_thread_exit(client_thread):
            cleanup_failures.append("MCP client thread did not retire")
        for launched_child in observer.records():
            if wait_for_process_exit(launched_child):
                continue
            cleanup_failures.append(
                f"NoKV MCP child process did not retire: pid={launched_child.pid}"
            )
            try:
                stopped = stop_process(launched_child)
            except Exception as error:  # pragma: no cover - failure-path evidence
                cleanup_failures.append(
                    f"NoKV MCP child cleanup failed for {launched_child.pid}: {error!r}"
                )
            else:
                if not stopped:
                    cleanup_failures.append(
                        f"NoKV MCP child remains after terminate/kill: pid={launched_child.pid}"
                    )
        if server is not None:
            try:
                _stop_server(server)
            except Exception as error:  # pragma: no cover - failure-path evidence
                cleanup_failures.append(f"NoKV metadata server stop failed: {error!r}")
            if server.poll() is None:
                cleanup_failures.append("NoKV metadata server did not retire")
        try:
            shutil.rmtree(fixture_root)
        except OSError as error:  # pragma: no cover - failure-path evidence
            cleanup_failures.append(f"fixture removal failed: {error!r}")
        if fixture_root.exists():
            cleanup_failures.append("fixture root remains after teardown")
        assert not cleanup_failures, "; ".join(cleanup_failures)
