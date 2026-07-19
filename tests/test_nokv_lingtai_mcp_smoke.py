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
import signal
import socket
import stat
import subprocess
import sys
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
_BUILD_TERMINATE_TIMEOUT_SECONDS = 5.0
_BUILD_KILL_TIMEOUT_SECONDS = 5.0
_SERVER_TIMEOUT_SECONDS = 10.0
_SERVER_START_ATTEMPTS = 3
_TEARDOWN_TIMEOUT_SECONDS = 10.0
_WORKSPACE_ID = "kernel-smoke-{agent_id}"
_WORKSPACE_ACTOR_ID = "kernel-smoke-{agent_dir}"
_WORKSPACE_ROLE = "reader"
_EXPECTED_READER_TOOL_COUNT = 20
_EXPECTED_READER_RAW_CONTRACT_SHA256 = (
    "e008fc0a776c3348ec0ddae3db9eebc01ea37eed3b723a86004eae110d94fc2f"
)


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
    script_dir = source / "scripts" / "lingtai-workbench"
    required_files = (
        source / "Cargo.toml",
        source / "Cargo.lock",
        script_dir / "workbench_contract.py",
        script_dir / "workbench_contract_schema.json",
        script_dir / "lingtai_contract_schema.json",
        script_dir / "workspace_grant.py",
    )
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


def _source_paths(root: Path) -> tuple[Path, ...]:
    raw = subprocess.check_output(
        [
            "git",
            "-C",
            str(root),
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
        ]
    )
    paths = tuple(
        sorted(
            (Path(os.fsdecode(value)) for value in raw.split(b"\0") if value),
            key=lambda path: os.fsencode(path.as_posix()),
        )
    )
    if not paths:
        pytest.fail(f"NoKV checkout has no tracked source files: {root}")
    for path in paths:
        if path.is_absolute() or ".." in path.parts:
            pytest.fail(f"NoKV checkout reported an unsafe source path: {path}")
    return paths


def _source_manifest_sha256(root: Path, paths: tuple[Path, ...]) -> str:
    """Hash the complete non-ignored source snapshot, including file modes."""

    digest = hashlib.sha256()
    for relative in paths:
        path = root / relative
        metadata = path.lstat()
        digest.update(os.fsencode(relative.as_posix()))
        digest.update(b"\0")
        if stat.S_ISLNK(metadata.st_mode):
            digest.update(b"symlink\0")
            digest.update(os.fsencode(os.readlink(path)))
        elif stat.S_ISREG(metadata.st_mode):
            digest.update(b"regular\0")
            digest.update(str(stat.S_IMODE(metadata.st_mode)).encode("ascii"))
            digest.update(b"\0")
            with path.open("rb") as handle:
                while chunk := handle.read(1 << 20):
                    digest.update(chunk)
        else:
            pytest.fail(f"NoKV source entry is not a regular file or symlink: {path}")
        digest.update(b"\0")
    return digest.hexdigest()


def _copy_source_snapshot(
    source: Path,
    snapshot: Path,
    paths: tuple[Path, ...],
) -> str:
    if snapshot.exists():
        pytest.fail(f"NoKV test-owned source snapshot already exists: {snapshot}")
    snapshot.mkdir()
    for relative in paths:
        source_path = source / relative
        target_path = snapshot / relative
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if source_path.is_symlink():
            target_path.symlink_to(os.readlink(source_path))
        else:
            shutil.copy2(source_path, target_path)
    source_manifest = _source_manifest_sha256(source, paths)
    snapshot_manifest = _source_manifest_sha256(snapshot, paths)
    if snapshot_manifest != source_manifest:
        pytest.fail("NoKV test-owned source snapshot differs from the selected checkout")
    return snapshot_manifest


def _process_group_exists(pgid: int) -> bool:
    if os.name != "posix":
        return False
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_for_process_group_exit(pgid: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _process_group_exists(pgid):
            return True
        time.sleep(0.05)
    return not _process_group_exists(pgid)


def _terminate_build_process(process: subprocess.Popen[str]) -> list[str]:
    """Bounded process-tree cleanup that preserves the primary build failure."""

    failures: list[str] = []
    if os.name == "posix":
        pgid = process.pid
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except OSError as error:
            failures.append(f"could not terminate Cargo process group: {error}")
        try:
            process.wait(timeout=_BUILD_TERMINATE_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            leader_reaped = False
        else:
            leader_reaped = True
        group_retired = leader_reaped and _wait_for_process_group_exit(
            pgid, _BUILD_TERMINATE_TIMEOUT_SECONDS
        )
        if not group_retired:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except OSError as error:
                failures.append(f"could not kill Cargo process group: {error}")
            if process.poll() is None:
                try:
                    process.wait(timeout=_BUILD_KILL_TIMEOUT_SECONDS)
                except subprocess.TimeoutExpired:
                    failures.append(f"Cargo process {process.pid} was not reaped")
            if not _wait_for_process_group_exit(pgid, _BUILD_KILL_TIMEOUT_SECONDS):
                failures.append(f"Cargo process group {pgid} did not retire")
    elif process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=_BUILD_TERMINATE_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            process.kill()
            try:
                process.wait(timeout=_BUILD_KILL_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                failures.append(f"Cargo process {process.pid} did not retire")
    if process.poll() is None:
        try:
            process.wait(timeout=_BUILD_KILL_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            failures.append(f"Cargo process {process.pid} was not reaped")
    return failures


def _build_smoke_binary(
    source: Path,
    fixture_root: Path,
) -> tuple[Path, tuple[str, str, bool], Path, str]:
    """Build from one verified test-owned source snapshot and empty target."""

    before = _checkout_fingerprint(source)
    source_paths = _source_paths(source)
    source_snapshot = fixture_root / "source-snapshot"
    source_manifest = _copy_source_snapshot(source, source_snapshot, source_paths)
    if _checkout_fingerprint(source) != before:
        pytest.fail("NoKV source checkout changed while its test snapshot was copied")
    target_dir = fixture_root / "cargo-target"
    if target_dir.exists():
        pytest.fail(f"NoKV test-owned Cargo target is not empty: {target_dir}")
    binary_name = "nokv.exe" if os.name == "nt" else "nokv"
    process = subprocess.Popen(
        [
            "cargo",
            "build",
            "--locked",
            "--target-dir",
            str(target_dir),
            "--manifest-path",
            str(source_snapshot / "Cargo.toml"),
            "-p",
            "nokv",
            "--bin",
            "nokv",
        ],
        cwd=source_snapshot,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=_BUILD_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired as error:
        cleanup_failures = _terminate_build_process(process)
        pytest.fail(
            f"NoKV source build exceeded {_BUILD_TIMEOUT_SECONDS:.0f}s: {error}; "
            f"cleanup={cleanup_failures or 'complete'}"
        )
    except BaseException:
        _terminate_build_process(process)
        raise
    if process.returncode != 0:
        cleanup_failures = _terminate_build_process(process)
        pytest.fail(
            "NoKV source build failed before real MCP activation:\n"
            f"stdout:\n{stdout}\nstderr:\n{stderr}\n"
            f"cleanup={cleanup_failures or 'complete'}"
        )
    after = _checkout_fingerprint(source)
    if after != before:
        pytest.fail("NoKV source checkout changed while the smoke binary was building")
    if _source_paths(source) != source_paths:
        pytest.fail("NoKV source file set changed while the smoke binary was building")
    if _source_manifest_sha256(source, source_paths) != source_manifest:
        pytest.fail("NoKV source content changed after its test snapshot was copied")
    compiled_binary = target_dir / "debug" / binary_name
    if not compiled_binary.is_file() or not os.access(compiled_binary, os.X_OK):
        pytest.fail(f"NoKV source build did not produce an executable: {compiled_binary}")
    binary = fixture_root / binary_name
    shutil.copy2(compiled_binary, binary)
    if hashlib.sha256(binary.read_bytes()).digest() != hashlib.sha256(
        compiled_binary.read_bytes()
    ).digest():
        pytest.fail("test-owned NoKV binary copy does not match the source build")
    return binary.resolve(), after, source_snapshot, source_manifest


def _load_source_module(source: Path, name: str) -> ModuleType:
    path = source / "scripts" / "lingtai-workbench" / f"{name}.py"
    source_suffix = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:12]
    qualified_name = f"nokv_lingtai_smoke_{name}_{source_suffix}"
    spec = importlib.util.spec_from_file_location(qualified_name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load NoKV source module from {path}")
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves annotation metadata through sys.modules while the
    # module is executing. Keep this source-qualified entry for the test run so
    # a different checkout can never reuse a stale grant implementation.
    sys.modules[qualified_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(qualified_name, None)
        raise
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
    workspace_grant: str,
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
        "--workspace-id",
        _WORKSPACE_ID,
        "--workspace-actor-id",
        _WORKSPACE_ACTOR_ID,
        "--workspace-grant",
        workspace_grant,
    ]
    template_arg_indices = [args.index("--workbench-root") + 1]
    record = {
        "name": "nokv-smoke",
        "summary": "Real NoKV MCP smoke fixture",
        "transport": "stdio",
        "command": str(binary),
        "args": args,
        "template_arg_indices": template_arg_indices,
        "source": "test",
    }
    return record, args


def _safe_activation_args_evidence(
    args: list[str] | tuple[str, ...],
) -> tuple[tuple[str, ...], str]:
    """Return redacted args plus a value-sensitive digest for safe assertions."""

    values = tuple(args)
    redacted = list(values)
    for index, value in enumerate(redacted):
        if value == "--workspace-grant" and index + 1 < len(redacted):
            redacted[index + 1] = "<redacted-workspace-grant>"
    digest = hashlib.sha256(
        json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return tuple(redacted), digest


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group regression")
def test_build_cleanup_retires_cargo_descendants() -> None:
    child_code = "import time; time.sleep(60)"
    parent_code = (
        "import subprocess,sys,time; "
        f"child=subprocess.Popen([sys.executable,'-c',{child_code!r}]); "
        "print(child.pid,flush=True); time.sleep(60)"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", parent_code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    assert process.stdout is not None
    child_pid = int(process.stdout.readline().strip())
    assert _terminate_build_process(process) == []
    assert process.poll() is not None
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)


def test_registered_lingtai_reader_profile_runs_frozen_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    record_property: pytest.RecordProperty,
) -> None:
    """Integration/lifecycle: real registered stdio activation and teardown."""

    source = _require_smoke_source()
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
        (
            binary,
            (source_head, source_fingerprint, source_dirty),
            source_snapshot,
            source_manifest,
        ) = _build_smoke_binary(source, fixture_root)
        contract = _load_source_module(source_snapshot, "workbench_contract")
        workspace_grant = _load_source_module(source_snapshot, "workspace_grant")
        expected_contract = contract.expected_profile_contract_evidence(
            "lingtai", role=_WORKSPACE_ROLE
        )
        assert expected_contract["tool_count"] == _EXPECTED_READER_TOOL_COUNT
        assert (
            expected_contract["raw_contract_sha256"]
            == _EXPECTED_READER_RAW_CONTRACT_SHA256
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
        record_property("nokv_source_manifest_sha256", source_manifest)
        record_property("nokv_binary", str(binary))
        record_property("nokv_binary_sha256", hashlib.sha256(binary.read_bytes()).hexdigest())
        contract_asset = (
            source_snapshot
            / "scripts"
            / "lingtai-workbench"
            / "lingtai_contract_schema.json"
        )
        record_property(
            "nokv_lingtai_contract_asset_sha256",
            hashlib.sha256(contract_asset.read_bytes()).hexdigest(),
        )

        agent_dir = fixture_root / ".lingtai" / "fixture-agent"
        agent_dir.mkdir(parents=True)
        server, endpoint, bucket = _start_server(
            binary=binary,
            meta_dir=meta_dir,
            object_root=object_root,
            server_log=server_log,
        )
        now_unix_ms = time.time_ns() // 1_000_000
        grant = workspace_grant.WorkspaceGrant(
            schema=workspace_grant.GRANT_SCHEMA,
            grant_id="kernel-smoke-reader",
            issuer=workspace_grant.GRANT_ISSUER,
            audience=workspace_grant.GRANT_AUDIENCE,
            workspace_id=_WORKSPACE_ID,
            actor_id=_WORKSPACE_ACTOR_ID,
            role=_WORKSPACE_ROLE,
            issued_at_unix_ms=now_unix_ms - 1_000,
            expires_at_unix_ms=now_unix_ms + 30 * 60 * 1_000,
        )
        encoded_grant = workspace_grant.encode_workspace_grant(grant)
        record_property("nokv_mcp_profile", "lingtai")
        record_property("nokv_workspace_role", _WORKSPACE_ROLE)
        record_property(
            "nokv_workspace_id_sha256",
            hashlib.sha256(_WORKSPACE_ID.encode("utf-8")).hexdigest(),
        )
        record_property(
            "nokv_workspace_actor_id_sha256",
            hashlib.sha256(_WORKSPACE_ACTOR_ID.encode("utf-8")).hexdigest(),
        )
        record_property(
            "nokv_workspace_grant_sha256",
            workspace_grant.workspace_grant_sha256(grant),
        )
        registry_record, activation_args = _registered_activation(
            binary=binary,
            endpoint=endpoint,
            bucket=bucket,
            object_root=object_root,
            workspace_grant=encoded_grant,
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
                            "template_arg_indices": registry_record[
                                "template_arg_indices"
                            ],
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
        assert len(stored_registry) == 1
        stored_record = stored_registry[0]
        assert {key: value for key, value in stored_record.items() if key != "args"} == {
            key: value for key, value in registry_record.items() if key != "args"
        }
        assert _safe_activation_args_evidence(stored_record["args"]) == (
            _safe_activation_args_evidence(registry_record["args"])
        )
        assert list(stored_init["mcp"]) == ["nokv-smoke"]
        assert stored_init["mcp"]["nokv-smoke"]["command"] == registry_record["command"]
        assert _safe_activation_args_evidence(
            stored_init["mcp"]["nokv-smoke"]["args"]
        ) == _safe_activation_args_evidence(registry_record["args"])
        assert stored_init["mcp"]["nokv-smoke"]["template_arg_indices"] == (
            registry_record["template_arg_indices"]
        )
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
        template_arg_indices = set(registry_record["template_arg_indices"])
        expected_child_args = tuple(
            value.replace("{agent_id}", agent_dir.name)
            if index in template_arg_indices
            else value
            for index, value in enumerate(activation_args)
        )
        child_args_evidence = _safe_activation_args_evidence(child.args)
        assert child_args_evidence == _safe_activation_args_evidence(expected_child_args)
        record_property("nokv_mcp_activation_args_sha256", child_args_evidence[1])
        safe_child_args = child_args_evidence[0]
        assert "/agents/fixture-agent/wb" in safe_child_args
        assert safe_child_args.count("--workspace-id") == 1
        assert safe_child_args.count("--workspace-actor-id") == 1
        assert safe_child_args.count("--workspace-grant") == 1
        assert safe_child_args[safe_child_args.index("--workspace-id") + 1] == (
            _WORKSPACE_ID
        )
        assert safe_child_args[
            safe_child_args.index("--workspace-actor-id") + 1
        ] == _WORKSPACE_ACTOR_ID

        # Compare the raw MCP list before LingTai strips top-level
        # additionalProperties while registering FunctionSchema instances.
        raw_tools = client.list_tools()
        live_contract = contract.profile_contract_evidence(
            raw_tools,
            "lingtai",
            role=_WORKSPACE_ROLE,
            schema_key="schema",
        )
        assert live_contract == expected_contract
        assert len(raw_tools) == _EXPECTED_READER_TOOL_COUNT
        assert live_contract["raw_contract_sha256"] == (
            _EXPECTED_READER_RAW_CONTRACT_SHA256
        )
        record_property("nokv_mcp_tool_count", str(len(raw_tools)))
        record_property(
            "nokv_mcp_raw_contract_sha256",
            live_contract["raw_contract_sha256"],
        )
        record_property(
            "nokv_mcp_semantic_contract_sha256",
            live_contract["contract_sha256"],
        )
        record_property(
            "nokv_mcp_contract_sha256",
            live_contract["contract_sha256"],
        )
        tool_names = [tool["name"] for tool in raw_tools]
        assert tool_names[-2:] == ["workspace_list", "workspace_read"]
        assert "workspace_put_file" not in tool_names
        assert "workspace_edit" not in tool_names
        assert "workspace_append" not in tool_names

        shared_root = client.call_tool("workspace_list", {})
        assert shared_root["status"] == "success"
        assert shared_root["workspace_id"] == _WORKSPACE_ID
        assert shared_root["path"] == ""
        assert shared_root["entries"] == []
        assert shared_root["total_entry_count"] == 0

        denied_write = client.call_tool(
            "workspace_put_file",
            {
                "path": "reader-must-not-write.txt",
                "operation_id": "kernel-smoke-denied-write",
                "base_generation": None,
                "text": "blocked",
            },
        )
        assert denied_write["status"] == "error"
        assert "WorkspacePermissionDenied" in denied_write["message"]
        unchanged_root = client.call_tool("workspace_list", {})
        assert unchanged_root == shared_root
        assert client.is_connected(), "permission denial must not kill the MCP client"

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
        remaining_children = observer.records()
        assert len(remaining_children) == 1, "one activation must remain one child"
        assert remaining_children[0].pid == child.pid
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
