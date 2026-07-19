"""Convergence gates for MCP stop, deep refresh, and refresh handoff."""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from lingtai.agent import Agent
from lingtai.kernel.llm.base import FunctionSchema
from lingtai.tools.system.preset import _refresh
from tests._service_helpers import make_gemini_mock_service
from tests._refresh_watcher_helpers import make_test_refresh_watcher
from tests._workdir_lease_helpers import RecordingWorkdirLease, make_test_lease
from tests.test_deep_refresh import _make_init


_ABSENT = object()


def _system_refresh_agent(
    tmp_path: Path,
    *,
    retry: object = _ABSENT,
    cleanup: object = _ABSENT,
) -> tuple[SimpleNamespace, list[dict], list[dict], list[None]]:
    """Build the smallest system-refresh surface and record lifecycle calls."""
    logs: list[dict] = []
    cleanup_calls: list[dict] = []
    handoffs: list[None] = []
    perform_attempts: list[dict] = []

    def _perform_refresh(**kwargs):
        perform_attempts.append(kwargs)
        before_spawn = kwargs.get("_before_spawn")
        if callable(before_spawn) and before_spawn() is not True:
            return False
        handoffs.append(None)
        return True

    agent = SimpleNamespace(
        _working_dir=tmp_path,
        _config=SimpleNamespace(language="en"),
        _log=lambda event, **fields: logs.append({"event": event, **fields}),
        _perform_refresh=_perform_refresh,
        _perform_refresh_attempts=perform_attempts,
    )

    if retry is not _ABSENT:
        setattr(
            agent,
            "_retry_failed_mcps",
            retry if callable(retry) else lambda: retry,
        )

    if cleanup is not _ABSENT:
        def _retire_all_mcp_clients(**kwargs):
            cleanup_calls.append(kwargs)
            return cleanup

        agent._retire_all_mcp_clients = _retire_all_mcp_clients

    return agent, logs, cleanup_calls, handoffs


def _converged_retry_report(*, healthy: list[str] | None = None) -> dict:
    return {
        "retried": [],
        "recovered": [],
        "still_failed": [],
        "healthy": healthy or [],
        "unresolved": [],
        "converged": True,
    }


def _converged_cleanup_report() -> dict:
    return {
        "context": "refresh",
        "attempted": [],
        "retired": [],
        "unresolved": [],
        "converged": True,
    }


def test_system_refresh_retry_exception_blocks_handoff_with_actionable_error(tmp_path):
    def retry():
        raise RuntimeError("retry transport is still shutting down")

    agent, _logs, cleanup_calls, handoffs = _system_refresh_agent(
        tmp_path,
        retry=retry,
        cleanup=_converged_cleanup_report(),
    )

    result = _refresh(agent, {"reason": "repair MCP"})

    assert result["status"] == "error"
    assert "MCP" in result["message"]
    assert "retry transport is still shutting down" in result["message"]
    assert cleanup_calls == []
    assert handoffs == []
    assert agent._perform_refresh_attempts == []
    assert not (tmp_path / ".refresh").exists()


@pytest.mark.parametrize(
    ("report", "actionable_fragment"),
    [
        (
            {
                **_converged_retry_report(),
                "still_failed": ["broken-search"],
            },
            "broken-search",
        ),
        (
            {
                **_converged_retry_report(),
                "unresolved": [
                    {
                        "client": "weather-client",
                        "phase": "retry",
                        "error": "close still blocked",
                    }
                ],
            },
            "close still blocked",
        ),
        (
            {
                **_converged_retry_report(),
                "converged": False,
            },
            "converg",
        ),
    ],
    ids=["still-failed", "unresolved-retirement", "false-convergence"],
)
def test_system_refresh_retry_nonconvergence_blocks_handoff(
    tmp_path,
    report,
    actionable_fragment,
):
    agent, _logs, cleanup_calls, handoffs = _system_refresh_agent(
        tmp_path,
        retry=report,
        cleanup=_converged_cleanup_report(),
    )

    result = _refresh(agent, {"reason": "repair MCP"})

    assert result["status"] == "error"
    assert "MCP" in result["message"]
    assert actionable_fragment.lower() in result["message"].lower()
    assert cleanup_calls == []
    assert handoffs == []
    assert agent._perform_refresh_attempts == []
    assert not (tmp_path / ".refresh").exists()


def test_system_refresh_cleanup_nonconvergence_blocks_handoff(tmp_path):
    cleanup_report = {
        "context": "refresh",
        "attempted": ["teardown-client"],
        "retired": [],
        "unresolved": [
            {
                "client": "teardown-client",
                "phase": "refresh",
                "error": "transport thread remains alive",
            }
        ],
        "converged": False,
    }
    agent, _logs, cleanup_calls, handoffs = _system_refresh_agent(
        tmp_path,
        retry=_converged_retry_report(healthy=["teardown-client"]),
        cleanup=cleanup_report,
    )

    result = _refresh(agent, {"reason": "replace process"})

    assert result["status"] == "error"
    assert "MCP" in result["message"]
    assert "transport thread remains alive" in result["message"]
    assert len(cleanup_calls) == 1
    assert handoffs == []
    assert agent._perform_refresh_attempts == []
    assert not (tmp_path / ".refresh").exists()
    assert not (tmp_path / ".refresh.taken").exists()


def test_system_refresh_cleanup_exception_never_enters_perform(tmp_path):
    """A retirement exception cannot create ACK or deferred relaunch state."""
    agent, _logs, _cleanup_calls, handoffs = _system_refresh_agent(
        tmp_path,
        retry=_converged_retry_report(),
    )

    def retire(**_kwargs):
        raise RuntimeError("retirement ownership proof failed")

    agent._retire_all_mcp_clients = retire

    result = _refresh(agent, {"reason": "retirement exception"})

    assert result["status"] == "error"
    assert "retirement" in result["message"]
    assert "ownership proof failed" in result["message"]
    assert handoffs == []
    assert agent._perform_refresh_attempts == []
    assert not (tmp_path / ".refresh").exists()
    assert not (tmp_path / ".refresh.taken").exists()


def test_system_refresh_healthy_mcp_teardown_then_handoff_once(tmp_path):
    agent, _logs, cleanup_calls, handoffs = _system_refresh_agent(
        tmp_path,
        retry=_converged_retry_report(healthy=["healthy-search"]),
        cleanup=_converged_cleanup_report(),
    )

    result = _refresh(agent, {"reason": "reload runtime"})

    assert result["status"] == "ok"
    assert len(cleanup_calls) == 1
    assert handoffs == [None]


def test_system_refresh_without_mcp_hooks_handoffs_once(tmp_path):
    agent, _logs, cleanup_calls, handoffs = _system_refresh_agent(tmp_path)

    result = _refresh(agent, {"reason": "base-agent refresh"})

    assert result["status"] == "ok"
    assert cleanup_calls == []
    assert handoffs == [None]


class _CloseProbe:
    def __init__(
        self,
        name: str,
        close_order: list[str],
        *,
        failures: int = 0,
    ) -> None:
        self.name = name
        self.close_order = close_order
        self.failures_remaining = failures
        self.close_calls = 0
        self.closed = False

    def call_tool(self, name: str, args: dict) -> dict:
        raise AssertionError(f"retiring client {self.name} must not dispatch {name}")

    def close(self) -> None:
        self.close_calls += 1
        self.close_order.append(self.name)
        if self.failures_remaining:
            self.failures_remaining -= 1
            raise RuntimeError(f"{self.name} close failed")
        self.closed = True

    def is_connected(self) -> bool:
        return not self.closed


def _agent(
    tmp_path: Path,
    *,
    cls: type[Agent] = Agent,
    workdir_lease=None,
    refresh_watcher=None,
) -> Agent:
    return cls(
        service=make_gemini_mock_service(),
        agent_name="mcp-refresh-teardown-test",
        working_dir=tmp_path / "agent",
        capabilities={"mcp": {}},
        event_journal=MagicMock(),
        workdir_lease=workdir_lease or make_test_lease(),
        refresh_watcher=refresh_watcher or make_test_refresh_watcher(),
    )


def _install_active_clients(agent: Agent, clients: list[_CloseProbe]) -> list[str]:
    names = [f"teardown_probe_{index}" for index, _ in enumerate(clients)]
    for name, client in zip(names, clients, strict=True):
        agent._tool_handlers[name] = agent._make_mcp_handler(client, name)
        agent._tool_schemas.append(
            FunctionSchema(
                name=name,
                description="MCP teardown probe",
                parameters={"type": "object", "properties": {}},
            )
        )
        agent._mcp_clients_by_tool[name] = client
        agent._mcp_tool_names.add(name)
    agent._mcp_clients = list(clients)
    agent._mcp_init_specs = {
        client.name: {
            "cfg": {"type": "stdio", "command": client.name},
            "source": "test",
            "client": client,
        }
        for client in clients
    }
    return names


def test_system_refresh_ack_failure_reports_after_exact_mcp_retirement(
    tmp_path, monkeypatch
):
    """Post-retirement ACK failure stays explicit and durably retryable."""
    close_order: list[str] = []
    client = _CloseProbe("ack-failure-client", close_order)
    watcher = make_test_refresh_watcher()
    lease = RecordingWorkdirLease()
    agent = _agent(
        tmp_path,
        workdir_lease=lease,
        refresh_watcher=watcher,
    )
    tool_names = _install_active_clients(agent, [client])
    agent._build_launch_cmd = lambda: ["python", "-c", "pass"]
    real_touch = Path.touch

    def fail_refresh_ack(path: Path, *args, **kwargs):
        if path.name == ".refresh.taken":
            raise OSError("workdir acknowledgement is read-only")
        return real_touch(path, *args, **kwargs)

    monkeypatch.setattr(Path, "touch", fail_refresh_ack)
    stopped = False
    try:
        result = _refresh(agent, {"reason": "ack failure probe"})

        assert result["status"] == "error"
        assert "handoff" in result["message"]
        assert "not established" in result["message"]
        assert watcher.spawned is False
        assert watcher.calls == []
        assert agent._shutdown.is_set() is False
        assert client.closed is True
        assert close_order == ["ack-failure-client"]
        assert agent._mcp_clients == []
        assert agent._mcp_retiring_clients == []
        assert all(name not in agent._tool_handlers for name in tool_names)
        assert (agent._working_dir / ".refresh").exists()
        assert lease.held is True
        assert lease.releases == 0

        agent.stop()
        stopped = True
        assert lease.held is False
        assert lease.releases == 1
    finally:
        if not stopped:
            agent._workdir_lease.release()


def test_system_refresh_spawn_failure_keeps_retry_signal_after_retirement(tmp_path):
    """Watcher failure is explicit; retired legacy ingress is not resurrected."""
    close_order: list[str] = []
    client = _CloseProbe("spawn-failure-client", close_order)
    watcher = make_test_refresh_watcher()
    watcher.spawn_detached = MagicMock(
        side_effect=RuntimeError("watcher process could not start")
    )
    lease = RecordingWorkdirLease()
    agent = _agent(tmp_path, workdir_lease=lease, refresh_watcher=watcher)
    _install_active_clients(agent, [client])
    # Model a legacy mcp/servers.json client rather than an init-spec retry.
    agent._mcp_init_specs = {}
    agent._build_launch_cmd = lambda: ["python", "-c", "pass"]
    stopped = False

    try:
        result = _refresh(agent, {"reason": "spawn failure probe"})

        assert result["status"] == "error"
        assert "watcher process could not start" in result["message"]
        assert watcher.spawn_detached.call_count == 1
        assert client.closed is True
        assert close_order == ["spawn-failure-client"]
        assert agent._mcp_clients == []
        assert (agent._working_dir / ".refresh").exists()
        assert not (agent._working_dir / ".refresh.taken").exists()
        assert agent._shutdown.is_set() is False
        assert lease.held is True

        agent.stop(timeout=1.0)
        stopped = True
        assert close_order == ["spawn-failure-client"]
        assert lease.held is False
        assert lease.releases == 1
    finally:
        if not stopped:
            lease.release()


def test_worker_hang_refresh_retires_mcp_without_touching_poisoned_chat(tmp_path):
    """Poison recovery closes transports but never syncs the live interface."""
    from lingtai.kernel.base_agent.worker_recovery import request_worker_hang_refresh

    close_order: list[str] = []
    client = _CloseProbe("poison-client", close_order)
    watcher = make_test_refresh_watcher()
    lease = RecordingWorkdirLease()
    agent = _agent(tmp_path, workdir_lease=lease, refresh_watcher=watcher)
    tool_names = _install_active_clients(agent, [client])
    update_tools = MagicMock(
        side_effect=AssertionError("poisoned ChatInterface must not be touched")
    )
    agent._chat = SimpleNamespace(update_tools=update_tools)
    agent._llm_worker_interface_poisoned = True
    agent._llm_worker_refresh_requested = False
    agent._build_launch_cmd = lambda: ["python", "-c", "pass"]
    stopped = False

    try:
        request_worker_hang_refresh(
            agent,
            artifact_relpath=(
                "history/unfinished_turns/worker_still_running_poison.json"
            ),
            source="run_loop",
        )

        assert agent._llm_worker_refresh_requested is True
        assert watcher.spawned is True
        assert client.closed is True
        assert close_order == ["poison-client"]
        assert update_tools.call_count == 0
        assert agent._mcp_clients == []
        assert agent._mcp_retiring_clients == []
        assert all(name not in agent._tool_handlers for name in tool_names)
        assert agent._mcp_inventory_sync_pending is True
        report = agent._last_mcp_cleanup_report
        assert report["transport_converged"] is True
        assert report["inventory_sync_deferred"] is True
        assert report["converged"] is False
        assert lease.held is True
        assert lease.releases == 0

        agent.stop()
        stopped = True
        assert update_tools.call_count == 0
        assert lease.held is False
        assert lease.releases == 1
    finally:
        if not stopped:
            lease.release()


def test_stop_keeps_lease_when_refresh_handoff_worker_has_not_joined(tmp_path):
    """Stop is retryable and retains liveness ownership until single-flight exits."""
    lease = RecordingWorkdirLease()
    agent = _agent(tmp_path, workdir_lease=lease)
    release_worker = threading.Event()
    worker = threading.Thread(target=release_worker.wait, daemon=True)
    agent._refresh_handoff_thread = worker
    worker.start()

    try:
        with pytest.raises(RuntimeError, match="refresh handoff remains in flight"):
            agent.stop(timeout=0.01)

        assert worker.is_alive()
        assert lease.held is True
        assert lease.releases == 0

        release_worker.set()
        worker.join(timeout=1.0)
        assert not worker.is_alive()

        agent.stop(timeout=1.0)

        assert lease.held is False
        assert lease.releases == 1
    finally:
        release_worker.set()
        worker.join(timeout=1.0)
        if lease.held:
            lease.release()


def test_stop_cancels_background_signal_handoff_before_watcher(tmp_path):
    """A concurrent ordinary stop must not relaunch after cleanup unblocks."""
    from lingtai.kernel.base_agent.lifecycle import _start_signal_refresh_handoff

    lease = RecordingWorkdirLease()
    agent = _agent(tmp_path, workdir_lease=lease)
    (agent._working_dir / ".refresh").touch()
    retirement_started = threading.Event()
    retire_calls: list[dict] = []

    def retire(**kwargs):
        retire_calls.append(kwargs)
        if kwargs["context"] == "heartbeat_refresh":
            retirement_started.set()
            assert agent._shutdown.wait(1.0)
            return {
                "context": "heartbeat_refresh",
                "attempted": [],
                "retired": [],
                "unresolved": [],
                "transport_converged": True,
                "inventory_sync_deferred": False,
                "converged": True,
            }
        return {
            "context": "stop",
            "attempted": [],
            "retired": [],
            "unresolved": [],
            "transport_converged": True,
            "inventory_sync_deferred": False,
            "converged": True,
        }

    agent._retire_all_mcp_clients = retire
    handoff_commits: list[None] = []

    def perform_refresh(**kwargs):
        before_spawn = kwargs.get("_before_spawn")
        assert callable(before_spawn)
        if before_spawn() is not True:
            return False
        handoff_commits.append(None)
        return True

    agent._perform_refresh = MagicMock(side_effect=perform_refresh)

    assert _start_signal_refresh_handoff(agent) is True
    assert retirement_started.wait(1.0)

    agent.stop(timeout=1.0)

    assert handoff_commits == []
    assert retire_calls[0] == {
        "context": "heartbeat_refresh",
        "sync_live_inventory": False,
    }
    assert retire_calls[1]["context"] == "stop"
    assert retire_calls[1]["sync_live_inventory"] is False
    assert 0.0 <= retire_calls[1]["activation_timeout"] <= 1.0
    assert agent._refresh_handoff_thread is None
    assert lease.held is False
    assert lease.releases == 1


def test_stop_wins_callback_to_spawn_commit_race(tmp_path):
    """Once stop commits its intent, a preflighted handoff cannot spawn."""
    from lingtai.kernel.base_agent.lifecycle import _request_refresh_handoff

    class CommitGateCondition:
        """Pause only the handoff's second condition entry (spawn commit)."""

        def __init__(self):
            self._condition = threading.Condition(threading.RLock())
            self._entries = 0
            self.commit_waiting = threading.Event()
            self.allow_commit = threading.Event()

        def __enter__(self):
            self._entries += 1
            if self._entries == 2:
                self.commit_waiting.set()
                assert self.allow_commit.wait(2.0)
            return self._condition.__enter__()

        def __exit__(self, *args):
            return self._condition.__exit__(*args)

        def notify_all(self):
            return self._condition.notify_all()

        def wait(self, timeout=None):
            return self._condition.wait(timeout)

    watcher = make_test_refresh_watcher()
    lease = RecordingWorkdirLease()
    agent = _agent(tmp_path, workdir_lease=lease, refresh_watcher=watcher)
    agent._build_launch_cmd = lambda: ["python", "-c", "pass"]
    refresh_signal = agent._working_dir / ".refresh"
    refresh_signal.write_text("stop-race payload", encoding="utf-8")
    gate = CommitGateCondition()
    agent._refresh_handoff_condition = gate
    handoff_result: list[bool] = []
    stop_errors: list[BaseException] = []

    handoff_thread = threading.Thread(
        target=lambda: handoff_result.append(
            _request_refresh_handoff(
                agent,
                context="commit_race",
                reconcile_failed_mcps=False,
                sync_live_inventory=False,
                retain_signal_on_failure=True,
            )
        ),
        daemon=True,
    )

    def stop_agent():
        try:
            agent.stop(timeout=2.0)
        except BaseException as exc:  # surfaced below with the original type
            stop_errors.append(exc)

    stop_thread = threading.Thread(target=stop_agent, daemon=True)
    handoff_thread.start()
    assert gate.commit_waiting.wait(1.0)
    stop_thread.start()

    deadline = threading.Event()
    for _ in range(100):
        if agent._refresh_stop_started:
            break
        deadline.wait(0.01)
    assert agent._refresh_stop_started is True
    gate.allow_commit.set()

    handoff_thread.join(timeout=2.0)
    stop_thread.join(timeout=2.0)

    assert not handoff_thread.is_alive()
    assert not stop_thread.is_alive()
    assert stop_errors == []
    assert handoff_result == [False]
    assert watcher.calls == []
    assert refresh_signal.read_text(encoding="utf-8") == "stop-race payload"
    assert not (agent._working_dir / ".refresh.taken").exists()
    assert lease.held is False
    assert lease.releases == 1


def test_direct_handoff_and_heartbeat_signal_share_single_flight(tmp_path):
    """A signal racing a direct handoff cannot run a second perform/spawn."""
    from lingtai.kernel.base_agent.lifecycle import (
        _join_refresh_handoff_thread,
        _request_refresh_handoff,
        _start_signal_refresh_handoff,
    )

    watcher = make_test_refresh_watcher()
    watcher.spawn_detached = MagicMock(
        side_effect=RuntimeError("detached watcher unavailable")
    )
    lease = RecordingWorkdirLease()
    agent = _agent(tmp_path, workdir_lease=lease, refresh_watcher=watcher)
    agent._build_launch_cmd = lambda: ["python", "-c", "pass"]
    refresh = agent._working_dir / ".refresh"
    refresh.write_text("original signal", encoding="utf-8")
    retirement_started = threading.Event()
    allow_retirement = threading.Event()
    retire_calls: list[str] = []

    def retire(**kwargs):
        context = kwargs["context"]
        retire_calls.append(context)
        if context == "system_refresh":
            retirement_started.set()
            assert allow_retirement.wait(2.0)
        return {
            "context": context,
            "attempted": [],
            "retired": [],
            "unresolved": [],
            "transport_converged": True,
            "inventory_sync_deferred": not kwargs.get(
                "sync_live_inventory", True
            ),
            "converged": kwargs.get("sync_live_inventory", True),
        }

    agent._retire_all_mcp_clients = retire
    real_perform = agent._perform_refresh
    perform_calls: list[dict] = []

    def perform_refresh(**kwargs):
        perform_calls.append(kwargs)
        return real_perform(**kwargs)

    agent._perform_refresh = perform_refresh
    direct_result: list[dict] = []
    direct_thread = threading.Thread(
        target=lambda: direct_result.append(
            _refresh(agent, {"reason": "diagnostic ownership race"})
        ),
        daemon=True,
    )
    direct_thread.start()
    assert retirement_started.wait(1.0)

    assert _start_signal_refresh_handoff(agent) is True
    assert _join_refresh_handoff_thread(agent, timeout=1.0)
    assert perform_calls == []
    assert retire_calls == ["system_refresh"]
    assert refresh.read_text(encoding="utf-8") == "original signal"
    assert getattr(agent, "_last_refresh_handoff_failure", None) is None

    allow_retirement.set()
    direct_thread.join(timeout=2.0)

    assert not direct_thread.is_alive()
    assert direct_result[0]["status"] == "error"
    assert "refresh handoff failed" in direct_result[0]["message"]
    assert "detached watcher unavailable" in direct_result[0]["message"]
    assert agent._last_refresh_handoff_failure == {
        "phase": "handoff_exception",
        "error": "detached watcher unavailable",
    }
    assert len(perform_calls) == 1
    assert watcher.spawn_detached.call_count == 1
    assert refresh.read_text(encoding="utf-8") == "original signal"
    assert not (agent._working_dir / ".refresh.taken").exists()
    assert agent._shutdown.is_set() is False

    agent.stop(timeout=1.0)
    assert lease.held is False
    assert lease.releases == 1


def test_stalled_watcher_spawn_keeps_stop_bounded_and_lease_retryable(tmp_path):
    """The external watcher Port cannot hold stop's coordination condition."""
    from lingtai.kernel.base_agent.lifecycle import _request_refresh_handoff

    spawn_entered = threading.Event()
    release_spawn = threading.Event()
    watcher = make_test_refresh_watcher()

    def stalled_spawn(request):
        watcher.calls.append(request)
        spawn_entered.set()
        assert release_spawn.wait(2.0)

    watcher.spawn_detached = stalled_spawn
    lease = RecordingWorkdirLease()
    agent = _agent(tmp_path, workdir_lease=lease, refresh_watcher=watcher)
    agent._build_launch_cmd = lambda: ["python", "-c", "pass"]
    handoff_results: list[bool] = []
    handoff_thread = threading.Thread(
        target=lambda: handoff_results.append(
            _request_refresh_handoff(
                agent,
                context="stalled_spawn",
                reconcile_failed_mcps=False,
                sync_live_inventory=False,
            )
        ),
        daemon=True,
    )
    handoff_thread.start()
    assert spawn_entered.wait(1.0)
    started = time.monotonic()
    try:
        with pytest.raises(RuntimeError, match="refresh handoff remains in flight"):
            agent.stop(timeout=0.05)
        assert time.monotonic() - started < 0.5
        assert lease.held is True
        assert lease.releases == 0

        release_spawn.set()
        handoff_thread.join(timeout=2.0)
        assert not handoff_thread.is_alive()
        assert handoff_results == [True]
        assert len(watcher.calls) == 1

        agent.stop(timeout=1.0)
        assert lease.held is False
        assert lease.releases == 1
    finally:
        release_spawn.set()
        handoff_thread.join(timeout=2.0)
        if lease.held:
            lease.release()


def test_post_commit_log_failure_cannot_downgrade_refresh_ack(tmp_path):
    """Once watcher+shutdown commit, telemetry failure remains best-effort."""
    from lingtai.kernel.base_agent.lifecycle import _request_refresh_handoff

    watcher = make_test_refresh_watcher()
    lease = RecordingWorkdirLease()
    agent = _agent(tmp_path, workdir_lease=lease, refresh_watcher=watcher)
    agent._build_launch_cmd = lambda: ["python", "-c", "pass"]
    real_log = agent._log

    def fail_commit_log(event, **fields):
        if event == "refresh_deferred_relaunch":
            raise RuntimeError("event journal unavailable after commit")
        return real_log(event, **fields)

    agent._log = fail_commit_log
    try:
        assert _request_refresh_handoff(
            agent,
            context="post_commit_log_failure",
            reconcile_failed_mcps=False,
            sync_live_inventory=False,
        ) is True
        assert len(watcher.calls) == 1
        assert agent._shutdown.is_set() is True
        assert getattr(agent, "_last_refresh_handoff_failure", None) is None

        agent.stop(timeout=1.0)
        assert lease.held is False
        assert lease.releases == 1
    finally:
        if lease.held:
            lease.release()


def test_stop_aggregates_active_mcp_cleanup_and_retries_only_pending(tmp_path):
    close_order: list[str] = []
    first = _CloseProbe("first", close_order, failures=1)
    second = _CloseProbe("second", close_order)
    lease = RecordingWorkdirLease()
    agent = _agent(tmp_path, workdir_lease=lease)
    tool_names = _install_active_clients(agent, [first, second])
    update_tools = MagicMock(
        side_effect=AssertionError("stop must not mutate live ChatInterface")
    )
    agent._chat = SimpleNamespace(update_tools=update_tools)

    try:
        with pytest.raises(RuntimeError, match="MCP cleanup remained unresolved during stop"):
            agent.stop()

        assert close_order == ["first", "second"]
        assert first.close_calls == 1
        assert second.close_calls == 1
        assert second.closed is True
        assert agent._mcp_clients == []
        assert agent._mcp_retiring_clients == [first]
        assert agent._last_mcp_cleanup_report["converged"] is False
        assert agent._last_mcp_cleanup_report["unresolved"]
        assert all(name not in agent._tool_handlers for name in tool_names)
        assert all(name not in agent._mcp_clients_by_tool for name in tool_names)
        assert all(name not in agent._mcp_tool_names for name in tool_names)
        assert all(
            schema.name not in tool_names for schema in agent._tool_schemas
        )
        assert all(
            spec["client"] is None for spec in agent._mcp_init_specs.values()
        )
        assert lease.held is True
        assert lease.releases == 0
        assert update_tools.call_count == 0

        agent.stop()

        assert close_order == ["first", "second", "first"]
        assert first.close_calls == 2
        assert second.close_calls == 1
        assert first.closed is True
        assert agent._mcp_retiring_clients == []
        assert agent._last_mcp_cleanup_report["transport_converged"] is True
        assert agent._last_mcp_cleanup_report["inventory_sync_deferred"] is True
        assert agent._last_mcp_cleanup_report["converged"] is False
        assert len(agent._last_mcp_cleanup_report["attempted"]) == 1
        assert update_tools.call_count == 0
        assert lease.held is False
        assert lease.releases == 1
    finally:
        agent._workdir_lease.release()


def test_deep_refresh_blocks_reconstruction_until_all_mcp_cleanup_converges(
    tmp_path,
):
    class TrackingAgent(Agent):
        def __init__(self, *args, **kwargs):
            self.mcp_load_calls = 0
            self.reconstruction_calls = 0
            super().__init__(*args, **kwargs)

        def _load_mcp_from_workdir(self) -> None:
            self.mcp_load_calls += 1
            self._mcp_init_specs = {}

        def _reload_prompt_sections(self, data=None) -> None:
            self.reconstruction_calls += 1
            super()._reload_prompt_sections(data)

    working_dir = tmp_path / "agent"
    working_dir.mkdir()
    init = _make_init(provider="gemini", model="gemini-test")
    (working_dir / "init.json").write_text(json.dumps(init), encoding="utf-8")
    agent = _agent(tmp_path, cls=TrackingAgent)

    close_order: list[str] = []
    first = _CloseProbe("first", close_order, failures=1)
    second = _CloseProbe("second", close_order)
    tool_names = _install_active_clients(agent, [first, second])
    initial_load_calls = agent.mcp_load_calls
    initial_reconstruction_calls = agent.reconstruction_calls

    try:
        with pytest.raises(
            RuntimeError,
            match="MCP cleanup remained unresolved during deep refresh",
        ):
            agent._setup_from_init()

        assert close_order == ["first", "second"]
        assert first.close_calls == 1
        assert second.close_calls == 1
        assert second.closed is True
        assert agent._mcp_clients == []
        assert agent._mcp_retiring_clients == [first]
        assert all(name not in agent._tool_handlers for name in tool_names)
        assert agent.mcp_load_calls == initial_load_calls
        assert agent.reconstruction_calls == initial_reconstruction_calls

        agent._setup_from_init()

        assert close_order == ["first", "second", "first"]
        assert first.close_calls == 2
        assert second.close_calls == 1
        assert first.closed is True
        assert agent._mcp_retiring_clients == []
        assert agent.mcp_load_calls == initial_load_calls + 1
        assert agent.reconstruction_calls == initial_reconstruction_calls + 1
    finally:
        agent._workdir_lease.release()
