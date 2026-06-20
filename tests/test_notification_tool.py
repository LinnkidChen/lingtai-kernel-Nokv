"""Tests for the standalone ``notification`` intrinsic.

The notification tool carves the notification-facing verbs out of ``system``
into a dedicated, mandatory-included tool.  It is a thin façade over the same
canonical implementation ``system`` uses, so:

* ``notification(action="check")`` returns the same placeholder shape as
  ``system(action="notification")`` and receives the same meta-block stamp;
* ``notification(action="dismiss", ...)`` produces results identical to
  ``system(action="dismiss", ...)`` (only the provenance log differs);
* ``notification(action="summarize", ...)`` is the same single-source
  summarize, including the auto-clear of large-result reminders.

#424 semantics (large_tool_result reminders undismissable; only successful
summarize clears them) must hold through BOTH the new tool and the old alias.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lingtai_kernel.intrinsics import (
    ALL_INTRINSICS,
    notification as notif_intrinsic,
    system as sys_intrinsic,
)
from lingtai_kernel.notifications import (
    collect_notifications,
    notification_fingerprint,
    publish,
)


# ---------------------------------------------------------------------------
# Stub agent — mirrors the one used by test_system_dismiss.py.
# ---------------------------------------------------------------------------


@dataclass
class _StubAgent:
    _working_dir: Path
    _logs: list[tuple[str, dict]] = field(default_factory=list)
    _notification_fp: tuple = ()
    _pending_notification_meta: str | None = "stale"
    _pending_notification_fp: tuple | None = (("soul.json", 1, 2),)
    _system_notification_lock: threading.Lock = field(default_factory=threading.Lock)

    def _log(self, event_type: str, **fields: Any) -> None:
        self._logs.append((event_type, fields))


def _events(agent: _StubAgent, name: str) -> list[dict]:
    return [fields for event, fields in agent._logs if event == name]


def _mark_delivered(agent: _StubAgent) -> None:
    agent._notification_fp = notification_fingerprint(agent._working_dir)


def _publish_large_result_reminder(
    tmp_path: Path,
    *,
    tool_call_id: str = "toolu_big",
    extra_events: list[dict] | None = None,
) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    events = [
        {
            "event_id": "evt_lr",
            "source": "large_tool_result",
            "ref_id": f"large_tool_result:{tool_call_id}",
            "body": "summarize me",
        }
    ]
    if extra_events:
        events = list(extra_events) + events
    publish(
        tmp_path,
        "system",
        {
            "header": f"{len(events)} system notifications",
            "icon": "🔔",
            "priority": "normal",
            "published_at": "2026-06-20T00:00:00Z",
            "data": {"events": events},
        },
    )


# ---------------------------------------------------------------------------
# Mandatory include + schema availability.
# ---------------------------------------------------------------------------


def test_notification_is_registered_like_system() -> None:
    """The notification intrinsic is in ALL_INTRINSICS — wired for every agent."""
    assert "notification" in ALL_INTRINSICS
    assert ALL_INTRINSICS["notification"]["module"] is notif_intrinsic


def test_notification_wired_into_every_agent() -> None:
    """_wire_intrinsics iterates ALL_INTRINSICS unconditionally → mandatory.

    There is no manifest gate: every key in ALL_INTRINSICS is wired into
    agent._intrinsics. Proving 'notification' lands there alongside 'system'
    is the mandatory-include proof.
    """
    from lingtai_kernel.base_agent import BaseAgent

    wired: dict[str, Any] = {}

    class _FakeAgent:
        _intrinsics = wired

        def _log(self, *a, **k):
            pass

    # Drive the real wiring routine against a fake agent.
    BaseAgent._wire_intrinsics(_FakeAgent())  # type: ignore[arg-type]

    assert "system" in wired
    assert "notification" in wired
    assert callable(wired["notification"])


def test_notification_schema_exposes_actions() -> None:
    schema = notif_intrinsic.get_schema("en")
    assert schema["properties"]["action"]["enum"] == ["check", "dismiss", "summarize"]
    assert schema["required"] == ["action"]
    # Shared params present for dismiss/summarize.
    for key in ("channel", "force", "event_id", "ref_id", "items"):
        assert key in schema["properties"]


def test_notification_schema_localized() -> None:
    for lang in ("en", "zh", "wen"):
        desc = notif_intrinsic.get_description(lang)
        assert desc and desc != "notification_tool.description"
        adesc = notif_intrinsic.get_schema(lang)["properties"]["action"]["description"]
        assert adesc and adesc != "notification_tool.action_description"


# ---------------------------------------------------------------------------
# check — placeholder shape mirrors system(action="notification").
# ---------------------------------------------------------------------------


def test_check_returns_placeholder_dict(tmp_path: Path) -> None:
    agent = _StubAgent(tmp_path)
    res = notif_intrinsic.handle(agent, {"action": "check"})
    assert res["_notification_placeholder"] is True
    assert "notification(action=check)" in res["message"]
    # No bare channel keys — payload arrives only via meta-block stamp.
    assert "notifications" not in res


def test_check_placeholder_matches_system_notification_shape(tmp_path: Path) -> None:
    agent = _StubAgent(tmp_path)
    notif_res = notif_intrinsic.handle(agent, {"action": "check"})
    sys_res = sys_intrinsic.handle(agent, {"action": "notification"})
    assert set(notif_res) == set(sys_res)
    assert notif_res["_notification_placeholder"] == sys_res["_notification_placeholder"]


def test_unknown_action_errors(tmp_path: Path) -> None:
    agent = _StubAgent(tmp_path)
    res = notif_intrinsic.handle(agent, {"action": "bogus"})
    assert res["status"] == "error"
    assert "Unknown notification action" in res["message"]


# ---------------------------------------------------------------------------
# dismiss — by channel / event_id / ref_id, via the new tool.
# ---------------------------------------------------------------------------


def test_dismiss_by_channel(tmp_path: Path) -> None:
    agent = _StubAgent(tmp_path)
    publish(tmp_path, "soul", {"header": "soul flow"})
    _mark_delivered(agent)

    res = notif_intrinsic.handle(agent, {"action": "dismiss", "channel": "soul"})

    assert res == {"status": "ok", "channel": "soul", "cleared": True, "forced": False}
    assert collect_notifications(tmp_path) == {}
    # Provenance: notification path logs invoked_by="notification" and does NOT
    # emit the system_dismiss extra line (that is reserved for the system tool).
    assert _events(agent, "notification_dismiss")[0]["invoked_by"] == "notification"
    assert _events(agent, "system_dismiss") == []


def test_dismiss_missing_channel(tmp_path: Path) -> None:
    agent = _StubAgent(tmp_path)
    res = notif_intrinsic.handle(agent, {"action": "dismiss"})
    assert res["status"] == "error"
    assert res["reason"] == "missing_channel"


def test_dismiss_by_event_id(tmp_path: Path) -> None:
    agent = _StubAgent(tmp_path)
    publish(
        tmp_path,
        "system",
        {
            "header": "2 system notifications",
            "data": {
                "events": [
                    {"event_id": "evt_a", "source": "daemon", "ref_id": "a", "body": "A"},
                    {"event_id": "evt_b", "source": "daemon", "ref_id": "b", "body": "B"},
                ]
            },
        },
    )
    _mark_delivered(agent)

    res = notif_intrinsic.handle(
        agent, {"action": "dismiss", "channel": "system", "event_id": "evt_b"}
    )

    assert res["status"] == "ok"
    assert res["removed"] == 1
    events = collect_notifications(tmp_path)["system"]["data"]["events"]
    assert [e["event_id"] for e in events] == ["evt_a"]


def test_dismiss_by_ref_id(tmp_path: Path) -> None:
    agent = _StubAgent(tmp_path)
    publish(
        tmp_path,
        "system",
        {"data": {"events": [{"event_id": "evt_a", "source": "goal.reminder", "ref_id": "goal:current"}]}},
    )
    _mark_delivered(agent)

    res = notif_intrinsic.handle(
        agent, {"action": "dismiss", "channel": "system", "ref_id": "goal:current"}
    )

    assert res["status"] == "ok"
    assert res["removed"] == 1
    assert not (tmp_path / ".notification" / "system.json").exists()


# ---------------------------------------------------------------------------
# Old system alias and new notification path are identical.
# ---------------------------------------------------------------------------


def test_old_alias_and_new_path_identical_result(tmp_path: Path) -> None:
    """Same input → byte-identical result dict through both tools."""
    # New tool.
    (tmp_path / "a").mkdir(parents=True, exist_ok=True)
    a1 = _StubAgent(tmp_path / "a")
    publish(a1._working_dir, "nudge", {"header": "ping"})
    _mark_delivered(a1)
    new_res = notif_intrinsic.handle(a1, {"action": "dismiss", "channel": "nudge"})

    # Old alias.
    (tmp_path / "b").mkdir(parents=True, exist_ok=True)
    a2 = _StubAgent(tmp_path / "b")
    publish(a2._working_dir, "nudge", {"header": "ping"})
    _mark_delivered(a2)
    old_res = sys_intrinsic.handle(a2, {"action": "dismiss", "channel": "nudge"})

    assert new_res == old_res


def test_old_alias_and_new_path_identical_for_event_dismiss(tmp_path: Path) -> None:
    def _seed(p: Path) -> _StubAgent:
        p.mkdir(parents=True, exist_ok=True)
        agent = _StubAgent(p)
        publish(
            p,
            "system",
            {
                "header": "2 system notifications",
                "data": {
                    "events": [
                        {"event_id": "evt_a", "source": "daemon", "ref_id": "a"},
                        {"event_id": "evt_b", "source": "daemon", "ref_id": "b"},
                    ]
                },
            },
        )
        _mark_delivered(agent)
        return agent

    new_res = notif_intrinsic.handle(
        _seed(tmp_path / "a"), {"action": "dismiss", "channel": "system", "event_id": "evt_a"}
    )
    old_res = sys_intrinsic.handle(
        _seed(tmp_path / "b"), {"action": "dismiss", "channel": "system", "event_id": "evt_a"}
    )
    assert new_res == old_res


# ---------------------------------------------------------------------------
# #424: large_tool_result guard via BOTH paths, including force.
# ---------------------------------------------------------------------------


def test_large_result_guard_new_tool_whole_channel(tmp_path: Path) -> None:
    agent = _StubAgent(tmp_path)
    _publish_large_result_reminder(tmp_path)
    _mark_delivered(agent)

    res = notif_intrinsic.handle(agent, {"action": "dismiss", "channel": "system"})

    assert res["status"] == "error"
    assert res["reason"] == "undismissable_large_result_reminder"
    assert (tmp_path / ".notification" / "system.json").exists()


def test_large_result_guard_new_tool_force(tmp_path: Path) -> None:
    agent = _StubAgent(tmp_path)
    _publish_large_result_reminder(tmp_path)
    _mark_delivered(agent)

    res = notif_intrinsic.handle(
        agent, {"action": "dismiss", "channel": "system", "force": True}
    )

    assert res["status"] == "error"
    assert res["reason"] == "undismissable_large_result_reminder"
    assert res["forced"] is True
    events = collect_notifications(tmp_path)["system"]["data"]["events"]
    assert any(ev["source"] == "large_tool_result" for ev in events)


def test_large_result_guard_new_tool_event_id_and_ref_id(tmp_path: Path) -> None:
    for kwargs in (
        {"event_id": "evt_lr"},
        {"ref_id": "large_tool_result:toolu_big"},
        {"ref_id": "large_tool_result:toolu_big", "force": True},
    ):
        agent = _StubAgent(tmp_path / json.dumps(kwargs, sort_keys=True))
        _publish_large_result_reminder(agent._working_dir)
        _mark_delivered(agent)
        res = notif_intrinsic.handle(
            agent, {"action": "dismiss", "channel": "system", **kwargs}
        )
        assert res["status"] == "error", kwargs
        assert res["reason"] == "undismissable_large_result_reminder", kwargs
        events = collect_notifications(agent._working_dir)["system"]["data"]["events"]
        assert any(ev["source"] == "large_tool_result" for ev in events), kwargs


def test_large_result_guard_old_alias_force(tmp_path: Path) -> None:
    """The old system alias must enforce the same guard (regression anchor)."""
    agent = _StubAgent(tmp_path)
    _publish_large_result_reminder(tmp_path)
    _mark_delivered(agent)

    res = sys_intrinsic.handle(
        agent, {"action": "dismiss", "channel": "system", "force": True}
    )

    assert res["status"] == "error"
    assert res["reason"] == "undismissable_large_result_reminder"


def test_guard_identical_through_both_paths(tmp_path: Path) -> None:
    a1 = _StubAgent(tmp_path / "a")
    _publish_large_result_reminder(a1._working_dir)
    _mark_delivered(a1)
    new_res = notif_intrinsic.handle(
        a1, {"action": "dismiss", "channel": "system", "force": True}
    )

    a2 = _StubAgent(tmp_path / "b")
    _publish_large_result_reminder(a2._working_dir)
    _mark_delivered(a2)
    old_res = sys_intrinsic.handle(
        a2, {"action": "dismiss", "channel": "system", "force": True}
    )

    assert new_res == old_res


# ---------------------------------------------------------------------------
# summarize — single source; success clears reminder, failure does not.
# ---------------------------------------------------------------------------


class _Block:
    def __init__(self, block_id: str, name: str, content: Any) -> None:
        self.id = block_id
        self.name = name
        self.content = content


class _Entry:
    def __init__(self, role: str, content: list) -> None:
        self.role = role
        self.content = content


class _Interface:
    def __init__(self, entries: list) -> None:
        self._entries = entries


class _Chat:
    def __init__(self, entries: list) -> None:
        self.interface = _Interface(entries)


@dataclass
class _SummarizeAgent:
    _working_dir: Path
    _chat: Any = None
    _logs: list = field(default_factory=list)
    _summarize_notification_threshold: int = 3000
    _system_notification_lock: threading.Lock = field(default_factory=threading.Lock)
    _pending_notification_meta: Any = None
    _pending_notification_fp: Any = None

    def _log(self, event_type: str, **fields: Any) -> None:
        self._logs.append((event_type, fields))

    def _save_chat_history(self, ledger_source: str | None = None) -> None:
        pass


def _make_summarize_agent(tmp_path: Path, tool_call_id: str) -> _SummarizeAgent:
    from lingtai_kernel.llm.interface import ToolResultBlock

    block = ToolResultBlock(id=tool_call_id, name="read", content="x" * 5000)
    entry = _Entry("user", [block])
    agent = _SummarizeAgent(_working_dir=tmp_path)
    agent._chat = _Chat([entry])
    return agent


def test_summarize_success_clears_large_result_reminder(tmp_path: Path) -> None:
    tool_call_id = "toolu_sum_ok"
    agent = _make_summarize_agent(tmp_path, tool_call_id)
    _publish_large_result_reminder(tmp_path, tool_call_id=tool_call_id)

    res = notif_intrinsic.handle(
        agent,
        {
            "action": "summarize",
            "items": [{"tool_call_id": tool_call_id, "summary": "digested"}],
        },
    )

    assert res["status"] == "ok"
    assert res["summarized"] == 1
    assert f"large_tool_result:{tool_call_id}" in res["cleared_reminders"]
    # Reminder file is gone (it was the only event).
    assert not (tmp_path / ".notification" / "system.json").exists()


def test_summarize_failure_does_not_clear_reminder(tmp_path: Path) -> None:
    """A failed summarize item (unknown tool_call_id) must NOT clear the reminder."""
    reminder_tcid = "toolu_real"
    agent = _make_summarize_agent(tmp_path, "toolu_real")
    _publish_large_result_reminder(tmp_path, tool_call_id=reminder_tcid)

    # Summarize a DIFFERENT, non-existent tool_call_id → that item fails.
    res = notif_intrinsic.handle(
        agent,
        {
            "action": "summarize",
            "items": [{"tool_call_id": "toolu_missing", "summary": "nope"}],
        },
    )

    assert res["status"] == "error"
    assert res["summarized"] == 0
    assert res["failed"] == 1
    assert res["cleared_reminders"] == []
    # Reminder for the real tool_call_id is untouched.
    events = collect_notifications(tmp_path)["system"]["data"]["events"]
    assert any(
        ev["ref_id"] == f"large_tool_result:{reminder_tcid}" for ev in events
    )


def test_summarize_identical_through_both_paths(tmp_path: Path) -> None:
    tcid = "toolu_both"

    a1 = _make_summarize_agent(tmp_path / "a", tcid)
    _publish_large_result_reminder(a1._working_dir, tool_call_id=tcid)
    new_res = notif_intrinsic.handle(
        a1, {"action": "summarize", "items": [{"tool_call_id": tcid, "summary": "s"}]}
    )

    a2 = _make_summarize_agent(tmp_path / "b", tcid)
    _publish_large_result_reminder(a2._working_dir, tool_call_id=tcid)
    old_res = sys_intrinsic.handle(
        a2, {"action": "summarize", "items": [{"tool_call_id": tcid, "summary": "s"}]}
    )

    # Per-call timestamps inside item bodies are not compared; the structural
    # status / counts / cleared reminders must match.
    assert new_res["status"] == old_res["status"]
    assert new_res["summarized"] == old_res["summarized"]
    assert new_res["cleared_reminders"] == old_res["cleared_reminders"]


# ---------------------------------------------------------------------------
# Producer ownership + stale/force boundary preserved through the new tool.
# ---------------------------------------------------------------------------


def test_guarded_channel_refuses_without_force_new_tool(tmp_path: Path) -> None:
    import lingtai_kernel.intrinsics.email  # noqa: F401 — registers the guard

    agent = _StubAgent(tmp_path)
    publish(tmp_path, "email", {"header": "1 unread"})

    res = notif_intrinsic.handle(agent, {"action": "dismiss", "channel": "email"})

    assert res["status"] == "error"
    assert res["reason"] == "guarded"
    # Producer surface untouched.
    assert "email" in collect_notifications(tmp_path)
    assert _events(agent, "notification_dismiss_guarded")[0]["invoked_by"] == "notification"


def test_stale_channel_refused_without_force_new_tool(tmp_path: Path) -> None:
    agent = _StubAgent(tmp_path)
    publish(tmp_path, "system", {"header": "one", "data": {"events": ["old"]}})
    _mark_delivered(agent)
    publish(tmp_path, "system", {"header": "two", "data": {"events": ["old", "new"]}})

    res = notif_intrinsic.handle(agent, {"action": "dismiss", "channel": "system"})

    assert res["status"] == "error"
    assert res["reason"] == "stale_channel_version"
    # Newer file preserved.
    assert collect_notifications(tmp_path)["system"]["header"] == "two"


def test_force_bypasses_stale_on_allowed_channel_new_tool(tmp_path: Path) -> None:
    agent = _StubAgent(tmp_path)
    publish(tmp_path, "system", {"header": "one", "data": {"events": ["old"]}})
    _mark_delivered(agent)
    publish(tmp_path, "system", {"header": "two", "data": {"events": ["old", "new"]}})

    res = notif_intrinsic.handle(
        agent, {"action": "dismiss", "channel": "system", "force": True}
    )

    assert res["status"] == "ok"
    assert res["forced"] is True
    assert "system" not in collect_notifications(tmp_path)


def test_protected_goal_channel_refused_new_tool(tmp_path: Path) -> None:
    agent = _StubAgent(tmp_path)
    publish(tmp_path, "goal", {"data": {"status": "active"}})
    agent._notification_fp = notification_fingerprint(tmp_path)

    res = notif_intrinsic.handle(
        agent, {"action": "dismiss", "channel": "goal", "force": True}
    )

    assert res["status"] == "error"
    assert res["reason"] == "protected_channel"
    assert (tmp_path / ".notification" / "goal.json").exists()
