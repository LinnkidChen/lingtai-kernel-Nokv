from lingtai_kernel.loop_guard import LoopGuard


def test_loop_guard_no_loop():
    """Distinct tool calls with different args should never be blocked."""
    guard = LoopGuard(max_total_calls=10)
    for i in range(5):
        verdict = guard.record_tool_call(f"tool_{i}", {"arg": i})
        assert not verdict.blocked


def test_loop_guard_detects_identical():
    """Identical calls at or beyond dup_hard_block should be blocked."""
    guard = LoopGuard(max_total_calls=20, dup_hard_block=3)
    guard.record_tool_call("read", {"path": "/foo"})
    guard.record_tool_call("read", {"path": "/foo"})
    verdict = guard.record_tool_call("read", {"path": "/foo"})  # 3rd = hard block
    assert verdict.blocked
    assert verdict.count == 3


def test_loop_guard_warning_before_block():
    """Calls between free passes and hard block should get a warning but not be blocked."""
    guard = LoopGuard(max_total_calls=20, dup_free_passes=1, dup_hard_block=5)
    guard.record_tool_call("read", {"path": "/foo"})  # count=1 (free pass)
    verdict = guard.record_tool_call("read", {"path": "/foo"})  # count=2 (warn)
    assert not verdict.blocked
    assert verdict.warning is not None


def test_loop_guard_check_limit():
    """check_limit returns a reason when total would be exceeded."""
    guard = LoopGuard(max_total_calls=3)
    guard.record_calls(3)
    reason = guard.check_limit(1)
    assert reason is not None
    assert "3" in reason


def test_loop_guard_invalid_tool():
    """Repeated invalid tool names should trigger a stop reason."""
    guard = LoopGuard(max_total_calls=20, invalid_tool_limit=2)
    guard.record_invalid_tool("ghost_tool")
    guard.record_invalid_tool("ghost_tool")
    guard.record_invalid_tool("ghost_tool")  # count=3 > limit=2
    reason = guard.check_invalid_tool_limit()
    assert reason is not None
    assert "ghost_tool" in reason


def test_loop_guard_progress_metadata_and_interval_notice():
    """ACTIVE-turn progress metadata is always available and adds soft notices at intervals."""
    guard = LoopGuard(max_total_calls=10_000, notice_interval=500)

    assert guard.record_calls(499) is None
    meta = guard.progress_metadata()
    assert meta == {"active_turn_tool_calls": 499}

    notice = guard.record_calls(1)
    assert notice is not None
    assert "Soft self-check" in notice
    assert "500 tool calls" in notice
    assert "may be repeating a loop" in notice
    assert "10000" not in notice
    assert "limit" not in notice.lower()
    meta = guard.progress_metadata()
    assert meta["active_turn_tool_calls"] == 500
    assert meta["active_turn_tool_call_notice"] == notice
    assert "active_turn_tool_call_limit" not in meta
    assert "active_turn_tool_call_notice_interval" not in meta

    guard.clear_progress_notice()
    assert "active_turn_tool_call_notice" not in guard.progress_metadata()


def test_loop_guard_default_emergency_limit_is_large():
    """The default total limit is an emergency fuse, not a 100-call guard."""
    guard = LoopGuard()
    assert guard.max_total_calls == 10_000
