"""
Active-turn tool-call progress meter and emergency fuse.

The total-call ceiling is deliberately large: it is an emergency fuse for a
single ACTIVE turn, not a normal workflow boundary.  The model sees progress
metadata on tool results and receives soft notices at regular intervals so it can
self-check without treating the fuse as a normal target.  Duplicate-call and
invalid-tool checks remain narrow loop detectors.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from .safety_limits import (
    ACTIVE_TURN_TOOL_CALL_EMERGENCY_LIMIT,
    ACTIVE_TURN_TOOL_CALL_NOTICE_INTERVAL,
)


@dataclass(frozen=True)
class DupVerdict:
    """Result of duplicate-call tracking for a single tool invocation.

    Attributes:
        count: Total times this (name, args) has been seen (including this call).
        blocked: If True, the caller should skip execution entirely.
        warning: Warning text to inject into the result dict, or None.
    """
    count: int
    blocked: bool
    warning: str | None


# Keys stripped from tool args before computing the dedup key.
# These carry metadata, not semantic intent.
_STRIP_KEYS = frozenset({"commentary", "_sync"})


class LoopGuard:
    """Prevents runaway tool-call loops in agent conversations.

    Usage:
        guard = LoopGuard()

        while True:
            # ... extract tool_calls from response ...

            reason = guard.check_limit(len(tool_calls))
            if reason:
                break

            # Count the batch before execution so tool results can expose the
            # post-batch ACTIVE-turn count.
            guard.record_calls(len(tool_calls))

            for tc in tool_calls:
                verdict = guard.record_tool_call(tc.name, tc.args)
                if verdict.blocked:
                    # skip execution, return blocked result
                    ...
                else:
                    # execute tool
                    ...
                    if verdict.warning:
                        result["_duplicate_warning"] = verdict.warning

            # ... send results back ...
            guard.clear_progress_notice()
    """

    def __init__(
        self,
        max_total_calls: int = ACTIVE_TURN_TOOL_CALL_EMERGENCY_LIMIT,
        dup_free_passes: int = 2,
        dup_hard_block: int = 8,
        invalid_tool_limit: int = 2,
        notice_interval: int | None = None,
        **_kwargs,
    ):
        self.max_total_calls = max_total_calls
        self.total_calls = 0
        if notice_interval is None:
            # Compatibility with the earlier progress-meter draft and any
            # downstream tests that passed the cadence under its old name.
            notice_interval = _kwargs.pop(
                "warning_interval", ACTIVE_TURN_TOOL_CALL_NOTICE_INTERVAL
            )
        self.notice_interval = notice_interval
        self._progress_notice: str | None = None
        self._dup_free_passes = dup_free_passes
        self._dup_hard_block = dup_hard_block
        self._dup_counts: dict[tuple[str, str], int] = {}
        # Track invalid/hallucinated tool names
        self._invalid_tool_limit = invalid_tool_limit
        self._invalid_tool_counts: dict[str, int] = {}
        self._total_invalid_tools = 0

    def check_limit(self, n_calls: int) -> str | None:
        """Check whether executing ``n_calls`` would exceed the emergency fuse.

        Returns a stop reason string or None to continue.  The ceiling is a
        kernel-owned ACTIVE-turn emergency limit, not an agent/user manifest
        setting.
        """
        if n_calls <= 0:
            return None
        if self.total_calls + n_calls > self.max_total_calls:
            return (
                "ACTIVE-turn tool-call safety fuse would be exceeded "
                f"after {self.total_calls} calls"
            )
        return None

    def record_invalid_tool(self, tool_name: str) -> None:
        """Record that a tool call was rejected as invalid/not-available."""
        self._invalid_tool_counts[tool_name] = (
            self._invalid_tool_counts.get(tool_name, 0) + 1
        )
        self._total_invalid_tools += 1

    def check_invalid_tool_limit(self) -> str | None:
        """Check if repeated invalid tool calls should stop the loop.

        Returns a stop reason string if any single tool name has been rejected
        more than ``invalid_tool_limit`` times, or if the total number of
        invalid tool calls exceeds ``invalid_tool_limit * 2``.
        Returns None to continue.
        """
        for name, count in self._invalid_tool_counts.items():
            if count > self._invalid_tool_limit:
                return (
                    f"tool '{name}' was rejected {count} times "
                    f"(limit: {self._invalid_tool_limit}) — "
                    f"the model is hallucinating tool names not in its schema"
                )
        if self._total_invalid_tools > self._invalid_tool_limit * 2:
            return (
                f"{self._total_invalid_tools} total invalid tool calls "
                f"(limit: {self._invalid_tool_limit * 2}) — "
                f"the model is hallucinating tool names"
            )
        return None

    def record_calls(self, n_calls: int) -> str | None:
        """Record ``n_calls`` in this ACTIVE turn and return any new soft notice."""
        if n_calls <= 0:
            self._progress_notice = None
            return None
        before = self.total_calls
        self.total_calls += n_calls
        self._progress_notice = self._build_progress_notice(before, self.total_calls)
        return self._progress_notice

    def _build_progress_notice(self, before: int, after: int) -> str | None:
        if self.notice_interval <= 0:
            return None
        crossed = after // self.notice_interval
        if after % self.notice_interval:
            crossed += 0
        previous = before // self.notice_interval
        if crossed <= previous:
            return None
        boundary = crossed * self.notice_interval
        if boundary <= 0:
            return None
        return (
            f"Soft self-check: this ACTIVE turn has made {after} tool calls "
            f"so far. Take a moment to notice whether you may be repeating a "
            f"loop; if the work is still progressing, continue normally."
        )

    def progress_metadata(self) -> dict:
        """Return LLM-visible ACTIVE-turn tool-call progress metadata."""
        meta = {
            "active_turn_tool_calls": self.total_calls,
        }
        if self._progress_notice:
            meta["active_turn_tool_call_notice"] = self._progress_notice
        return meta

    def clear_progress_notice(self) -> None:
        """Clear the batch-scoped progress notice after result construction."""
        self._progress_notice = None

    # ------------------------------------------------------------------
    # Duplicate call tracking
    # ------------------------------------------------------------------

    @staticmethod
    def _dedup_key(name: str, args: dict | None) -> tuple[str, str]:
        """Create a hashable key for duplicate detection.

        Strips metadata keys (commentary, _sync) before serializing,
        so that semantically identical calls with different metadata
        are treated as duplicates.
        """
        if not args:
            cleaned = {}
        else:
            cleaned = {k: v for k, v in args.items() if k not in _STRIP_KEYS}
        try:
            args_str = json.dumps(cleaned, sort_keys=True, default=str)
        except (TypeError, ValueError):
            args_str = str(sorted(cleaned.items()))
        return (name, args_str)

    def record_tool_call(self, name: str, args: dict | None) -> DupVerdict:
        """Record a tool call and return a verdict on whether to execute it.

        Call this *before* executing each tool. The verdict tells the caller:
          - Whether to skip execution (verdict.blocked)
          - Whether to inject a warning into the result (verdict.warning)
        """
        key = self._dedup_key(name, args)
        count = self._dup_counts.get(key, 0) + 1
        self._dup_counts[key] = count

        if count <= self._dup_free_passes:
            return DupVerdict(count=count, blocked=False, warning=None)

        if count >= self._dup_hard_block:
            return DupVerdict(
                count=count,
                blocked=True,
                warning=self._warning_for_count(name, count),
            )

        return DupVerdict(
            count=count,
            blocked=False,
            warning=self._warning_for_count(name, count),
        )

    def _warning_for_count(self, name: str, count: int) -> str:
        """Generate an escalating warning message for duplicate calls."""
        if count < self._dup_free_passes + 3:
            # Mild warning (passes N+1 to N+2)
            return (
                f"You have called '{name}' with identical arguments {count} times. "
                f"Consider whether this is necessary — repeated identical calls "
                f"waste tokens and rarely produce new information."
            )
        if count < self._dup_hard_block:
            # Strong warning (passes N+3+)
            return (
                f"STOP POLLING: '{name}' called {count} times with identical arguments. "
                f"This is a polling loop that wastes tokens. Do NOT call this tool again "
                f"with the same arguments. If you are waiting for a result, it will be "
                f"delivered to you automatically — do not poll for it."
            )
        # At or past hard block
        return (
            f"BLOCKED: '{name}' has been called {count} times with identical arguments "
            f"(hard limit: {self._dup_hard_block}). Execution was SKIPPED. "
            f"This call was a polling loop. Do NOT repeat it. "
            f"Proceed with a different approach or respond to the user."
        )

    @property
    def dup_counts(self) -> dict[tuple[str, str], int]:
        """Expose duplicate counts for testing/debugging."""
        return dict(self._dup_counts)
