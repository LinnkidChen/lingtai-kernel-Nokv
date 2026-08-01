"""Public typed contract for coordinated refresh handoff outcomes."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RefreshHandoffStatus(str, Enum):
    """Terminal truth for one coordinated deferred-relaunch request."""

    COMMITTED = "committed"
    COMMITTED_DEGRADED = "committed_degraded"
    LIFECYCLE_BLOCKED = "lifecycle_blocked"
    PREPARATION_FAILED = "preparation_failed"
    OPERATION_FAILED = "operation_failed"
    INVALID_OUTCOME = "invalid_outcome"
    NO_LAUNCH_COMMAND = "no_launch_command"
    ACK_FAILED = "ack_failed"
    WATCHER_SPAWN_FAILED = "watcher_spawn_failed"


@dataclass(frozen=True)
class RefreshHandoffOutcome:
    """Explicit result of one lifecycle-owned refresh handoff attempt."""

    status: RefreshHandoffStatus
    message: str

    @property
    def committed(self) -> bool:
        return self.status in {
            RefreshHandoffStatus.COMMITTED,
            RefreshHandoffStatus.COMMITTED_DEGRADED,
        }

    @property
    def degraded(self) -> bool:
        return self.status is RefreshHandoffStatus.COMMITTED_DEGRADED
