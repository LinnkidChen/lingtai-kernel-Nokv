"""Notification intrinsic — the standalone notification surface.

This intrinsic carves the notification-facing verbs out of the ``system``
tool into a dedicated, mandatory-included tool.  It is a thin façade: every
action delegates to the single canonical implementation that ``system``
already uses, so there is exactly one source of truth for notification
behavior and the two tools stay byte-for-byte equivalent in their results.

Actions:
    check     — voluntary read of the live notification surface.  Returns a
                placeholder dict; the turn loop's meta-block post-hook stamps
                the canonical ``notifications`` + ``_notification_guidance``
                payload onto this same result (identical to the old
                ``system(action="notification")``).
    dismiss   — clear one ``.notification/<channel>.json`` surface, or a single
                ``system`` event by ``event_id``/``ref_id``.  Delegates to the
                shared :func:`lingtai_kernel.notifications.dismiss_channel`
                with ``invoked_by="notification"``.  Producer-owned state is
                never touched; #424 large-result reminders remain undismissable.
    summarize — agent-authored context summarization.  Delegates to the single
                ``system.summarize`` implementation, so a successful summarize
                is still the only path that auto-clears a large-result reminder.

The old ``system(action="notification"|"dismiss"|"summarize")`` verbs remain as
compatibility aliases; both tools route into the same functions.
"""
from __future__ import annotations

# Schema (tool registration).
from .schema import get_description, get_schema  # noqa: F401

# Single-source delegates — imported from the canonical implementations the
# ``system`` tool already uses.  No behavior is reimplemented here.
from ..system.notification import _dismiss as _system_dismiss
from ..system.summarize import _summarize


# Placeholder returned by ``check`` — mirrors the shape the ``system`` tool
# returns for ``action="notification"`` so the meta-block stamp lands here
# identically.  The kernel never returns bare channel keys from the handler;
# the live payload arrives only via the meta-block path.
_CHECK_PLACEHOLDER_MESSAGE = (
    "Voluntary notification(action=check) read. The live notification payload "
    "is delivered via the kernel meta-block under the `notifications` and "
    "`_notification_guidance` keys on this same result. If those keys are "
    "absent, no notifications are active."
)


def _check(agent, args: dict) -> dict:
    """Voluntary read of the notification surface — returns a placeholder.

    The canonical live payload (``notifications`` + ``_notification_guidance``)
    is stamped onto this same result dict by ``attach_active_notifications`` in
    the turn loop.  Returning a dict (not a string) is what makes that stamp
    possible: the meta-block walks backward for the freshest *dict-shaped* tool
    result, so this placeholder receives exactly the same payload the old
    ``system(action="notification")`` placeholder did.
    """
    return {
        "_notification_placeholder": True,
        "message": _CHECK_PLACEHOLDER_MESSAGE,
    }


def _dismiss(agent, args: dict) -> dict:
    """Dismiss via the notification tool.

    Delegates to the shared ``system`` dismiss implementation but stamps
    ``_invoked_by="notification"`` so logs attribute the clear to this tool.
    The decision logic (guards, stale-version refusal, large-result reminder
    protection, producer-ownership) is identical regardless of ``_invoked_by``;
    only the provenance log line differs.
    """
    delegated = dict(args)
    delegated.setdefault("_invoked_by", "notification")
    return _system_dismiss(agent, delegated)


def handle(agent, args: dict) -> dict:
    """Handle the standalone ``notification`` tool."""
    action = args.get("action")
    if action == "check":
        return _check(agent, args)
    handler = {
        "dismiss": _dismiss,
        "summarize": _summarize,
    }.get(action)
    if handler is None:
        return {
            "status": "error",
            "message": f"Unknown notification action: {action}",
        }
    return handler(agent, args)
