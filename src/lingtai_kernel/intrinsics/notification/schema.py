"""Schema — tool registration for the standalone ``notification`` tool.

The notification tool exposes only the notification-facing verbs.  Where a
parameter is shared with the ``system`` tool (``channel``, ``force``,
``event_id``, ``ref_id``, ``reason``, ``items``), the same i18n string is
reused so the two tools describe identical behavior.
"""
from __future__ import annotations


def get_description(lang: str = "en") -> str:
    from ...i18n import t
    return t(lang, "notification_tool.description")


def get_schema(lang: str = "en") -> dict:
    from ...i18n import t
    return {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["check", "dismiss", "summarize"],
                "description": t(lang, "notification_tool.action_description"),
            },
            "channel": {
                "type": "string",
                "description": t(lang, "system_tool.channel_description"),
            },
            "force": {
                "type": "boolean",
                "description": t(lang, "system_tool.force_description"),
            },
            "event_id": {
                "type": "string",
                "description": t(lang, "system_tool.event_id_description"),
            },
            "ref_id": {
                "type": "string",
                "description": t(lang, "system_tool.ref_id_description"),
            },
            "reason": {
                "type": "string",
                "description": t(lang, "system_tool.reason_description"),
            },
            "items": {
                "type": "array",
                "description": t(lang, "system_tool.items_description"),
                "items": {
                    "type": "object",
                    "properties": {
                        "tool_call_id": {
                            "type": "string",
                            "description": "The id of the prior tool-result block to summarize.",
                        },
                        "summary": {
                            "type": "string",
                            "description": "Your agent-authored summary of that tool result.",
                        },
                    },
                    "required": ["tool_call_id", "summary"],
                },
            },
        },
        "required": ["action"],
    }
