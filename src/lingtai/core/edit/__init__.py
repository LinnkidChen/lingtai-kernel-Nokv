"""Edit capability — exact string replacement in a file.

Usage: Agent(capabilities=["edit"]) or capabilities=["file"]
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ...i18n import t
from ...services.nokv import is_nokv_uri

if TYPE_CHECKING:
    from lingtai_kernel.base_agent import BaseAgent


def get_description(lang: str = "en") -> str:
    return t(lang, "edit.description")


def get_schema(lang: str = "en") -> dict:
    return {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": t(lang, "edit.file_path")},
            "old_string": {"type": "string", "description": t(lang, "edit.old_string")},
            "new_string": {"type": "string", "description": t(lang, "edit.new_string")},
            "replace_all": {"type": "boolean", "description": t(lang, "edit.replace_all"), "default": False},
        },
        "required": ["file_path", "old_string", "new_string"],
    }



def setup(agent: "BaseAgent") -> None:
    """Set up the edit capability on an agent."""
    lang = agent._config.language

    def handle_edit(args: dict) -> dict:
        path = args.get("file_path", "")
        if not path:
            return {"status": "error", "message": "file_path is required"}
        if not is_nokv_uri(path) and not Path(path).is_absolute():
            path = str(agent._working_dir / path)
        old = args.get("old_string", "")
        new = args.get("new_string", "")
        replace_all = args.get("replace_all", False)
        is_routed = False
        route_check = getattr(agent._file_io, "is_routed_to_nokv", None)
        if callable(route_check):
            try:
                is_routed = bool(route_check(path))
            except Exception:
                is_routed = False
        if is_routed and replace_all:
            return {
                "status": "error",
                "message": "replace_all is not supported for NoKV-routed paths; provide a unique old_string",
            }
        if not replace_all:
            try:
                agent._file_io.edit(path, old, new)
            except FileNotFoundError:
                return {"status": "error", "message": f"File not found: {path}"}
            except Exception as e:
                message = str(e)
                if not is_routed and "appears" in message and "replace_all" not in message:
                    message = f"{message} Use replace_all=true or provide more context."
                return {"status": "error", "message": message}
            return {"status": "ok", "replacements": 1}
        try:
            content = agent._file_io.read(path)
        except FileNotFoundError:
            return {"status": "error", "message": f"File not found: {path}"}
        except Exception as e:
            return {"status": "error", "message": f"Cannot read {path}: {e}"}
        count = content.count(old)
        if count == 0:
            return {"status": "error", "message": f"old_string not found in {path}"}
        updated = content.replace(old, new)
        try:
            agent._file_io.write(path, updated)
        except Exception as e:
            return {"status": "error", "message": f"Cannot write {path}: {e}"}
        return {"status": "ok", "replacements": count}

    agent.add_tool("edit", schema=get_schema(lang), handler=handle_edit, description=get_description(lang))
