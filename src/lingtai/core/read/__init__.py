"""Read capability — read text file contents.

Usage: Agent(capabilities=["read"]) or capabilities=["file"]
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ...i18n import t

if TYPE_CHECKING:
    from lingtai_kernel.base_agent import BaseAgent

PROVIDERS = {"providers": [], "default": "builtin"}


def get_description(lang: str = "en") -> str:
    return t(lang, "read.description")


def get_schema(lang: str = "en") -> dict:
    return {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": t(lang, "read.file_path")},
            "offset": {"type": "integer", "description": t(lang, "read.offset"), "default": 1},
            "limit": {"type": "integer", "description": t(lang, "read.limit"), "default": 2000},
        },
        "required": ["file_path"],
    }



def setup(agent: "BaseAgent") -> None:
    """Set up the read capability on an agent."""
    lang = agent._config.language

    def handle_read(args: dict) -> dict:
        path = args.get("file_path", "")
        if not path:
            return {"status": "error", "message": "file_path is required"}
        if not Path(path).is_absolute():
            path = str(agent._working_dir / path)
        offset = args.get("offset", 1)
        limit = args.get("limit", 2000)
        try:
            content = agent._file_io.read(path)
        except FileNotFoundError:
            # Spill-aware messaging: if the missing file is under
            # tmp/tool-results/, it was an ephemeral sidecar artifact
            # that has been cleaned up.  Give a specific hint instead
            # of the generic "File not found".
            rel = Path(path)
            try:
                rel = Path(path).relative_to(agent._working_dir)
            except ValueError:
                pass
            parts = rel.parts
            if len(parts) >= 3 and parts[0] == "tmp" and parts[1] == "tool-results":
                return {
                    "status": "error",
                    "message": (
                        "Spill artifact expired: this tmp/tool-results/ sidecar file "
                        "no longer exists. The original tool result content is "
                        "unavailable. Use the preview from the manifest or rerun the "
                        "source tool."
                    ),
                }
            return {"status": "error", "message": f"File not found: {path}"}
        except Exception as e:
            return {"status": "error", "message": f"Cannot read {path}: {e}"}
        lines = content.splitlines(keepends=True)
        start = max(0, offset - 1)
        selected = lines[start:start + limit]
        numbered = "".join(f"{start + i + 1}\t{line}" for i, line in enumerate(selected))
        return {"content": numbered, "total_lines": len(lines), "lines_shown": len(selected)}

    agent.add_tool("read", schema=get_schema(lang), handler=handle_read, description=get_description(lang))
