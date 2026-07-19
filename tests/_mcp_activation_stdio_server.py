"""Hermetic real-stdio MCP server for activation lifecycle tests."""
from __future__ import annotations

import asyncio
import sys
import time

from mcp import types
from mcp.server import Server
from mcp.server.stdio import stdio_server


TOOL_NAME = "activation_probe"


def build_server(mode: str) -> Server:
    server: Server = Server("lingtai-activation-fixture")

    @server.list_tools()
    async def _list_tools() -> list[types.Tool]:
        if mode == "list_fail":
            raise RuntimeError("fixture tools/list failure")
        return [
            types.Tool(
                name=TOOL_NAME,
                description="One valid activation lifecycle probe.",
                inputSchema={"type": "object", "properties": {}},
            )
        ]

    return server


async def serve(mode: str) -> None:
    server = build_server(mode)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    selected = sys.argv[1] if len(sys.argv) > 1 else "valid"
    if selected == "hang_start":
        # The production client's bounded startup timeout owns retirement.
        time.sleep(60)
    else:
        asyncio.run(serve(selected))
