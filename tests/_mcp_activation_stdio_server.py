"""Hermetic stdio MCP server whose tools/list operation fails."""
from __future__ import annotations

import asyncio

from mcp.server import Server
from mcp.server.stdio import stdio_server


def build_server() -> Server:
    server: Server = Server("lingtai-activation-failure-fixture")

    @server.list_tools()
    async def list_tools():
        raise RuntimeError("intentional tools/list failure")

    return server


async def serve() -> None:
    server = build_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(serve())
