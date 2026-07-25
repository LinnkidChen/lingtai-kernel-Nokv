"""Hermetic stdio MCP server for structured-result transport coverage."""
from __future__ import annotations

import asyncio
from typing import Any

from mcp import types
from mcp.server import Server
from mcp.server.stdio import stdio_server

TOOL_NAME = "typed_failure"
STRUCTURED_ERROR = {
    "status": "success",
    "code": "HermeticTypedFailure",
    "message": "real stdio preserved this error",
    "retryable": True,
    "details": {"attempt": 2, "operation_id": "stdio-structured-1"},
}


def build_server() -> Server:
    server: Server = Server("lingtai-structured-result-fixture")

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name=TOOL_NAME,
                description="Return one typed structured error.",
                inputSchema={"type": "object", "properties": {}},
            )
        ]

    @server.call_tool()
    async def call_tool(
        name: str, arguments: dict[str, Any]
    ) -> types.CallToolResult:
        del arguments
        if name != TOOL_NAME:
            raise ValueError(f"unknown tool: {name!r}")
        return types.CallToolResult(
            isError=True,
            structuredContent=dict(STRUCTURED_ERROR),
            content=[
                types.TextContent(
                    type="text",
                    text='{"code":"WrongTextFallback","message":"wrong"}',
                )
            ],
        )

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
