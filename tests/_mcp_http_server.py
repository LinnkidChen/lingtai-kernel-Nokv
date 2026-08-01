"""Small local streamable-HTTP MCP server used by lifecycle tests."""
from __future__ import annotations

import argparse

from mcp.server.fastmcp import FastMCP


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()

    server = FastMCP(
        "lifecycle-test",
        host="127.0.0.1",
        port=args.port,
        streamable_http_path="/mcp",
        stateless_http=True,
        json_response=True,
        log_level="ERROR",
    )

    @server.tool()
    def ping(value: str = "ok") -> dict[str, str]:
        return {"value": value}

    server.run(transport="streamable-http")


if __name__ == "__main__":
    main()
