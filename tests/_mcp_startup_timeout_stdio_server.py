"""Hermetic stdio process that never completes the MCP startup handshake."""
from __future__ import annotations

import time


def main() -> None:
    time.sleep(60)


if __name__ == "__main__":
    main()
