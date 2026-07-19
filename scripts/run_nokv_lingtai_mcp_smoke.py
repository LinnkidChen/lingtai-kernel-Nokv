#!/usr/bin/env python3
"""Fail-closed runner for the real NoKV x LingTai MCP smoke.

Use this rather than invoking the opt-in pytest module directly. It supplies
the activation environment required for the test to build NoKV from the named
checkout and launch a real stdio child; a missing or invalid checkout fails
before pytest can report a skipped smoke as green.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nokv-source",
        default=os.environ.get("NOKV_LINGTAI_MCP_SOURCE"),
        help="NoKV Git checkout to build and exercise (or NOKV_LINGTAI_MCP_SOURCE).",
    )
    parser.add_argument(
        "--junit-xml",
        default=os.environ.get("NOKV_LINGTAI_MCP_JUNIT_XML"),
        help=(
            "Required durable output for the source/binary and LingTai checkout "
            "evidence (or NOKV_LINGTAI_MCP_JUNIT_XML)."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if not args.nokv_source:
        raise SystemExit("--nokv-source (or NOKV_LINGTAI_MCP_SOURCE) is required")
    if not args.junit_xml:
        raise SystemExit("--junit-xml (or NOKV_LINGTAI_MCP_JUNIT_XML) is required")

    source = Path(args.nokv_source).expanduser().resolve()
    required = (source / "Cargo.toml", source / "Cargo.lock")
    if not source.joinpath(".git").exists() or not all(path.is_file() for path in required):
        raise SystemExit(f"--nokv-source is not a usable NoKV checkout: {source}")
    junit_xml = Path(args.junit_xml).expanduser().resolve()
    junit_xml.parent.mkdir(parents=True, exist_ok=True)

    kernel_root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment["NOKV_LINGTAI_MCP_SMOKE"] = "1"
    environment["NOKV_LINGTAI_MCP_SOURCE"] = str(source)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-o",
            "junit_family=legacy",
            f"--junitxml={junit_xml}",
            "tests/test_nokv_lingtai_mcp_smoke.py",
        ],
        cwd=kernel_root,
        env=environment,
        check=False,
    )
    if completed.returncode == 0:
        print(f"evidence_junit_xml: {junit_xml}")
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
