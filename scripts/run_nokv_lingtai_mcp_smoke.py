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
import stat
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


_EXPECTED_TESTCASES = {
    (
        "tests.test_nokv_lingtai_mcp_smoke",
        "test_build_cleanup_retires_cargo_descendants",
    ),
    (
        "tests.test_nokv_lingtai_mcp_smoke",
        "test_registered_lingtai_reader_profile_runs_frozen_contract",
    ),
}
_REAL_SMOKE_NAME = "test_registered_lingtai_reader_profile_runs_frozen_contract"
_REQUIRED_REAL_SMOKE_PROPERTIES = {
    "lingtai_kernel_head",
    "lingtai_kernel_dirty",
    "lingtai_kernel_fingerprint",
    "nokv_cargo_version",
    "nokv_rustc_version",
    "nokv_source_head",
    "nokv_source_dirty",
    "nokv_source_fingerprint",
    "nokv_source_manifest_sha256",
    "nokv_binary",
    "nokv_binary_sha256",
    "nokv_lingtai_contract_asset_sha256",
    "nokv_mcp_profile",
    "nokv_workspace_role",
    "nokv_workspace_id_sha256",
    "nokv_workspace_actor_id_sha256",
    "nokv_workspace_grant_sha256",
    "nokv_mcp_activation_args_sha256",
    "nokv_mcp_tool_count",
    "nokv_mcp_raw_contract_sha256",
    "nokv_mcp_semantic_contract_sha256",
    "nokv_mcp_contract_sha256",
}


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


def _validate_junit_evidence(path: Path) -> None:
    """Require one complete, unfiltered execution of both smoke testcases."""

    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as error:
        raise ValueError(f"cannot read pytest JUnit evidence: {error}") from error

    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    if not suites:
        raise ValueError("pytest JUnit evidence contains no testsuite")

    totals = {field: 0 for field in ("tests", "failures", "errors", "skipped")}
    cases: list[ET.Element] = []
    for suite in suites:
        for field in totals:
            try:
                totals[field] += int(suite.attrib[field])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"pytest JUnit testsuite has invalid {field!r} total"
                ) from error
        cases.extend(suite.findall("testcase"))

    if totals != {"tests": 2, "failures": 0, "errors": 0, "skipped": 0}:
        raise ValueError(f"pytest JUnit totals do not prove both smoke gates: {totals}")
    observed = {
        (case.attrib.get("classname"), case.attrib.get("name")) for case in cases
    }
    if len(cases) != 2 or observed != _EXPECTED_TESTCASES:
        raise ValueError(
            "pytest JUnit testcases do not exactly match the required smoke gates"
        )
    if any(
        case.find(outcome) is not None
        for case in cases
        for outcome in ("failure", "error", "skipped")
    ):
        raise ValueError("pytest JUnit contains a non-passing smoke testcase")

    real_case = next(
        case for case in cases if case.attrib.get("name") == _REAL_SMOKE_NAME
    )
    properties = {
        node.attrib.get("name"): node.attrib.get("value")
        for node in real_case.findall("./properties/property")
    }
    missing = sorted(_REQUIRED_REAL_SMOKE_PROPERTIES - properties.keys())
    if missing:
        raise ValueError(
            "pytest JUnit real smoke is missing required evidence properties: "
            + ", ".join(missing)
        )
    empty = sorted(
        name for name in _REQUIRED_REAL_SMOKE_PROPERTIES if not properties[name]
    )
    if empty:
        raise ValueError(
            "pytest JUnit real smoke has empty evidence properties: "
            + ", ".join(empty)
        )
    if properties["nokv_mcp_tool_count"] != "20":
        raise ValueError("pytest JUnit real smoke did not prove the 20-tool contract")
    if properties["nokv_mcp_profile"] != "lingtai":
        raise ValueError("pytest JUnit real smoke did not prove the LingTai profile")
    if properties["nokv_workspace_role"] != "reader":
        raise ValueError("pytest JUnit real smoke did not prove the reader role")


def _lexical_absolute_path(value: str) -> Path:
    """Make a path absolute without dereferencing its final component."""

    path = Path(value).expanduser()
    return path if path.is_absolute() else Path.cwd() / path


def _require_replaceable_destination(path: Path) -> None:
    """Allow only entries that an evidence file may safely replace."""

    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return
    except OSError as error:
        raise ValueError(f"cannot inspect pytest JUnit destination: {error}") from error
    if stat.S_ISREG(mode) or stat.S_ISLNK(mode):
        return
    raise ValueError(
        "pytest JUnit destination must be absent, a regular file, or a symlink"
    )


def _validate_private_junit_evidence(path: Path) -> None:
    """Reject missing or non-regular staged output before parsing it."""

    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise ValueError(f"cannot read pytest JUnit evidence: {error}") from error
    if not stat.S_ISREG(mode):
        raise ValueError("pytest JUnit evidence is not a private regular file")
    _validate_junit_evidence(path)


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
    junit_xml = _lexical_absolute_path(args.junit_xml)
    try:
        junit_xml.parent.mkdir(parents=True, exist_ok=True)
        _require_replaceable_destination(junit_xml)
    except (OSError, ValueError) as error:
        print(f"cannot prepare pytest JUnit destination: {error}", file=sys.stderr)
        return 2

    kernel_root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    # The runner owns test selection. Inherited addopts can otherwise turn a
    # filtered subset into a false-green evidence artifact.
    environment.pop("PYTEST_ADDOPTS", None)
    environment["NOKV_LINGTAI_MCP_SMOKE"] = "1"
    environment["NOKV_LINGTAI_MCP_SOURCE"] = str(source)
    try:
        with tempfile.TemporaryDirectory(
            dir=junit_xml.parent,
            prefix=".nokv-lingtai-mcp-smoke-",
        ) as temporary_directory:
            staging_directory = Path(temporary_directory)
            os.chmod(staging_directory, 0o700)
            staging_mode = staging_directory.lstat().st_mode
            if not stat.S_ISDIR(staging_mode) or stat.S_IMODE(staging_mode) != 0o700:
                raise ValueError("pytest JUnit staging directory is not private")
            private_junit_xml = staging_directory / "evidence.xml"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    "-o",
                    "addopts=",
                    "-o",
                    "junit_family=legacy",
                    f"--junitxml={private_junit_xml}",
                    "tests/test_nokv_lingtai_mcp_smoke.py",
                ],
                cwd=kernel_root,
                env=environment,
                check=False,
            )
            if completed.returncode != 0:
                return completed.returncode
            try:
                _validate_private_junit_evidence(private_junit_xml)
            except ValueError as error:
                print(f"invalid smoke evidence: {error}", file=sys.stderr)
                return 2
            _require_replaceable_destination(junit_xml)
            os.replace(private_junit_xml, junit_xml)
    except (OSError, ValueError) as error:
        print(f"cannot publish pytest JUnit evidence: {error}", file=sys.stderr)
        return 2
    print(f"evidence_junit_xml: {junit_xml}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
