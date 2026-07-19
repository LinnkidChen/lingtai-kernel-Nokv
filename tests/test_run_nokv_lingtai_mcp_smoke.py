"""Regression tests for the fail-closed real NoKV smoke runner."""
from __future__ import annotations

import importlib.util
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest


_RUNNER_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_nokv_lingtai_mcp_smoke.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "run_nokv_lingtai_mcp_smoke_for_test", _RUNNER_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
runner = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(runner)


def _write_junit(
    path: Path,
    *,
    include_cleanup: bool = True,
    include_real: bool = True,
    skipped_real: bool = False,
    omit_property: str | None = None,
) -> None:
    cases: list[ET.Element] = []
    classname = "tests.test_nokv_lingtai_mcp_smoke"
    if include_cleanup:
        cases.append(
            ET.Element(
                "testcase",
                classname=classname,
                name="test_build_cleanup_retires_cargo_descendants",
            )
        )
    if include_real:
        real = ET.Element(
            "testcase",
            classname=classname,
            name="test_registered_lingtai_reader_profile_runs_frozen_contract",
        )
        properties = ET.SubElement(real, "properties")
        for name in sorted(runner._REQUIRED_REAL_SMOKE_PROPERTIES):
            if name == omit_property:
                continue
            values = {
                "lingtai_kernel_dirty": "false",
                "nokv_source_dirty": "false",
                "nokv_cargo_version": "cargo test",
                "nokv_rustc_version": "rustc test",
                "nokv_binary": "/tmp/nokv",
                "nokv_mcp_profile": "lingtai",
                "nokv_workspace_role": "reader",
                "nokv_mcp_tool_count": "20",
            }
            value = values.get(name, "a" * 64)
            ET.SubElement(properties, "property", name=name, value=value)
        if skipped_real:
            ET.SubElement(real, "skipped")
        cases.append(real)

    skipped = int(skipped_real and include_real)
    suite = ET.Element(
        "testsuite",
        tests=str(len(cases)),
        failures="0",
        errors="0",
        skipped=str(skipped),
    )
    suite.extend(cases)
    root = ET.Element("testsuites")
    root.append(suite)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def _source_checkout(tmp_path: Path) -> Path:
    source = tmp_path / "nokv"
    source.mkdir()
    (source / ".git").mkdir()
    (source / "Cargo.toml").write_text("[workspace]\n", encoding="utf-8")
    (source / "Cargo.lock").write_text("# lock\n", encoding="utf-8")
    return source


def test_main_neutralizes_ambient_selection_and_requires_fresh_exact_junit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = _source_checkout(tmp_path)
    junit = tmp_path / "evidence.xml"
    junit.write_text("stale evidence", encoding="utf-8")
    monkeypatch.setenv("PYTEST_ADDOPTS", "-k build_cleanup")

    def fake_run(command, *, cwd, env, check):
        assert not junit.exists(), "runner must remove stale evidence before pytest"
        assert "PYTEST_ADDOPTS" not in env
        assert ["-o", "addopts="] == command[4:6]
        assert cwd == _RUNNER_PATH.parents[1]
        assert check is False
        _write_junit(junit)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    assert runner.main(
        ["--nokv-source", str(source), "--junit-xml", str(junit)]
    ) == 0
    captured = capsys.readouterr()
    assert f"evidence_junit_xml: {junit}" in captured.out
    assert captured.err == ""


@pytest.mark.parametrize(
    "junit_kwargs",
    (
        {"include_real": False},
        {"skipped_real": True},
        {"omit_property": "nokv_binary_sha256"},
    ),
)
def test_main_rejects_false_green_junit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    junit_kwargs: dict[str, object],
) -> None:
    source = _source_checkout(tmp_path)
    junit = tmp_path / "evidence.xml"

    def fake_run(command, *, cwd, env, check):
        _write_junit(junit, **junit_kwargs)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    assert runner.main(
        ["--nokv-source", str(source), "--junit-xml", str(junit)]
    ) == 2
    captured = capsys.readouterr()
    assert "evidence_junit_xml:" not in captured.out
    assert "invalid smoke evidence:" in captured.err


def test_main_rejects_success_without_junit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = _source_checkout(tmp_path)
    junit = tmp_path / "missing.xml"
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0),
    )

    assert runner.main(
        ["--nokv-source", str(source), "--junit-xml", str(junit)]
    ) == 2
    captured = capsys.readouterr()
    assert "evidence_junit_xml:" not in captured.out
    assert "cannot read pytest JUnit evidence" in captured.err
