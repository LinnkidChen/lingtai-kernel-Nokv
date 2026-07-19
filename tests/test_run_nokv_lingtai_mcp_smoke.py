"""Regression tests for the fail-closed real NoKV smoke runner."""
from __future__ import annotations

import importlib.util
import stat
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


def _junit_path_from_command(command: list[str]) -> Path:
    junit_arguments = [item for item in command if item.startswith("--junitxml=")]
    assert len(junit_arguments) == 1
    return Path(junit_arguments[0].partition("=")[2])


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
        private_junit = _junit_path_from_command(command)
        assert private_junit.parent.parent == junit.parent
        assert private_junit.parent.name.startswith(".nokv-lingtai-mcp-smoke-")
        assert stat.S_IMODE(private_junit.parent.stat().st_mode) == 0o700
        assert not private_junit.exists()
        assert junit.read_text(encoding="utf-8") == "stale evidence"
        assert "PYTEST_ADDOPTS" not in env
        assert ["-o", "addopts="] == command[4:6]
        assert cwd == _RUNNER_PATH.parents[1]
        assert check is False
        _write_junit(private_junit)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    assert runner.main(
        ["--nokv-source", str(source), "--junit-xml", str(junit)]
    ) == 0
    captured = capsys.readouterr()
    assert f"evidence_junit_xml: {junit}" in captured.out
    assert captured.err == ""
    runner._validate_junit_evidence(junit)
    assert not list(tmp_path.glob(".nokv-lingtai-mcp-smoke-*"))


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
        _write_junit(_junit_path_from_command(command), **junit_kwargs)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    assert runner.main(
        ["--nokv-source", str(source), "--junit-xml", str(junit)]
    ) == 2
    captured = capsys.readouterr()
    assert "evidence_junit_xml:" not in captured.out
    assert "invalid smoke evidence:" in captured.err


def test_main_does_not_accept_stale_destination_as_fresh_junit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = _source_checkout(tmp_path)
    junit = tmp_path / "evidence.xml"
    _write_junit(junit)
    stale_bytes = junit.read_bytes()
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
    assert junit.read_bytes() == stale_bytes
    assert not list(tmp_path.glob(".nokv-lingtai-mcp-smoke-*"))


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


def test_main_rejects_symlinked_private_junit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = _source_checkout(tmp_path)
    outside_junit = tmp_path / "outside.xml"
    _write_junit(outside_junit)
    junit = tmp_path / "evidence.xml"

    def fake_run(command, *, cwd, env, check):
        private_junit = _junit_path_from_command(command)
        try:
            private_junit.symlink_to(outside_junit)
        except OSError as error:
            pytest.skip(f"symlinks unavailable: {error}")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    assert runner.main(
        ["--nokv-source", str(source), "--junit-xml", str(junit)]
    ) == 2
    captured = capsys.readouterr()
    assert "pytest JUnit evidence is not a private regular file" in captured.err
    assert not junit.exists()
    runner._validate_junit_evidence(outside_junit)
    assert not list(tmp_path.glob(".nokv-lingtai-mcp-smoke-*"))


def test_main_cleans_private_junit_when_atomic_publish_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = _source_checkout(tmp_path)
    junit = tmp_path / "evidence.xml"

    def fake_run(command, *, cwd, env, check):
        _write_junit(_junit_path_from_command(command))
        return subprocess.CompletedProcess(command, 0)

    def fake_replace(source_path: Path, destination_path: Path) -> None:
        assert source_path.is_file()
        assert destination_path == junit
        raise PermissionError("publish denied")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    monkeypatch.setattr(runner.os, "replace", fake_replace)

    assert runner.main(
        ["--nokv-source", str(source), "--junit-xml", str(junit)]
    ) == 2
    captured = capsys.readouterr()
    assert "cannot publish pytest JUnit evidence: publish denied" in captured.err
    assert not junit.exists()
    assert not list(tmp_path.glob(".nokv-lingtai-mcp-smoke-*"))


def test_main_replaces_destination_symlink_without_touching_its_victim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source_checkout(tmp_path)
    victim = tmp_path / "victim.xml"
    victim.write_text("victim must survive\n", encoding="utf-8")
    junit = tmp_path / "evidence.xml"
    try:
        junit.symlink_to(victim)
    except OSError as error:
        pytest.skip(f"symlinks unavailable: {error}")

    def fake_run(command, *, cwd, env, check):
        assert junit.is_symlink()
        assert victim.read_text(encoding="utf-8") == "victim must survive\n"
        _write_junit(_junit_path_from_command(command))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    assert runner.main(
        ["--nokv-source", str(source), "--junit-xml", str(junit)]
    ) == 0
    assert victim.read_text(encoding="utf-8") == "victim must survive\n"
    assert not junit.is_symlink()
    assert stat.S_ISREG(junit.lstat().st_mode)
    runner._validate_junit_evidence(junit)
    assert not list(tmp_path.glob(".nokv-lingtai-mcp-smoke-*"))


def test_main_creates_missing_destination_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source_checkout(tmp_path)
    junit = tmp_path / "new" / "nested" / "evidence.xml"

    def fake_run(command, *, cwd, env, check):
        private_junit = _junit_path_from_command(command)
        assert private_junit.parent.parent == junit.parent
        _write_junit(private_junit)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    assert runner.main(
        ["--nokv-source", str(source), "--junit-xml", str(junit)]
    ) == 0
    runner._validate_junit_evidence(junit)
    assert not list(junit.parent.glob(".nokv-lingtai-mcp-smoke-*"))


def test_main_rejects_non_directory_destination_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = _source_checkout(tmp_path)
    parent = tmp_path / "not-a-directory"
    parent.write_text("occupied\n", encoding="utf-8")
    junit = parent / "evidence.xml"

    def unexpected_run(*args, **kwargs):
        pytest.fail("pytest must not run with a non-directory evidence parent")

    monkeypatch.setattr(runner.subprocess, "run", unexpected_run)

    assert runner.main(
        ["--nokv-source", str(source), "--junit-xml", str(junit)]
    ) == 2
    captured = capsys.readouterr()
    assert "cannot prepare pytest JUnit destination:" in captured.err
    assert parent.read_text(encoding="utf-8") == "occupied\n"


def test_main_rejects_directory_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = _source_checkout(tmp_path)
    junit = tmp_path / "evidence.xml"
    junit.mkdir()

    def unexpected_run(*args, **kwargs):
        pytest.fail("pytest must not run with a directory evidence destination")

    monkeypatch.setattr(runner.subprocess, "run", unexpected_run)

    assert runner.main(
        ["--nokv-source", str(source), "--junit-xml", str(junit)]
    ) == 2
    captured = capsys.readouterr()
    assert "must be absent, a regular file, or a symlink" in captured.err
    assert junit.is_dir()
