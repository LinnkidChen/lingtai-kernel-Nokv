"""Tests for the intrinsic lingtai-doctor script."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCTOR = ROOT / "src" / "lingtai" / "intrinsic_skills" / "lingtai-doctor" / "scripts" / "doctor.py"


def test_lingtai_doctor_self_test_passes():
    proc = subprocess.run(
        [sys.executable, str(DOCTOR), "--self-test"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "self-test OK" in proc.stdout


def test_lingtai_doctor_json_redacts_env_secrets(tmp_path):
    agent = tmp_path / "project" / ".lingtai" / "mimo"
    agent.mkdir(parents=True)
    (agent / ".agent.json").write_text(
        json.dumps({"name": "mimo", "state": "idle"}), encoding="utf-8"
    )
    (agent / ".agent.heartbeat").write_text("ok", encoding="utf-8")
    (agent / "init.json").write_text(
        json.dumps(
            {
                "mcp": {
                    "telegram": {
                        "type": "stdio",
                        "command": "/definitely/missing/python",
                        "env": {"BOT_TOKEN": "secret-value", "CONFIG_PATH": ".secrets/tg.json"},
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    proc = subprocess.run(
        [sys.executable, str(DOCTOR), "--agent-dir", str(agent), "--json"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert proc.returncode == 1
    assert "secret-value" not in proc.stdout
    data = json.loads(proc.stdout)
    assert data["severity"] == "FAIL"
    assert any(section["name"] == "mcp/addons" for section in data["sections"])
