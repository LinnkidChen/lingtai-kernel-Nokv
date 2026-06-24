"""Tests for generic MCP account-identity discoverability.

The curated addon servers (telegram, feishu, wechat, whatsapp) each persist a
non-secret identity document to ``system/mcp_identities/<name>.json`` using the
shared ``lingtai.mcp.identity.v1`` schema. Before this change those documents
were only reachable via each addon's own ``accounts`` action — invisible from
the generic ``mcp(action="show")`` surface that agents use to discover which
MCP servers they have.

These tests cover the generic reader (``read_identities``) and its surfacing
through ``mcp(action="show")`` and the system-prompt registry XML, and — most
importantly — prove that NO secret-shaped fields are ever propagated, even when
a (hypothetical) malformed identity file on disk contains them.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from lingtai.agent import Agent
from lingtai.core.mcp import (
    IDENTITY_SAFE_ACCOUNT_KEYS,
    read_identities,
)
from tests._service_helpers import make_gemini_mock_service as make_mock_service


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _write_identity(workdir: Path, name: str, payload: dict) -> Path:
    path = workdir / "system" / "mcp_identities" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _telegram_identity() -> dict:
    return {
        "schema": "lingtai.mcp.identity.v1",
        "mcp": "telegram",
        "generated_at": "2026-06-24T10:00:00+00:00",
        "last_verified_at": "2026-06-24T09:59:00+00:00",
        "accounts": [
            {
                "alias": "main",
                "bot_id": 123456789,
                "bot_username": "my_agent_bot",
                "bot_display_name": "My Agent",
                "is_bot": True,
                "last_verified_at": "2026-06-24T09:59:00+00:00",
                "allowed_users_count": 2,
                "contact_count": 5,
            }
        ],
    }


def _mk_agent(tmp_path: Path, *, addons=None):
    workdir = tmp_path / "agent"
    return Agent(
        service=make_mock_service(),
        agent_name="test",
        working_dir=workdir,
        capabilities={"mcp": {}},
        addons=addons,
    ), workdir


# ---------------------------------------------------------------------------
# read_identities — the generic reader
# ---------------------------------------------------------------------------

def test_read_identities_empty_when_no_dir(tmp_path):
    assert read_identities(tmp_path / "agent") == {}


def test_read_identities_surfaces_safe_fields(tmp_path):
    workdir = tmp_path / "agent"
    _write_identity(workdir, "telegram", _telegram_identity())

    ids = read_identities(workdir)
    assert "telegram" in ids
    tg = ids["telegram"]
    assert tg["mcp"] == "telegram"
    assert tg["account_count"] == 1
    assert tg["last_verified_at"] == "2026-06-24T09:59:00+00:00"
    acct = tg["accounts"][0]
    assert acct["alias"] == "main"
    assert acct["bot_username"] == "my_agent_bot"
    assert acct["bot_id"] == 123456789
    assert acct["bot_display_name"] == "My Agent"


def test_read_identities_ignores_wrong_schema(tmp_path):
    workdir = tmp_path / "agent"
    _write_identity(
        workdir,
        "telegram",
        {"schema": "something.else.v9", "mcp": "telegram", "accounts": []},
    )
    assert read_identities(workdir) == {}


def test_read_identities_skips_malformed_json(tmp_path):
    workdir = tmp_path / "agent"
    path = workdir / "system" / "mcp_identities" / "telegram.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    # Malformed file must not crash the reader; it is simply skipped.
    assert read_identities(workdir) == {}


def test_read_identities_handles_multiple_servers(tmp_path):
    workdir = tmp_path / "agent"
    _write_identity(workdir, "telegram", _telegram_identity())
    _write_identity(
        workdir,
        "feishu",
        {
            "schema": "lingtai.mcp.identity.v1",
            "mcp": "feishu",
            "generated_at": "2026-06-24T10:00:00+00:00",
            "accounts": [{"alias": "ops", "app_id": "cli_xxx"}],
        },
    )
    ids = read_identities(workdir)
    assert set(ids) == {"telegram", "feishu"}


# ---------------------------------------------------------------------------
# SECRET SAFETY — the heart of the change
# ---------------------------------------------------------------------------

SECRET_KEYS = [
    "bot_token",
    "token",
    "password",
    "email_password",
    "app_secret",
    "client_secret",
    "refresh_token",
    "access_token",
    "api_key",
    "secret",
    "headers",
    "authorization",
]


def test_secret_fields_are_stripped_from_accounts(tmp_path):
    """Even if an identity file on disk somehow contains secret-shaped keys,
    the generic reader must project to the safe allowlist and drop them."""
    workdir = tmp_path / "agent"
    poisoned_account = {
        "alias": "main",
        "bot_username": "my_agent_bot",
        "bot_id": 1,
        # Secrets that must NEVER survive projection:
        "bot_token": "123456:AAH-supersecrettoken",
        "password": "hunter2",
        "app_secret": "shhh",
        "refresh_token": "rt_abc",
        "access_token": "at_xyz",
        "api_key": "sk-live-leak",
        "headers": {"Authorization": "Bearer leak"},
        "authorization": "Bearer leak",
        "secret": "leak",
    }
    _write_identity(
        workdir,
        "telegram",
        {
            "schema": "lingtai.mcp.identity.v1",
            "mcp": "telegram",
            "accounts": [poisoned_account],
        },
    )

    ids = read_identities(workdir)
    acct = ids["telegram"]["accounts"][0]
    blob = json.dumps(ids)

    # Safe fields survive.
    assert acct["alias"] == "main"
    assert acct["bot_username"] == "my_agent_bot"
    # No secret-shaped key survives, anywhere in the serialized output.
    for key in SECRET_KEYS:
        assert key not in acct, f"secret key {key!r} leaked into account"
    assert "supersecrettoken" not in blob
    assert "hunter2" not in blob
    assert "Bearer leak" not in blob
    assert "sk-live-leak" not in blob


def test_allowlist_keys_are_all_non_secret():
    """Guard: the safe-key allowlist must not contain any secret-shaped key."""
    for key in IDENTITY_SAFE_ACCOUNT_KEYS:
        low = key.lower()
        assert "token" not in low
        assert "secret" not in low
        assert "password" not in low
        assert low not in {"api_key", "headers", "authorization"}


def test_unknown_account_fields_are_dropped_by_default(tmp_path):
    """Defense in depth: keys not on the allowlist are dropped, not passed
    through — so a future producer adding a sensitive field cannot leak it
    through the generic reader without an explicit allowlist update."""
    workdir = tmp_path / "agent"
    _write_identity(
        workdir,
        "telegram",
        {
            "schema": "lingtai.mcp.identity.v1",
            "mcp": "telegram",
            "accounts": [{"alias": "main", "some_future_private_field": "x"}],
        },
    )
    acct = read_identities(workdir)["telegram"]["accounts"][0]
    assert acct == {"alias": "main"}


# ---------------------------------------------------------------------------
# mcp(action="show") surfacing
# ---------------------------------------------------------------------------

def test_show_action_includes_identity_when_present(tmp_path):
    agent, workdir = _mk_agent(tmp_path, addons=["telegram"])
    _write_identity(workdir, "telegram", _telegram_identity())

    handler = agent._tool_handlers.get("mcp")
    result = handler({"action": "show"})

    registered = {r["name"]: r for r in result["registered"]}
    assert "telegram" in registered
    ident = registered["telegram"].get("identity")
    assert ident is not None
    assert ident["account_count"] == 1
    assert ident["accounts"][0]["bot_username"] == "my_agent_bot"
    # No secret anywhere in the identity-bearing portion of the payload.
    # (The `mcp_manual` field embeds SKILL.md docs that legitimately mention
    # field names like "bot_token", so we scope the check to `registered`.)
    assert "bot_token" not in json.dumps(result["registered"])


def test_show_action_omits_identity_when_absent(tmp_path):
    agent, workdir = _mk_agent(tmp_path, addons=["telegram"])
    # No identity file written.
    handler = agent._tool_handlers.get("mcp")
    result = handler({"action": "show"})
    registered = {r["name"]: r for r in result["registered"]}
    assert "identity" not in registered["telegram"]


def test_show_action_ignores_identity_without_registry_match(tmp_path):
    """An identity file for an MCP not in the registry should not invent a
    registered entry."""
    agent, workdir = _mk_agent(tmp_path, addons=["telegram"])
    _write_identity(workdir, "ghost", _telegram_identity())
    handler = agent._tool_handlers.get("mcp")
    result = handler({"action": "show"})
    names = {r["name"] for r in result["registered"]}
    assert "ghost" not in names


# ---------------------------------------------------------------------------
# System-prompt XML surfacing
# ---------------------------------------------------------------------------

def test_identity_rendered_into_prompt_xml(tmp_path):
    agent, workdir = _mk_agent(tmp_path, addons=["telegram"])
    _write_identity(workdir, "telegram", _telegram_identity())

    # Re-render the registry now that the identity file exists.
    handler = agent._tool_handlers.get("mcp")
    handler({"action": "show"})

    section = agent._prompt_manager._sections.get("mcp")
    body = section.body if hasattr(section, "body") else str(section)
    assert "<identity>" in body
    assert "my_agent_bot" in body
    # Secret must never reach the prompt.
    assert "bot_token" not in body


def test_prompt_xml_has_no_secret_even_with_poisoned_file(tmp_path):
    agent, workdir = _mk_agent(tmp_path, addons=["telegram"])
    _write_identity(
        workdir,
        "telegram",
        {
            "schema": "lingtai.mcp.identity.v1",
            "mcp": "telegram",
            "accounts": [
                {"alias": "main", "bot_username": "b", "bot_token": "123:SECRET"}
            ],
        },
    )
    handler = agent._tool_handlers.get("mcp")
    handler({"action": "show"})
    section = agent._prompt_manager._sections.get("mcp")
    body = section.body if hasattr(section, "body") else str(section)
    assert "SECRET" not in body
    assert "bot_token" not in body
