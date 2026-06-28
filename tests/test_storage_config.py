from __future__ import annotations

import json

import pytest


def _storage_api():
    from lingtai.services.storage_config import (
        StorageConfigError,
        parse_storage_config,
    )

    return parse_storage_config, StorageConfigError


def _enabled_config(**overrides):
    storage = {
        "enabled": True,
        "backend": "nokv",
        "nokv": {
            "namespace_root": "/lingtai/projects/${project_hash}/agents/${agent_name}",
            "metadata_addr_env": "NOKV_METADATA_ADDR",
            "bucket_env": "NOKV_BUCKET",
            "endpoint_env": "NOKV_ENDPOINT",
            "access_key_id_env": "AWS_ACCESS_KEY_ID",
            "secret_access_key_env": "AWS_SECRET_ACCESS_KEY",
            "region_env": "AWS_REGION",
        },
        "mounts": ["artifacts", "reports", "checkpoints", "knowledge"],
    }
    storage.update(overrides)
    return {"storage": storage}


def _env(**overrides):
    values = {
        "NOKV_METADATA_ADDR": "127.0.0.1:7777",
        "NOKV_BUCKET": "nokv",
        "NOKV_ENDPOINT": "http://127.0.0.1:9000",
        "AWS_ACCESS_KEY_ID": "test-access-key",
        "AWS_SECRET_ACCESS_KEY": "super-secret-value",
        "AWS_REGION": "us-west-2",
    }
    values.update(overrides)
    return values


def test_missing_storage_block_defaults_to_disabled_local(tmp_path):
    parse_storage_config, _ = _storage_api()

    resolved = parse_storage_config(
        {},
        agent_dir=tmp_path / ".lingtai" / "alice",
        project_root=tmp_path,
        project_hash="abc123",
        agent_name="alice",
        environ={},
    )

    assert resolved.enabled is False
    assert resolved.backend == "local"
    assert resolved.routes == []
    assert resolved.to_status()["enabled"] is False


@pytest.mark.parametrize("enabled_value", ["false", "0"])
def test_storage_enabled_rejects_non_boolean_values(tmp_path, enabled_value):
    parse_storage_config, StorageConfigError = _storage_api()

    with pytest.raises(StorageConfigError, match="storage.enabled"):
        parse_storage_config(
            _enabled_config(enabled=enabled_value),
            agent_dir=tmp_path / ".lingtai" / "alice",
            project_root=tmp_path,
            project_hash="abc123",
            agent_name="alice",
            environ=_env(),
        )


def test_enabled_storage_expands_default_selected_mounts(tmp_path):
    parse_storage_config, _ = _storage_api()
    agent_dir = tmp_path / ".lingtai" / "alice"

    resolved = parse_storage_config(
        _enabled_config(),
        agent_dir=agent_dir,
        project_root=tmp_path,
        project_hash="abc123",
        agent_name="alice",
        environ=_env(),
    )

    assert resolved.enabled is True
    assert [route.mount for route in resolved.routes] == [
        "artifacts",
        "reports",
        "checkpoints",
        "knowledge",
    ]
    assert [route.local_root for route in resolved.routes] == [
        agent_dir / "artifacts",
        agent_dir / "reports",
        agent_dir / "checkpoints",
        agent_dir / "knowledge",
    ]
    assert [route.remote_root for route in resolved.routes] == [
        "/lingtai/projects/abc123/agents/alice/artifacts",
        "/lingtai/projects/abc123/agents/alice/reports",
        "/lingtai/projects/abc123/agents/alice/checkpoints",
        "/lingtai/projects/abc123/agents/alice/knowledge",
    ]


def test_enabled_storage_rejects_unknown_mount(tmp_path):
    parse_storage_config, StorageConfigError = _storage_api()

    with pytest.raises(StorageConfigError, match="mailbox"):
        parse_storage_config(
            _enabled_config(mounts=["artifacts", "mailbox"]),
            agent_dir=tmp_path / ".lingtai" / "alice",
            project_root=tmp_path,
            project_hash="abc123",
            agent_name="alice",
            environ=_env(),
        )


def test_enabled_storage_requires_absolute_namespace_root(tmp_path):
    parse_storage_config, StorageConfigError = _storage_api()

    config = _enabled_config(
        nokv={
            "namespace_root": "relative/projects/${project_hash}",
            "metadata_addr_env": "NOKV_METADATA_ADDR",
            "bucket_env": "NOKV_BUCKET",
            "endpoint_env": "NOKV_ENDPOINT",
            "access_key_id_env": "AWS_ACCESS_KEY_ID",
            "secret_access_key_env": "AWS_SECRET_ACCESS_KEY",
            "region_env": "AWS_REGION",
        }
    )
    with pytest.raises(StorageConfigError, match="absolute"):
        parse_storage_config(
            config,
            agent_dir=tmp_path / ".lingtai" / "alice",
            project_root=tmp_path,
            project_hash="abc123",
            agent_name="alice",
            environ=_env(),
        )


def test_enabled_storage_rejects_unresolved_namespace_template_variables(tmp_path):
    parse_storage_config, StorageConfigError = _storage_api()

    config = _enabled_config(
        nokv={
            "namespace_root": "/lingtai/projects/${project}/agents/${agent_name}",
            "metadata_addr_env": "NOKV_METADATA_ADDR",
            "bucket_env": "NOKV_BUCKET",
            "endpoint_env": "NOKV_ENDPOINT",
            "access_key_id_env": "AWS_ACCESS_KEY_ID",
            "secret_access_key_env": "AWS_SECRET_ACCESS_KEY",
            "region_env": "AWS_REGION",
        }
    )
    with pytest.raises(StorageConfigError, match="allowed variables"):
        parse_storage_config(
            config,
            agent_dir=tmp_path / ".lingtai" / "alice",
            project_root=tmp_path,
            project_hash="abc123",
            agent_name="alice",
            environ=_env(),
        )


def test_enabled_storage_requires_declared_environment(tmp_path):
    parse_storage_config, StorageConfigError = _storage_api()
    environ = _env()
    del environ["NOKV_BUCKET"]

    with pytest.raises(StorageConfigError, match="NOKV_BUCKET"):
        parse_storage_config(
            _enabled_config(),
            agent_dir=tmp_path / ".lingtai" / "alice",
            project_root=tmp_path,
            project_hash="abc123",
            agent_name="alice",
            environ=environ,
        )


def test_resolved_storage_status_does_not_leak_secret_values(tmp_path):
    parse_storage_config, _ = _storage_api()

    resolved = parse_storage_config(
        _enabled_config(),
        agent_dir=tmp_path / ".lingtai" / "alice",
        project_root=tmp_path,
        project_hash="abc123",
        agent_name="alice",
        environ=_env(),
    )
    status_text = json.dumps(resolved.to_status(), sort_keys=True)

    assert "super-secret-value" not in status_text
    assert "test-access-key" not in status_text
    assert "AWS_SECRET_ACCESS_KEY" not in status_text
    assert "secret_access_key" not in status_text
    assert "127.0.0.1:7777" in status_text
    assert "nokv" in status_text


def test_enabled_storage_rejects_public_endpoint_env_mapped_to_secret_name(tmp_path):
    parse_storage_config, StorageConfigError = _storage_api()
    secret_value = "endpoint-secret-must-not-echo"
    config = _enabled_config(
        nokv={
            "namespace_root": "/lingtai/projects/${project_hash}/agents/${agent_name}",
            "metadata_addr_env": "NOKV_METADATA_ADDR",
            "bucket_env": "NOKV_BUCKET",
            "endpoint_env": "AWS_SECRET_ACCESS_KEY",
            "access_key_id_env": "AWS_ACCESS_KEY_ID",
            "secret_access_key_env": "AWS_SECRET_ACCESS_KEY",
            "region_env": "AWS_REGION",
        }
    )

    with pytest.raises(StorageConfigError) as excinfo:
        parse_storage_config(
            config,
            agent_dir=tmp_path / ".lingtai" / "alice",
            project_root=tmp_path,
            project_hash="abc123",
            agent_name="alice",
            environ=_env(AWS_SECRET_ACCESS_KEY=secret_value),
        )

    assert secret_value not in str(excinfo.value)


@pytest.mark.parametrize("secret_env_name", ["FOO_TOKEN", "PASSWORD"])
def test_enabled_storage_rejects_public_metadata_env_mapped_to_secret_name(
    tmp_path, secret_env_name
):
    parse_storage_config, StorageConfigError = _storage_api()
    secret_value = "metadata-secret-must-not-echo"
    config = _enabled_config(
        nokv={
            "namespace_root": "/lingtai/projects/${project_hash}/agents/${agent_name}",
            "metadata_addr_env": secret_env_name,
            "bucket_env": "NOKV_BUCKET",
            "endpoint_env": "NOKV_ENDPOINT",
            "access_key_id_env": "AWS_ACCESS_KEY_ID",
            "secret_access_key_env": "AWS_SECRET_ACCESS_KEY",
            "region_env": "AWS_REGION",
        }
    )

    with pytest.raises(StorageConfigError) as excinfo:
        parse_storage_config(
            config,
            agent_dir=tmp_path / ".lingtai" / "alice",
            project_root=tmp_path,
            project_hash="abc123",
            agent_name="alice",
            environ=_env(**{secret_env_name: secret_value}),
        )

    assert secret_value not in str(excinfo.value)


def test_enabled_storage_rejects_credential_bearing_endpoint_without_secret_echo(tmp_path):
    parse_storage_config, StorageConfigError = _storage_api()
    secret_endpoint = "https://user:pass@example.com?token=secret"

    with pytest.raises(StorageConfigError) as excinfo:
        parse_storage_config(
            _enabled_config(),
            agent_dir=tmp_path / ".lingtai" / "alice",
            project_root=tmp_path,
            project_hash="abc123",
            agent_name="alice",
            environ=_env(NOKV_ENDPOINT=secret_endpoint),
        )

    error_text = str(excinfo.value)
    assert "endpoint" in error_text
    assert "user" not in error_text
    assert "pass" not in error_text
    assert "token" not in error_text
    assert "secret" not in error_text
    assert secret_endpoint not in error_text


def test_enabled_storage_rejects_credential_bearing_metadata_addr_without_status_echo(tmp_path):
    parse_storage_config, StorageConfigError = _storage_api()
    secret_metadata_addr = "https://user:pass@example.com?token=secret"
    config = _enabled_config(
        nokv={
            "namespace_root": "/lingtai/projects/${project_hash}/agents/${agent_name}",
            "metadata_addr_env": "CUSTOM_METADATA_ADDR",
            "bucket_env": "NOKV_BUCKET",
            "endpoint_env": "NOKV_ENDPOINT",
            "access_key_id_env": "AWS_ACCESS_KEY_ID",
            "secret_access_key_env": "AWS_SECRET_ACCESS_KEY",
            "region_env": "AWS_REGION",
        }
    )

    with pytest.raises(StorageConfigError) as excinfo:
        parse_storage_config(
            config,
            agent_dir=tmp_path / ".lingtai" / "alice",
            project_root=tmp_path,
            project_hash="abc123",
            agent_name="alice",
            environ=_env(CUSTOM_METADATA_ADDR=secret_metadata_addr),
        )

    error_text = str(excinfo.value)
    assert "metadata" in error_text
    assert "user" not in error_text
    assert "pass" not in error_text
    assert "token" not in error_text
    assert "secret" not in error_text
    assert secret_metadata_addr not in error_text


def test_enabled_storage_rejects_bare_credential_bearing_metadata_addr_without_echo(tmp_path):
    parse_storage_config, StorageConfigError = _storage_api()
    secret_metadata_addr = "user:pass@example.com:7777"
    config = _enabled_config(
        nokv={
            "namespace_root": "/lingtai/projects/${project_hash}/agents/${agent_name}",
            "metadata_addr_env": "CUSTOM_METADATA_ADDR",
            "bucket_env": "NOKV_BUCKET",
            "endpoint_env": "NOKV_ENDPOINT",
            "access_key_id_env": "AWS_ACCESS_KEY_ID",
            "secret_access_key_env": "AWS_SECRET_ACCESS_KEY",
            "region_env": "AWS_REGION",
        }
    )

    with pytest.raises(StorageConfigError) as excinfo:
        parse_storage_config(
            config,
            agent_dir=tmp_path / ".lingtai" / "alice",
            project_root=tmp_path,
            project_hash="abc123",
            agent_name="alice",
            environ=_env(CUSTOM_METADATA_ADDR=secret_metadata_addr),
        )

    error_text = str(excinfo.value)
    assert "metadata" in error_text
    assert "user" not in error_text
    assert "pass" not in error_text
    assert "example.com" not in error_text
    assert "7777" not in error_text
    assert secret_metadata_addr not in error_text


def test_enabled_storage_rejects_bare_credential_bearing_endpoint_without_echo(tmp_path):
    parse_storage_config, StorageConfigError = _storage_api()
    secret_endpoint = "user:pass@example.com:9000"

    with pytest.raises(StorageConfigError) as excinfo:
        parse_storage_config(
            _enabled_config(),
            agent_dir=tmp_path / ".lingtai" / "alice",
            project_root=tmp_path,
            project_hash="abc123",
            agent_name="alice",
            environ=_env(NOKV_ENDPOINT=secret_endpoint),
        )

    error_text = str(excinfo.value)
    assert "endpoint" in error_text
    assert "user" not in error_text
    assert "pass" not in error_text
    assert "example.com" not in error_text
    assert "9000" not in error_text
    assert secret_endpoint not in error_text
