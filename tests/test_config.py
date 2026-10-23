"""Tests for optional local-service configuration."""

import pytest

from app.config import Settings


def test_settings_default_to_free_tier_gemini_model(monkeypatch):
    for name in (
        "GEMINI_MODEL",
        "QDRANT_URL",
        "QDRANT_CLUSTER_ENDPOINT",
        "UPSTASH_REDIS_REST_URL",
        "UPSTASH_REDIS_REST_TOKEN",
    ):
        monkeypatch.delenv(name, raising=False)
    config = Settings(_env_file=None, GEMINI_API_KEY="test-key")
    assert config.GEMINI_MODEL == "gemini-flash-lite-latest"
    assert config.QDRANT_URL == "http://localhost:6333"
    assert config.redis_url is None


def test_empty_optional_credentials_are_unset():
    config = Settings(
        _env_file=None,
        GEMINI_API_KEY="test-key",
        JINA_API_KEY="",
        PORTKEY_API_KEY="",
        PORTKEY_PRIMARY_SLUG="",
        PORTKEY_FALLBACK_SLUG="",
        PORTKEY_PRIMARY_CONFIG_ID="",
        JUDGE_GEMINI_API_KEY="",
        NEON_DB_URL="",
        UPSTASH_REDIS_REST_URL="",
        UPSTASH_REDIS_REST_TOKEN="",
    )
    assert config.JINA_API_KEY is None
    assert config.PORTKEY_API_KEY is None
    assert config.PORTKEY_PRIMARY_CONFIG_ID is None
    assert config.NEON_DB_URL is None
    assert config.redis_url is None
    assert config.judge_gemini_api_key == "test-key"


def test_gemini_judge_settings_and_local_backend_defaults():
    config = Settings(
        _env_file=None,
        GEMINI_API_KEY="runtime-key",
        JUDGE_PROVIDER="gemini",
        JUDGE_GEMINI_API_KEY="judge-key",
        JUDGE_GEMINI_MODEL="gemini-eval-model",
        BACKEND_URL="",
    )
    assert config.judge_gemini_api_key == "judge-key"
    assert config.JUDGE_GEMINI_MODEL == "gemini-eval-model"
    assert config.backend_url == "http://localhost:8000"


def test_logfire_base_url_uses_explicit_setting_or_eu_token():
    eu_config = Settings(
        _env_file=None,
        GEMINI_API_KEY="test-key",
        LOGFIRE_TOKEN="pylf_v2_eu_test",
    )
    explicit_config = Settings(
        _env_file=None,
        GEMINI_API_KEY="test-key",
        LOGFIRE_TOKEN="pylf_v2_eu_test",
        LOGFIRE_BASE_URL="https://logfire.example.com",
    )
    assert eu_config.logfire_base_url == "https://logfire-eu.pydantic.dev"
    assert explicit_config.logfire_base_url == "https://logfire.example.com"


def test_neon_uri_requires_configuration():
    config = Settings(_env_file=None, GEMINI_API_KEY="test-key")
    with pytest.raises(ValueError, match="NEON_DB_URL is not configured"):
        _ = config.postgres_uri


def test_qdrant_cluster_endpoint_alias_is_supported():
    config = Settings(
        _env_file=None,
        GEMINI_API_KEY="test-key",
        QDRANT_CLUSTER_ENDPOINT="https://qdrant.example.com",
    )
    assert config.QDRANT_URL == "https://qdrant.example.com"
