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
        NEON_DB_URL="",
        UPSTASH_REDIS_REST_URL="",
        UPSTASH_REDIS_REST_TOKEN="",
    )
    assert config.JINA_API_KEY is None
    assert config.NEON_DB_URL is None
    assert config.redis_url is None


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
