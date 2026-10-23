"""Tests for app.services.health.connection_checker."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import requests

from app.services.health.connection_checker import (
    ConnectionResult,
    _check_jina_embeddings,
    _check_jina_reranker,
    _check_neon_postgres,
    _check_portkey,
    _check_qdrant,
    _check_upstash_redis,
    check_all_connections,
)


def _ok(name: str) -> ConnectionResult:
    return ConnectionResult(name, True, "ok")


def _fail(name: str, message: str = "fail") -> ConnectionResult:
    return ConnectionResult(name, False, message)


def test_check_neon_postgres_success():
    mock_pool = MagicMock()
    mock_conn = MagicMock()
    mock_pool.getconn.return_value = mock_conn
    with (
        patch("app.services.health.connection_checker.settings.NEON_DB_URL", "postgresql://test"),
        patch("app.services.health.connection_checker.ConnectionPool", return_value=mock_pool),
    ):
        result = _check_neon_postgres()
    assert result.healthy is True
    assert result.name == "postgres"


def test_check_neon_postgres_failure():
    with (
        patch("app.services.health.connection_checker.settings.NEON_DB_URL", "postgresql://test"),
        patch("app.services.health.connection_checker.ConnectionPool", side_effect=Exception("conn refused")),
    ):
        result = _check_neon_postgres()
    assert result.healthy is False
    assert result.required is False
    assert "Exception" in result.message


def test_check_neon_postgres_unconfigured_uses_memory_saver():
    with patch("app.services.health.connection_checker.settings.NEON_DB_URL", None):
        result = _check_neon_postgres()
    assert result.healthy is False
    assert result.required is False
    assert "in-memory checkpointing" in result.message


def test_check_upstash_redis_success():
    mock_redis = MagicMock()
    with (
        patch("app.services.health.connection_checker.settings.UPSTASH_REDIS_REST_URL", "https://test.example.com"),
        patch("app.services.health.connection_checker.settings.UPSTASH_REDIS_REST_TOKEN", "test-token"),
        patch("app.services.health.connection_checker.Redis.from_url", return_value=mock_redis),
    ):
        result = _check_upstash_redis()
    assert result.healthy is True
    assert result.name == "redis"
    mock_redis.ping.assert_called_once()


def test_check_upstash_redis_failure():
    with (
        patch("app.services.health.connection_checker.settings.UPSTASH_REDIS_REST_URL", "https://test.example.com"),
        patch("app.services.health.connection_checker.settings.UPSTASH_REDIS_REST_TOKEN", "test-token"),
        patch("app.services.health.connection_checker.Redis.from_url", side_effect=Exception("timeout")),
    ):
        result = _check_upstash_redis()
    assert result.healthy is False


def test_check_upstash_redis_unconfigured_uses_memory():
    with (
        patch("app.services.health.connection_checker.settings.UPSTASH_REDIS_REST_URL", None),
        patch("app.services.health.connection_checker.settings.UPSTASH_REDIS_REST_TOKEN", None),
    ):
        result = _check_upstash_redis()
    assert result.healthy is False
    assert result.required is False
    assert "in-memory rate limiting" in result.message


def test_check_qdrant_success():
    mock_client = MagicMock()
    mock_client.get_collection.return_value = SimpleNamespace(
        config=SimpleNamespace(params=SimpleNamespace(vectors=SimpleNamespace(size=1024)))
    )
    with (
        patch("app.services.health.connection_checker.QdrantClient", return_value=mock_client),
        patch("app.services.health.connection_checker.get_embedding_dim", return_value=1024),
    ):
        result = _check_qdrant()
    assert result.healthy is True
    assert result.name == "qdrant"
    mock_client.collection_exists.assert_called_once()


def test_check_qdrant_requires_collection():
    mock_client = MagicMock()
    mock_client.collection_exists.return_value = False
    with patch("app.services.health.connection_checker.QdrantClient", return_value=mock_client):
        result = _check_qdrant()
    assert result.healthy is False
    assert "does not exist" in result.message


def test_check_qdrant_failure():
    with patch("app.services.health.connection_checker.QdrantClient", side_effect=Exception("unauthorized")):
        result = _check_qdrant()
    assert result.healthy is False


def test_check_portkey_success():
    mock_resp = MagicMock(content="hi")
    with patch("app.services.health.connection_checker.get_portkey_llm") as mock_llm:
        mock_llm.return_value.bind.return_value.invoke.return_value = mock_resp
        result = _check_portkey()
    assert result.healthy is True
    assert result.name == "portkey"


def test_check_portkey_empty_response():
    mock_resp = MagicMock(content="")
    with patch("app.services.health.connection_checker.get_portkey_llm") as mock_llm:
        mock_llm.return_value.bind.return_value.invoke.return_value = mock_resp
        result = _check_portkey()
    assert result.healthy is False


def test_check_jina_embeddings_success():
    mock_response = MagicMock()
    mock_response.json.return_value = {"data": [{"embedding": [0.1] * 1024}]}
    mock_response.raise_for_status = MagicMock()
    with (
        patch("app.services.health.connection_checker.settings.JINA_API_KEY", "test-key"),
        patch("app.services.health.connection_checker.requests.post", return_value=mock_response),
    ):
        result = _check_jina_embeddings()
    assert result.healthy is True
    assert result.name == "jina_embeddings"


def test_check_jina_embeddings_no_key():
    with patch("app.services.health.connection_checker.settings.JINA_API_KEY", None):
        result = _check_jina_embeddings()
    assert result.healthy is True
    assert result.required is False
    assert "JINA_API_KEY not set" in result.message


def test_check_jina_reranker_success():
    mock_response = MagicMock()
    mock_response.json.return_value = {"results": [{"index": 0}]}
    mock_response.raise_for_status = MagicMock()
    with (
        patch("app.services.health.connection_checker.settings.JINA_API_KEY", "test-key"),
        patch("app.services.health.connection_checker.requests.post", return_value=mock_response),
    ):
        result = _check_jina_reranker()
    assert result.healthy is True
    assert result.name == "jina_reranker"


def test_check_jina_reranker_failure():
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = requests.HTTPError("service unavailable")
    with (
        patch("app.services.health.connection_checker.settings.JINA_API_KEY", "test-key"),
        patch("app.services.health.connection_checker.requests.post", return_value=mock_response),
    ):
        result = _check_jina_reranker()
    assert result.healthy is False
    assert result.required is False
    assert result.name == "jina_reranker"


def test_check_all_connections_returns_all_results():
    mock_checkers = [
        lambda: _ok("postgres"),
        lambda: _ok("redis"),
        lambda: _ok("qdrant"),
        lambda: _ok("gemini"),
        lambda: _ok("jina_embeddings"),
        lambda: _fail("jina_reranker"),
    ]
    with patch("app.services.health.connection_checker._CHECKERS", mock_checkers):
        results = check_all_connections()
    assert len(results) == 6
    assert results["postgres"].healthy is True
    assert results["jina_reranker"].healthy is False
