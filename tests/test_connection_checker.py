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
    mock_conn = MagicMock()
    mock_connect = MagicMock()
    mock_connect.return_value.__enter__.return_value = mock_conn
    with (
        patch("app.services.health.connection_checker.settings.NEON_DB_URL", "postgresql://test"),
        patch("app.services.health.connection_checker.psycopg.connect", mock_connect),
    ):
        result = _check_neon_postgres()
    assert result.healthy is True
    assert result.name == "postgres"
    mock_connect.assert_called_once()
    assert mock_connect.call_args.kwargs["connect_timeout"] == 5
    mock_conn.cursor.return_value.__enter__.return_value.execute.assert_called_once_with("SELECT 1")
    mock_connect.return_value.__exit__.assert_called_once()


def test_check_neon_postgres_failure_is_sanitized():
    error = RuntimeError("connection failed for postgresql://user:secret@host/db")
    error.sqlstate = "08006"
    with (
        patch("app.services.health.connection_checker.settings.NEON_DB_URL", "postgresql://test"),
        patch("app.services.health.connection_checker.psycopg.connect", side_effect=error),
        patch("app.services.health.connection_checker.logfire.warning") as warning,
    ):
        result = _check_neon_postgres()
    assert result.healthy is False
    assert result.required is False
    assert result.message == "RuntimeError (SQLSTATE 08006)"
    assert "secret" not in result.message
    assert "postgresql://" not in warning.call_args.args[0]


def test_check_neon_postgres_failure_without_sqlstate_reports_exception_type():
    with (
        patch("app.services.health.connection_checker.settings.NEON_DB_URL", "postgresql://test"),
        patch("app.services.health.connection_checker.psycopg.connect", side_effect=TimeoutError("timed out")),
    ):
        result = _check_neon_postgres()
    assert result.healthy is False
    assert result.message == "TimeoutError"


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


def test_check_logfire_reports_successful_export():
    with (
        patch("app.services.health.connection_checker.settings.LOGFIRE_TOKEN", "test-token"),
        patch(
            "app.services.health.connection_checker.probe_logfire_authentication",
            return_value=(True, "Logfire token authenticated by API"),
        ) as probe,
    ):
        from app.services.health.connection_checker import _check_logfire

        result = _check_logfire()
    assert result.healthy is True
    assert "authenticated" in result.message
    probe.assert_called_once()


def test_check_logfire_does_not_treat_configured_token_as_authenticated():
    with (
        patch("app.services.health.connection_checker.settings.LOGFIRE_TOKEN", "test-token"),
        patch(
            "app.services.health.connection_checker.probe_logfire_authentication",
            return_value=(False, "Logfire rejected or could not validate the configured token (HTTP 401)"),
        ),
    ):
        from app.services.health.connection_checker import _check_logfire

        result = _check_logfire()
    assert result.healthy is False
    assert result.required is False
    assert "HTTP 401" in result.message
