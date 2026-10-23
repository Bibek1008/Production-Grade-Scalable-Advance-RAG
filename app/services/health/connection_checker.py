"""Standalone and startup-time connection health checks for all external services.

Run from the command line:
    python -m app.services.health.connection_checker
"""

from __future__ import annotations

import sys
import threading
import time
from typing import Callable

import logfire
import psycopg
import requests
from qdrant_client import QdrantClient
from redis import Redis

from app.config import settings
from app.gateway import get_portkey_llm
from app.observability import probe_logfire_authentication
from app.postgres import postgres_error_summary
from app.services.retrieval.embedding import get_embedding_dim


class ConnectionResult:
    """Result of a single connectivity check."""

    def __init__(self, name: str, healthy: bool, message: str = "", required: bool = True):
        self.name = name
        self.healthy = healthy
        self.message = message
        self.required = required

    def to_dict(self) -> dict[str, object]:
        status = "ok" if self.healthy else "unavailable"
        if self.message:
            status = f"{status}: {self.message}"
        return {
            "status": status,
            "healthy": self.healthy,
            "message": self.message,
            "required": self.required,
        }


def _failure_message(error: Exception) -> str:
    """Return error metadata without leaking connection credentials."""
    response = getattr(error, "response", None)
    status_code = getattr(response, "status_code", None)
    suffix = f" (HTTP {status_code})" if status_code else ""
    return f"{type(error).__name__}{suffix}"


def _check_neon_postgres() -> ConnectionResult:
    """Verify Neon Postgres is reachable and accepts queries."""
    if not settings.NEON_DB_URL:
        return ConnectionResult("postgres", False, "NEON_DB_URL not set; using in-memory checkpointing", False)
    try:
        with psycopg.connect(settings.postgres_uri, connect_timeout=5) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        return ConnectionResult("postgres", True, "Neon Postgres reachable", False)
    except Exception as e:
        diagnostic = postgres_error_summary(e)
        logfire.warning(f"Postgres health check failed: {diagnostic}")
        return ConnectionResult("postgres", False, diagnostic, False)


def _check_upstash_redis() -> ConnectionResult:
    """Verify Upstash Redis is reachable."""
    if not settings.redis_url:
        return ConnectionResult("redis", False, "Upstash not configured; using in-memory rate limiting", False)
    try:
        r = Redis.from_url(
            settings.redis_url,
            socket_connect_timeout=5,
            socket_timeout=5,
        )
        r.ping()
        return ConnectionResult("redis", True, "Upstash Redis reachable", False)
    except Exception as e:
        logfire.warning(f"Redis health check failed: {_failure_message(e)}")
        return ConnectionResult("redis", False, _failure_message(e), False)


def _check_qdrant() -> ConnectionResult:
    """Verify Qdrant and the configured collection are reachable."""
    try:
        client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
            timeout=5,
        )
        if not client.collection_exists(settings.QDRANT_COLLECTION):
            return ConnectionResult(
                "qdrant",
                False,
                f"Collection '{settings.QDRANT_COLLECTION}' does not exist; ingest documents first.",
            )
        collection = client.get_collection(settings.QDRANT_COLLECTION)
        vectors = collection.config.params.vectors
        if isinstance(vectors, dict) or vectors.size != get_embedding_dim():
            return ConnectionResult(
                "qdrant",
                False,
                f"Configured collection vector dimensions do not match the {get_embedding_dim()}-dimensional retriever.",
            )
        return ConnectionResult("qdrant", True, "Qdrant collection reachable with compatible vector dimensions")
    except Exception as e:
        logfire.warning(f"Qdrant health check failed: {_failure_message(e)}")
        return ConnectionResult("qdrant", False, _failure_message(e))


def _check_portkey() -> ConnectionResult:
    """Verify the configured Portkey route returns a Gemini response."""
    try:
        response = get_portkey_llm().bind(max_tokens=8).invoke("Say hello in one word.")
        if isinstance(response.content, str) and response.content.strip():
            return ConnectionResult("portkey", True, "Configured Portkey generation route reachable")
        raise RuntimeError("Portkey returned an empty response")
    except Exception as e:
        logfire.warning(f"Portkey health check failed: {_failure_message(e)}")
        return ConnectionResult("portkey", False, _failure_message(e))


def _check_jina_embeddings() -> ConnectionResult:
    """Verify Jina Embeddings API accepts a probe request."""
    if not settings.JINA_API_KEY:
        return ConnectionResult("jina_embeddings", True, "JINA_API_KEY not set; local fallback selected", False)
    try:
        response = requests.post(
            "https://api.jina.ai/v1/embeddings",
            headers={
                "Authorization": f"Bearer {settings.JINA_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "jina-embeddings-v3",
                "task": "retrieval.query",
                "normalized": True,
                "input": ["probe"],
            },
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()
        vectors = [item.get("embedding", []) for item in payload.get("data", [])]
        if len(vectors) != 1 or len(vectors[0]) != 1024:
            raise RuntimeError("unexpected embedding count or dimensions")
        return ConnectionResult("jina_embeddings", True, "Jina Embeddings API reachable")
    except Exception as e:
        logfire.warning(f"Jina Embeddings health check failed: {_failure_message(e)}")
        return ConnectionResult("jina_embeddings", False, _failure_message(e))


def _check_jina_reranker() -> ConnectionResult:
    """Verify Jina Reranker API accepts a probe request."""
    if not settings.JINA_API_KEY:
        return ConnectionResult("jina_reranker", True, "JINA_API_KEY not set; reranking is disabled", False)
    try:
        response = requests.post(
            "https://api.jina.ai/v1/rerank",
            headers={
                "Authorization": f"Bearer {settings.JINA_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "jina-reranker-v3",
                "query": "health check",
                "documents": ["document one", "document two"],
                "top_n": 2,
                "return_documents": True,
            },
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()
        if "results" not in payload:
            raise RuntimeError("missing results key")
        return ConnectionResult("jina_reranker", True, "Jina Reranker API reachable", False)
    except Exception as e:
        logfire.warning(f"Jina Reranker health check failed: {_failure_message(e)}")
        return ConnectionResult("jina_reranker", False, _failure_message(e), False)


def _check_logfire() -> ConnectionResult:
    """Verify that Logfire accepts the configured token, not merely that it exists."""
    if not settings.LOGFIRE_TOKEN:
        return ConnectionResult("logfire", False, "LOGFIRE_TOKEN not set — spans dropped", False)
    healthy, message = probe_logfire_authentication()
    return ConnectionResult("logfire", healthy, message, False)


def _check_langsmith() -> ConnectionResult:
    """Verify LangSmith API key is valid and the endpoint is reachable."""
    if not settings.LANGSMITH_API_KEY:
        return ConnectionResult("langsmith", False, "LANGSMITH_API_KEY not set — tracing disabled", False)
    try:
        response = requests.get(
            f"{settings.LANGSMITH_ENDPOINT}/ok",
            headers={"x-api-key": settings.LANGSMITH_API_KEY},
            timeout=5,
        )
        response.raise_for_status()
        return ConnectionResult(
            "langsmith",
            True,
            f"LangSmith reachable (project: {settings.LANGSMITH_PROJECT})",
            False,
        )
    except Exception as e:
        logfire.warning(f"LangSmith health check failed: {_failure_message(e)}")
        return ConnectionResult("langsmith", False, _failure_message(e), False)


# Ordered list of all checks to run during startup and /ready.
_CHECKERS: list[Callable[[], ConnectionResult]] = [
    _check_neon_postgres,
    _check_upstash_redis,
    _check_qdrant,
    _check_portkey,
    _check_jina_embeddings,
    _check_jina_reranker,
    _check_logfire,
    _check_langsmith,
]

_CACHE_TTL_SECONDS = 30
_connection_cache: dict[str, ConnectionResult] = {}
_connection_cache_time = 0.0
_connection_cache_lock = threading.Lock()


def check_all_connections(*, refresh: bool = False) -> dict[str, ConnectionResult]:
    """Return checks cached for 30 seconds to limit cost and quota-consuming probes."""
    global _connection_cache, _connection_cache_time
    with _connection_cache_lock:
        now = time.monotonic()
        if not refresh and now - _connection_cache_time < _CACHE_TTL_SECONDS and _connection_cache:
            return dict(_connection_cache)
        results: dict[str, ConnectionResult] = {}
        for checker in _CHECKERS:
            result = checker()
            results[result.name] = result
        _connection_cache = results
        _connection_cache_time = time.monotonic()
        return dict(results)


def log_connection_summary(results: dict[str, ConnectionResult]) -> bool:
    """Log a human-readable summary. Returns True if required checks passed."""
    healthy = all(r.healthy for r in results.values() if r.required)
    for name, result in results.items():
        icon = "✅" if result.healthy else ("⚠️" if not result.required else "❌")
        logfire.info(f"{icon} {name}: {result.message or result.to_dict()['status']}")
    if healthy:
        logfire.info("🟢 All required connections healthy.")
    else:
        logfire.warning("🟠 Some external connections are unavailable.")
    return healthy


def _print_cli_report(results: dict[str, ConnectionResult]) -> int:
    """Print a CLI report and return an exit code."""
    healthy = True
    print("\nExternal Connection Health Report")
    print("=" * 50)
    for name, result in results.items():
        status = "OK" if result.healthy else ("OPTIONAL" if not result.required else "FAIL")
        print(f"{status:4} {name:20} {result.message}")
        if not result.healthy and result.required:
            healthy = False
    print("=" * 50)
    if healthy:
        print("All required connections healthy.")
        return 0
    print("One or more connections failed.")
    return 1


if __name__ == "__main__":
    sys.exit(_print_cli_report(check_all_connections(refresh=True)))
