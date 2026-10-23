"""Tests for health and readiness endpoints."""

from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app
from app.services.health.connection_checker import ConnectionResult


def _ok(name: str) -> ConnectionResult:
    return ConnectionResult(name, True, "")


def _fail(name: str, message: str = "unavailable") -> ConnectionResult:
    return ConnectionResult(name, False, message)


def test_health_returns_ok():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ready_when_all_dependencies_healthy():
    results = {
        "postgres": _ok("postgres"),
        "redis": _ok("redis"),
        "qdrant": _ok("qdrant"),
        "gemini": _ok("gemini"),
        "jina_embeddings": _ok("jina_embeddings"),
        "jina_reranker": _ok("jina_reranker"),
    }
    with patch("app.health.check_all_connections", return_value=results):
        client = TestClient(app)
        response = client.get("/ready")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert all(v == "ok" for v in data["checks"].values())


def test_ready_returns_503_when_any_dependency_fails():
    results = {
        "postgres": _ok("postgres"),
        "redis": _ok("redis"),
        "qdrant": _fail("qdrant", "qdrant down"),
        "gemini": _ok("gemini"),
        "jina_embeddings": _ok("jina_embeddings"),
        "jina_reranker": _ok("jina_reranker"),
    }
    with patch("app.health.check_all_connections", return_value=results):
        client = TestClient(app)
        response = client.get("/ready")

    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "not_ready"
    assert data["checks"]["qdrant"].startswith("unavailable")


def test_ready_ignores_unavailable_optional_services():
    results = {
        "qdrant": _ok("qdrant"),
        "gemini": _ok("gemini"),
        "redis": ConnectionResult("redis", False, "not configured", required=False),
        "postgres": ConnectionResult("postgres", False, "not configured", required=False),
    }
    with patch("app.health.check_all_connections", return_value=results):
        response = TestClient(app).get("/ready")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"
