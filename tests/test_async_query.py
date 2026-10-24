"""Tests for synchronous /query endpoint."""

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.main import app


def test_query_returns_direct_answer():
    """/query should run the pipeline synchronously and return the answer."""
    mock_state = {
        "final_answer": "hello",
        "plan": ["Intent: Technical", "Search Term: hi"],
        "status": "Response generated.",
        "documents": ["doc1"],
    }
    mock_agent = MagicMock()
    mock_agent.invoke.return_value = mock_state

    # Ensure rag_agent exists on app.state before TestClient uses it.
    app.state.rag_agent = mock_agent

    client = TestClient(app)
    with (
        patch("app.main.guard") as mock_guard,
        patch("app.main.log_query_stage") as stage_log,
    ):
        mock_guard.return_value = (False, "")
        response = client.post("/query", json={"q": "hi", "thread_id": "t1"})

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "hello"
    assert data["status"] == "Response generated."
    assert data["sources"] == ["doc1"]
    mock_agent.invoke.assert_called_once()
    assert [(call.args[0], call.args[1]) for call in stage_log.call_args_list] == [
        ("query", "received_and_validated"),
        ("guardrails", "started"),
        ("guardrails", "completed"),
        ("graph", "started"),
        ("graph", "completed"),
    ]


def test_query_blocks_guardrails():
    """/query should return a blocked response when guardrails fire."""
    client = TestClient(app)
    with (
        patch("app.main.guard") as mock_guard,
        patch("app.main.log_query_stage") as stage_log,
    ):
        mock_guard.return_value = (True, "Blocked message")
        response = client.post("/query", json={"q": "bad prompt", "thread_id": "t2"})

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "Blocked by guardrails."
    assert data["answer"] == "Blocked message"
    assert [(call.args[0], call.args[1]) for call in stage_log.call_args_list] == [
        ("query", "received_and_validated"),
        ("guardrails", "started"),
        ("guardrails", "completed"),
    ]


def test_query_stage_log_has_request_id_and_elapsed_time():
    from app import logging as request_logging

    request_id = "2c80512f-8d09-4c2c-9096-1f34c7753f0a"
    with request_logging.request_id_context(request_id):
        with patch("app.logging.logfire.info") as info:
            request_logging.log_query_stage(
                "guardrails",
                "completed",
                started_at=request_logging.time.perf_counter(),
                outcome="allowed",
                api_key="test-secret",
                query="private query",
                provider="private query",
                candidate_count=1_000_001,
                error=RuntimeError("private query test-secret"),
            )

        attributes = info.call_args.kwargs
        assert attributes["request_id"] == request_id
        assert attributes["stage"] == "guardrails"
        assert attributes["event"] == "completed"
        assert attributes["outcome"] == "allowed"
        assert 0 <= attributes["elapsed_ms"] <= 86_400_000
        assert attributes["error_type"] == "RuntimeError"
        assert "api_key" not in attributes
        assert "query" not in attributes
        assert "provider" not in attributes
        assert "candidate_count" not in attributes
        assert "private query" not in str(info.call_args)
        assert "test-secret" not in str(info.call_args)
    assert request_logging.get_request_id() is None


def test_query_stage_ids_are_unique_per_request():
    app.state.rag_agent = MagicMock()
    app.state.rag_agent.invoke.return_value = {
        "final_answer": "hello",
        "plan": [],
        "status": "Response generated.",
        "documents": [],
    }
    client = TestClient(app)
    with (
        patch("app.main.guard", return_value=(False, None)),
        patch("app.logging.logfire.info") as info,
    ):
        first = client.post("/query", json={"q": "first private query"})
        second = client.post("/query", json={"q": "second private query"})

    assert first.status_code == second.status_code == 200
    stage_calls = [call for call in info.call_args_list if call.args[0] == "RAG query stage"]
    request_ids = [call.kwargs["request_id"] for call in stage_calls]
    assert len(request_ids) == 10
    assert len(set(request_ids[:5])) == len(set(request_ids[5:])) == 1
    assert request_ids[0] != request_ids[5]
    assert "first private query" not in str(stage_calls)
    assert "second private query" not in str(stage_calls)


def test_query_failure_logs_safe_stage_and_preserves_server_error():
    from app.logging import get_request_id

    app.state.rag_agent = MagicMock()
    client = TestClient(app, raise_server_exceptions=False)
    with (
        patch("app.main.guard", side_effect=RuntimeError("private query and secret")),
        patch("app.logging.logfire.info") as info,
    ):
        response = client.post("/query", json={"q": "private query"})

    assert response.status_code == 500
    stage_calls = [call for call in info.call_args_list if call.args[0] == "RAG query stage"]
    failed = next(call for call in stage_calls if call.kwargs["event"] == "failed")
    assert failed.kwargs["stage"] == "guardrails"
    assert failed.kwargs["error_type"] == "RuntimeError"
    assert failed.kwargs["request_id"]
    assert "private query" not in str(stage_calls)
    assert "secret" not in str(stage_calls)
    assert get_request_id() is None


def test_graph_failure_logs_safe_stage_and_preserves_response():
    from app.logging import get_request_id

    app.state.rag_agent = MagicMock()
    app.state.rag_agent.invoke.side_effect = RuntimeError("private query and secret")
    client = TestClient(app)
    with (
        patch("app.main.guard", return_value=(False, None)),
        patch("app.logging.logfire.info") as info,
    ):
        response = client.post("/query", json={"q": "private query"})

    assert response.status_code == 500
    assert response.json()["status"] == "error"
    stage_calls = [call for call in info.call_args_list if call.args[0] == "RAG query stage"]
    failed = next(call for call in stage_calls if call.kwargs["event"] == "failed")
    assert failed.kwargs["stage"] == "graph"
    assert failed.kwargs["error_type"] == "RuntimeError"
    assert "private query" not in str(stage_calls)
    assert "secret" not in str(stage_calls)
    assert get_request_id() is None
