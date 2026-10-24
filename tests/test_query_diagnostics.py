"""Tests for safe diagnostics emitted from graph pipeline stages."""

from types import SimpleNamespace
from unittest.mock import patch

from app.agents.nodes.planner import planner_node
from app.agents.nodes.responder import generate_node
from app.agents.nodes.retriever import retrieve_node
from app.services.retrieval import embedding, qdrant_service, ranking_service


def test_planner_logs_stages_without_query_content():
    state = {"messages": [{"role": "user", "content": "private query"}]}
    with (
        patch("app.agents.nodes.planner.llm") as llm,
        patch("app.agents.nodes.planner.log_query_stage") as stage_log,
    ):
        llm.invoke.return_value = SimpleNamespace(content="CONVERSATIONAL")
        planner_node(state)

    events = [call.args[1] for call in stage_log.call_args_list]
    assert events == ["started", "completed"]
    assert all("private query" not in str(call) for call in stage_log.call_args_list)
    assert stage_log.call_args_list[-1].kwargs["intent"] == "conversational"
    assert stage_log.call_args_list[-1].kwargs["started_at"] > 0


def test_retriever_logs_safe_counts_and_reranking_stages():
    state = {"current_query": "private query", "plan": []}
    documents = [{"content": "private document", "source": "source.txt", "score": 1.0}]
    with (
        patch("app.agents.nodes.retriever.search_enterprise_knowledge", return_value=documents),
        patch("app.agents.nodes.retriever.rerank_documents", return_value=["private document"]),
        patch("app.agents.nodes.retriever.log_query_stage") as stage_log,
    ):
        result = retrieve_node(state)

    assert result["documents"] == documents
    assert [call.args[1] for call in stage_log.call_args_list] == ["started", "completed"]
    assert stage_log.call_args_list[-1].kwargs["candidate_count"] == 1
    assert stage_log.call_args_list[-1].kwargs["selected_count"] == 1
    assert all(
        "private query" not in str(call) and "private document" not in str(call) for call in stage_log.call_args_list
    )


def test_embedding_logs_provider_and_dimensions(monkeypatch):
    monkeypatch.setattr(embedding, "_model_type", "jina")
    with (
        patch("app.services.retrieval.embedding._embed", return_value=[[0.0] * 1024]),
        patch("app.services.retrieval.embedding.log_query_stage") as stage_log,
    ):
        vector = embedding.embed_query("private query")

    assert len(vector) == 1024
    assert [call.args[1] for call in stage_log.call_args_list] == ["started", "completed"]
    assert stage_log.call_args_list[0].kwargs["provider"] == "jina"
    assert stage_log.call_args_list[-1].kwargs["dimensions"] == 1024
    assert all("private query" not in str(call) for call in stage_log.call_args_list)


def test_qdrant_search_logs_result_count(monkeypatch):
    monkeypatch.setattr(qdrant_service, "embed_query", lambda query: [0.0] * 1024)
    response = SimpleNamespace(points=[SimpleNamespace(payload={"text": "doc"}, score=0.9)])
    with (
        patch("app.services.retrieval.qdrant_service.client.query_points", return_value=response),
        patch("app.services.retrieval.qdrant_service.log_query_stage") as stage_log,
    ):
        results = qdrant_service._search_enterprise_knowledge("private query", limit=5)

    assert len(results) == 1
    assert [call.args[1] for call in stage_log.call_args_list] == ["started", "completed"]
    assert stage_log.call_args_list[-1].kwargs["result_count"] == 1
    assert all("private query" not in str(call) for call in stage_log.call_args_list)


def test_reranking_logs_safe_counts(monkeypatch):
    monkeypatch.setattr(ranking_service.settings, "JINA_API_KEY", "test-key")
    with (
        patch("app.services.retrieval.ranking_service._rerank", return_value=["document"]),
        patch("app.services.retrieval.ranking_service.log_query_stage") as stage_log,
    ):
        result = ranking_service.rerank_documents("private query", ["document"], top_n=1)

    assert result == ["document"]
    assert [call.args[1] for call in stage_log.call_args_list] == ["started", "completed"]
    assert stage_log.call_args_list[0].kwargs["document_count"] == 1
    assert stage_log.call_args_list[-1].kwargs["selected_count"] == 1
    assert all(
        "private query" not in str(call) and "private document" not in str(call) for call in stage_log.call_args_list
    )


def test_generation_logs_synthesis_stage(monkeypatch):
    state = {
        "current_query": "CONVERSATIONAL",
        "messages": [{"role": "user", "content": "private query"}],
        "documents": [],
        "plan": [],
    }
    with (
        patch("app.agents.nodes.responder._generate_response", return_value=SimpleNamespace(content="answer")),
        patch("app.agents.nodes.responder.log_query_stage") as stage_log,
    ):
        result = generate_node(state)

    assert result["final_answer"] == "answer"
    assert [call.args[1] for call in stage_log.call_args_list] == ["started", "completed"]
    assert stage_log.call_args_list[-1].kwargs["started_at"] > 0
    assert all("private query" not in str(call) for call in stage_log.call_args_list)
