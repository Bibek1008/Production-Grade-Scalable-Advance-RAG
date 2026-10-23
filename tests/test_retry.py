"""Tests for retry-with-backoff behavior on external service calls."""

from unittest.mock import MagicMock, patch

import pytest

from app.services.retrieval.qdrant_service import search_enterprise_knowledge
from app.services.retrieval.ranking_service import _JinaReranker, rerank_documents


def test_jina_reranker_extracts_text_from_document_objects():
    response = MagicMock()
    response.json.return_value = {
        "results": [
            {"index": 1, "document": {"text": "doc two"}},
            {"index": 0, "document": "doc one"},
        ]
    }

    with patch("app.services.retrieval.ranking_service.requests.post", return_value=response):
        results = _JinaReranker().rerank("query", ["doc one", "doc two"], top_n=2)

    assert results == ["doc two", "doc one"]


def test_qdrant_search_retries_then_raises():
    """Qdrant search should surface persistent failures after retrying."""
    with (
        patch("app.services.retrieval.qdrant_service.client") as mock_client,
        patch("app.services.retrieval.qdrant_service.embed_query") as mock_embed,
    ):
        mock_embed.return_value = [0.0] * 10
        mock_client.query_points.side_effect = RuntimeError("transient")

        with pytest.raises(RuntimeError, match="Knowledge-base search failed"):
            search_enterprise_knowledge("test query", limit=1)
        # Tenacity default: initial call + 2 retries = 3 attempts
        assert mock_client.query_points.call_count == 3


def test_qdrant_missing_collection_fails_before_embedding():
    with (
        patch("app.services.retrieval.qdrant_service.client") as mock_client,
        patch("app.services.retrieval.qdrant_service.embed_query") as mock_embed,
    ):
        mock_client.collection_exists.return_value = False
        with pytest.raises(RuntimeError, match="Ingest documents before querying"):
            search_enterprise_knowledge("test query", limit=1)

    mock_embed.assert_not_called()


def test_rerank_retries_then_falls_back():
    """Reranker should retry transient failures and fall back to input order."""
    docs = ["doc one", "doc two", "doc three"]
    with patch("app.services.retrieval.ranking_service._get_ranker") as mock_get_ranker:
        mock_ranker = MagicMock()
        mock_ranker.rerank.side_effect = RuntimeError("transient")
        mock_get_ranker.return_value = mock_ranker

        results = rerank_documents("query", docs, top_n=2)

        assert results == docs[:2]
        assert mock_ranker.rerank.call_count == 3
