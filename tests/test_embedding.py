"""Tests for consistent embedding provider selection and response validation."""

from unittest.mock import MagicMock, patch

import pytest
import requests

from app.services.retrieval import embedding


def test_jina_batch_validates_embedding_dimensions():
    response = MagicMock()
    response.json.return_value = {"data": [{"index": 0, "embedding": [0.1] * 1024}]}
    with (
        patch("app.services.retrieval.embedding.settings.JINA_API_KEY", "test-key"),
        patch("app.services.retrieval.embedding.requests.post", return_value=response),
    ):
        vectors = embedding._embed_jina_batch(["test"], "retrieval.query")

    assert len(vectors) == 1
    assert len(vectors[0]) == 1024


def test_jina_quota_error_is_not_retried():
    response = requests.Response()
    response.status_code = 429
    error = requests.HTTPError(response=response)
    with (
        patch("app.services.retrieval.embedding.settings.JINA_API_KEY", "test-key"),
        patch("app.services.retrieval.embedding.requests.post") as post,
    ):
        post.return_value.raise_for_status.side_effect = error
        with pytest.raises(requests.HTTPError):
            embedding._embed_jina_batch(["test"], "retrieval.query")

    post.assert_called_once()


def test_jina_failure_does_not_switch_to_local_vectors():
    with (
        patch.object(embedding, "_model_type", "jina"),
        patch.object(embedding, "_embed_jina", side_effect=requests.HTTPError("quota exhausted")),
        patch.object(embedding, "_load_fallback") as load_fallback,
    ):
        with pytest.raises(RuntimeError, match="no alternate vector model was used"):
            embedding._embed(["test"], "retrieval.query")

    load_fallback.assert_not_called()
