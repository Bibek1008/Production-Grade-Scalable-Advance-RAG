"""Tests for the Gemini-backed RAGAS judge configuration."""

from unittest.mock import patch

import pytest

from app.config import settings
from evals.metrics import (
    _build_judge,
    _ConfiguredEvaluationEmbeddings,
)


def test_build_judge_uses_gemini_without_openai_credentials(monkeypatch):
    monkeypatch.setattr(settings, "JUDGE_PROVIDER", "gemini")
    monkeypatch.setattr(settings, "JUDGE_GEMINI_API_KEY", "judge-key")
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "runtime-key")
    monkeypatch.setattr(settings, "JUDGE_GEMINI_MODEL", "gemini-eval-model")

    with (
        patch("evals.metrics.AsyncOpenAI") as mock_client_class,
        patch("evals.metrics.llm_factory") as mock_factory,
    ):
        judge, embeddings = _build_judge()

    assert judge is mock_factory.return_value
    assert isinstance(embeddings, _ConfiguredEvaluationEmbeddings)
    mock_client_class.assert_called_once_with(
        api_key="judge-key",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        timeout=90,
        max_retries=1,
    )
    mock_factory.assert_called_once_with(
        "gemini-eval-model",
        provider="openai",
        client=mock_client_class.return_value,
        adapter="instructor",
        temperature=0,
    )


def test_build_judge_uses_runtime_key_when_dedicated_key_is_empty(monkeypatch):
    monkeypatch.setattr(settings, "JUDGE_PROVIDER", "gemini")
    monkeypatch.setattr(settings, "JUDGE_GEMINI_API_KEY", None)
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "runtime-key")

    with (
        patch("evals.metrics.AsyncOpenAI") as mock_client_class,
        patch("evals.metrics.llm_factory") as mock_factory,
    ):
        _build_judge()

    assert mock_client_class.call_args.kwargs["api_key"] == "runtime-key"
    assert mock_factory.call_args.args[0] == settings.JUDGE_GEMINI_MODEL


def test_build_judge_rejects_openai_provider(monkeypatch):
    monkeypatch.setattr(settings, "JUDGE_PROVIDER", "openai")
    with pytest.raises(ValueError, match="JUDGE_PROVIDER must be 'gemini'"):
        _build_judge()
