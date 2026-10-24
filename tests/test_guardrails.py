"""Regression tests for the Gemini-backed NeMo Guardrails configuration."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from nemoguardrails import LLMRails
from nemoguardrails.embeddings.basic import BasicEmbeddingsIndex
from nemoguardrails.embeddings.index import IndexItem
from nemoguardrails.llm.prompts import get_prompt
from nemoguardrails.llm.types import Task

import app.guardrails.rails as guardrails
from app.config import settings


@pytest.mark.asyncio
async def test_default_embeddings_index_builds_and_searches_numpy_matrix(monkeypatch):
    index = BasicEmbeddingsIndex()
    vectors = {
        "first": [3.0, 0.0],
        "second": [0.0, 4.0],
        "query": [0.0, 2.0],
    }

    async def get_embeddings(texts):
        return [vectors[text] for text in texts]

    monkeypatch.setattr(index, "_get_embeddings", get_embeddings)
    first = IndexItem(text="first")
    second = IndexItem(text="second")
    await index.add_items([first, second])

    await index.build()

    built_index = index.embeddings_index
    assert isinstance(built_index, np.ndarray)
    assert built_index.dtype == np.float32
    np.testing.assert_allclose(built_index, [[1.0, 0.0], [0.0, 1.0]])
    assert await index.search("query", max_results=1) == [second]


def test_first_guardrails_query_builds_index_and_completes(monkeypatch):
    monkeypatch.setattr(guardrails, "_rails", None)
    monkeypatch.setattr(
        guardrails,
        "get_portkey_llm",
        lambda: FakeListChatModel(responses=["express greeting"] * 20),
    )

    async def get_embeddings(self, texts):
        return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr(BasicEmbeddingsIndex, "_get_embeddings", get_embeddings)
    guardrails.initialize_rails()

    assert isinstance(guardrails._rails, LLMRails)
    blocked, message = guardrails.guard("hello")
    assert (blocked, message) == (False, None)


def test_guardrails_use_configured_gemini_prompt(monkeypatch):
    monkeypatch.setattr(settings, "GEMINI_MODEL", "gemini-test-model")
    monkeypatch.setattr(guardrails, "_rails", None)
    monkeypatch.setattr(guardrails, "get_portkey_llm", lambda: object())
    monkeypatch.setattr(guardrails, "LLMRails", lambda config, llm: config)

    guardrails.initialize_rails()

    prompt = get_prompt(guardrails._rails, Task.GENERATE_NEXT_STEPS)
    assert guardrails._rails.models[0].model == "gemini-test-model"
    assert prompt.content is not None
    assert prompt.messages is None


def test_guardrails_intent_policy_allows_in_domain_abstention(monkeypatch):
    monkeypatch.setattr(guardrails, "_rails", None)
    monkeypatch.setattr(guardrails, "get_portkey_llm", lambda: object())
    monkeypatch.setattr(guardrails, "LLMRails", lambda config, llm: config)

    guardrails.initialize_rails()

    general_instructions = next(
        instruction.content for instruction in guardrails._rails.instructions if instruction.type == "general"
    )
    assert "Classify a request by its subject" in general_instructions
    assert "explicitly request an abstention if evidence is missing" in general_instructions
    assert "Refuse only requests whose subject is clearly unrelated" in general_instructions


@pytest.mark.parametrize(
    ("query", "response", "blocked"),
    [
        (
            "What does kube-scheduler do?",
            "user ask technical question",
            False,
        ),
        (
            "According to the indexed documents, what Kubernetes feature was introduced in version 9.99? "
            "If the documents don't establish this, explicitly say that the answer was not found in the documents.",
            "user ask technical question",
            False,
        ),
        (
            "What is the best restaurant near me?",
            "I'm an Enterprise IT Assistant focused on Kubernetes, Intel hardware, and networking. "
            "I can't help with that — but ask me anything technical!",
            True,
        ),
    ],
    ids=("in-domain", "in-domain-abstention", "out-of-domain"),
)
def test_guardrails_routes_domain_and_evidence_requests(query, response, blocked, monkeypatch):
    rails = MagicMock()
    rails.generate.return_value = {"content": response}
    monkeypatch.setattr(guardrails, "_rails", rails)

    actual_blocked, message = guardrails.guard(query)

    assert actual_blocked is blocked
    assert message == (response if blocked else None)
    rails.generate.assert_called_once_with(messages=[{"role": "user", "content": query}])


@pytest.mark.parametrize(
    "response",
    [
        "Hello! I'm your Enterprise IT Assistant.",
        "bot explain capabilities: I'm an Enterprise AI Assistant with deep expertise in Kubernetes.",
        "Goodbye! Feel free to return whenever you have more enterprise IT questions.",
    ],
)
def test_dialogue_responses_do_not_block_technical_requests(monkeypatch, response):
    rails = MagicMock()
    rails.generate.return_value = {"content": response}
    monkeypatch.setattr(guardrails, "_rails", rails)

    blocked, message = guardrails.guard("How do I troubleshoot Kubernetes networking?")

    assert blocked is False
    assert message is None


@pytest.mark.parametrize(
    "response",
    [
        "I'm an Enterprise IT Assistant focused on Kubernetes. I can't help with that — but ask me anything technical!",
        "I maintain consistent guidelines regardless of how I am prompted.",
    ],
)
def test_refusal_responses_still_block_requests(monkeypatch, response):
    rails = MagicMock()
    rails.generate.return_value = {"content": response}
    monkeypatch.setattr(guardrails, "_rails", rails)

    blocked, message = guardrails.guard("Tell me a joke")

    assert blocked is True
    assert message == response


def test_guardrails_logs_block_without_user_content(monkeypatch):
    rails = MagicMock()
    rails.generate.return_value = {"content": "I cannot help with that request."}
    monkeypatch.setattr(guardrails, "_rails", rails)
    query = "private user question"

    with patch("app.guardrails.rails.logfire.info") as info:
        guardrails.guard(query)

    assert all(query not in str(call) for call in info.call_args_list)
