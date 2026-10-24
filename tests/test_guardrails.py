"""Regression tests for the Gemini-backed NeMo Guardrails configuration."""

from unittest.mock import MagicMock, patch

import pytest
from nemoguardrails.llm.prompts import get_prompt
from nemoguardrails.llm.types import Task

import app.guardrails.rails as guardrails
from app.config import settings


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
