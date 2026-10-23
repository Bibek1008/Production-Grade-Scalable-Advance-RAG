"""Tests for the Gemini chat model configuration."""

from unittest.mock import patch

from app.gateway.client import get_async_gemini_client, get_gemini_llm


def test_get_gemini_llm_uses_configured_credentials_and_model():
    with (
        patch("app.gateway.client.settings.GEMINI_API_KEY", "test-key"),
        patch("app.gateway.client.settings.GEMINI_MODEL", "gemini-test"),
        patch("app.gateway.client.ChatOpenAI") as mock_chat,
    ):
        result = get_gemini_llm()

    assert result is mock_chat.return_value
    mock_chat.assert_called_once_with(
        model="gemini-test",
        api_key="test-key",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        temperature=0,
        timeout=90,
        max_retries=1,
    )


def test_get_async_gemini_client_uses_google_compatible_endpoint():
    with (
        patch("app.gateway.client.settings.GEMINI_API_KEY", "test-key"),
        patch("app.gateway.client.AsyncOpenAI") as mock_client,
    ):
        result = get_async_gemini_client()

    assert result is mock_client.return_value
    mock_client.assert_called_once_with(
        api_key="test-key",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        timeout=90,
        max_retries=1,
    )
