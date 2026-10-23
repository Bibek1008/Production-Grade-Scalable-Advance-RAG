"""Tests for the Portkey chat model configuration."""

from unittest.mock import patch

from app.gateway.client import get_portkey_async_client, get_portkey_llm


def test_get_portkey_llm_uses_configured_route_and_auth_headers():
    with (
        patch("app.gateway.client.settings.GEMINI_API_KEY", "test-key"),
        patch("app.gateway.client.settings.GEMINI_MODEL", "gemini-test"),
        patch("app.gateway.client.settings.PORTKEY_API_KEY", "test-portkey-key"),
        patch("app.gateway.client.settings.PORTKEY_PRIMARY_SLUG", "primary-gemini"),
        patch("app.gateway.client.settings.PORTKEY_FALLBACK_SLUG", "fallback-gemini"),
        patch("app.gateway.client.settings.PORTKEY_PRIMARY_CONFIG_ID", "pc-test"),
        patch("app.gateway.client.ChatOpenAI") as mock_chat,
    ):
        result = get_portkey_llm()

    assert result is mock_chat.return_value
    mock_chat.assert_called_once_with(
        model="@primary-gemini/gemini-test",
        api_key="test-portkey-key",
        base_url="https://api.portkey.ai/v1",
        default_headers={
            "x-portkey-api-key": "test-portkey-key",
            "x-portkey-config": "pc-test",
        },
        temperature=0,
        timeout=90,
        max_retries=1,
    )


def test_get_async_portkey_client_uses_saved_configuration_headers():
    with (
        patch("app.gateway.client.settings.PORTKEY_API_KEY", "test-portkey-key"),
        patch("app.gateway.client.settings.PORTKEY_PRIMARY_SLUG", "primary-gemini"),
        patch("app.gateway.client.settings.PORTKEY_FALLBACK_SLUG", "fallback-gemini"),
        patch("app.gateway.client.settings.PORTKEY_PRIMARY_CONFIG_ID", "pc-test"),
        patch("app.gateway.client.AsyncOpenAI") as mock_client,
    ):
        result = get_portkey_async_client()

    assert result is mock_client.return_value
    mock_client.assert_called_once_with(
        api_key="test-portkey-key",
        base_url="https://api.portkey.ai/v1",
        default_headers={
            "x-portkey-api-key": "test-portkey-key",
            "x-portkey-config": "pc-test",
        },
        timeout=90,
        max_retries=1,
    )


def test_portkey_client_fails_clearly_when_route_is_incomplete():
    with (
        patch("app.gateway.client.settings.PORTKEY_API_KEY", "test-key"),
        patch("app.gateway.client.settings.PORTKEY_PRIMARY_SLUG", None),
        patch("app.gateway.client.settings.PORTKEY_FALLBACK_SLUG", None),
        patch("app.gateway.client.settings.PORTKEY_PRIMARY_CONFIG_ID", None),
    ):
        try:
            get_portkey_llm()
        except RuntimeError as error:
            assert "PORTKEY_PRIMARY_SLUG" in str(error)
            assert "PORTKEY_FALLBACK_SLUG" in str(error)
            assert "PORTKEY_PRIMARY_CONFIG_ID" in str(error)
        else:
            raise AssertionError("Missing Portkey configuration should fail")
