"""Tests for Logfire configuration and token authentication checks."""

from unittest.mock import MagicMock, patch

import requests

from app.observability import configure_logfire, probe_logfire_authentication


def test_configure_logfire_uses_token_and_configured_region_endpoint():
    with (
        patch("app.observability.settings.LOGFIRE_TOKEN", "test-token"),
        patch("app.observability.settings.LOGFIRE_BASE_URL", "https://logfire-eu.pydantic.dev"),
        patch("app.observability.logfire.AdvancedOptions") as options,
        patch("app.observability.logfire.configure") as configure,
    ):
        configure_logfire()

    configure.assert_called_once_with(
        token="test-token",
        send_to_logfire="if-token-present",
        advanced=options.return_value,
    )
    options.assert_called_once_with(base_url="https://logfire-eu.pydantic.dev")


def test_probe_logfire_authentication_uses_region_endpoint_and_token():
    advanced = MagicMock()
    advanced.generate_base_url.return_value = "https://logfire-eu.pydantic.dev"
    response = MagicMock(status_code=200)
    with (
        patch("app.observability.settings.LOGFIRE_TOKEN", "test-token"),
        patch("app.observability.settings.LOGFIRE_BASE_URL", "https://logfire-eu.pydantic.dev"),
        patch("app.observability.logfire.AdvancedOptions", return_value=advanced) as options,
        patch("app.observability.requests.get", return_value=response) as get,
    ):
        assert probe_logfire_authentication() == (True, "Logfire token authenticated by API")

    options.assert_called_once_with(base_url="https://logfire-eu.pydantic.dev")
    advanced.generate_base_url.assert_called_once_with("test-token", warn_unknown_region=False)
    get.assert_called_once_with(
        "https://logfire-eu.pydantic.dev/v1/info",
        headers={"Authorization": "test-token"},
        timeout=5,
    )


def test_probe_logfire_authentication_reports_invalid_token_without_response_body():
    response = MagicMock(status_code=401, text="token contents should not be logged")
    advanced = MagicMock()
    advanced.generate_base_url.return_value = "https://logfire-api.pydantic.dev"
    with (
        patch("app.observability.settings.LOGFIRE_TOKEN", "test-token"),
        patch("app.observability.settings.LOGFIRE_BASE_URL", None),
        patch("app.observability.logfire.AdvancedOptions", return_value=advanced),
        patch("app.observability.requests.get", return_value=response),
    ):
        healthy, message = probe_logfire_authentication()

    assert healthy is False
    assert "HTTP 401" in message
    assert "token contents" not in message


def test_probe_logfire_authentication_sanitizes_request_errors():
    with (
        patch("app.observability.settings.LOGFIRE_TOKEN", "test-token"),
        patch("app.observability.settings.LOGFIRE_BASE_URL", "https://logfire-eu.pydantic.dev"),
        patch(
            "app.observability.requests.get",
            side_effect=requests.ConnectionError("request failed for https://user:secret@example.test"),
        ),
    ):
        healthy, message = probe_logfire_authentication()

    assert healthy is False
    assert message == "Logfire authentication probe failed (ConnectionError)"
    assert "secret" not in message
