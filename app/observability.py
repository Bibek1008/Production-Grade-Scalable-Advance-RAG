"""Logfire setup and token-authentication checks for readiness."""

from urllib.parse import urljoin

import logfire
import requests

from app.config import settings


def configure_logfire() -> None:
    """Configure Logfire with the configured token and region endpoint."""
    advanced = logfire.AdvancedOptions(base_url=settings.logfire_base_url) if settings.logfire_base_url else None
    logfire.configure(
        token=settings.LOGFIRE_TOKEN,
        send_to_logfire="if-token-present",
        advanced=advanced,
    )


def probe_logfire_authentication() -> tuple[bool, str]:
    """Check the configured token against Logfire's authenticated project-info endpoint."""
    token = settings.LOGFIRE_TOKEN
    if not token:
        return False, "LOGFIRE_TOKEN not set"

    try:
        base_url = logfire.AdvancedOptions(base_url=settings.logfire_base_url).generate_base_url(
            token,
            warn_unknown_region=False,
        )
        response = requests.get(
            urljoin(base_url, "/v1/info"),
            headers={"Authorization": token},
            timeout=5,
        )
    except Exception as error:
        return False, f"Logfire authentication probe failed ({type(error).__name__})"

    if response.status_code == 200:
        return True, "Logfire token authenticated by API"
    return False, f"Logfire rejected or could not validate the configured token (HTTP {response.status_code})"
