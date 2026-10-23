"""Portkey-backed LLM clients for the application generation pipeline."""

from langchain_openai import ChatOpenAI
from openai import AsyncOpenAI
from portkey_ai import PORTKEY_GATEWAY_URL, createHeaders

from app.config import settings

PORTKEY_TIMEOUT_SECONDS = 90
PORTKEY_MAX_RETRIES = 1


def _portkey_config() -> tuple[str, dict[str, str]]:
    missing = [
        name
        for name, value in (
            ("PORTKEY_API_KEY", settings.PORTKEY_API_KEY),
            ("PORTKEY_PRIMARY_SLUG", settings.PORTKEY_PRIMARY_SLUG),
            ("PORTKEY_FALLBACK_SLUG", settings.PORTKEY_FALLBACK_SLUG),
            ("PORTKEY_PRIMARY_CONFIG_ID", settings.PORTKEY_PRIMARY_CONFIG_ID),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(f"Portkey generation is not configured; set {', '.join(missing)}.")
    # The saved config header controls fallback behavior; the model selects its
    # primary catalog entry so the config can apply its configured fallback.
    model = f"@{settings.PORTKEY_PRIMARY_SLUG}/{settings.GEMINI_MODEL}"
    headers = createHeaders(
        api_key=settings.PORTKEY_API_KEY,
        config=settings.PORTKEY_PRIMARY_CONFIG_ID,
    )
    return model, headers


def get_portkey_llm() -> ChatOpenAI:
    """Create the LangChain model routed through the configured Portkey config."""
    model, headers = _portkey_config()
    return ChatOpenAI(
        model=model,
        api_key=settings.PORTKEY_API_KEY,
        base_url=PORTKEY_GATEWAY_URL,
        default_headers=headers,
        temperature=0,
        timeout=PORTKEY_TIMEOUT_SECONDS,
        max_retries=PORTKEY_MAX_RETRIES,
    )


def get_portkey_async_client() -> AsyncOpenAI:
    """Create an async OpenAI-compatible client routed through Portkey."""
    _, headers = _portkey_config()
    return AsyncOpenAI(
        api_key=settings.PORTKEY_API_KEY,
        base_url=PORTKEY_GATEWAY_URL,
        default_headers=headers,
        timeout=PORTKEY_TIMEOUT_SECONDS,
        max_retries=PORTKEY_MAX_RETRIES,
    )
