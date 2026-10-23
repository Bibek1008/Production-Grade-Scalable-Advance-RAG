from langchain_openai import ChatOpenAI
from openai import AsyncOpenAI

from app.config import settings

GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_TIMEOUT_SECONDS = 90
GEMINI_MAX_RETRIES = 1


def get_gemini_llm() -> ChatOpenAI:
    """Create a LangChain chat model for Gemini's OpenAI-compatible API."""
    return ChatOpenAI(
        model=settings.GEMINI_MODEL,
        api_key=settings.GEMINI_API_KEY,
        base_url=GEMINI_OPENAI_BASE_URL,
        temperature=0,
        timeout=GEMINI_TIMEOUT_SECONDS,
        max_retries=GEMINI_MAX_RETRIES,
    )


def get_async_gemini_client() -> AsyncOpenAI:
    """Create an async Gemini client using Google's OpenAI-compatible endpoint."""
    return AsyncOpenAI(
        api_key=settings.GEMINI_API_KEY,
        base_url=GEMINI_OPENAI_BASE_URL,
        timeout=GEMINI_TIMEOUT_SECONDS,
        max_retries=GEMINI_MAX_RETRIES,
    )


def get_async_openai_client(feature: str = "rag") -> AsyncOpenAI:
    """Compatibility wrapper for existing callers; requests are sent to Gemini."""
    return get_async_gemini_client()
