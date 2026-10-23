"""Provide non-secret defaults so unit tests never depend on local credentials."""

import os

import pytest

os.environ["GEMINI_API_KEY"] = "test-only-key"
os.environ.setdefault("LOGFIRE_IGNORE_NO_CONFIG", "1")

from app.config import settings

settings.LOGFIRE_TOKEN = None
settings.LANGSMITH_TRACING = "false"
settings.LANGSMITH_API_KEY = None
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ.pop("LANGCHAIN_API_KEY", None)


@pytest.fixture(autouse=True)
def clear_api_key_for_unit_tests(monkeypatch):
    """Keep local `.env` authentication credentials from leaking into unit tests."""
    monkeypatch.setattr(settings, "API_KEY", None)
