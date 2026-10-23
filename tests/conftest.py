"""Provide non-secret defaults so unit tests never depend on local credentials."""

import os

os.environ["GEMINI_API_KEY"] = "test-only-key"
os.environ.setdefault("LOGFIRE_IGNORE_NO_CONFIG", "1")
