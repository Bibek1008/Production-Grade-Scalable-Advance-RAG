"""Centralized, Pydantic-validated application settings."""

import os
from urllib.parse import quote, urlunsplit

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Load and validate environment variables from `.env`.

    - `extra="ignore"` lets `.env` keep legacy keys without failing startup.
    - Required fields raise a clear validation error at import time if missing.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_ignore_empty=True,
    )

    # --- JINA AI (embeddings + reranker) ---
    JINA_API_KEY: str | None = None

    # --- GEMINI LLM ---
    GEMINI_API_KEY: str
    GEMINI_MODEL: str = "gemini-flash-lite-latest"

    # --- OPTIONAL OPENAI EVAL JUDGE ---
    OPENAI_API_KEY: str | None = None
    JUDGE_OPENAI_API_KEY: str | None = None

    # --- QDRANT VECTOR DB ---
    QDRANT_URL: str = Field(
        default="http://localhost:6333",
        validation_alias=AliasChoices("QDRANT_URL", "QDRANT_CLUSTER_ENDPOINT"),
    )
    QDRANT_API_KEY: str | None = None
    QDRANT_COLLECTION: str = "enterprise_rag"

    # --- NEON SERVERLESS POSTGRES (LangGraph checkpointer) ---
    NEON_DB_URL: str | None = None

    # --- UPSTASH REDIS (rate limiting) ---
    UPSTASH_REDIS_REST_URL: str | None = None
    UPSTASH_REDIS_REST_TOKEN: str | None = None

    # --- API SAFETY ---
    API_KEY: str | None = Field(default=None, alias="RAG_API_KEY")
    RATE_LIMIT_PER_MINUTE: int = 20
    STRICT_STARTUP: bool = False

    # --- OBSERVABILITY ---
    LOGFIRE_TOKEN: str | None = None
    LOGFIRE_BASE_URL: str | None = None  # e.g. https://logfire-eu.pydantic.dev for EU tokens
    LANGSMITH_TRACING: str = "true"
    LANGSMITH_API_KEY: str | None = None
    LANGSMITH_PROJECT: str = "rag_scale_test"
    LANGSMITH_ENDPOINT: str = "https://api.smith.langchain.com"

    @field_validator(
        "JINA_API_KEY",
        "OPENAI_API_KEY",
        "JUDGE_OPENAI_API_KEY",
        "QDRANT_API_KEY",
        "NEON_DB_URL",
        "UPSTASH_REDIS_REST_URL",
        "UPSTASH_REDIS_REST_TOKEN",
        mode="before",
    )
    @classmethod
    def _empty_optional_setting_as_none(cls, value):
        """Treat empty optional environment values as unset."""
        if value == "" or value is None:
            return None
        return value

    @property
    def judge_api_key(self) -> str | None:
        """Dedicated judge key, falling back to the optional OpenAI key."""
        return self.JUDGE_OPENAI_API_KEY or self.OPENAI_API_KEY

    @property
    def postgres_uri(self) -> str:
        """LangGraph Postgres checkpointer URI (Neon).

        Serverless Postgres closes idle connections, so append TCP keepalive
        options to keep the connection pool healthy between requests.
        """
        if not self.NEON_DB_URL:
            raise ValueError("NEON_DB_URL is not configured.")
        base = self.NEON_DB_URL.rstrip("/")
        keepalive = "keepalives=1&keepalives_idle=30&keepalives_interval=10&keepalives_count=5"
        if "?" in base:
            return f"{base}&{keepalive}"
        return f"{base}?{keepalive}"

    @property
    def redis_url(self) -> str | None:
        """TLS Redis URL derived from Upstash REST credentials.

        Upstash exposes the same host for REST and TLS Redis. The REST token is
        used as the Redis password under the default username. The result is
        passed to `limits` for rate limiting and to the health checker.
        """
        if not self.UPSTASH_REDIS_REST_URL or not self.UPSTASH_REDIS_REST_TOKEN:
            return None
        host = self.UPSTASH_REDIS_REST_URL.replace("https://", "").rstrip("/")
        token = quote(self.UPSTASH_REDIS_REST_TOKEN, safe="")
        netloc = f"default:{token}@{host}"
        return urlunsplit(("rediss", netloc, "/0", "ssl_cert_reqs=required", ""))


# Singleton used across the app.
settings = Settings()


def apply_langchain_env():
    """Write LangSmith/LangChain settings to os.environ for automatic tracing.

    Tracing is only activated when both LANGSMITH_TRACING and LANGSMITH_API_KEY
    are set — enabling tracing without a key causes LangChain to emit 401 noise
    on every LangGraph step.
    """
    if settings.LANGSMITH_TRACING and settings.LANGSMITH_API_KEY:
        os.environ.setdefault("LANGCHAIN_TRACING_V2", settings.LANGSMITH_TRACING)
        os.environ.setdefault("LANGCHAIN_API_KEY", settings.LANGSMITH_API_KEY)
    if settings.LANGSMITH_PROJECT:
        os.environ.setdefault("LANGCHAIN_PROJECT", settings.LANGSMITH_PROJECT)
    if settings.LANGSMITH_ENDPOINT:
        os.environ.setdefault("LANGCHAIN_ENDPOINT", settings.LANGSMITH_ENDPOINT)


apply_langchain_env()
