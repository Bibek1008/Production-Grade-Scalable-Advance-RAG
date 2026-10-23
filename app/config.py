"""Centralized, Pydantic-validated application settings."""

import os
from pathlib import Path
from urllib.parse import quote, urlunsplit

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Load and validate environment variables from `.env`.

    - `extra="ignore"` lets `.env` keep legacy keys without failing startup.
    - Required fields raise a clear validation error at import time if missing.
    """

    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parent.parent / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_ignore_empty=True,
    )

    # --- JINA AI (embeddings + reranker) ---
    JINA_API_KEY: str | None = None

    # --- GEMINI LLM ---
    GEMINI_API_KEY: str
    GEMINI_MODEL: str = "gemini-flash-lite-latest"

    # --- PORTKEY GENERATION ROUTING ---
    PORTKEY_API_KEY: str | None = None
    PORTKEY_PRIMARY_SLUG: str | None = None
    PORTKEY_FALLBACK_SLUG: str | None = None
    PORTKEY_PRIMARY_CONFIG_ID: str | None = None

    # --- RAGAS GEMINI EVALUATION JUDGE ---
    JUDGE_PROVIDER: str = "gemini"
    JUDGE_GEMINI_API_KEY: str | None = None
    JUDGE_GEMINI_MODEL: str = "gemini-2.5-flash"

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

    # --- DEPLOYMENT ---
    BACKEND_URL: str = "http://localhost:8000"

    @field_validator(
        "JINA_API_KEY",
        "PORTKEY_API_KEY",
        "PORTKEY_PRIMARY_SLUG",
        "PORTKEY_FALLBACK_SLUG",
        "PORTKEY_PRIMARY_CONFIG_ID",
        "JUDGE_GEMINI_API_KEY",
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
    def judge_gemini_api_key(self) -> str | None:
        """Use the dedicated Gemini judge key, falling back to the runtime key."""
        return self.JUDGE_GEMINI_API_KEY or self.GEMINI_API_KEY

    @property
    def backend_url(self) -> str:
        """Return the configured backend origin, defaulting to local development."""
        return self.BACKEND_URL.strip().rstrip("/") or "http://localhost:8000"

    @property
    def logfire_base_url(self) -> str | None:
        """Return an explicit Logfire endpoint or infer the EU endpoint from its token."""
        if self.LOGFIRE_BASE_URL:
            return self.LOGFIRE_BASE_URL
        if self.LOGFIRE_TOKEN and self.LOGFIRE_TOKEN.startswith("pylf_v2_eu_"):
            return "https://logfire-eu.pydantic.dev"
        return None

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
    tracing_enabled = settings.LANGSMITH_TRACING.lower() in {"true", "1", "yes", "on"}
    os.environ["LANGCHAIN_TRACING_V2"] = str(tracing_enabled and bool(settings.LANGSMITH_API_KEY)).lower()
    if settings.LANGSMITH_API_KEY:
        os.environ["LANGCHAIN_API_KEY"] = settings.LANGSMITH_API_KEY
    if settings.LANGSMITH_PROJECT:
        os.environ["LANGCHAIN_PROJECT"] = settings.LANGSMITH_PROJECT
    if settings.LANGSMITH_ENDPOINT:
        os.environ["LANGCHAIN_ENDPOINT"] = settings.LANGSMITH_ENDPOINT


apply_langchain_env()
