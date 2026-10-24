"""Request-context logging helpers."""

import time
from contextlib import contextmanager
from contextvars import ContextVar
from math import isfinite
from typing import Iterator
from uuid import UUID

import logfire

_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)

_STAGES = {
    "query",
    "guardrails",
    "graph",
    "planner",
    "retrieval",
    "embedding",
    "qdrant_search",
    "reranking",
    "generation",
}
_EVENTS = {"received_and_validated", "started", "completed", "failed", "skipped"}
_ATTRIBUTE_VALUES = {
    "intent": {"conversational", "technical"},
    "outcome": {"allowed", "blocked"},
    "provider": {"jina", "fallback"},
    "reason": {"no_documents", "provider_not_configured"},
}
_COUNT_ATTRIBUTES = {
    "candidate_count",
    "dimensions",
    "document_count",
    "limit",
    "result_count",
    "selected_count",
    "top_n",
}
_MAX_COUNT = 1_000_000
_MAX_ELAPSED_MS = 86_400_000


@contextmanager
def request_id_context(request_id: str) -> Iterator[None]:
    """Set a request id for the current context and restore its previous value afterward."""
    token = _request_id.set(request_id)
    try:
        yield
    finally:
        _request_id.reset(token)


def get_request_id() -> str | None:
    """Get the current request id, if any."""
    return _request_id.get()


def log_query_stage(
    stage: str,
    event: str,
    *,
    started_at: float | None = None,
    error: BaseException | None = None,
    **attributes: str | int | float | bool | None,
) -> None:
    """Log bounded query-stage metadata without request content or credentials."""
    if stage not in _STAGES or event not in _EVENTS:
        return

    fields: dict[str, str | int | float | bool | None] = {
        "stage": stage,
        "event": event,
    }
    request_id = get_request_id()
    if request_id is not None:
        try:
            fields["request_id"] = str(UUID(request_id))
        except (ValueError, TypeError, AttributeError):
            pass

    if type(started_at) in (int, float) and isfinite(started_at):
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        if isfinite(elapsed_ms) and 0 <= elapsed_ms <= _MAX_ELAPSED_MS:
            fields["elapsed_ms"] = round(elapsed_ms, 2)
    if isinstance(error, BaseException):
        error_type = type(error).__name__
        if len(error_type) <= 64 and error_type.isascii() and error_type.isidentifier():
            fields["error_type"] = error_type

    for name, value in attributes.items():
        if name in _COUNT_ATTRIBUTES and type(value) is int and 0 <= value <= _MAX_COUNT:
            fields[name] = value
        elif name in _ATTRIBUTE_VALUES and isinstance(value, str):
            allowed_values = _ATTRIBUTE_VALUES[name]
            if value in allowed_values:
                fields[name] = value

    logfire.info("RAG query stage", **fields)
