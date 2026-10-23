"""Health and readiness checks for the Enterprise RAG API."""

import logfire
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from langgraph.checkpoint.memory import MemorySaver

from app.agents.graph import is_postgres_checkpointer
from app.services.health.connection_checker import check_all_connections

router = APIRouter(tags=["health"])


@router.get("/health")
def health():
    """Liveness probe — returns 200 if the process is running."""
    return {"status": "ok"}


@router.get("/ready")
def ready(request: Request):
    """
    Readiness probe — verifies that critical runtime dependencies are reachable.
    Optional persistence, rate limiting, tracing, and reranking failures are reported
    but do not make the API unready. The active checkpointer's durability is reported
    separately from database reachability.
    """
    results = check_all_connections()
    checks = {name: result.to_dict()["status"] for name, result in results.items()}
    rag_agent = getattr(request.app.state, "rag_agent", None)
    checkpointer = getattr(rag_agent, "checkpointer", None)
    if isinstance(checkpointer, MemorySaver):
        checks["checkpointer"] = "unavailable: MemorySaver active; checkpoint data is not durable"
    elif is_postgres_checkpointer(checkpointer):
        checks["checkpointer"] = "ok: PostgreSQL checkpointer active"
    elif checkpointer is not None:
        checks["checkpointer"] = "unavailable: active checkpointer durability is unknown"
    healthy = all(r.healthy for r in results.values() if r.required)

    if not healthy:
        logfire.warning("Readiness check failed", checks=checks)

    status_code = 200 if healthy else 503
    return JSONResponse(
        status_code=status_code,
        content={"status": "ready" if healthy else "not_ready", "checks": checks},
    )
