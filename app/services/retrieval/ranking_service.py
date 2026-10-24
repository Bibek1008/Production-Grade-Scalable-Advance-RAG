import time

import logfire
import requests
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_exponential

from app.config import settings
from app.logging import log_query_stage

_JINA_RERANK_URL = "https://api.jina.ai/v1/rerank"
_JINA_RERANK_MODEL = "jina-reranker-v3"

_ranker = None


class _JinaReranker:
    """Thin wrapper around the Jina Reranker API."""

    def rerank(self, query: str, documents: list[str], top_n: int) -> list[str]:
        """Score and reorder documents against the query via the Jina API."""
        response = requests.post(
            _JINA_RERANK_URL,
            headers={
                "Authorization": f"Bearer {settings.JINA_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": _JINA_RERANK_MODEL,
                "query": query,
                "documents": documents,
                "top_n": top_n,
                "return_documents": True,
            },
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()

        results = payload.get("results", [])
        # Results are already sorted by relevance_score descending
        reranked_docs = []
        for res in results[:top_n]:
            doc_text = res.get("document")
            if isinstance(doc_text, dict):
                doc_text = doc_text.get("text")
            if not isinstance(doc_text, str):
                # Fallback to original index if document text is missing
                index = res.get("index")
                if index is not None and 0 <= index < len(documents):
                    doc_text = documents[index]
            if isinstance(doc_text, str):
                reranked_docs.append(doc_text)

        return reranked_docs


def _get_ranker() -> _JinaReranker:
    """Returns the Jina Reranker wrapper (lazy singleton)."""
    global _ranker
    if _ranker is None:
        logfire.info("🧠 Initializing Jina Reranker v3 via API...")
        _ranker = _JinaReranker()
    return _ranker


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=5),
    reraise=True,
    before_sleep=before_sleep_log(logfire, "warning"),
)
def _rerank(query: str, documents: list[str], top_n: int) -> list[str]:
    """Core Jina API reranking with retry on transient failures."""
    ranker = _get_ranker()
    return ranker.rerank(query, documents, top_n)


def rerank_documents(query: str, documents: list[str], top_n: int = 5) -> list[str]:
    """
    Refines retrieval results by re-scoring documents against the query semantically.
    Retries transient failures and falls back to the original Qdrant order if
    reranking ultimately fails, ensuring the user still receives an answer.
    """
    if not documents:
        log_query_stage("reranking", "skipped", reason="no_documents", top_n=top_n)
        return []

    if not settings.JINA_API_KEY:
        log_query_stage("reranking", "skipped", reason="provider_not_configured", top_n=top_n)
        return documents[:top_n]

    started_at = time.perf_counter()
    log_query_stage("reranking", "started", document_count=len(documents), top_n=top_n)

    try:
        reranked_docs = _rerank(query, documents, top_n)
        log_query_stage(
            "reranking",
            "completed",
            started_at=started_at,
            selected_count=len(reranked_docs),
        )
        return reranked_docs
    except Exception as e:
        log_query_stage(
            "reranking",
            "failed",
            started_at=started_at,
            error=e,
        )
        # Fallback to the original Qdrant order to ensure the user still gets an answer
        return documents[:top_n]
