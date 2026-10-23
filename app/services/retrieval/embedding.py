"""Consistent Jina or local embeddings for document indexing and retrieval."""

import threading

import logfire
import requests
from tenacity import before_sleep_log, retry, retry_if_exception, stop_after_attempt, wait_exponential

from app.config import settings

BATCH_SIZE = 64
_EMBEDDING_DIM = 1024
_JINA_EMBEDDING_URL = "https://api.jina.ai/v1/embeddings"
_JINA_MODEL = "jina-embeddings-v3"
_FALLBACK_MODEL = "mixedbread-ai/mxbai-embed-large-v1"

_active_model = None
_model_type: str | None = None
_model_lock = threading.Lock()


def _load_fallback():
    """Load the local mxbai fallback model."""
    from sentence_transformers import SentenceTransformer

    logfire.info(f"Loading fallback embedding model ({_FALLBACK_MODEL}, {_EMBEDDING_DIM}-dim).")
    return SentenceTransformer(_FALLBACK_MODEL)


def _init():
    """Choose one embedding vector space once per process."""
    global _model_type
    if _model_type is not None:
        return

    with _model_lock:
        if _model_type is None:
            if settings.JINA_API_KEY:
                _model_type = "jina"
            else:
                logfire.info("JINA_API_KEY not set — using local fallback embeddings.")
                _model_type = "fallback"


def get_embedding_dim() -> int:
    """Return the dimension shared by the selected embedding provider."""
    _init()
    return _EMBEDDING_DIM


def _is_transient_jina_error(error: BaseException) -> bool:
    """Retry network errors and server failures, but not auth or quota errors."""
    if isinstance(error, requests.HTTPError):
        response = error.response
        return response is not None and (response.status_code == 408 or response.status_code >= 500)
    return isinstance(error, (requests.Timeout, requests.ConnectionError))


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=5),
    reraise=True,
    before_sleep=before_sleep_log(logfire, "warning"),
    retry=retry_if_exception(_is_transient_jina_error),
)
def _embed_jina_batch(texts: list[str], task: str) -> list[list[float]]:
    """Call Jina and validate its vector count and dimensions."""
    response = requests.post(
        _JINA_EMBEDDING_URL,
        headers={
            "Authorization": f"Bearer {settings.JINA_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": _JINA_MODEL,
            "task": task,
            "normalized": True,
            "input": texts,
        },
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    results = sorted(payload.get("data", []), key=lambda item: item.get("index", 0))
    embeddings = [item["embedding"] for item in results]
    if len(embeddings) != len(texts) or any(len(vector) != _EMBEDDING_DIM for vector in embeddings):
        raise RuntimeError(
            f"Jina returned an unexpected embedding shape; expected {len(texts)} vectors "
            f"of {_EMBEDDING_DIM} dimensions."
        )
    return embeddings


def _embed_jina(texts: list[str], task: str) -> list[list[float]]:
    """Embed texts via Jina in bounded batches."""
    all_embeddings: list[list[float]] = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        with logfire.span("Embed batch via Jina API", start=start, size=len(batch)):
            all_embeddings.extend(_embed_jina_batch(batch, task))
    return all_embeddings


def _embed_fallback_batch(texts: list[str]) -> list[list[float]]:
    """Embed texts using the local mxbai model."""
    global _active_model
    if _active_model is None:
        with _model_lock:
            if _active_model is None:
                _active_model = _load_fallback()
    embeddings = _active_model.encode(texts, show_progress_bar=False).tolist()
    if any(len(vector) != _EMBEDDING_DIM for vector in embeddings):
        raise RuntimeError(f"Local fallback returned vectors that are not {_EMBEDDING_DIM}-dimensional.")
    return embeddings


def _embed_fallback(texts: list[str]) -> list[list[float]]:
    """Embed texts via the local fallback model in bounded batches."""
    all_embeddings: list[list[float]] = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        with logfire.span("Embed batch via fallback model", start=start, size=len(batch)):
            all_embeddings.extend(_embed_fallback_batch(batch))
    return all_embeddings


def _embed(texts: list[str], task: str) -> list[list[float]]:
    """Embed using one process-wide provider; never mix vector spaces."""
    _init()
    if not texts:
        return []

    if _model_type == "jina":
        try:
            return _embed_jina(texts, task)
        except Exception as error:
            logfire.error(f"Jina Embeddings API failed for the configured vector model: {error}")
            raise RuntimeError(
                "Jina embedding failed. Verify JINA_API_KEY, account quota, and API availability; "
                "no alternate vector model was used."
            ) from error
    return _embed_fallback(texts)


def embed_query(query: str) -> list[float]:
    """Embed one retrieval query."""
    return _embed([query], task="retrieval.query")[0]


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed document texts for indexing."""
    return _embed(texts, task="retrieval.passage")
