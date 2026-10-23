"""Run opt-in, minimal, read-only integration probes for configured services."""

import argparse
import asyncio
import sys
from pathlib import Path

import logfire
import requests
from qdrant_client import QdrantClient
from redis import Redis

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings

logfire.configure(
    token=settings.LOGFIRE_TOKEN,
    service_name="integration-smoke-test",
    send_to_logfire="if-token-present",
    advanced=logfire.AdvancedOptions(base_url=settings.logfire_base_url) if settings.logfire_base_url else None,
)

from app.gateway import get_portkey_llm
from app.services.retrieval.embedding import embed_query, get_embedding_dim

JINA_EMBEDDING_URL = "https://api.jina.ai/v1/embeddings"
JINA_RERANK_URL = "https://api.jina.ai/v1/rerank"


def _http_error_code(error: Exception) -> str:
    response = getattr(error, "response", None)
    status_code = getattr(response, "status_code", None)
    return f" HTTP {status_code}" if status_code else ""


def _run_check(name: str, check, *, required: bool = True) -> bool:
    try:
        check()
    except Exception as error:
        level = "FAIL" if required else "OPTIONAL"
        print(f"{level}: {name} ({type(error).__name__}{_http_error_code(error)})")
        return not required
    print(f"OK: {name}")
    return True


def _check_portkey_generation() -> None:
    response = get_portkey_llm().bind(max_tokens=8).invoke("Reply with OK.")
    if not isinstance(response.content, str) or not response.content.strip():
        raise RuntimeError("Portkey returned an empty response")


def _check_gemini_ragas_judge() -> None:
    from ragas.metrics.collections import Faithfulness

    from evals.metrics import _build_judge

    judge_llm, embeddings = _build_judge()
    embedding = embeddings.embed_text("RAGAS embedding smoke test")
    if len(embedding) != get_embedding_dim():
        raise RuntimeError("RAGAS evaluation embeddings returned an unexpected vector dimension")
    score = asyncio.run(
        Faithfulness(llm=judge_llm).ascore(
            user_input="What color is the sky?",
            response="The sky is blue.",
            retrieved_contexts=["On a clear day, the sky appears blue."],
        )
    )
    if not 0 <= float(score.value) <= 1:
        raise RuntimeError("RAGAS Gemini judge returned an invalid faithfulness score")


def _check_jina_embeddings(api_key: str) -> None:
    response = requests.post(
        JINA_EMBEDDING_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": "jina-embeddings-v3",
            "task": "retrieval.query",
            "normalized": True,
            "input": ["integration smoke test"],
        },
        timeout=30,
    )
    response.raise_for_status()
    vectors = [item.get("embedding", []) for item in response.json().get("data", [])]
    if len(vectors) != 1 or len(vectors[0]) != 1024:
        raise RuntimeError("Jina returned an unexpected embedding shape")


def _check_jina_reranker(api_key: str) -> None:
    response = requests.post(
        JINA_RERANK_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": "jina-reranker-v3",
            "query": "integration smoke test",
            "documents": ["A short test document.", "Another short test document."],
            "top_n": 1,
            "return_documents": False,
        },
        timeout=30,
    )
    response.raise_for_status()
    if not response.json().get("results"):
        raise RuntimeError("Jina returned no reranking results")


def _check_qdrant() -> None:
    client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY, timeout=10)
    if not client.collection_exists(settings.QDRANT_COLLECTION):
        raise RuntimeError(f"Configured collection '{settings.QDRANT_COLLECTION}' does not exist")
    collection = client.get_collection(settings.QDRANT_COLLECTION)
    vectors = collection.config.params.vectors
    if isinstance(vectors, dict):
        raise RuntimeError("Configured retrieval expects a default vector, but Qdrant uses named vectors")
    if vectors.size != get_embedding_dim():
        raise RuntimeError(
            f"Qdrant vector dimension ({vectors.size}) does not match retrieval embeddings ({get_embedding_dim()})"
        )
    response = client.query_points(
        collection_name=settings.QDRANT_COLLECTION,
        query=embed_query("read-only integration smoke test"),
        limit=1,
        with_payload=False,
    )
    if not response.points:
        raise RuntimeError("Qdrant retrieval returned no points for the read-only probe")


def _check_neon() -> None:
    import psycopg

    database_url = settings.NEON_DB_URL
    if not database_url:
        raise ValueError("NEON_DB_URL is not configured")
    with psycopg.connect(database_url, connect_timeout=5) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
            if cursor.fetchone() != (1,):
                raise RuntimeError("Neon SELECT 1 returned an unexpected result")


def _check_upstash() -> None:
    if not settings.redis_url:
        raise ValueError("Upstash credentials are not configured")
    client = Redis.from_url(settings.redis_url, socket_connect_timeout=5, socket_timeout=5)
    if not client.ping():
        raise RuntimeError("Upstash did not respond to PING")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live",
        action="store_true",
        required=True,
        help="explicitly allow small external API probes that may consume quota",
    )
    parser.parse_args()
    healthy = True
    healthy &= _run_check("Portkey generation (configured Gemini route)", _check_portkey_generation)
    healthy &= _run_check("Gemini RAGAS judge", _check_gemini_ragas_judge)
    jina_key = settings.JINA_API_KEY
    if jina_key:
        healthy &= _run_check("Jina embeddings", lambda: _check_jina_embeddings(jina_key))
        healthy &= _run_check("Jina reranker", lambda: _check_jina_reranker(jina_key), required=False)
    else:
        print("SKIP: Jina APIs (JINA_API_KEY not configured; local embedding fallback selected)")
    healthy &= _run_check("Qdrant collection (read-only)", _check_qdrant)
    if settings.NEON_DB_URL:
        healthy &= _run_check("Neon SELECT 1 (read-only)", _check_neon, required=False)
    else:
        print("SKIP: Neon (NEON_DB_URL not configured)")
    if settings.UPSTASH_REDIS_REST_URL and settings.UPSTASH_REDIS_REST_TOKEN:
        healthy &= _run_check("Upstash PING", _check_upstash, required=False)
    else:
        print("SKIP: Upstash (credentials not configured; process-local rate limiting selected)")
    return 0 if healthy else 1


if __name__ == "__main__":
    sys.exit(main())
