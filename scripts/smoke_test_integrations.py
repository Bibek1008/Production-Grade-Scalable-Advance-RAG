"""Run opt-in, minimal, read-only integration probes for configured services."""

import argparse
import os
import sys
from urllib.parse import quote, urlsplit, urlunsplit

import requests
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from redis import Redis

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
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


def _check_gemini() -> None:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not configured")
    client = OpenAI(
        api_key=api_key,
        base_url=GEMINI_BASE_URL,
        timeout=20,
        max_retries=0,
    )
    response = client.chat.completions.create(
        model=os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest"),
        messages=[{"role": "user", "content": "Reply with OK."}],
        max_tokens=8,
    )
    if not response.choices or not response.choices[0].message.content:
        raise RuntimeError("Gemini returned an empty response")


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
    url = os.getenv("QDRANT_URL") or os.getenv("QDRANT_CLUSTER_ENDPOINT") or "http://localhost:6333"
    collection = os.getenv("QDRANT_COLLECTION", "enterprise_rag")
    client = QdrantClient(url=url, api_key=os.getenv("QDRANT_API_KEY") or None, timeout=10)
    if not client.collection_exists(collection):
        raise RuntimeError(f"Configured collection '{collection}' does not exist")


def _check_neon() -> None:
    import psycopg

    database_url = os.getenv("NEON_DB_URL")
    if not database_url:
        raise ValueError("NEON_DB_URL is not configured")
    with psycopg.connect(database_url, connect_timeout=5) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
            if cursor.fetchone() != (1,):
                raise RuntimeError("Neon SELECT 1 returned an unexpected result")


def _check_upstash() -> None:
    endpoint = os.getenv("UPSTASH_REDIS_REST_URL")
    token = os.getenv("UPSTASH_REDIS_REST_TOKEN")
    if not endpoint or not token:
        raise ValueError("Upstash credentials are not configured")
    host = urlsplit(endpoint).netloc or urlsplit("https://" + endpoint).netloc
    redis_url = urlunsplit(("rediss", f"default:{quote(token, safe='')}@{host}", "/0", "ssl_cert_reqs=required", ""))
    client = Redis.from_url(redis_url, socket_connect_timeout=5, socket_timeout=5)
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
    load_dotenv()

    healthy = True
    healthy &= _run_check("Gemini generation", _check_gemini)
    jina_key = os.getenv("JINA_API_KEY")
    if jina_key:
        healthy &= _run_check("Jina embeddings", lambda: _check_jina_embeddings(jina_key))
        healthy &= _run_check("Jina reranker", lambda: _check_jina_reranker(jina_key), required=False)
    else:
        print("SKIP: Jina APIs (JINA_API_KEY not configured; local embedding fallback selected)")
    healthy &= _run_check("Qdrant collection (read-only)", _check_qdrant)
    if os.getenv("NEON_DB_URL"):
        healthy &= _run_check("Neon SELECT 1 (read-only)", _check_neon, required=False)
    else:
        print("SKIP: Neon (NEON_DB_URL not configured)")
    if os.getenv("UPSTASH_REDIS_REST_URL") and os.getenv("UPSTASH_REDIS_REST_TOKEN"):
        healthy &= _run_check("Upstash PING", _check_upstash, required=False)
    else:
        print("SKIP: Upstash (credentials not configured; process-local rate limiting selected)")
    return 0 if healthy else 1


if __name__ == "__main__":
    sys.exit(main())
