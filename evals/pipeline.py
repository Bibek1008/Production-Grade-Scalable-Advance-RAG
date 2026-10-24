"""
Phase 1 — Live Pipeline.
Calls the running FastAPI /query endpoint for each golden sample.
Captures: actual_response (truncated to 300 chars), actual_contexts (from sources),
and actual_tools_called (detected from thought_process).
"""

import copy
import json
import os
import time

import logfire
import requests

from app.config import settings

API_URL = f"{settings.backend_url}/query"
STATUS_URL_TEMPLATE = f"{settings.backend_url}/query/status/{{job_id}}"
RESPONSE_TRUNCATE = 300
DELAY_BETWEEN_CALLS = 10  # seconds — avoid exhausting provider quotas during evals
REQUEST_TIMEOUT = 120  # seconds — guardrails + LangGraph can take >60s
POLL_INTERVAL = 3  # seconds between job status polls
MAX_POLL_ATTEMPTS = 60  # ~3 minutes max wait per sample


class BackendAPIError(RuntimeError):
    """An API request failed with a non-success HTTP status."""


def _raise_for_status(response: requests.Response, endpoint: str) -> None:
    try:
        response.raise_for_status()
    except requests.HTTPError:
        raise BackendAPIError(f"FastAPI {endpoint} returned HTTP {response.status_code}.") from None


def _api_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {settings.API_KEY}"} if settings.API_KEY else {}


def detect_tool(thought_process: list) -> str:
    """
    Maps the thought_process list from /query response to a tool name.
    Planner sets:  'Intent: Technical' + 'Search Term: ...' → retrieve_documents
                   'Intent: Conversational/Memory'           → direct_answer
    main.py sets:  'Intent: Guardrails Fired'                → guardrails
    """
    joined = " ".join(thought_process).lower()
    if "guardrails fired" in joined:
        return "guardrails"
    if "intent: technical" in joined or "search term:" in joined or "context retrieved" in joined:
        return "retrieve_documents"
    if "conversational" in joined or "memory" in joined:
        return "direct_answer"
    return "unknown"


def _poll_for_result(job_id: str) -> dict:
    """Poll /query/status until the Celery job completes or times out."""
    url = STATUS_URL_TEMPLATE.format(job_id=job_id)
    for attempt in range(MAX_POLL_ATTEMPTS):
        with logfire.span("🔄 Eval polling job", job_id=job_id, attempt=attempt + 1):
            resp = requests.get(url, headers=_api_headers(), timeout=REQUEST_TIMEOUT)
            _raise_for_status(resp, "/query/status")
            data = resp.json()

        status = data.get("status", "UNKNOWN")
        if status == "SUCCESS":
            return data.get("result", {})
        if status == "FAILURE":
            error = data.get("error", "unknown failure")
            raise RuntimeError(f"RAG job failed: {error}")
        time.sleep(POLL_INTERVAL)

    raise RuntimeError(f"Polling timed out for job {job_id}")


def _fetch_query_result(question: str, thread_id: str) -> dict:
    """Submit a query and return the final result (handling sync block + async jobs)."""
    resp = requests.post(
        API_URL,
        json={"q": question, "thread_id": thread_id},
        headers=_api_headers(),
        timeout=REQUEST_TIMEOUT,
    )
    _raise_for_status(resp, "/query")
    data = resp.json()

    # Guardrails can block synchronously without creating a job.
    if data.get("status") == "Blocked by guardrails." or "answer" in data:
        return data

    job_id = data.get("job_id")
    if not job_id:
        raise RuntimeError(f"Unexpected /query response: {data}")

    return _poll_for_result(job_id)


def run_pipeline(golden_dataset: dict, progress_callback=None) -> dict:
    """
    Enriches each rag_sample in golden_dataset with live API results.
    Returns a deep copy with actual_response, actual_contexts, actual_tools_called filled.
    progress_callback(i, total, question, stage, response="") is called per step.
    """
    dataset = copy.deepcopy(golden_dataset)
    samples = dataset["rag_samples"]
    n = len(samples)

    with logfire.span("🚀 Eval Phase 1 — Live Pipeline", total_samples=n):
        for i, sample in enumerate(samples):
            question = sample["question"]

            if progress_callback:
                progress_callback(i, n, question, "calling")

            with logfire.span(
                f"📤 Live Query {i + 1}/{n}",
                question=question[:80],
                domain=sample.get("domain", ""),
            ):
                try:
                    data = _fetch_query_result(question, thread_id=f"eval_run_{i}")

                    raw_answer = data.get("answer") or ""
                    thought_process = data.get("thought_process") or []
                    sources = data.get("sources") or []

                    sample["actual_response"] = raw_answer[:RESPONSE_TRUNCATE]
                    sample["actual_contexts"] = [
                        source.get("content", "") if isinstance(source, dict) else source for source in sources[:5]
                    ]
                    sample["actual_tools_called"] = [detect_tool(thought_process)]

                    logfire.info(
                        "✅ Response captured",
                        tool=sample["actual_tools_called"][0],
                        response_chars=len(raw_answer),
                        context_chunks=len(sources),
                    )

                except requests.exceptions.ConnectionError as error:
                    raise RuntimeError(f"Cannot reach the FastAPI backend at {settings.backend_url}.") from error

            if progress_callback:
                progress_callback(i, n, question, "done", sample["actual_response"])

            if i < n - 1:
                time.sleep(DELAY_BETWEEN_CALLS)

    return dataset


def save_results(dataset: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(dataset, f, indent=2)


def load_golden_dataset() -> dict:
    golden_path = os.path.join(os.path.dirname(__file__), "golden_dataset.json")
    with open(golden_path, encoding="utf-8") as f:
        return json.load(f)
