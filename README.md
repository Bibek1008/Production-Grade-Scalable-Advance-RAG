# Enterprise Agentic RAG

Agentic retrieval-augmented generation built with LangGraph, Google Gemini, Qdrant, Jina AI, and NeMo Guardrails.

## Current runtime

- Planner, responder, and guardrails use Gemini through Google's OpenAI-compatible endpoint:
  `https://generativelanguage.googleapis.com/v1beta/openai/`.
- Default model: `gemini-flash-lite-latest`; override with `GEMINI_MODEL`.
- Qdrant stores document vectors. Jina `jina-embeddings-v3` uses 1024-dimensional vectors.
- If `JINA_API_KEY` is absent, embeddings use local `mixedbread-ai/mxbai-embed-large-v1` instead. Do not switch embedding providers for an existing collection: matching dimensions do not mean matching vector spaces.
- Jina reranking, Neon conversation persistence, Upstash distributed rate limiting, Logfire, and LangSmith are optional.
- Without Neon, LangGraph uses in-memory conversation state, which is lost on process restart. Without Upstash, rate limiting is process-local.

## Windows PowerShell setup

Python 3.12 is recommended. Do not read or commit `.env`; copy the template and add your own credentials:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
notepad .env
```

If PowerShell does not allow virtual-environment activation, do not change the machine execution policy. Invoke `.\.venv\Scripts\python.exe` directly instead.

Set `GEMINI_API_KEY` and your Qdrant connection in `.env`. Configure `JINA_API_KEY` if you want Jina embeddings and reranking. Leave Neon and Upstash values empty for local in-memory operation.

Docker Compose starts a local Qdrant service and routes the API container to it. To use a cloud Qdrant from Compose, set `DOCKER_QDRANT_URL` and `DOCKER_QDRANT_API_KEY` in `.env`; these override the local service route.

**Windows Application Control note:** Some Windows policies block the native `uuid_utils` extension imported by LangChain. Do not disable or bypass Application Control. If Python reports that this module is blocked, use WSL for the Python environment and commands below.

## Prepare Qdrant and run

First ingest the documents you intend to query. This command does not delete the existing collection:

```powershell
.\.venv\Scripts\python.exe -m app.ingestion.processor DATA
```

Do **not** add `--wipe` unless you have deliberately decided to delete all existing vectors in that Qdrant collection. The ingestion command creates the collection when it is missing and appends indexed points. Keep the same embedding provider for both ingestion and query.

Start the API and UI in separate PowerShell terminals:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
```

```powershell
.\.venv\Scripts\python.exe -m streamlit run ui/app.py
```

Check liveness at `http://localhost:8000/health` and readiness at `http://localhost:8000/ready`. Readiness requires Gemini, Qdrant, and the configured Qdrant collection. Optional-service failures are reported but do not make the API unready. Readiness performs small Gemini/Jina probes and caches their results for 30 seconds.

## Verify and test

An explicit integration smoke test sends small, read-only requests to configured external services:

```powershell
.\.venv\Scripts\python.exe scripts/smoke_test_integrations.py --live
```

This does not create infrastructure or write/delete Qdrant data. Calls may consume provider quota.

```powershell
.\.venv\Scripts\ruff.exe check app tests evals
.\.venv\Scripts\ruff.exe format --check app tests evals
.\.venv\Scripts\python.exe -m pytest
```

The OpenAI key is only used as an optional RAGAS evaluation judge; it is not needed for application inference.
