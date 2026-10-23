# Enterprise Agentic RAG

Agentic retrieval-augmented generation built with LangGraph, Portkey-routed Google Gemini, Qdrant, Jina AI, and NeMo Guardrails.

## Current runtime

- Planner, responder, and guardrails use the Portkey OpenAI-compatible gateway. `PORTKEY_PRIMARY_CONFIG_ID`
  selects the saved primary/fallback configuration, and `PORTKEY_PRIMARY_SLUG` selects the Gemini model route.
- Default generation model: `gemini-flash-lite-latest`; override with `GEMINI_MODEL`.
- RAGAS evaluation uses Gemini independently through Google's OpenAI-compatible endpoint. Set
  `JUDGE_PROVIDER=gemini`, `JUDGE_GEMINI_MODEL`, and optionally `JUDGE_GEMINI_API_KEY` (otherwise
  the judge uses `GEMINI_API_KEY`). No OpenAI key is required.
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

Set `GEMINI_API_KEY`, all four `PORTKEY_*` values, and your Qdrant connection in `.env`. Configure
`JINA_API_KEY` if you want Jina embeddings and reranking. Leave Neon and Upstash values empty for
local in-memory operation. The saved Portkey config must retain the intended Gemini fallback route.

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

Check liveness at `http://localhost:8000/health` and readiness at `http://localhost:8000/ready`. Readiness
requires the configured Portkey route, Qdrant collection, and matching vector dimensions. Optional-service
failures are reported but do not make the API unready. Readiness performs small Portkey/Jina probes and
caches their results for 30 seconds.

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

The live integration smoke test includes a minimal Gemini-backed RAGAS faithfulness score, a Portkey
generation request, Qdrant retrieval, and read-only Neon/Upstash checks. These provider calls may consume quota.

## Render free-tier experiment

The root `render.yaml` defines the FastAPI service only. Use this branch's Blueprint when creating
the experiment; it does not configure or invoke the AWS deployment workflow. Render uses Python
3.12 from `.python-version` and installs the locked production dependencies with `uv sync --locked
--no-dev`. The start command binds Uvicorn to `0.0.0.0` and Render's runtime-provided `$PORT`.

Set the `sync: false` variables in the Render dashboard; do not put credentials in this repository.
Required API values are `GEMINI_API_KEY`, all four `PORTKEY_*` values, `JINA_API_KEY`, and
`QDRANT_URL` for the existing collection. Set `QDRANT_API_KEY` if that Qdrant cluster requires
authentication. Keep
`QDRANT_COLLECTION=enterprise_rag` unless the intended existing collection has another name. The
collection must already contain 1024-dimensional Jina vectors; deployment does not create, reset,
or migrate it. `JINA_API_KEY` is especially important on this small instance: without it, the
application selects its local mxbai embedding fallback, which loads sentence-transformers/PyTorch
when a query first needs an embedding.

Set `RAG_API_KEY` to protect the API. Add `JUDGE_GEMINI_API_KEY` only if the RAGAS evaluation
scripts are run with a separate judge credential; otherwise the judge uses `GEMINI_API_KEY`.
The API's `--no-dev` install intentionally excludes RAGAS; the Gemini RAGAS judge remains in the
development/evaluation dependency set and is not part of the serving request path.
`NEON_DB_URL`, both `UPSTASH_REDIS_REST_*` values, `LOGFIRE_TOKEN`/`LOGFIRE_BASE_URL`, and
`LANGSMITH_API_KEY` are optional integrations. Neon enables persistent LangGraph checkpoints;
if Neon is unavailable or checkpointer setup fails, `/ready` reports the active `MemorySaver` as
non-durable and conversation checkpoints disappear on restart. Logfire readiness verifies token
authentication through its region-aware `/v1/info` endpoint rather than treating token presence
as proof. Qdrant and provider credentials must refer to existing external services. Do not enter
credentials in source files or logs.

If you deploy Streamlit as a separate Render web service, set its `BACKEND_URL` to the API's full
HTTPS origin (for example, `https://<api-service>.onrender.com`) and set the same `RAG_API_KEY`.
The UI already reads `BACKEND_URL` and sends it to `/query`; do not use `localhost` for a remote
backend. Bind Streamlit to `0.0.0.0` and its runtime `$PORT`.

This is a constrained experiment, not a verified free-tier deployment. Render Free provides
512 MB RAM and 0.1 CPU and spins idle services down; NeMo Guardrails is imported and initialized
on every API start, while the project also retains sentence-transformers and PyTorch for its local
fallback. Jina prevents the fallback model from being loaded during ordinary retrieval, but does
not remove NeMo's startup footprint or the large dependency install. No Render instance has been
deployed or memory-profiled from this branch, so the API must not be considered free-tier-ready
until an actual deploy's peak memory and restart behavior are measured. If it exceeds 512 MB, the
smallest safe compute change that increases RAM is Render's `1c-2g` plan (legacy Standard, 1 CPU /
2 GB); the `0.5c-512mb` plan has the same memory limit. Keep NeMo and the local fallback unless a
separate behavior change is approved and tested.
