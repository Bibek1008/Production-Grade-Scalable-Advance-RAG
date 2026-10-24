# Enterprise Agentic RAG

Agentic retrieval-augmented generation built with FastAPI, LangGraph, Portkey-routed Gemini, Qdrant, Jina, and NeMo Guardrails.

## Deployment architecture

- **Frontend:** Next.js, TypeScript, and Tailwind CSS in [`frontend/`](./frontend), deployed on Vercel.
- **Backend:** FastAPI and the RAG pipeline, deployed on Render at <https://enterprise-rag-api-af4b.onrender.com>.
- **CI:** GitHub Actions runs the Python tests and Ruff checks, plus the Next.js lint, typecheck, and production build.

The Next.js app calls its same-origin `/api/query` route. That server-side proxy forwards only `POST /query` to the backend, and keeps `RAG_API_KEY` server-side. The proxy still has the existing Render URL as a fallback; configure `BACKEND_API_URL` explicitly in Vercel.

## Frontend on Vercel

1. Import this repository into Vercel and set **Root Directory** to `frontend`.
2. Keep the **Next.js** framework preset and default install/build commands.
3. Set the server-side environment variable `BACKEND_API_URL` to the Render API URL.
4. If the backend has `RAG_API_KEY` configured, set the matching `RAG_API_KEY` in Vercel. Do not use a `NEXT_PUBLIC_` prefix.

Local frontend development:

```powershell
cd frontend
npm ci
Copy-Item .env.example .env.local
npm run dev
```

`frontend/.env.example` contains the default backend URL and an empty optional API key. Never commit `.env.local`.

Run frontend checks from `frontend/`:

```powershell
npm run lint
npm run typecheck
npm run build
```

## Backend on Render

The existing API is available at <https://enterprise-rag-api-af4b.onrender.com>. Its liveness and readiness endpoints are `/health` and `/ready`. `/query` accepts authenticated `POST` requests when `RAG_API_KEY` is configured.

There is no checked-in `render.yaml`; Render's active service settings are managed outside this repository and are intentionally unchanged. The repository's root [`Dockerfile`](./Dockerfile) is the available container build definition. It installs the backend dependencies from `pyproject.toml` and starts:

```text
uvicorn app.main:app --host 0.0.0.0 --port 8080 --timeout-graceful-shutdown 5
```

For a local container build, run `docker build -t enterprise-rag -f Dockerfile .` from the repository root. If the existing Render service is configured as a Docker service, it should build from this root `Dockerfile` and use its `CMD`; confirm the actual dashboard build/runtime settings before changing them. This cleanup does not replace its external configuration.

### Backend configuration

Configure secrets in Render's environment settings, not in source control. The main settings are:

- **Generation:** `GEMINI_API_KEY`, `PORTKEY_API_KEY`, `PORTKEY_PRIMARY_SLUG`, `PORTKEY_FALLBACK_SLUG`, `PORTKEY_PRIMARY_CONFIG_ID`; optional `GEMINI_MODEL`.
- **Retrieval:** `QDRANT_URL` (or `QDRANT_CLUSTER_ENDPOINT`), `QDRANT_API_KEY` when required by the cluster, `QDRANT_COLLECTION`, and optional `JINA_API_KEY`.
- **Optional persistence and rate limiting:** `NEON_DB_URL`, `UPSTASH_REDIS_REST_URL`, and `UPSTASH_REDIS_REST_TOKEN`.
- **API protection:** `RAG_API_KEY` and optional `RATE_LIMIT_PER_MINUTE`.
- **Optional observability:** `LOGFIRE_TOKEN`, `LOGFIRE_BASE_URL`, `LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`, and `LANGSMITH_ENDPOINT`.

The backend relies on Gemini through Portkey and Qdrant for generation and vector retrieval. Jina supplies remote embeddings and reranking when configured; otherwise the existing local embedding fallback is used. Neon PostgreSQL persistence, Upstash distributed rate limiting, Logfire, and LangSmith are optional. Readiness is determined by the live `/ready` check; this README does not assert current service health.

### Local backend setup and checks

Python 3.12 is recommended. Do not read or commit `.env`; start from the example and fill it with your own local settings:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

Start the API:

```powershell
python -m uvicorn app.main:app --reload --port 8000
```

Start the frontend separately from `frontend/` with `npm run dev`.

Run Python tests and checks from the repository root:

```powershell
python -m pytest tests/
ruff check app tests evals
ruff format --check app tests evals
uv lock --check
```

The evaluation tooling, including its Streamlit-based evaluation UI, remains separate from the user-facing Next.js application. To run that UI locally, install the development extra and use `streamlit run evals/app.py`.

Evaluation runs use `BACKEND_URL` to locate the running FastAPI API and `JUDGE_PROVIDER`, `JUDGE_GEMINI_MODEL`, and optionally `JUDGE_GEMINI_API_KEY` for the RAGAS judge. When the judge key is empty, evaluation uses `GEMINI_API_KEY`.
