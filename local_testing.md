# Local Testing Guide

This guide uses Windows PowerShell. Python 3.12 is recommended. If Windows Application Control blocks LangChain's native `uuid_utils` extension, do not disable or bypass the policy; use a WSL Python environment instead.

## 1. Set up the environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
notepad .env
```

If activation is blocked, leave the execution policy unchanged and invoke `.\.venv\Scripts\python.exe` directly.

Set `GEMINI_API_KEY` and Qdrant credentials in `.env`. `GEMINI_MODEL` defaults to `gemini-flash-lite-latest`. `JINA_API_KEY` is optional: if omitted, the app uses local `mixedbread-ai/mxbai-embed-large-v1` embeddings and skips reranking. Use the same embedding provider to index and search one Qdrant collection; the providers do not share a vector space.

Neon (`NEON_DB_URL`) and Upstash (`UPSTASH_REDIS_REST_URL` and `UPSTASH_REDIS_REST_TOKEN`) are optional for local development. Without Neon, conversation history is in memory and is lost on restart. Without Upstash, rate limits are process-local.

## 2. Run static checks and unit tests

```powershell
.\.venv\Scripts\ruff.exe check app tests evals
.\.venv\Scripts\ruff.exe format --check app tests evals
.\.venv\Scripts\python.exe -m pytest
```

The unit tests do not need live Gemini, Jina, Qdrant, Neon, or Upstash credentials. If Windows Application Control blocks a native dependency during test collection, use WSL rather than modifying the policy or installed system files.

## 3. Run an explicit integration smoke test

```powershell
.\.venv\Scripts\python.exe scripts/smoke_test_integrations.py --live
```

This sends small requests to configured providers, including one Gemini generation and (when configured) Jina embedding and reranking probes. It performs a read-only Qdrant collection check and Neon `SELECT 1`; it does not write or delete remote data. Provider calls may consume quota. The script skips optional services whose credentials are absent and never prints secret values.

The built-in startup and `/ready` checks also probe configured external APIs. Results are cached for 30 seconds to limit API use. `/health` is a liveness-only endpoint. `/ready` requires Gemini, Qdrant connectivity, and the configured Qdrant collection; optional service failures are reported without blocking readiness.

## 4. Ingest documents safely

Check the configured Qdrant URL, collection, and embedding provider before indexing. The following command is non-destructive and creates the collection only if it is missing:

```powershell
.\.venv\Scripts\python.exe -m app.ingestion.processor DATA
```

For a single source folder:

```powershell
.\.venv\Scripts\python.exe -m app.ingestion.processor DATA\true_data true
```

Do not add `--wipe`: it deletes the configured Qdrant collection and all its points. Only use that flag when deletion is explicitly intended and backed up. Confirm indexing from the Qdrant dashboard or a read-only collection inspection before querying.

## 5. Start and query the API

Start the API in one PowerShell terminal:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
```

Start the Next.js frontend in another terminal:

```powershell
Set-Location frontend
npm ci
Copy-Item .env.example .env.local
npm run dev
```

Set `BACKEND_API_URL` in `frontend/.env.local` to the local API origin. If authentication is enabled, configure `RAG_API_KEY` there; it is used only by the server-side proxy.

Check service state:

```powershell
Invoke-RestMethod http://localhost:8000/health
Invoke-RestMethod http://localhost:8000/ready
```

Send a query:

```powershell
$body = @{ q = "How do I configure Redis?"; thread_id = "local-user" } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://localhost:8000/query `
  -ContentType "application/json" -Body $body
```

If `RAG_API_KEY` is configured, add an `Authorization: Bearer <key>` header to protected API requests.

The separate Streamlit evaluation dashboard remains available at `evals/app.py`; it is not the user-facing chat application.

## Troubleshooting

| Symptom | Likely cause and action |
|---|---|
| Gemini authentication or quota error | Check `GEMINI_API_KEY`, confirm `GEMINI_MODEL` is available, and review the Google AI Studio quota. |
| Jina 401/403 | Verify `JINA_API_KEY`. The app will surface embedding errors and will not silently change vector spaces. |
| Jina 429 | Check account quota/rate limits. Retry after quota resets; auth and quota failures are not automatically retried. |
| Qdrant collection missing | Run the safe ingestion command after confirming the target cluster and collection. |
| No search results | Confirm documents were ingested with the same embedding provider and configured collection. |
| Windows blocked `_uuid_utils` | Run the environment in WSL; do not disable Application Control or patch the installed package. |
