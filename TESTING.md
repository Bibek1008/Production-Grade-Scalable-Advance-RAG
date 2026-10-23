# Testing Guide

Use Windows PowerShell commands below. Python 3.12 is recommended. If Windows Application Control blocks the LangChain `uuid_utils` native extension, run the project in WSL; do not disable/bypass the policy or patch installed Python packages.

## Unit tests and static checks

From the repository root:

```powershell
.\.venv\Scripts\ruff.exe check app tests evals
.\.venv\Scripts\ruff.exe format --check app tests evals
.\.venv\Scripts\python.exe -m compileall -q app tests evals
.\.venv\Scripts\python.exe -m pytest
```

These checks are intended to run without live service credentials. Unit-test collection may still be prevented if Application Control blocks a native dependency on Windows; use WSL in that case.

## Live integration smoke test

Run only when you intend to consume provider quota:

```powershell
.\.venv\Scripts\python.exe scripts/smoke_test_integrations.py --live
```

The test performs one minimal Gemini generation, Jina embed/rerank probes when configured, Qdrant collection inspection, Neon `SELECT 1`, and Upstash `PING` when configured. Qdrant and Neon operations are read-only. It does not ingest data, create infrastructure, or delete collections.

## Local app checks

Start the API:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
```

From another terminal:

```powershell
Invoke-RestMethod http://localhost:8000/health
Invoke-RestMethod http://localhost:8000/ready
$body = @{ q = "How do I configure the service?"; thread_id = "test-user" } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://localhost:8000/query `
  -ContentType "application/json" -Body $body
```

`/health` is process liveness. `/ready` requires Gemini, Qdrant, and the configured Qdrant collection. The probe results are cached for 30 seconds. Neon, Upstash, observability, local fallback embeddings, and Jina reranking are optional as configured.

## Safe document ingestion

After confirming `.env` points to the intended Qdrant cluster and collection, run:

```powershell
.\.venv\Scripts\python.exe -m app.ingestion.processor DATA
```

This creates the collection if missing and indexes documents without deleting existing data. **Do not pass `--wipe`** unless deleting all points in the collection is intentional and backed up.

Keep one embedding provider consistent between indexing and querying. Jina and the local mxbai fallback both produce 1024 dimensions, but their vector spaces are incompatible.
