# Enterprise Agentic RAG Frontend

Standalone Next.js frontend for the existing Streamlit chat interface. It recreates the Agent OS sidebar, conversation thread, guardrail status, thought steps, markdown answers, and expandable retrieved sources. The backend and Streamlit app remain separate and unchanged.

## Requirements

- Node.js 20.9 or newer
- npm

## Install and run locally

From this directory:

```powershell
npm ci
Copy-Item .env.example .env.local
notepad .env.local
npm run dev
```

Open <http://localhost:3000>.

`BACKEND_API_URL` defaults to the deployed Render API shown in `.env.example`. If the API requires authentication, set `RAG_API_KEY` in `.env.local` to the existing API bearer key. It is read only by the Next.js server route; it must never be renamed to a `NEXT_PUBLIC_*` variable. Do not commit `.env.local`.

## API behavior

The browser sends `POST /api/query` to the same-origin Next.js server. That route forwards only the supported `POST /query` request to `BACKEND_API_URL`, using `{ "q": string, "thread_id": string }`. When configured, the server adds `Authorization: Bearer <RAG_API_KEY>`. The upstream status, response body, and relevant error headers are returned to the browser. No other backend paths are proxied.

The chat thread ID lives in browser session storage. Clearing history resets the visible messages and starts a new backend thread ID, matching the existing Streamlit behavior. It does not delete old checkpoints from the backend.

## Checks

```powershell
npm run lint
npm run typecheck
npm run build
```

## Deploy to Vercel

1. Import `Bibek1008/Production-Grade-Scalable-Advance-RAG` into Vercel.
2. Set the Vercel project's **Root Directory** to `frontend`.
3. Keep the framework preset as Next.js and the install/build commands at their defaults (`npm install`/`npm run build`, or use `npm ci` where supported).
4. Add `BACKEND_API_URL` with value `https://enterprise-rag-api-af4b.onrender.com`.
5. If Render requires API authentication, add `RAG_API_KEY` as a server-side Vercel environment variable. Do not use a `NEXT_PUBLIC_` prefix.
6. For initial verification, deploy the `render-free-tier-test` branch as a Vercel preview deployment. Do not merge it into `main` as part of this setup.
7. Open the preview URL and test a normal in-domain query, an evidence/abstention query, an out-of-domain query, source expansion, and the clear-history action. Check the browser response and Render logs for any backend errors.

This repository change does not push a branch or create a Vercel deployment. The Render API must allow requests from Vercel's server runtime and have its existing credentials configured; no backend CORS change is needed because the browser calls the same-origin Next.js proxy.
