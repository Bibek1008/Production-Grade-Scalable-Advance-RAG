# Enterprise Agentic RAG Frontend

The canonical user-facing application is a Next.js frontend in `frontend/`, deployed on Vercel. The FastAPI backend and RAG pipeline remain on Render.

## Requirements and local development

- Node.js 20.19 or newer (required by the locked ESLint version)
- npm

From this directory:

```powershell
npm ci
Copy-Item .env.example .env.local
npm run dev
```

Open <http://localhost:3000>.

Set `BACKEND_API_URL` to the FastAPI backend origin. It currently defaults to the existing Render API URL for compatibility; configure it explicitly in Vercel. If the backend requires authentication, set `RAG_API_KEY` as a server-side environment variable. The key is read only by the Next.js route and must never be prefixed with `NEXT_PUBLIC_`. Do not commit `.env.local`.

## Backend proxy

The browser sends `POST /api/query` to the same-origin Next.js server. The route validates a bounded JSON request and forwards only `q` and the optional `thread_id` to `POST /query` at `BACKEND_API_URL`. When configured, the server sends `RAG_API_KEY` in the `Authorization: Bearer` header. The upstream status and response body, along with selected safe headers, are returned to the browser. No other backend paths are proxied.

The chat thread ID lives in browser session storage. Clearing history resets the visible messages and starts a new backend thread ID; it does not delete old checkpoints on the backend.

## Checks

Run from this directory:

```powershell
npm run lint
npm run typecheck
npm run build
```

## Deploy to Vercel

1. Import `Bibek1008/Production-Grade-Scalable-Advance-RAG` into Vercel.
2. Set the Vercel project's **Root Directory** to `frontend`.
3. Select the **Next.js** framework preset and retain the standard npm install/build settings.
4. Configure `BACKEND_API_URL` with the Render API origin, `https://enterprise-rag-api-af4b.onrender.com`.
5. If the backend uses API authentication, configure the matching `RAG_API_KEY` as a server-side Vercel environment variable. Never expose it through a `NEXT_PUBLIC_*` variable.

No Vercel deployment is created by repository changes. The Render API must already be configured with its existing credentials; the browser reaches it through the same-origin proxy, so no browser-side API key or cross-origin backend request is needed.
