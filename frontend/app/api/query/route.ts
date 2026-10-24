import { NextRequest, NextResponse } from "next/server";

export const maxDuration = 300;

const DEFAULT_BACKEND_URL = "https://enterprise-rag-api-af4b.onrender.com";

function getQueryEndpoint(): URL | null {
  try {
    const backend = new URL(process.env.BACKEND_API_URL ?? DEFAULT_BACKEND_URL);
    if (
      !["http:", "https:"].includes(backend.protocol) ||
      backend.username ||
      backend.password ||
      backend.search ||
      backend.hash
    ) {
      return null;
    }
    return new URL("/query", backend);
  } catch {
    return null;
  }
}

export async function POST(request: NextRequest) {
  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ detail: "Request body must be valid JSON." }, { status: 400 });
  }

  if (!body || typeof body !== "object" || Array.isArray(body)) {
    return NextResponse.json({ detail: "Request body must be a JSON object." }, { status: 400 });
  }

  const payload = body as Record<string, unknown>;
  if (typeof payload.q !== "string" || payload.q.trim().length === 0) {
    return NextResponse.json({ detail: "Question must be a non-empty string." }, { status: 422 });
  }
  if (payload.thread_id !== undefined && payload.thread_id !== null && typeof payload.thread_id !== "string") {
    return NextResponse.json({ detail: "thread_id must be a string or null." }, { status: 422 });
  }

  const endpoint = getQueryEndpoint();
  if (!endpoint) {
    return NextResponse.json({ detail: "The backend URL is not configured correctly." }, { status: 500 });
  }

  const headers = new Headers({ "Content-Type": "application/json" });
  const apiKey = process.env.RAG_API_KEY;
  if (apiKey) {
    headers.set("Authorization", `Bearer ${apiKey}`);
  }

  const threadId = payload.thread_id;
  const upstreamPayload: { q: string; thread_id?: string | null } = { q: payload.q };
  if (threadId !== undefined) {
    upstreamPayload.thread_id = threadId;
  }

  let upstream: Response;
  try {
    upstream = await fetch(endpoint, {
      method: "POST",
      headers,
      body: JSON.stringify(upstreamPayload),
      cache: "no-store",
      signal: AbortSignal.timeout(300_000),
    });
  } catch (error) {
    if (error instanceof Error && error.name === "TimeoutError") {
      return NextResponse.json({ detail: "The backend request timed out. Please try again." }, { status: 504 });
    }
    return NextResponse.json({ detail: "Could not reach the RAG backend." }, { status: 502 });
  }

  const responseHeaders = new Headers();
  for (const name of ["content-type", "retry-after", "www-authenticate", "x-request-id"]) {
    const value = upstream.headers.get(name);
    if (value) responseHeaders.set(name, value);
  }
  responseHeaders.set("Cache-Control", "no-store");

  return new Response(upstream.body, {
    status: upstream.status,
    headers: responseHeaders,
  });
}
