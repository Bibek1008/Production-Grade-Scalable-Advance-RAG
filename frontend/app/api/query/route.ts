import { NextRequest, NextResponse } from "next/server";

export const maxDuration = 300;

const DEFAULT_BACKEND_URL = "https://enterprise-rag-api-af4b.onrender.com";
const MAX_REQUEST_BYTES = 16 * 1024;
const MAX_QUESTION_LENGTH = 8_000;
const MAX_THREAD_ID_LENGTH = 256;

function getQueryEndpoint(): URL | null {
  try {
    const backend = new URL(process.env.BACKEND_API_URL ?? DEFAULT_BACKEND_URL);
    if (
      !["http:", "https:"].includes(backend.protocol) ||
      backend.username ||
      backend.password ||
      backend.pathname !== "/" ||
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

async function readRequestBody(request: NextRequest): Promise<string | null> {
  const reader = request.body?.getReader();
  if (!reader) return "";

  const chunks: Uint8Array[] = [];
  let totalBytes = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    totalBytes += value.byteLength;
    if (totalBytes > MAX_REQUEST_BYTES) {
      await reader.cancel().catch(() => undefined);
      return null;
    }
    chunks.push(value);
  }

  const bytes = new Uint8Array(totalBytes);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return new TextDecoder("utf-8", { fatal: true }).decode(bytes);
}

export async function POST(request: NextRequest) {
  const contentLength = request.headers.get("content-length");
  if (contentLength !== null) {
    const declaredLength = Number(contentLength);
    if (!Number.isSafeInteger(declaredLength) || declaredLength < 0) {
      return NextResponse.json({ detail: "Content-Length is invalid." }, { status: 400 });
    }
    if (declaredLength > MAX_REQUEST_BYTES) {
      return NextResponse.json({ detail: "Request body is too large." }, { status: 413 });
    }
  }

  let body: unknown;
  try {
    const rawBody = await readRequestBody(request);
    if (rawBody === null) {
      return NextResponse.json({ detail: "Request body is too large." }, { status: 413 });
    }
    body = JSON.parse(rawBody);
  } catch {
    return NextResponse.json({ detail: "Request body must be valid JSON." }, { status: 400 });
  }

  if (!body || typeof body !== "object" || Array.isArray(body)) {
    return NextResponse.json({ detail: "Request body must be a JSON object." }, { status: 400 });
  }

  const payload = body as Record<string, unknown>;
  if (Object.keys(payload).some((key) => key !== "q" && key !== "thread_id")) {
    return NextResponse.json({ detail: "Request contains unsupported fields." }, { status: 422 });
  }
  if (typeof payload.q !== "string" || payload.q.trim().length === 0) {
    return NextResponse.json({ detail: "Question must be a non-empty string." }, { status: 422 });
  }
  if (payload.q.length > MAX_QUESTION_LENGTH) {
    return NextResponse.json({ detail: "Question is too long." }, { status: 422 });
  }
  if (payload.thread_id !== undefined && payload.thread_id !== null && typeof payload.thread_id !== "string") {
    return NextResponse.json({ detail: "thread_id must be a string or null." }, { status: 422 });
  }
  if (typeof payload.thread_id === "string" && payload.thread_id.length > MAX_THREAD_ID_LENGTH) {
    return NextResponse.json({ detail: "thread_id is too long." }, { status: 422 });
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
