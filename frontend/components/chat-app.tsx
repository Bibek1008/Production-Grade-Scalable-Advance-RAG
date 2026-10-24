"use client";

import ReactMarkdown from "react-markdown";
import { FormEvent, KeyboardEvent, useEffect, useRef, useState, useSyncExternalStore } from "react";

type Source = string | { content?: string; source?: string; score?: number };

type QueryResponse = {
  answer?: string | null;
  thought_process?: string[] | null;
  status?: string;
  sources?: Source[] | null;
};

type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  blocked?: boolean;
  steps?: string[];
  sources?: Source[];
  failed?: boolean;
};

const THREAD_STORAGE_KEY = "enterprise-rag-thread-id";
let fallbackThreadId = "";
const threadIdListeners = new Set<() => void>();

function readThreadId(): string {
  if (typeof window === "undefined") return "";
  try {
    const storedId = window.sessionStorage.getItem(THREAD_STORAGE_KEY);
    if (storedId) fallbackThreadId = storedId;
  } catch {
    return fallbackThreadId;
  }
  return fallbackThreadId;
}

function setStoredThreadId(threadId: string) {
  fallbackThreadId = threadId;
  if (typeof window !== "undefined") {
    try {
      window.sessionStorage.setItem(THREAD_STORAGE_KEY, threadId);
    } catch {
      // Keep the conversation usable if browser storage is unavailable.
    }
  }
  threadIdListeners.forEach((listener) => listener());
}

function subscribeToThreadId(listener: () => void) {
  threadIdListeners.add(listener);
  if (!readThreadId()) setStoredThreadId(window.crypto.randomUUID());
  return () => threadIdListeners.delete(listener);
}

function getErrorMessage(body: unknown, status: number): string {
  if (body && typeof body === "object") {
    const errorBody = body as { detail?: unknown; message?: unknown };
    if (typeof errorBody.detail === "string") return errorBody.detail;
    if (typeof errorBody.message === "string") return errorBody.message;
  }
  return `The backend returned an error (HTTP ${status}).`;
}

export function ChatApp() {
  const threadId = useSyncExternalStore(subscribeToThreadId, readThreadId, () => "");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [question, setQuestion] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, isLoading]);

  function clearConversation() {
    if (isLoading) return;
    const nextId = window.crypto.randomUUID();
    setStoredThreadId(nextId);
    setMessages([]);
    setQuestion("");
  }

  async function submitQuestion(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const submittedQuestion = question;
    if (!submittedQuestion.trim() || !threadId || isLoading) return;

    const userMessage: ChatMessage = {
      id: window.crypto.randomUUID(),
      role: "user",
      content: submittedQuestion,
    };
    setMessages((current) => [...current, userMessage]);
    setQuestion("");
    setIsLoading(true);

    try {
      const response = await fetch("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ q: submittedQuestion, thread_id: threadId }),
      });
      const responseText = await response.text();
      let responseBody: unknown;
      try {
        responseBody = responseText ? JSON.parse(responseText) : {};
      } catch {
        responseBody = null;
      }

      if (!response.ok) {
        throw new Error(getErrorMessage(responseBody, response.status));
      }
      if (!responseBody || typeof responseBody !== "object") {
        throw new Error("The backend returned an invalid response.");
      }

      const data = responseBody as QueryResponse;
      const isBlocked = data.status === "Blocked by guardrails.";
      setMessages((current) => [
        ...current,
        {
          id: window.crypto.randomUUID(),
          role: "assistant",
          content: data.answer ?? "No response.",
          blocked: isBlocked,
          steps: data.thought_process ?? [],
          sources: data.sources ?? [],
        },
      ]);
    } catch (error) {
      const message = error instanceof Error ? error.message : "The request failed. Please try again.";
      setMessages((current) => [
        ...current,
        {
          id: window.crypto.randomUUID(),
          role: "assistant",
          content: message,
          failed: true,
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  }

  function handleInputKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      event.currentTarget.form?.requestSubmit();
    }
  }

  return (
    <div className="app-shell flex min-h-screen">
      {sidebarOpen && (
        <aside className="sidebar sticky top-0 flex h-screen w-[300px] shrink-0 flex-col px-5 py-[30px]">
          <div className="sidebar-heading">
            <span aria-hidden="true">🧠</span>
            <h2>Agent OS</h2>
            <button
              aria-label="Close sidebar"
              className="sidebar-toggle mobile-only"
              onClick={() => setSidebarOpen(false)}
              type="button"
            >
              ×
            </button>
          </div>
          <div className="sidebar-divider" />
          <div className="sidebar-status unavailable-status" title="The backend does not expose Logfire status to the frontend.">
            <span className="status-dot unavailable" />
            <span>Logfire: Status unavailable</span>
          </div>
          <div className="memory-card">
            <span>Memory ID:</span>
            <code>{threadId ? threadId.slice(0, 8) : "--------"}</code>
          </div>
          <button className="clear-button" disabled={!threadId || isLoading} onClick={clearConversation} type="button">
            <span aria-hidden="true">▤</span>
            Clear History &amp; Memory
          </button>
        </aside>
      )}

      <main className="main-panel flex min-w-0 flex-1 flex-col">
        <header className="topbar">
          {!sidebarOpen && (
            <button
              aria-label="Open sidebar"
              className="sidebar-toggle"
              onClick={() => setSidebarOpen(true)}
              type="button"
            >
              ☰
            </button>
          )}
          <div className="topbar-label">ENTERPRISE RAG</div>
        </header>

        <section
          aria-label="Assistant conversation"
          className="chat-area mx-auto flex min-h-[calc(100vh-64px)] w-full max-w-[1060px] flex-1 flex-col px-10 py-7"
        >
          <div className="chat-content">
            <h1>🤖 Enterprise Agentic Assistant</h1>

            <div aria-live="polite" className="message-list">
              {messages.map((message) => (
                <article className={`message-row ${message.role}`} key={message.id}>
                  <div aria-hidden="true" className={`avatar ${message.role}`}>
                    {message.role === "user" ? "👤" : "🤖"}
                  </div>
                  <div className="message-body">
                    {message.role === "user" ? (
                      <div className="user-bubble">{message.content}</div>
                    ) : (
                      <>
                        <details className={`answer-status ${message.failed ? "failed" : message.blocked ? "blocked" : ""}`}>
                          <summary>
                            <span aria-hidden="true">
                              {message.failed ? "⚠️" : message.blocked ? "🛡️" : "✅"}
                            </span>
                            {message.failed
                              ? "Request failed"
                              : message.blocked
                                ? "Blocked by guardrails"
                                : "Answer Synthesized"}
                          </summary>
                          {message.steps && message.steps.length > 0 && (
                            <ul>
                              {message.steps.map((step, index) => (
                                <li key={`${message.id}-step-${index}`}>{step}</li>
                              ))}
                            </ul>
                          )}
                        </details>
                        <div className={`assistant-answer ${message.failed ? "error-answer" : ""}`}>
                          <ReactMarkdown>{message.content}</ReactMarkdown>
                        </div>
                        {message.sources && message.sources.length > 0 && (
                          <details className="sources">
                            <summary>📄 View Retrieved Context ({message.sources.length} chunks)</summary>
                            <div className="source-list">
                              {message.sources.map((source, index) => {
                                const item = typeof source === "string" ? { content: source } : source;
                                const sourceName = item.source || "Retrieved context";
                                const content = item.content || "";
                                const score =
                                  typeof item.score === "number" && Number.isFinite(item.score)
                                    ? item.score.toFixed(3)
                                    : null;
                                return (
                                  <details className="source-item" key={`${message.id}-source-${index}`}>
                                    <summary>
                                      <span>Chunk {index + 1}: {sourceName}</span>
                                      {score && <span className="source-score">{score}</span>}
                                    </summary>
                                    <div className="source-content">
                                      <ReactMarkdown>{content}</ReactMarkdown>
                                    </div>
                                  </details>
                                );
                              })}
                            </div>
                          </details>
                        )}
                      </>
                    )}
                  </div>
                </article>
              ))}

              {isLoading && (
                <div className="message-row assistant" role="status">
                  <div aria-hidden="true" className="avatar assistant">🤖</div>
                  <div className="loading-card">
                    <span className="spinner" />
                    Agent is thinking...
                  </div>
                </div>
              )}
            </div>
            <div ref={bottomRef} />
          </div>

          <form className="composer" onSubmit={submitQuestion}>
            <label className="sr-only" htmlFor="question">Ask about your documentation</label>
            <textarea
              autoComplete="off"
              id="question"
              onChange={(event) => setQuestion(event.target.value)}
              onKeyDown={handleInputKeyDown}
              placeholder="Ask about your documentation..."
              rows={1}
              value={question}
            />
            <button aria-label="Send message" disabled={!question.trim() || isLoading || !threadId} type="submit">
              ↑
            </button>
            <p className="composer-hint">Enter to send · Shift + Enter for a new line</p>
          </form>
        </section>
      </main>
    </div>
  );
}
