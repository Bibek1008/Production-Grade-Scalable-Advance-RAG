import logfire
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_exponential

from app.agents.state import AgentState
from app.gateway import get_portkey_llm

llm = get_portkey_llm()


def generate_node(state: AgentState):
    """
    Synthesizes a response using both Documentation Context AND Conversation History.
    Uses Gemini through LangChain for answer generation.
    """
    query = state["current_query"]

    history_str = ""
    for msg in state["messages"][:-1]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_str += f"{role}: {msg['content']}\n"

    user_msg = state["messages"][-1]["content"] if state["messages"] else ""

    if query == "CONVERSATIONAL":
        logfire.info("Generating conversational response using memory.")
        prompt = f"""
        You are a friendly and helpful Enterprise AI Assistant.
        Answer the user's latest message using the CONVERSATION HISTORY below.

        CONVERSATION HISTORY:
        {history_str}

        LATEST MESSAGE:
        "{user_msg}"
        """
    else:
        logfire.info("Generating technical RAG response.")
        max_context_chars = 25000
        full_context = ""

        for doc in state["documents"]:
            content = f"Source: {doc['source']}\n{doc['content']}"
            if len(full_context) + len(content) < max_context_chars:
                full_context += content + "\n\n"
            else:
                logfire.warning("Context truncated to fit the configured Gemini request limit.")
                break

        if not full_context.strip():
            return {
                "final_answer": (
                    "I couldn't find relevant documents in the knowledge base for this question. "
                    "Please ingest the relevant documents or ask a conversational question."
                ),
                "status": "No documents retrieved.",
                "plan": state["plan"] + ["No matching documents found"],
                "messages": [
                    {
                        "role": "assistant",
                        "content": (
                            "I couldn't find relevant documents in the knowledge base for this question. "
                            "Please ingest the relevant documents or ask a conversational question."
                        ),
                    }
                ],
            }

        prompt = f"""
        You are a Senior Technical Architect.
        Answer using only the TECHNICAL CONTEXT below and cite its source filenames.
        For troubleshooting questions, turn documented expected behaviors and architecture
        into a short list of checks when they are relevant. Clearly label these as checks
        derived from the documentation, not as documented procedures. If the context lacks
        specific commands, diagnostic steps, or fixes, say so explicitly rather than
        inventing them. Do not claim that the documents contain no useful information when
        they provide relevant facts.

        TECHNICAL CONTEXT:
        {full_context}

        CONVERSATION HISTORY:
        {history_str}

        USER QUESTION:
        "{user_msg}"
        """

    with logfire.span("✍️ LLM Synthesis"):
        try:
            response = _generate_response(prompt)
            content = response.content
            if not isinstance(content, str) or not content.strip():
                raise RuntimeError("Gemini returned an empty response.")
            logfire.info("Response synthesised via the configured Portkey Gemini route.")

            return {
                "final_answer": content,
                "status": "Response generated.",
                "plan": state["plan"],
                "messages": [{"role": "assistant", "content": content}],
            }

        except Exception as e:
            logfire.error(f"LLM Generation failed after retries: {e}")
            raise


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=5),
    reraise=True,
    before_sleep=before_sleep_log(logfire, "warning"),
)
def _generate_response(prompt: str):
    """Call Gemini with retry logic for transient failures."""
    return llm.invoke(prompt)
