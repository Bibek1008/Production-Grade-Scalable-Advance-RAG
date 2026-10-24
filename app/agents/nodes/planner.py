import time

import logfire

from app.agents.state import AgentState
from app.gateway import get_portkey_llm
from app.logging import log_query_stage

llm = get_portkey_llm()


def planner_node(state: AgentState):
    """
    The Planner determines if a search is needed based on the ENTIRE conversation.
    """
    # Get the conversation history (excluding the latest message)
    history = ""
    for msg in state["messages"][:-1]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history += f"{role}: {msg['content']}\n"

    user_message = state["messages"][-1]["content"] if state["messages"] else ""

    prompt = f"""
    You are an intelligent Assistant Planner.
    Analyze the conversation history and the latest user message.

    CONVERSATION HISTORY:
    {history}

    LATEST MESSAGE:
    "{user_message}"

    Task:
    1. If the latest message is a greeting (hi, hello) or a question that can be answered using ONLY the conversation history above (e.g., "what is my name"), respond with 'CONVERSATIONAL'.
    2. If it is a technical question about Kubernetes, Intel, or Networking that requires fresh documentation, output a refined search query.

    Output ONLY 'CONVERSATIONAL' or the search query.
    """

    with logfire.span("🧠 Planner Decision"):
        started_at = time.perf_counter()
        log_query_stage("planner", "started")
        try:
            decision = llm.invoke(prompt).content.strip()
        except Exception as error:
            log_query_stage("planner", "failed", started_at=started_at, error=error)
            raise
        log_query_stage(
            "planner",
            "completed",
            started_at=started_at,
            intent="conversational" if decision == "CONVERSATIONAL" else "technical",
        )

    if decision == "CONVERSATIONAL":
        return {
            "current_query": "CONVERSATIONAL",
            "status": "Handling conversationally (using memory)...",
            "plan": ["Intent: Conversational/Memory", "Retrieval: Skipped"],
        }

    return {
        "current_query": decision,
        "status": f"Technical research needed. Searching for: {decision}",
        "plan": ["Intent: Technical", f"Search Term: {decision}"],
    }
