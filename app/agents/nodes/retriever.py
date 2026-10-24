import time

import logfire

from app.agents.state import AgentState
from app.logging import log_query_stage
from app.services.retrieval.qdrant_service import search_enterprise_knowledge
from app.services.retrieval.ranking_service import rerank_documents


def retrieve_node(state: AgentState):
    """
    Performs vector search and semantic reranking for technical queries.
    """
    query = state["current_query"]

    # Standard Retrieval Logic
    with logfire.span("🔍 Knowledge Retrieval"):
        started_at = time.perf_counter()
        log_query_stage("retrieval", "started")
        try:
            raw_results = search_enterprise_knowledge(query, limit=15)
            doc_contents = [doc["content"] for doc in raw_results]

            with logfire.span("⚖️ Semantic Reranking"):
                reranked_contents = rerank_documents(query, doc_contents, top_n=5)

            documents_by_content = {doc["content"]: doc for doc in raw_results}
            formatted_docs = [documents_by_content[content] for content in reranked_contents]
        except Exception as error:
            log_query_stage("retrieval", "failed", started_at=started_at, error=error)
            raise
        log_query_stage(
            "retrieval",
            "completed",
            started_at=started_at,
            candidate_count=len(raw_results),
            selected_count=len(formatted_docs),
        )

    return {
        "documents": formatted_docs,
        "status": "Found technical context.",
        "plan": state["plan"] + ["Context Retrieved"],
    }
