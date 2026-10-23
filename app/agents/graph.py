import logfire
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from app.agents.nodes.planner import planner_node
from app.agents.nodes.responder import generate_node
from app.agents.nodes.retriever import retrieve_node
from app.agents.state import AgentState
from app.config import settings
from app.postgres import postgres_error_summary


def create_checkpointer() -> BaseCheckpointSaver:
    """
    Create a Postgres checkpointer when available.
    Falls back to non-durable in-memory MemorySaver when Postgres setup fails.

    Note: we run setup() through a single autocommit connection because
    LangGraph's migrations include CREATE INDEX CONCURRENTLY, which Neon
    rejects when run inside a transaction (the default ConnectionPool mode).
    """
    pool = None
    try:
        from langgraph.checkpoint.postgres import PostgresSaver
        from psycopg_pool import ConnectionPool

        pool = ConnectionPool(
            conninfo=settings.postgres_uri,
            kwargs={"connect_timeout": 5},
            min_size=1,
            max_size=2,
            open=False,
            timeout=10,
            num_workers=1,
            check=ConnectionPool.check_connection,
            max_idle=240,
        )
        pool.open(wait=True, timeout=10)

        # Run migrations on a separate autocommit connection so that
        # CREATE INDEX CONCURRENTLY succeeds on Neon.
        with PostgresSaver.from_conn_string(settings.postgres_uri) as setup_saver:
            setup_saver.setup()

        checkpointer = PostgresSaver(pool)
        logfire.info("🗄️ Postgres checkpointer configured.")
        return checkpointer
    except Exception as error:
        if pool is not None:
            try:
                pool.close()
            except Exception as close_error:
                logfire.warning(f"Postgres pool cleanup failed: {postgres_error_summary(close_error)}")
        logfire.warning(
            f"Postgres checkpointer unavailable ({postgres_error_summary(error)}); using MemorySaver. "
            "Checkpoint data is in memory and will be lost on restart."
        )
        return MemorySaver()


def close_checkpointer(checkpointer: BaseCheckpointSaver) -> None:
    """Close the connection pool owned by a PostgresSaver, if present."""
    connection = getattr(checkpointer, "conn", None)
    close = getattr(connection, "close", None)
    if callable(close):
        close()


def is_postgres_checkpointer(checkpointer: BaseCheckpointSaver) -> bool:
    """Return whether the active saver is a PostgreSQL-backed checkpointer."""
    from langgraph.checkpoint.postgres import PostgresSaver

    return isinstance(checkpointer, PostgresSaver)


def build_graph(checkpointer: BaseCheckpointSaver | None = None) -> StateGraph:
    """
    Build and compile the LangGraph RAG agent.

    Args:
        checkpointer: Optional checkpointer. If None, a Postgres-backed
            checkpointer is created. Pass a MemorySaver in tests.
    """
    if checkpointer is None:
        checkpointer = create_checkpointer()

    # 1. Initialize the State Graph
    workflow = StateGraph(AgentState)

    # 2. Define the Nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("retriever", retrieve_node)
    workflow.add_node("responder", generate_node)

    # 3. Define the Edges & Routing Logic
    def route_planner(state: AgentState):
        """
        Routes the workflow based on the planner's decision.
        """
        if state["current_query"] == "CONVERSATIONAL":
            return "responder"
        return "retriever"

    workflow.set_entry_point("planner")

    # Conditional Edge: Planner -> Router -> (Retriever OR Responder)
    workflow.add_conditional_edges("planner", route_planner, {"retriever": "retriever", "responder": "responder"})

    workflow.add_edge("retriever", "responder")
    workflow.add_edge("responder", END)

    # 4. Compile the Graph with Memory
    return workflow.compile(checkpointer=checkpointer)


# Production code should call build_graph() explicitly (see app.main.startup_event).
# The old module-level `rag_agent` has been removed to avoid constructing two
# checkpointers and to make dependency injection/testability cleaner.
