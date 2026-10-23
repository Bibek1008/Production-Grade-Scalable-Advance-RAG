"""Tests for LangGraph checkpointer configuration."""

from unittest.mock import MagicMock, patch

from langgraph.checkpoint.memory import MemorySaver

from app.agents.graph import build_graph, close_checkpointer, create_checkpointer, is_postgres_checkpointer


def test_build_graph_uses_provided_checkpointer():
    """build_graph should compile using the supplied checkpointer."""
    saver = MemorySaver()
    agent = build_graph(checkpointer=saver)
    assert agent is not None
    # The compiled graph should expose checkpointer
    assert agent.checkpointer is saver


def test_create_checkpointer_prefers_postgres_when_available():
    """When Postgres deps are available, a PostgresSaver is built."""
    mock_pg = MagicMock(name="langgraph.checkpoint.postgres")
    mock_pg.PostgresSaver = MagicMock(name="PostgresSaver")

    mock_pool = MagicMock(name="psycopg_pool")
    mock_pool.ConnectionPool = MagicMock(name="ConnectionPool")

    modules = {
        "langgraph.checkpoint.postgres": mock_pg,
        "psycopg_pool": mock_pool,
    }

    with (
        patch.dict("sys.modules", modules),
        patch("app.agents.graph.settings.NEON_DB_URL", "postgresql://test"),
        patch("app.agents.graph.logfire"),
    ):
        cp = create_checkpointer()

    mock_pool.ConnectionPool.assert_called_once()
    mock_pg.PostgresSaver.assert_called_once_with(mock_pool.ConnectionPool.return_value)
    assert cp is mock_pg.PostgresSaver.return_value


def test_create_checkpointer_uses_small_pool_and_waits_for_initial_connection():
    mock_pg = MagicMock(name="langgraph.checkpoint.postgres")
    mock_pg.PostgresSaver = MagicMock(name="PostgresSaver")
    setup_saver = MagicMock()
    mock_pg.PostgresSaver.from_conn_string.return_value.__enter__.return_value = setup_saver

    mock_pool = MagicMock(name="psycopg_pool")
    mock_pool.ConnectionPool = MagicMock(name="ConnectionPool")
    modules = {
        "langgraph.checkpoint.postgres": mock_pg,
        "psycopg_pool": mock_pool,
    }

    with (
        patch.dict("sys.modules", modules),
        patch("app.agents.graph.settings.NEON_DB_URL", "postgresql://test"),
        patch("app.agents.graph.logfire"),
    ):
        cp = create_checkpointer()

    options = mock_pool.ConnectionPool.call_args.kwargs
    assert options["min_size"] == 1
    assert options["max_size"] == 2
    assert options["num_workers"] == 1
    assert options["kwargs"] == {"connect_timeout": 5}
    mock_pool.ConnectionPool.return_value.open.assert_called_once_with(wait=True, timeout=10)
    setup_saver.setup.assert_called_once()
    mock_pool.ConnectionPool.return_value.getconn.assert_not_called()
    assert cp is mock_pg.PostgresSaver.return_value


def test_create_checkpointer_closes_pool_if_setup_fails():
    mock_pg = MagicMock(name="langgraph.checkpoint.postgres")
    mock_pg.PostgresSaver = MagicMock(name="PostgresSaver")
    mock_pg.PostgresSaver.from_conn_string.return_value.__enter__.side_effect = RuntimeError("setup failed")
    mock_pool = MagicMock(name="psycopg_pool")
    mock_pool.ConnectionPool = MagicMock(name="ConnectionPool")
    modules = {
        "langgraph.checkpoint.postgres": mock_pg,
        "psycopg_pool": mock_pool,
    }

    with (
        patch.dict("sys.modules", modules),
        patch("app.agents.graph.settings.NEON_DB_URL", "postgresql://test"),
        patch("app.agents.graph.logfire"),
    ):
        cp = create_checkpointer()

    mock_pool.ConnectionPool.return_value.close.assert_called_once_with()
    assert isinstance(cp, MemorySaver)


def test_create_checkpointer_falls_back_to_memory():
    """If Postgres dependencies are missing, fall back to MemorySaver."""
    with patch.dict("sys.modules", {"langgraph.checkpoint.postgres": None}), patch("app.agents.graph.logfire"):
        cp = create_checkpointer()
        assert isinstance(cp, MemorySaver)


def test_close_checkpointer_closes_postgres_pool():
    saver = MagicMock()
    close_checkpointer(saver)
    saver.conn.close.assert_called_once_with()


def test_memory_saver_is_not_reported_as_postgres_checkpointer():
    assert is_postgres_checkpointer(MemorySaver()) is False


def test_postgres_saver_is_reported_as_postgres_checkpointer():
    from langgraph.checkpoint.postgres import PostgresSaver

    assert is_postgres_checkpointer(PostgresSaver(MagicMock())) is True
