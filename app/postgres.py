"""PostgreSQL helpers that avoid exposing connection details in diagnostics."""


def postgres_error_summary(error: Exception) -> str:
    """Describe a PostgreSQL failure without including its message or connection URI."""
    sqlstate = getattr(error, "sqlstate", None)
    if sqlstate is None:
        sqlstate = getattr(getattr(error, "diag", None), "sqlstate", None)
    suffix = f" (SQLSTATE {sqlstate})" if sqlstate else ""
    return f"{type(error).__name__}{suffix}"
