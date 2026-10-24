"""Regression tests for the psycopg Python and binary driver pairing."""

import psycopg.waiting


def test_psycopg_c_wait_accepts_timeout_keyword():
    def completed_connection():
        if False:
            yield

    assert psycopg.waiting.wait_c(completed_connection(), 0, timeout=0) is None
