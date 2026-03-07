"""Tests for the nondeterministic SQL INSERT warning mode."""

from __future__ import annotations

import sqlite3
import threading

import pytest

from frontrun._sql_cursor import (
    clear_insert_tables,
    get_insert_tables,
    patch_sql,
    record_insert_table,
    unpatch_sql,
)
from frontrun.common import NondeterministicSQLError


class TestInsertTracking:
    """Unit tests for the INSERT table tracking infrastructure."""

    def test_record_and_get(self) -> None:
        clear_insert_tables()
        record_insert_table("users")
        record_insert_table("orders")
        assert get_insert_tables() == {"users", "orders"}
        clear_insert_tables()

    def test_clear(self) -> None:
        clear_insert_tables()
        record_insert_table("users")
        clear_insert_tables()
        assert get_insert_tables() == set()

    def test_duplicate_tables(self) -> None:
        clear_insert_tables()
        record_insert_table("users")
        record_insert_table("users")
        assert get_insert_tables() == {"users"}
        clear_insert_tables()

    def test_thread_safety(self) -> None:
        clear_insert_tables()
        barrier = threading.Barrier(4)

        def add_table(name: str) -> None:
            barrier.wait()
            record_insert_table(name)

        threads = [threading.Thread(target=add_table, args=(f"t{i}",)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert get_insert_tables() == {"t0", "t1", "t2", "t3"}
        clear_insert_tables()


class TestInsertDetectionInCursor:
    """Test that INSERT statements are automatically tracked when SQL patching is active."""

    def setup_method(self) -> None:
        clear_insert_tables()
        patch_sql()

    def teardown_method(self) -> None:
        unpatch_sql()
        clear_insert_tables()

    def test_insert_tracked_via_cursor(self) -> None:
        """An INSERT through a patched cursor records the table."""
        from frontrun._io_detection import set_io_reporter

        reported: list[tuple[str, str]] = []

        def reporter(res_id: str, kind: str) -> None:
            reported.append((res_id, kind))

        set_io_reporter(reporter)
        try:
            conn = sqlite3.connect(":memory:")
            conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
            conn.execute("INSERT INTO users (id, name) VALUES (1, 'alice')")
            conn.close()
        finally:
            set_io_reporter(None)

        assert "users" in get_insert_tables()

    def test_select_not_tracked(self) -> None:
        """A SELECT does not record any insert tables."""
        from frontrun._io_detection import set_io_reporter

        def reporter(res_id: str, kind: str) -> None:
            pass

        set_io_reporter(reporter)
        try:
            conn = sqlite3.connect(":memory:")
            conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
            conn.execute("INSERT INTO users (id, name) VALUES (1, 'alice')")
            clear_insert_tables()  # Reset after setup
            conn.execute("SELECT * FROM users")
            conn.close()
        finally:
            set_io_reporter(None)

        assert get_insert_tables() == set()

    def test_update_not_tracked(self) -> None:
        """An UPDATE does not record insert tables."""
        from frontrun._io_detection import set_io_reporter

        def reporter(res_id: str, kind: str) -> None:
            pass

        set_io_reporter(reporter)
        try:
            conn = sqlite3.connect(":memory:")
            conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
            conn.execute("INSERT INTO users (id, name) VALUES (1, 'alice')")
            clear_insert_tables()  # Reset after setup
            conn.execute("UPDATE users SET name = 'bob' WHERE id = 1")
            conn.close()
        finally:
            set_io_reporter(None)

        assert get_insert_tables() == set()


class TestNondeterministicSQLError:
    """Test the exception itself."""

    def test_exception_is_importable(self) -> None:
        from frontrun.common import NondeterministicSQLError as E

        assert issubclass(E, Exception)

    def test_exception_message(self) -> None:
        err = NondeterministicSQLError("INSERT detected on: users")
        assert "users" in str(err)
