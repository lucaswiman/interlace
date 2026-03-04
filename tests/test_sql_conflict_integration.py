"""Integration tests for SQL-level conflict detection with real databases.

These tests verify that the SQL cursor patching correctly detects table-level
(and row-level) conflicts when used with real database drivers.

Uses sqlite3 (in-memory) — no external dependencies required.

The tests operate at the _intercept_execute level by:
1. Setting up sqlite3 with patch_sql()
2. Creating threads that execute SQL against real in-memory databases
3. Checking that the I/O reporter receives the correct resource IDs

This verifies that the SQL conflict detection correctly identifies:
- Threads touching different tables as independent (no conflict)
- Threads writing to the same table as conflicting
- Threads reading the same table as independent (reads don't conflict)
- A thread reading and another writing the same table as conflicting
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Generator
from typing import Any

import pytest

import frontrun._sql_cursor as sql_cursor_mod
from frontrun._io_detection import _io_tls, set_io_reporter
from frontrun._sql_cursor import (
    _ORIGINAL_METHODS,
    _PATCHES,
    _suppress_tids,
    patch_sql,
    unpatch_sql,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class IOLog:
    """Collects IO events reported to the reporter callback.

    Thread-safe: all access to ``events`` is protected by ``_lock``.
    """

    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []
        self._lock = threading.Lock()

    def __call__(self, resource_id: str, kind: str) -> None:
        with self._lock:
            self.events.append((resource_id, kind))

    def clear(self) -> None:
        with self._lock:
            self.events.clear()

    @property
    def resource_ids(self) -> list[str]:
        with self._lock:
            return [r for r, _ in self.events]

    @property
    def kinds(self) -> list[str]:
        with self._lock:
            return [k for _, k in self.events]

    def events_for_table(self, table: str) -> list[tuple[str, str]]:
        prefix = f"sql:{table}"
        with self._lock:
            return [(r, k) for r, k in self.events if r == prefix or r.startswith(f"{prefix}:")]

    def has_write_for(self, table: str) -> bool:
        return any(k == "write" for _, k in self.events_for_table(table))

    def has_read_for(self, table: str) -> bool:
        return any(k == "read" for _, k in self.events_for_table(table))

    def tables_accessed(self) -> set[str]:
        with self._lock:
            return {r.split(":", 1)[1].split(":")[0] for r, _ in self.events if r.startswith("sql:")}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _cleanup_sql_patch() -> Generator[None, None, None]:
    """Ensure SQL patching is cleaned up between tests."""
    yield
    unpatch_sql()
    _ORIGINAL_METHODS.clear()
    _PATCHES.clear()
    _suppress_tids.clear()
    sql_cursor_mod._sql_patched = False
    set_io_reporter(None)
    if hasattr(_io_tls, "_sql_suppress"):
        _io_tls._sql_suppress = False


def _make_db_with_tables(*table_names: str) -> sqlite3.Connection:
    """Create a patched in-memory SQLite database with the given tables.

    Each table is a simple ``(id INTEGER PRIMARY KEY, value INTEGER)`` schema.
    Must be called after ``patch_sql()`` so the connection is traced.
    """
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    # Use the raw cursor execute to avoid polluting event logs during setup.
    orig_execute = sqlite3.Cursor.execute
    cur = conn.cursor()
    for name in table_names:
        orig_execute(cur, f"CREATE TABLE {name} (id INTEGER PRIMARY KEY, value INTEGER)")
        orig_execute(cur, f"INSERT INTO {name} VALUES (1, 0)")
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# 1. Different tables are independent
# ---------------------------------------------------------------------------


class TestDifferentTablesIndependent:
    """Two threads touching different tables produce non-overlapping resource IDs.

    This verifies that the SQL patching correctly scopes conflict detection to
    the table level — threads that never touch the same table are independent.
    """

    def test_insert_into_different_tables_reports_different_resources(self) -> None:
        """Each thread inserts into its own table; resource IDs must not overlap."""
        log = IOLog()
        set_io_reporter(log)
        patch_sql()

        conn_a = sqlite3.connect(":memory:", check_same_thread=False)
        conn_a.execute("CREATE TABLE accounts (id INTEGER, balance INTEGER)")

        conn_b = sqlite3.connect(":memory:", check_same_thread=False)
        conn_b.execute("CREATE TABLE products (id INTEGER, stock INTEGER)")

        log.clear()

        errors: list[Exception] = []
        lock = threading.Lock()

        def thread_a() -> None:
            thread_log = IOLog()
            set_io_reporter(thread_log)
            try:
                conn_a.execute("INSERT INTO accounts VALUES (1, 100)")
            except Exception as exc:
                with lock:
                    errors.append(exc)
            finally:
                with lock:
                    log.events.extend(thread_log.events)

        def thread_b() -> None:
            thread_log = IOLog()
            set_io_reporter(thread_log)
            try:
                conn_b.execute("INSERT INTO products VALUES (1, 50)")
            except Exception as exc:
                with lock:
                    errors.append(exc)
            finally:
                with lock:
                    log.events.extend(thread_log.events)

        t1 = threading.Thread(target=thread_a)
        t2 = threading.Thread(target=thread_b)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        conn_a.close()
        conn_b.close()

        assert errors == [], f"Thread errors: {errors}"

        # Thread A touched "accounts", Thread B touched "products" — no overlap
        accounts_events = log.events_for_table("accounts")
        products_events = log.events_for_table("products")

        assert len(accounts_events) >= 1, "accounts table should have been reported"
        assert len(products_events) >= 1, "products table should have been reported"
        assert log.has_write_for("accounts"), "INSERT into accounts should be a write"
        assert log.has_write_for("products"), "INSERT into products should be a write"

    def test_different_tables_have_no_shared_resource_ids(self) -> None:
        """Resource IDs for separate tables never intersect."""
        patch_sql()

        shared_results: dict[str, list[tuple[str, str]]] = {"a": [], "b": []}
        lock = threading.Lock()

        conn_a = sqlite3.connect(":memory:", check_same_thread=False)
        conn_a.execute("CREATE TABLE orders (id INTEGER, total INTEGER)")

        conn_b = sqlite3.connect(":memory:", check_same_thread=False)
        conn_b.execute("CREATE TABLE inventory (id INTEGER, qty INTEGER)")

        def thread_a() -> None:
            thread_log = IOLog()
            set_io_reporter(thread_log)
            conn_a.execute("SELECT * FROM orders")
            conn_a.execute("INSERT INTO orders VALUES (2, 200)")
            with lock:
                shared_results["a"].extend(thread_log.events)

        def thread_b() -> None:
            thread_log = IOLog()
            set_io_reporter(thread_log)
            conn_b.execute("SELECT * FROM inventory")
            conn_b.execute("UPDATE inventory SET qty = 10 WHERE id = 1")
            with lock:
                shared_results["b"].extend(thread_log.events)

        t1 = threading.Thread(target=thread_a)
        t2 = threading.Thread(target=thread_b)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        conn_a.close()
        conn_b.close()

        resources_a = {r for r, _ in shared_results["a"]}
        resources_b = {r for r, _ in shared_results["b"]}

        # No shared resources — these threads are independent
        assert resources_a.isdisjoint(resources_b), (
            f"Expected no shared resource IDs for different tables, but got overlap: {resources_a & resources_b}"
        )

    def test_concurrent_inserts_different_tables_both_succeed(self) -> None:
        """Real SQL executes correctly in both threads with no interference."""
        patch_sql()

        conn_a = sqlite3.connect(":memory:", check_same_thread=False)
        conn_a.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")

        conn_b = sqlite3.connect(":memory:", check_same_thread=False)
        conn_b.execute("CREATE TABLE sessions (id INTEGER PRIMARY KEY, token TEXT)")

        errors: list[Exception] = []
        lock = threading.Lock()

        def thread_a() -> None:
            log = IOLog()
            set_io_reporter(log)
            try:
                conn_a.execute("INSERT INTO users VALUES (1, 'Alice')")
                conn_a.execute("INSERT INTO users VALUES (2, 'Bob')")
                conn_a.commit()
            except Exception as exc:
                with lock:
                    errors.append(exc)

        def thread_b() -> None:
            log = IOLog()
            set_io_reporter(log)
            try:
                conn_b.execute("INSERT INTO sessions VALUES (1, 'abc123')")
                conn_b.execute("INSERT INTO sessions VALUES (2, 'def456')")
                conn_b.commit()
            except Exception as exc:
                with lock:
                    errors.append(exc)

        t1 = threading.Thread(target=thread_a)
        t2 = threading.Thread(target=thread_b)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert errors == [], f"Thread errors: {errors}"

        # Verify actual data was written correctly in both connections
        cur_a = conn_a.cursor()
        cur_a.execute("SELECT COUNT(*) FROM users")
        assert cur_a.fetchone()[0] == 2  # type: ignore[index]

        cur_b = conn_b.cursor()
        cur_b.execute("SELECT COUNT(*) FROM sessions")
        assert cur_b.fetchone()[0] == 2  # type: ignore[index]

        conn_a.close()
        conn_b.close()


# ---------------------------------------------------------------------------
# 2. Same table write-write conflict
# ---------------------------------------------------------------------------


class TestSameTableWriteWriteConflict:
    """Two threads updating the same table produce the same resource ID.

    This verifies that write-write access on the same table is detected as
    a conflict — both threads report the same ``sql:<table>`` resource ID
    with kind ``"write"``.
    """

    def test_two_threads_insert_same_table_both_report_write(self) -> None:
        """Both threads INSERT into the same table → both report sql:items write."""
        patch_sql()

        conn = sqlite3.connect(":memory:", check_same_thread=False)
        conn.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT)")

        results: dict[str, list[tuple[str, str]]] = {"a": [], "b": []}
        lock = threading.Lock()
        errors: list[Exception] = []

        def thread_a() -> None:
            thread_log = IOLog()
            set_io_reporter(thread_log)
            try:
                conn.execute("INSERT INTO items VALUES (1, 'item_a')")
            except Exception as exc:
                with lock:
                    errors.append(exc)
            finally:
                with lock:
                    results["a"].extend(thread_log.events)

        def thread_b() -> None:
            thread_log = IOLog()
            set_io_reporter(thread_log)
            try:
                conn.execute("INSERT INTO items VALUES (2, 'item_b')")
            except Exception as exc:
                with lock:
                    errors.append(exc)
            finally:
                with lock:
                    results["b"].extend(thread_log.events)

        t1 = threading.Thread(target=thread_a)
        t2 = threading.Thread(target=thread_b)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert errors == [], f"Thread errors: {errors}"

        # Both threads should have reported a write to "items"
        writes_a = [(r, k) for r, k in results["a"] if r == "sql:items" and k == "write"]
        writes_b = [(r, k) for r, k in results["b"] if r == "sql:items" and k == "write"]

        assert len(writes_a) >= 1, "Thread A should report a write to sql:items"
        assert len(writes_b) >= 1, "Thread B should report a write to sql:items"

    def test_two_threads_update_same_table_report_shared_resource(self) -> None:
        """Both threads UPDATE the same table → shared resource ID sql:counters."""
        patch_sql()

        conn = sqlite3.connect(":memory:", check_same_thread=False)
        orig_execute = sqlite3.Cursor.execute
        cur = conn.cursor()
        orig_execute(cur, "CREATE TABLE counters (id INTEGER PRIMARY KEY, val INTEGER)")
        orig_execute(cur, "INSERT INTO counters VALUES (1, 0)")
        conn.commit()

        results: dict[str, list[tuple[str, str]]] = {"a": [], "b": []}
        lock = threading.Lock()

        def thread_a() -> None:
            thread_log = IOLog()
            set_io_reporter(thread_log)
            conn.execute("UPDATE counters SET val = val + 1 WHERE id = 1")
            with lock:
                results["a"].extend(thread_log.events)

        def thread_b() -> None:
            thread_log = IOLog()
            set_io_reporter(thread_log)
            conn.execute("UPDATE counters SET val = val + 10 WHERE id = 1")
            with lock:
                results["b"].extend(thread_log.events)

        t1 = threading.Thread(target=thread_a)
        t2 = threading.Thread(target=thread_b)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        resources_a = {r for r, _ in results["a"]}
        resources_b = {r for r, _ in results["b"]}

        # Both threads touched the same table → same resource IDs
        assert "sql:counters" in resources_a, f"Thread A resources: {resources_a}"
        assert "sql:counters" in resources_b, f"Thread B resources: {resources_b}"

        # UPDATE reports both read (WHERE clause) and write (SET clause)
        assert any(k == "write" for r, k in results["a"] if r == "sql:counters"), (
            "Thread A UPDATE should report a write to sql:counters"
        )
        assert any(k == "write" for r, k in results["b"] if r == "sql:counters"), (
            "Thread B UPDATE should report a write to sql:counters"
        )

    def test_write_write_conflict_detected_same_resource_id(self) -> None:
        """Two writes to the same table produce identical resource IDs — a conflict."""
        patch_sql()

        conn = sqlite3.connect(":memory:", check_same_thread=False)
        conn.execute("CREATE TABLE events (id INTEGER PRIMARY KEY, data TEXT)")

        all_writes: list[str] = []
        lock = threading.Lock()

        def thread_fn(row_id: int) -> None:
            thread_log = IOLog()
            set_io_reporter(thread_log)
            conn.execute("INSERT INTO events VALUES (?, 'payload')", (row_id,))
            with lock:
                all_writes.extend(r for r, k in thread_log.events if k == "write")

        threads = [threading.Thread(target=thread_fn, args=(i,)) for i in range(1, 4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All writes are to the same resource — conflicting
        assert all(r == "sql:events" for r in all_writes), (
            f"Expected all writes to be sql:events, got: {set(all_writes)}"
        )
        assert len(all_writes) >= 3, f"Expected at least 3 write reports, got: {all_writes}"

    def test_delete_same_table_both_threads_report_write(self) -> None:
        """Both threads DELETE from the same table report write to sql:logs."""
        patch_sql()

        conn = sqlite3.connect(":memory:", check_same_thread=False)
        orig_execute = sqlite3.Cursor.execute
        cur = conn.cursor()
        orig_execute(cur, "CREATE TABLE logs (id INTEGER PRIMARY KEY, msg TEXT)")
        orig_execute(cur, "INSERT INTO logs VALUES (1, 'msg1')")
        orig_execute(cur, "INSERT INTO logs VALUES (2, 'msg2')")
        conn.commit()

        results: dict[str, list[tuple[str, str]]] = {"a": [], "b": []}
        lock = threading.Lock()

        def thread_a() -> None:
            thread_log = IOLog()
            set_io_reporter(thread_log)
            conn.execute("DELETE FROM logs WHERE id = 1")
            with lock:
                results["a"].extend(thread_log.events)

        def thread_b() -> None:
            thread_log = IOLog()
            set_io_reporter(thread_log)
            conn.execute("DELETE FROM logs WHERE id = 2")
            with lock:
                results["b"].extend(thread_log.events)

        t1 = threading.Thread(target=thread_a)
        t2 = threading.Thread(target=thread_b)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # DELETE reports both read (WHERE) and write (the deletion)
        writes_a = [r for r, k in results["a"] if k == "write" and r == "sql:logs"]
        writes_b = [r for r, k in results["b"] if k == "write" and r == "sql:logs"]

        assert len(writes_a) >= 1, "Thread A DELETE should report write to sql:logs"
        assert len(writes_b) >= 1, "Thread B DELETE should report write to sql:logs"


# ---------------------------------------------------------------------------
# 3. Same table read-read independent
# ---------------------------------------------------------------------------


class TestSameTableReadReadIndependent:
    """Two threads SELECT-ing from the same table both report reads.

    Reads don't conflict with each other — both threads observe the same
    resource ID with kind ``"read"``. A DPOR engine would treat this as
    independent and explore only one interleaving.
    """

    def test_two_selects_same_table_both_report_read(self) -> None:
        """Both threads SELECT from the same table → both report sql:products read."""
        patch_sql()

        conn = sqlite3.connect(":memory:", check_same_thread=False)
        conn.execute("CREATE TABLE products (id INTEGER, name TEXT, price REAL)")
        conn.execute("INSERT INTO products VALUES (1, 'Widget', 9.99)")
        conn.execute("INSERT INTO products VALUES (2, 'Gadget', 19.99)")
        conn.commit()

        results: dict[str, list[tuple[str, str]]] = {"a": [], "b": []}
        lock = threading.Lock()
        rows: dict[str, Any] = {}

        def thread_a() -> None:
            thread_log = IOLog()
            set_io_reporter(thread_log)
            cur = conn.cursor()
            cur.execute("SELECT * FROM products WHERE id = 1")
            rows["a"] = cur.fetchone()
            with lock:
                results["a"].extend(thread_log.events)

        def thread_b() -> None:
            thread_log = IOLog()
            set_io_reporter(thread_log)
            cur = conn.cursor()
            cur.execute("SELECT * FROM products WHERE id = 2")
            rows["b"] = cur.fetchone()
            with lock:
                results["b"].extend(thread_log.events)

        t1 = threading.Thread(target=thread_a)
        t2 = threading.Thread(target=thread_b)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Both threads report reads from products — reads don't conflict
        reads_a = [(r, k) for r, k in results["a"] if r == "sql:products" and k == "read"]
        reads_b = [(r, k) for r, k in results["b"] if r == "sql:products" and k == "read"]

        assert len(reads_a) >= 1, f"Thread A should report a read from sql:products, got {results['a']}"
        assert len(reads_b) >= 1, f"Thread B should report a read from sql:products, got {results['b']}"

        # No writes reported — pure SELECTs
        writes_a = [k for _, k in results["a"] if k == "write"]
        writes_b = [k for _, k in results["b"] if k == "write"]
        assert writes_a == [], f"Thread A SELECT should report no writes, got {writes_a}"
        assert writes_b == [], f"Thread B SELECT should report no writes, got {writes_b}"

    def test_concurrent_selects_read_correct_data(self) -> None:
        """Read-only threads see correct data despite concurrent access."""
        patch_sql()

        conn = sqlite3.connect(":memory:", check_same_thread=False)
        conn.execute("CREATE TABLE catalog (id INTEGER PRIMARY KEY, item TEXT)")
        conn.execute("INSERT INTO catalog VALUES (1, 'alpha')")
        conn.execute("INSERT INTO catalog VALUES (2, 'beta')")
        conn.execute("INSERT INTO catalog VALUES (3, 'gamma')")
        conn.commit()

        results: dict[int, Any] = {}
        lock = threading.Lock()
        errors: list[Exception] = []

        def reader(row_id: int) -> None:
            thread_log = IOLog()
            set_io_reporter(thread_log)
            try:
                cur = conn.cursor()
                cur.execute("SELECT item FROM catalog WHERE id = ?", (row_id,))
                row = cur.fetchone()
                with lock:
                    results[row_id] = row[0] if row else None  # type: ignore[index]
            except Exception as exc:
                with lock:
                    errors.append(exc)

        threads = [threading.Thread(target=reader, args=(i,)) for i in range(1, 4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"
        assert results[1] == "alpha"
        assert results[2] == "beta"
        assert results[3] == "gamma"

    def test_multiple_threads_select_same_table_all_report_read(self) -> None:
        """Many concurrent readers all report read access — same resource, no conflict."""
        patch_sql()

        conn = sqlite3.connect(":memory:", check_same_thread=False)
        conn.execute("CREATE TABLE stats (id INTEGER, count INTEGER)")
        conn.execute("INSERT INTO stats VALUES (1, 42)")
        conn.commit()

        all_events: list[tuple[str, str]] = []
        lock = threading.Lock()

        def reader() -> None:
            thread_log = IOLog()
            set_io_reporter(thread_log)
            cur = conn.cursor()
            cur.execute("SELECT count FROM stats WHERE id = 1")
            cur.fetchone()
            with lock:
                all_events.extend(thread_log.events)

        threads = [threading.Thread(target=reader) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        conn.close()

        # Every event should be a read from sql:stats
        stats_events = [(r, k) for r, k in all_events if r == "sql:stats"]
        assert len(stats_events) == 5, f"Expected 5 read events (one per thread), got {len(stats_events)}"
        assert all(k == "read" for _, k in stats_events), (
            f"All events should be reads, got: {[(r, k) for r, k in stats_events if k != 'read']}"
        )


# ---------------------------------------------------------------------------
# 4. Read-write conflict
# ---------------------------------------------------------------------------


class TestReadWriteConflict:
    """One thread reads, one thread writes the same table.

    This verifies that a read-write pair on the same table is detected as
    a conflict — the reader reports ``"read"`` and the writer reports
    ``"write"`` for the same ``sql:<table>`` resource ID.
    """

    def test_read_and_write_same_table_both_report_resource(self) -> None:
        """Reader and writer both report the same resource ID — they conflict."""
        patch_sql()

        conn = sqlite3.connect(":memory:", check_same_thread=False)
        conn.execute("CREATE TABLE balances (id INTEGER PRIMARY KEY, amount INTEGER)")
        conn.execute("INSERT INTO balances VALUES (1, 1000)")
        conn.commit()

        results: dict[str, list[tuple[str, str]]] = {"reader": [], "writer": []}
        lock = threading.Lock()

        def reader_thread() -> None:
            thread_log = IOLog()
            set_io_reporter(thread_log)
            cur = conn.cursor()
            cur.execute("SELECT amount FROM balances WHERE id = 1")
            cur.fetchone()
            with lock:
                results["reader"].extend(thread_log.events)

        def writer_thread() -> None:
            thread_log = IOLog()
            set_io_reporter(thread_log)
            conn.execute("UPDATE balances SET amount = 900 WHERE id = 1")
            with lock:
                results["writer"].extend(thread_log.events)

        t1 = threading.Thread(target=reader_thread)
        t2 = threading.Thread(target=writer_thread)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Reader reports read to balances
        reader_reads = [(r, k) for r, k in results["reader"] if r == "sql:balances" and k == "read"]
        assert len(reader_reads) >= 1, f"Reader should report read from sql:balances, got {results['reader']}"

        # Writer reports write (and read, from the WHERE clause) to balances
        writer_writes = [(r, k) for r, k in results["writer"] if r == "sql:balances" and k == "write"]
        assert len(writer_writes) >= 1, f"Writer should report write to sql:balances, got {results['writer']}"

    def test_read_write_conflict_shared_resource_id(self) -> None:
        """Reader and writer share the resource ID — DPOR would see a conflict."""
        patch_sql()

        conn = sqlite3.connect(":memory:", check_same_thread=False)
        conn.execute("CREATE TABLE queue (id INTEGER PRIMARY KEY, task TEXT, done INTEGER)")
        conn.execute("INSERT INTO queue VALUES (1, 'process_a', 0)")
        conn.execute("INSERT INTO queue VALUES (2, 'process_b', 0)")
        conn.commit()

        all_events: list[tuple[str, str]] = []
        lock = threading.Lock()

        def reader_thread() -> None:
            thread_log = IOLog()
            set_io_reporter(thread_log)
            cur = conn.cursor()
            cur.execute("SELECT * FROM queue WHERE done = 0")
            cur.fetchall()
            with lock:
                all_events.extend(thread_log.events)

        def writer_thread() -> None:
            thread_log = IOLog()
            set_io_reporter(thread_log)
            conn.execute("UPDATE queue SET done = 1 WHERE id = 1")
            with lock:
                all_events.extend(thread_log.events)

        t1 = threading.Thread(target=reader_thread)
        t2 = threading.Thread(target=writer_thread)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        conn.close()

        queue_events = [(r, k) for r, k in all_events if r == "sql:queue"]
        resource_ids = {r for r, _ in queue_events}
        kinds = {k for _, k in queue_events}

        # Both threads touched the same resource
        assert "sql:queue" in resource_ids, f"Expected sql:queue in resources, got {resource_ids}"

        # Read-write conflict: both kinds present
        assert "read" in kinds, f"Expected at least one read, got kinds {kinds}"
        assert "write" in kinds, f"Expected at least one write, got kinds {kinds}"

    def test_insert_while_select_running_reports_conflict_resources(self) -> None:
        """INSERT and SELECT on the same table both report the same table resource."""
        patch_sql()

        conn = sqlite3.connect(":memory:", check_same_thread=False)
        conn.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY, body TEXT)")
        conn.execute("INSERT INTO messages VALUES (1, 'hello')")
        conn.commit()

        barrier = threading.Barrier(2)
        results: dict[str, list[tuple[str, str]]] = {"select": [], "insert": []}
        lock = threading.Lock()

        def select_thread() -> None:
            thread_log = IOLog()
            set_io_reporter(thread_log)
            # Give both threads a chance to start before executing
            barrier.wait()
            cur = conn.cursor()
            cur.execute("SELECT * FROM messages")
            cur.fetchall()
            with lock:
                results["select"].extend(thread_log.events)

        def insert_thread() -> None:
            thread_log = IOLog()
            set_io_reporter(thread_log)
            barrier.wait()
            conn.execute("INSERT INTO messages VALUES (2, 'world')")
            with lock:
                results["insert"].extend(thread_log.events)

        t1 = threading.Thread(target=select_thread)
        t2 = threading.Thread(target=insert_thread)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        conn.close()

        # SELECT reports a read to messages
        select_reads = [k for r, k in results["select"] if r == "sql:messages" and k == "read"]
        assert len(select_reads) >= 1, f"SELECT should report a read, got {results['select']}"

        # INSERT reports a write to messages
        insert_writes = [k for r, k in results["insert"] if r == "sql:messages" and k == "write"]
        assert len(insert_writes) >= 1, f"INSERT should report a write, got {results['insert']}"

    def test_read_write_conflict_correct_sql_execution(self) -> None:
        """Reader and writer execute correctly — no data corruption from patching."""
        patch_sql()

        conn = sqlite3.connect(":memory:", check_same_thread=False)
        conn.execute("CREATE TABLE config (key TEXT PRIMARY KEY, value INTEGER)")
        conn.execute("INSERT INTO config VALUES ('threshold', 100)")
        conn.commit()

        read_value: list[Any] = []
        errors: list[Exception] = []
        lock = threading.Lock()

        def reader_thread() -> None:
            thread_log = IOLog()
            set_io_reporter(thread_log)
            try:
                cur = conn.cursor()
                cur.execute("SELECT value FROM config WHERE key = ?", ("threshold",))
                row = cur.fetchone()
                with lock:
                    read_value.append(row[0] if row else None)  # type: ignore[index]
            except Exception as exc:
                with lock:
                    errors.append(exc)

        def writer_thread() -> None:
            thread_log = IOLog()
            set_io_reporter(thread_log)
            try:
                conn.execute("UPDATE config SET value = 200 WHERE key = 'threshold'")
                conn.commit()
            except Exception as exc:
                with lock:
                    errors.append(exc)

        t1 = threading.Thread(target=reader_thread)
        t2 = threading.Thread(target=writer_thread)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        conn.close()

        assert errors == [], f"Thread errors: {errors}"
        # Reader saw either the old value (100) or the new value (200)
        assert len(read_value) == 1
        assert read_value[0] in (100, 200), f"Unexpected read value: {read_value[0]}"


# ---------------------------------------------------------------------------
# 5. Multi-table and cross-thread resource ID correctness
# ---------------------------------------------------------------------------


class TestMultiTableResourceIds:
    """Verify resource IDs are correct for multi-table queries and joins."""

    def test_join_query_reports_all_touched_tables(self) -> None:
        """A JOIN query reports reads from all joined tables."""
        patch_sql()

        log = IOLog()
        set_io_reporter(log)

        conn = sqlite3.connect(":memory:", check_same_thread=False)
        conn.execute("CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT)")
        conn.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER, amount REAL)")
        conn.execute("INSERT INTO customers VALUES (1, 'Alice')")
        conn.execute("INSERT INTO orders VALUES (1, 1, 50.0)")
        log.clear()

        conn.execute("SELECT c.name, o.amount FROM customers c JOIN orders o ON c.id = o.customer_id")

        cust_events = log.events_for_table("customers")
        order_events = log.events_for_table("orders")

        assert len(cust_events) >= 1, "JOIN should report access to customers table"
        assert len(order_events) >= 1, "JOIN should report access to orders table"
        assert all(k == "read" for _, k in cust_events), "customers read should be kind=read"
        assert all(k == "read" for _, k in order_events), "orders read should be kind=read"

        conn.close()

    def test_threads_on_partially_overlapping_tables_separate_correctly(self) -> None:
        """Thread A touches (users, orders), Thread B touches (orders, shipments).

        The overlap is on 'orders' — that is the conflict point.
        """
        patch_sql()

        conn_a = sqlite3.connect(":memory:", check_same_thread=False)
        conn_a.execute("CREATE TABLE users (id INTEGER, name TEXT)")
        conn_a.execute("CREATE TABLE orders (id INTEGER, user_id INTEGER, amount REAL)")
        conn_a.execute("INSERT INTO users VALUES (1, 'Alice')")
        conn_a.execute("INSERT INTO orders VALUES (1, 1, 99.0)")

        conn_b = sqlite3.connect(":memory:", check_same_thread=False)
        conn_b.execute("CREATE TABLE orders (id INTEGER, user_id INTEGER, amount REAL)")
        conn_b.execute("CREATE TABLE shipments (id INTEGER, order_id INTEGER, status TEXT)")
        conn_b.execute("INSERT INTO orders VALUES (1, 1, 99.0)")
        conn_b.execute("INSERT INTO shipments VALUES (1, 1, 'pending')")

        results: dict[str, list[tuple[str, str]]] = {"a": [], "b": []}
        lock = threading.Lock()

        def thread_a() -> None:
            thread_log = IOLog()
            set_io_reporter(thread_log)
            cur = conn_a.cursor()
            cur.execute("SELECT u.name, o.amount FROM users u JOIN orders o ON u.id = o.user_id")
            cur.fetchall()
            with lock:
                results["a"].extend(thread_log.events)

        def thread_b() -> None:
            thread_log = IOLog()
            set_io_reporter(thread_log)
            cur = conn_b.cursor()
            cur.execute("SELECT o.amount, s.status FROM orders o JOIN shipments s ON o.id = s.order_id")
            cur.fetchall()
            with lock:
                results["b"].extend(thread_log.events)

        t1 = threading.Thread(target=thread_a)
        t2 = threading.Thread(target=thread_b)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        conn_a.close()
        conn_b.close()

        tables_a = {r.split(":", 1)[1] for r, _ in results["a"] if r.startswith("sql:")}
        tables_b = {r.split(":", 1)[1] for r, _ in results["b"] if r.startswith("sql:")}

        # Thread A touched users and orders
        assert "users" in tables_a, f"Thread A should touch users, got {tables_a}"
        assert "orders" in tables_a, f"Thread A should touch orders, got {tables_a}"

        # Thread B touched orders and shipments
        assert "orders" in tables_b, f"Thread B should touch orders, got {tables_b}"
        assert "shipments" in tables_b, f"Thread B should touch shipments, got {tables_b}"

        # The shared table "orders" is the conflict point
        assert "orders" in (tables_a & tables_b), f"orders should appear in both threads: A={tables_a}, B={tables_b}"

        # "users" is exclusive to Thread A, "shipments" to Thread B
        assert "users" not in tables_b, f"users should only be in Thread A, but also in B: {tables_b}"
        assert "shipments" not in tables_a, f"shipments should only be in Thread B, but also in A: {tables_a}"

    def test_parameterized_writes_report_same_resource(self) -> None:
        """Parameterized INSERT statements report the same table resource regardless of row values."""
        patch_sql()

        conn = sqlite3.connect(":memory:", check_same_thread=False)
        conn.execute("CREATE TABLE audit (id INTEGER PRIMARY KEY, action TEXT, ts INTEGER)")

        all_write_resources: list[str] = []
        lock = threading.Lock()

        def writer(row_id: int, action: str) -> None:
            thread_log = IOLog()
            set_io_reporter(thread_log)
            conn.execute("INSERT INTO audit VALUES (?, ?, ?)", (row_id, action, row_id * 1000))
            with lock:
                all_write_resources.extend(r for r, k in thread_log.events if k == "write")

        threads = [threading.Thread(target=writer, args=(i, f"action_{i}")) for i in range(1, 5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        conn.close()

        # All parameterized INSERT statements report the same table resource
        assert len(all_write_resources) >= 4, f"Expected at least 4 write reports, got {len(all_write_resources)}"
        assert all(r == "sql:audit" for r in all_write_resources), (
            f"All writes should be to sql:audit, got: {set(all_write_resources)}"
        )
