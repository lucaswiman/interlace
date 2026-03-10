"""Reproduction test for FRONTRUN_DEFECTS.md #1: LD_PRELOAD shared socket deadlock.

When the LD_PRELOAD library is active, two DPOR threads that each open their
own psycopg2 connection to the same PostgreSQL server (via Unix socket) are
incorrectly identified as sharing a socket.  The LD_PRELOAD library tracks
connections by socket path (resource_id), so both threads connecting to
``/var/run/postgresql/.s.PGSQL.5432`` look like the same connection.

This triggers:
  - "DPOR threads {0} and 1 share socket ..." warning
  - Deadlock in scheduler.report_and_wait() as both threads block

This test demonstrates the bug by using explore_dpor with detect_io=True
and two threads that each independently connect to PostgreSQL and run SQL.
The test expects a timeout (deadlock) or the shared-socket warning.
"""

from __future__ import annotations

import os
import warnings

import pytest

try:
    import psycopg2
except ImportError:
    pytest.skip("psycopg2 not installed", allow_module_level=True)

from frontrun.cli import require_active
from frontrun.dpor import explore_dpor

_DB_NAME = os.environ.get("FRONTRUN_TEST_DB", "frontrun_test")
_DB_URL = os.environ.get("DATABASE_URL", f"dbname={_DB_NAME}")


def _pg_available() -> bool:
    """Check if PostgreSQL is reachable."""
    try:
        conn = psycopg2.connect(_DB_URL)
        conn.close()
        return True
    except Exception:
        return False


@pytest.fixture(scope="module", autouse=True)
def _check_pg():
    if not _pg_available():
        pytest.skip(f"PostgreSQL not available at {_DB_URL}")


@pytest.fixture(autouse=True)
def _setup_table():
    """Create a simple test table, drop it after the test."""
    conn = psycopg2.connect(_DB_URL)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS defect1_counter")
    cur.execute("CREATE TABLE defect1_counter (id INT PRIMARY KEY, val INT)")
    cur.execute("INSERT INTO defect1_counter VALUES (1, 0)")
    cur.close()
    conn.close()
    yield
    conn = psycopg2.connect(_DB_URL)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS defect1_counter")
    cur.close()
    conn.close()


class TestSharedSocketDeadlock:
    """Reproduce defect #1: LD_PRELOAD + per-thread psycopg2 connections deadlock."""

    def test_two_threads_own_connections_deadlocks_with_preload(self) -> None:
        """Two DPOR threads each opening their own psycopg2 connection.

        This simulates what django_dpor does: each thread closes the shared
        connection and opens a fresh one.  With LD_PRELOAD active, both
        threads connect to the same Unix socket path, causing the preload
        bridge to think they share a connection → deadlock.

        Expected behavior: this test should pass (DPOR explores interleavings).
        Actual behavior (defect #1): deadlocks or emits shared-socket warning.
        """
        require_active("test_two_threads_own_connections_deadlocks_with_preload")

        class State:
            pass

        def thread_fn(state: State) -> None:
            # Each thread opens its own independent connection — simulating
            # Django's conn.close() + conn.ensure_connection() pattern.
            conn = psycopg2.connect(_DB_URL)
            conn.autocommit = True
            cur = conn.cursor()
            cur.execute("UPDATE defect1_counter SET val = val + 1 WHERE id = 1")
            cur.close()
            conn.close()

        def invariant(state: State) -> bool:
            conn = psycopg2.connect(_DB_URL)
            conn.autocommit = True
            cur = conn.cursor()
            cur.execute("SELECT val FROM defect1_counter WHERE id = 1")
            val = cur.fetchone()[0]
            cur.close()
            conn.close()
            return val == 2

        # Use a short timeout to detect the deadlock quickly rather than
        # hanging for the full test suite timeout.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = explore_dpor(
                setup=State,
                threads=[thread_fn, thread_fn],
                invariant=invariant,
                detect_io=True,
                timeout_per_run=5.0,
                total_timeout=15.0,
                max_executions=5,
                reproduce_on_failure=0,
            )

        # Check for the shared-socket warning — this is the signature of defect #1.
        shared_socket_warnings = [
            w for w in caught if "share socket" in str(w.message)
        ]

        # If we get here without deadlocking, check results.
        # The defect manifests as either:
        #   (a) Deadlock → timeout_per_run fires → test sees error in result
        #   (b) Shared-socket warning → incorrect conflict tracking
        #
        # If the bug is fixed, DPOR should explore >1 interleaving and the
        # invariant should hold (UPDATE is atomic in PostgreSQL).
        if shared_socket_warnings:
            pytest.fail(
                f"Defect #1 reproduced: shared-socket warning fired.\n"
                f"Warning: {shared_socket_warnings[0].message}\n"
                f"DPOR explored {result.num_explored} interleavings."
            )

        if result.num_explored <= 1:
            pytest.fail(
                f"Defect #1 likely present: DPOR explored only "
                f"{result.num_explored} interleaving(s), suggesting I/O "
                f"conflicts were not detected despite LD_PRELOAD being active."
            )

        # If we get here, the bug is fixed — DPOR correctly explored
        # interleavings with per-thread connections.
        assert result.property_holds, (
            f"Unexpected invariant violation: {result.explanation}"
        )
