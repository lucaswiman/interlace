"""Minimal reproduction of frontrun defect #6: DPOR cooperative scheduler
deadlocks with PostgreSQL row locks.

This is a general explore_dpor bug, NOT Django-specific. Any explore_dpor test
where two threads contend on the same PostgreSQL row lock will deadlock.

The scenario:
1. Thread A INSERTs a row (acquires PG row lock via UNIQUE constraint)
2. DPOR suspends thread A at the next scheduling point
3. DPOR resumes thread B
4. Thread B INSERTs the same row — PG blocks thread B waiting for A's lock
5. Thread B never reaches a DPOR scheduling point (blocked in kernel)
6. DPOR waits for thread B to yield — deadlock

Without lock_timeout, this test hangs forever (or until deadlock_timeout).
With lock_timeout, PG raises an error and the thread returns to DPOR.

Running:

    # This will HANG (demonstrating the bug):
    PYTHONPATH=libraries .venv/bin/frontrun pytest \
      frontrun-bugs/tests/test_defect6_scheduler_deadlock.py::test_deadlock_without_lock_timeout \
      -v --timeout=30

    # This WORKS (demonstrating the workaround):
    PYTHONPATH=libraries .venv/bin/frontrun pytest \
      frontrun-bugs/tests/test_defect6_scheduler_deadlock.py::test_workaround_with_lock_timeout \
      -v --timeout=30
"""
from __future__ import annotations

import os

import psycopg2
import pytest
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

from frontrun.dpor import explore_dpor

_DB_NAME = os.environ.get("FRONTRUN_TEST_DB", "frontrun_test")
_DSN = f"dbname={_DB_NAME}"


@pytest.fixture(scope="module")
def _pg_available():
    """Create test table in Postgres."""
    try:
        conn = psycopg2.connect(_DSN)
    except Exception:
        pytest.skip(f"Postgres not available at {_DB_NAME}")

    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS defect6_test")
        cur.execute("""
            CREATE TABLE defect6_test (
                id TEXT PRIMARY KEY,
                value TEXT
            )
        """)
    conn.close()
    yield


class _State:
    """Shared state: two threads race to INSERT the same row."""

    def __init__(self) -> None:
        conn = psycopg2.connect(_DSN)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        with conn.cursor() as cur:
            cur.execute("DELETE FROM defect6_test")
        conn.close()
        self.results: list[str | None] = [None, None]


def _make_thread_fn(idx: int, use_lock_timeout: bool = False):
    """Each thread opens its own connection and tries to INSERT the same row.

    This is the minimal TOCTOU: SELECT to check existence, then INSERT.
    """

    def _thread_fn(state: _State) -> None:
        conn = psycopg2.connect(_DSN)
        try:
            conn.autocommit = False
            with conn.cursor() as cur:
                if use_lock_timeout:
                    cur.execute("SET lock_timeout = '2s'")

                # CHECK: does the row exist?
                cur.execute(
                    "SELECT id FROM defect6_test WHERE id = %s", ("row1",)
                )
                row = cur.fetchone()
                if row is not None:
                    state.results[idx] = "already_exists"
                    conn.rollback()
                    return

                # ACT: insert the row
                try:
                    cur.execute(
                        "INSERT INTO defect6_test (id, value) VALUES (%s, %s)",
                        ("row1", f"thread_{idx}"),
                    )
                    conn.commit()
                    state.results[idx] = "inserted"
                except Exception as exc:
                    conn.rollback()
                    state.results[idx] = f"error: {type(exc).__name__}: {exc}"
        finally:
            conn.close()

    return _thread_fn


def _invariant(state: _State) -> bool:
    """Invariant: no errors should occur if the TOCTOU is properly serialized.

    When the race is triggered, one thread gets a PG error (IntegrityError
    or LockNotAvailable with lock_timeout).
    """
    r0, r1 = state.results
    has_error = (r0 is not None and r0.startswith("error")) or (
        r1 is not None and r1.startswith("error")
    )
    return not has_error


def test_deadlock_without_lock_timeout(_pg_available) -> None:
    """Demonstrates defect #6: this test HANGS because DPOR and PG deadlock.

    DPOR's cooperative scheduler suspends thread A (which holds a PG row lock),
    then resumes thread B which tries to acquire the same lock. PG blocks
    thread B in the kernel, and DPOR waits for thread B to yield.

    Expected: this test should time out (proving the deadlock).
    """
    result = explore_dpor(
        setup=_State,
        threads=[_make_thread_fn(0, use_lock_timeout=False),
                 _make_thread_fn(1, use_lock_timeout=False)],
        invariant=_invariant,
        deadlock_timeout=10.0,
        timeout_per_run=15.0,
    )

    # If we get here without hanging, the bug is fixed!
    # The race SHOULD be detected (property_holds=False).
    assert not result.property_holds, (
        "Expected race to be detected, but DPOR reported property holds. "
        "If this test passed without hanging, defect #6 may be fixed."
    )


def test_workaround_with_lock_timeout(_pg_available) -> None:
    """Demonstrates the lock_timeout workaround: PG fails fast instead of
    blocking, allowing thread B to return to a DPOR scheduling point.

    This test should PASS (detecting the race) without hanging.
    """
    result = explore_dpor(
        setup=_State,
        threads=[_make_thread_fn(0, use_lock_timeout=True),
                 _make_thread_fn(1, use_lock_timeout=True)],
        invariant=_invariant,
        deadlock_timeout=10.0,
        timeout_per_run=15.0,
    )

    assert not result.property_holds, (
        f"Expected race to be detected (invariant violated by PG error).\n"
        f"num_explored={result.num_explored}\n{result.explanation}"
    )


def test_explore_dpor_lock_timeout_parameter(_pg_available) -> None:
    """Test that explore_dpor's lock_timeout parameter automatically injects
    SET lock_timeout on PostgreSQL connections, preventing the DPOR/PG deadlock.

    Threads do NOT set lock_timeout themselves — explore_dpor handles it.
    Without the lock_timeout parameter, this scenario hangs (see
    test_deadlock_without_lock_timeout). With it, PG raises an error on the
    blocked thread, allowing it to return to a DPOR scheduling point.
    """
    result = explore_dpor(
        setup=_State,
        threads=[_make_thread_fn(0, use_lock_timeout=False),
                 _make_thread_fn(1, use_lock_timeout=False)],
        invariant=_invariant,
        deadlock_timeout=10.0,
        timeout_per_run=15.0,
        lock_timeout=2000,
    )

    assert not result.property_holds, (
        f"Expected race to be detected with explore_dpor lock_timeout.\n"
        f"num_explored={result.num_explored}\n{result.explanation}"
    )
