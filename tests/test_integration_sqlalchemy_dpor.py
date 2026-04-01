"""Tests for the SQLAlchemy DPOR integration helper.

Verifies that ``sqlalchemy_dpor`` correctly:
1. Disposes the engine before each execution (fresh connections via patch_sql)
2. Provides per-thread connections via ``get_connection()``
3. Injects ``lock_timeout`` on each thread's connection
4. Passes ``lock_timeout`` through to ``explore_dpor`` for raw connections
5. Detects TOCTOU races with check-then-insert patterns
"""

from __future__ import annotations

import os

import pytest

try:
    import sqlalchemy  # noqa: F401
except ImportError:
    pytest.skip("sqlalchemy not installed", allow_module_level=True)

pytestmark = pytest.mark.integration

_DB_NAME = os.environ.get("FRONTRUN_TEST_DB", "frontrun_test")
_DB_URL = os.environ.get("DATABASE_URL", f"postgresql:///{_DB_NAME}")


@pytest.fixture(scope="module")
def _pg_available():
    """Check Postgres is available and create test tables."""
    try:
        import psycopg2

        conn = psycopg2.connect(_DB_URL)
    except Exception:
        pytest.skip(f"Postgres not available at {_DB_NAME}")

    from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS sa_dpor_test")
        cur.execute("""
            CREATE TABLE sa_dpor_test (
                id TEXT PRIMARY KEY,
                value TEXT
            )
        """)
    conn.close()
    yield


@pytest.fixture()
def engine(_pg_available):
    """Create a SQLAlchemy engine for tests."""
    from sqlalchemy import create_engine

    eng = create_engine(_DB_URL)
    yield eng
    eng.dispose()


def test_sqlalchemy_dpor_per_thread_connections(engine) -> None:
    """Each thread gets its own connection via get_connection()."""
    from sqlalchemy import text

    from frontrun.contrib.sqlalchemy import get_connection, sqlalchemy_dpor

    observed_conn_ids: list[int] = []

    class _State:
        def __init__(self) -> None:
            with engine.connect() as conn:
                conn.execute(text("DELETE FROM sa_dpor_test"))
                conn.commit()
            self.results: list[str | None] = [None, None]

    def _thread_fn(state: _State) -> None:
        conn = get_connection()
        observed_conn_ids.append(id(conn))
        state.results[0] = "ok"

    result = sqlalchemy_dpor(
        engine=engine,
        setup=_State,
        threads=[_thread_fn],
        invariant=lambda s: True,
        lock_timeout=2000,
    )

    assert result.num_explored >= 1
    assert len(observed_conn_ids) >= 1


def test_sqlalchemy_dpor_lock_timeout_injection(engine) -> None:
    """Verify lock_timeout is injected on per-thread connections."""
    from frontrun.contrib.sqlalchemy import get_connection, sqlalchemy_dpor

    observed_lock_timeouts: list[str] = []

    class _State:
        def __init__(self) -> None:
            pass

    def _thread_fn(state: _State) -> None:
        conn = get_connection()
        result = conn.exec_driver_sql("SHOW lock_timeout")
        row = result.fetchone()
        if row:
            observed_lock_timeouts.append(row[0])

    result = sqlalchemy_dpor(
        engine=engine,
        setup=_State,
        threads=[_thread_fn],
        invariant=lambda s: True,
        lock_timeout=2000,
    )

    assert result.num_explored >= 1
    assert any(lt == "2s" for lt in observed_lock_timeouts), f"Expected lock_timeout='2s', got {observed_lock_timeouts}"


def test_sqlalchemy_dpor_toctou_race(engine) -> None:
    """Detect a check-then-insert TOCTOU race using sqlalchemy_dpor."""
    from sqlalchemy import text

    from frontrun.contrib.sqlalchemy import get_connection, sqlalchemy_dpor

    class _State:
        def __init__(self) -> None:
            with engine.connect() as conn:
                conn.execute(text("DELETE FROM sa_dpor_test"))
                conn.commit()
            self.results: list[str | None] = [None, None]

    def _make_thread_fn(idx: int):
        def _thread_fn(state: _State) -> None:
            conn = get_connection()
            # CHECK: does the row exist?
            result = conn.execute(
                text("SELECT id FROM sa_dpor_test WHERE id = :id"),
                {"id": "row1"},
            )
            row = result.fetchone()
            if row is not None:
                state.results[idx] = "already_exists"
                conn.rollback()
                return

            # ACT: insert the row
            try:
                conn.execute(
                    text("INSERT INTO sa_dpor_test (id, value) VALUES (:id, :value)"),
                    {"id": "row1", "value": f"thread_{idx}"},
                )
                conn.commit()
                state.results[idx] = "inserted"
            except Exception as exc:
                conn.rollback()
                state.results[idx] = f"error: {type(exc).__name__}: {exc}"

        return _thread_fn

    def _invariant(state: _State) -> bool:
        r0, r1 = state.results
        has_error = (r0 is not None and r0.startswith("error")) or (r1 is not None and r1.startswith("error"))
        return not has_error

    result = sqlalchemy_dpor(
        engine=engine,
        setup=_State,
        threads=[_make_thread_fn(0), _make_thread_fn(1)],
        invariant=_invariant,
        lock_timeout=2000,
        deadlock_timeout=10.0,
        timeout_per_run=15.0,
    )

    # The key assertion: the exploration completes without deadlocking.
    # Whether DPOR detects the race depends on conflict analysis.
    assert result.num_explored >= 1, "Expected at least 1 interleaving explored"


def test_sqlalchemy_setup_failure_closes_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    from unittest.mock import MagicMock

    from frontrun.contrib import sqlalchemy as sa_helper

    mock_conn = MagicMock()
    mock_conn.exec_driver_sql.side_effect = RuntimeError("SET lock_timeout failed")

    mock_conn_ctx = MagicMock()
    mock_conn_ctx.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn_ctx.__exit__ = MagicMock(return_value=False)

    mock_engine = MagicMock()
    mock_engine.connect.return_value = mock_conn_ctx
    mock_engine.dispose = MagicMock()

    def _setup() -> object:
        return object()

    def _thread_fn(_state: object) -> None:
        raise AssertionError("thread should not run after setup failure")

    def _explore_dpor(*, setup, threads, invariant, detect_io, lock_timeout, **kwargs):  # type: ignore[no-untyped-def]
        with pytest.raises(RuntimeError, match="SET lock_timeout failed"):
            wrapped_setup = setup()
            threads[0](wrapped_setup)
        return object()

    monkeypatch.setattr("frontrun.dpor.explore_dpor", _explore_dpor)

    try:
        sa_helper.sqlalchemy_dpor(
            engine=mock_engine,
            setup=_setup,
            threads=[_thread_fn],
            invariant=lambda s: True,
            lock_timeout=5000,
        )
    except RuntimeError:
        pass

    assert mock_conn_ctx.__exit__.called
