"""Integration test: DPOR failure with nondeterministic IDs (sequences).

DPOR relies on deterministic resource IDs across runs to match events.
If a test inserts a row and gets ID=1 in run A and ID=2 in run B
(because the DB sequence wasn't reset), the resource IDs change,
breaking DPOR's ability to map events between runs.
"""

from __future__ import annotations

import os
import threading

import pytest

try:
    from sqlalchemy import String, create_engine, text
    from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column
except ImportError:
    pytest.skip("sqlalchemy not installed", allow_module_level=True)

from frontrun.dpor import explore_dpor

_DB_NAME = os.environ.get("FRONTRUN_TEST_DB", "frontrun_test")
_DB_URL = os.environ.get("DATABASE_URL", f"postgresql:///{_DB_NAME}")


# ---------------------------------------------------------------------------
# SQLAlchemy model
# ---------------------------------------------------------------------------


class Base(DeclarativeBase):
    pass


class Item(Base):
    __tablename__ = "items"

    # SERIAL primary key
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def engine():
    """Create the engine and verify Postgres is reachable."""
    eng = create_engine(_DB_URL)
    try:
        with eng.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception:
        pytest.skip(f"Postgres not available at {_DB_URL}")
    Base.metadata.drop_all(eng)
    Base.metadata.create_all(eng)
    yield eng
    Base.metadata.drop_all(eng)
    eng.dispose()


# ---------------------------------------------------------------------------
# Test reproduction
# ---------------------------------------------------------------------------


class TestNondeterministicIds:
    """Demonstrate DPOR failure when IDs are not deterministic."""

    def test_nondeterministic_ids_break_dpor(self, engine) -> None:
        """DPOR should fail to find the race if IDs change between runs."""

        class _State:
            def __init__(self) -> None:
                # We intentionally do NOT reset the sequence here.
                # We only truncate the table.
                engine.dispose()
                with engine.connect() as conn:
                    conn.execute(text("TRUNCATE TABLE items"))
                    conn.commit()

        def _thread_fn(_state: _State) -> None:
            with Session(engine) as session:
                # This insert will generate a new ID from the sequence.
                # Run 1: ID=1, Run 2: ID=2, etc.
                item = Item(name="test")
                session.add(item)
                session.commit()
                # Row-level predicate will be sql:items:(('id', '1'),) in run 1
                # and sql:items:(('id', '2'),) in run 2.

        def _invariant(_state: _State) -> bool:
            # We don't really care about the invariant here, we just want to
            # see if DPOR can explore more than one interleaving or if it
            # gets confused.
            return True

        # In a deterministic run, two inserts to the same table (without row-level
        # predicates matching) would be seen as conflicting on the table level
        # if they both write.
        # But here, DPOR's row-level detection will extract the generated ID.

        result = explore_dpor(
            setup=_State,
            threads=[_thread_fn, _thread_fn],
            invariant=_invariant,
            detect_io=True,
            strict_determinism=True,
        )

        # If DPOR works correctly (with deterministic IDs), it should find
        # that these two threads conflict and explore both orders.
        # But if IDs are nondeterministic, DPOR might:
        # 1. Fail with an error (if we add the check)
        # 2. Silently fail to explore (if it can't match events)
        # 3. Crash in the Rust engine if assumptions are violated.

        # For now, we expect it to AT LEAST complete, but it might not
        # explore as many states as it should if it were deterministic.
        # Actually, if IDs are different, DPOR thinks they are DIFFERENT resources
        # and thus they DON'T conflict (if they only touch those specific rows).
        # But `INSERT` usually touches the table resource too.

        print(f"Explored {result.num_explored} executions")
        # With 2 threads and table-level conflict, we expect 2 executions.
        # With row-level conflict on DIFFERENT IDs, they look independent!
        # So it might only do 1 execution.
        assert result.num_explored == 1, (
            f"Expected only 1 execution because nondeterministic IDs "
            f"made operations look independent, but got {result.num_explored}"
        )
