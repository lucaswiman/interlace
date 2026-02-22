"""
Django-style ORM field clobber — real Postgres race condition
=============================================================

Django's default ``Model.save()`` writes ALL fields back to the database,
even if only one field was modified.  When two concurrent request handlers
update different fields of the same record, the second ``save()`` silently
overwrites the first handler's changes.

This is arguably the most common database race condition in Django apps,
and it's hard to catch because:

1. The two operations touch different fields (no obvious conflict)
2. Each operation succeeds (no error, no exception)
3. It only manifests under concurrent load (~10% with naive threading)
4. The clobbered field silently reverts to its old value

Scenario
--------
A ``user_profiles`` table with columns ``name``, ``email``, ``bio``.
Two concurrent HTTP request handlers:

* **Admin panel** — admin changes user's email
* **Profile page** — user updates their own bio

Django view code (the real-world pattern)::

    # views.py — admin panel
    def admin_change_email(request, user_id):
        user = UserProfile.objects.get(id=user_id)   # SELECT *
        user.email = request.POST['email']
        user.save()                                   # UPDATE SET all fields

    # views.py — profile page
    def user_edit_profile(request, user_id):
        user = UserProfile.objects.get(id=user_id)   # SELECT *
        user.bio = request.POST['bio']
        user.save()                                   # UPDATE SET all fields

Race window::

    Admin handler:  SELECT * → email="old"  bio="Original"
    User handler:   SELECT * → email="old"  bio="Original"
    Admin saves:    UPDATE SET email="new", bio="Original"      ← correct
    User saves:     UPDATE SET email="old", bio="I love hiking" ← clobbers email!

Fix: ``user.save(update_fields=['email'])`` — write only the changed
field.  Available since Django 1.5, but the default still writes
everything.

This example demonstrates detection using frontrun:

1. **Trace markers** (deterministic) — exact schedule forces the race
2. **Bytecode exploration** (automatic) — random opcode-level schedules
3. **Naive threading** — shows intermittent failure rate

Requirements::

    pip install psycopg2-binary
    createdb frontrun_test   # or: sudo -u postgres createdb frontrun_test

Running::

    python examples/django_orm_race.py
"""

from __future__ import annotations

import random
import threading
import time

import psycopg2

from frontrun.bytecode import explore_interleavings
from frontrun.common import Schedule, Step
from frontrun.trace_markers import TraceExecutor

_DSN = "dbname=frontrun_test"
_SEP = "=" * 70


# ---------------------------------------------------------------------------
# Database setup
# ---------------------------------------------------------------------------


def _setup_table() -> None:
    """(Re-)create the user_profiles table and seed one row."""
    conn = psycopg2.connect(_DSN)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS user_profiles")
        cur.execute(
            "CREATE TABLE user_profiles ("
            "  id SERIAL PRIMARY KEY,"
            "  name TEXT NOT NULL,"
            "  email TEXT NOT NULL,"
            "  bio TEXT NOT NULL"
            ")"
        )
        cur.execute(
            "INSERT INTO user_profiles (name, email, bio) VALUES (%s, %s, %s)",
            ("Alice", "alice@old.com", "Original bio"),
        )
    conn.close()


def _reset_row() -> None:
    """Reset the row to initial state between attempts."""
    conn = psycopg2.connect(_DSN)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE user_profiles SET name = 'Alice', email = 'alice@old.com', bio = 'Original bio' WHERE id = 1"
        )
    conn.close()


def _read_row() -> tuple[str, str, str]:
    """Read the final state of the row."""
    conn = psycopg2.connect(_DSN)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("SELECT name, email, bio FROM user_profiles WHERE id = 1")
        row = cur.fetchone()
    conn.close()
    assert row is not None
    return row  # type: ignore[return-value]


# ============================================================================
# Demo 1: Exact reproduction with trace markers
# ============================================================================
#
# Each handler opens its own psycopg2 connection (separate Postgres session).
# The # frontrun: markers on the cur.execute() calls let TraceExecutor
# control when each SQL statement fires.
#
# Schedule:
#   1. Admin handler: SELECT * (reads old email, old bio)
#   2. User handler:  SELECT * (reads old email, old bio)
#   3. Admin handler: UPDATE SET all (writes new email, old bio)
#   4. User handler:  UPDATE SET all (writes old email, new bio) ← clobbers!


def demo_trace_markers() -> None:
    """Reproduce the ORM field clobber deterministically against real Postgres."""
    print(_SEP)
    print("Demo 1: Django ORM Field Clobber  (TraceExecutor + psycopg2 — deterministic)")
    print(_SEP)
    print()
    print("  Scenario:")
    print("    Admin changes email:  alice@old.com → alice@new.com")
    print("    User changes bio:     Original bio  → I love hiking")
    print("    Expected: BOTH changes persist")
    print()

    _setup_table()

    conn_admin = psycopg2.connect(_DSN)
    conn_user = psycopg2.connect(_DSN)
    conn_admin.autocommit = False
    conn_user.autocommit = False

    errors: dict[str, str] = {}

    def admin_update_email() -> None:
        """Admin changes user's email — Django's Model.save() pattern."""
        try:
            with conn_admin.cursor() as cur:
                cur.execute("SELECT name, email, bio FROM user_profiles WHERE id = 1")  # frontrun: orm_load
                name, email, bio = cur.fetchone()  # type: ignore[misc]
            email = "alice@new.com"
            with conn_admin.cursor() as cur:
                cur.execute("UPDATE user_profiles SET name=%s, email=%s, bio=%s WHERE id = 1", (name, email, bio))  # frontrun: orm_save
            conn_admin.commit()
        except Exception as exc:
            errors["admin"] = str(exc).strip()
            conn_admin.rollback()

    def user_update_bio() -> None:
        """User updates their bio — Django's Model.save() pattern."""
        try:
            with conn_user.cursor() as cur:
                cur.execute("SELECT name, email, bio FROM user_profiles WHERE id = 1")  # frontrun: orm_load
                name, email, bio = cur.fetchone()  # type: ignore[misc]
            bio = "I love hiking"
            with conn_user.cursor() as cur:
                cur.execute("UPDATE user_profiles SET name=%s, email=%s, bio=%s WHERE id = 1", (name, email, bio))  # frontrun: orm_save
            conn_user.commit()
        except Exception as exc:
            errors["user"] = str(exc).strip()
            conn_user.rollback()

    # Force both SELECTs before either UPDATE
    schedule = Schedule(
        [
            Step("admin", "orm_load"),  # Admin does SELECT * (stale snapshot)
            Step("user", "orm_load"),  # User does SELECT * (stale snapshot)
            Step("admin", "orm_save"),  # Admin saves all fields (new email, stale bio)
            Step("user", "orm_save"),  # User saves all fields (stale email, new bio) ← clobbers!
        ]
    )

    executor = TraceExecutor(schedule)
    executor.run("admin", admin_update_email)
    executor.run("user", user_update_bio)
    executor.wait(timeout=10.0)

    conn_admin.close()
    conn_user.close()

    if errors:
        for name, msg in errors.items():
            print(f"  {name} error: {msg}")
        print()

    _name, email, bio = _read_row()

    print(f"  Final email: {email!r}  (expected 'alice@new.com')")
    print(f"  Final bio:   {bio!r}  (expected 'I love hiking')")
    print()

    if email != "alice@new.com":
        print("  FIELD CLOBBER confirmed: admin's email update was lost.")
        print("  The user's save() wrote back the stale email it loaded")
        print("  before the admin made their change.")
        print()
        print("  Reproducibility: 100% — the Schedule deterministically forces")
        print("  both SELECTs to run before either UPDATE on every execution.")
    elif bio != "I love hiking":
        print("  FIELD CLOBBER confirmed: user's bio update was lost.")
    else:
        print("  No clobber (both updates persisted).")
    print()


# ============================================================================
# Demo 2: Automatic detection with bytecode exploration
# ============================================================================
#
# explore_interleavings generates random opcode-level schedules and runs
# both transactions under controlled interleaving.  psycopg2 releases the
# GIL during C-level I/O, so the scheduler has real interleaving leverage.


def demo_bytecode_exploration() -> None:
    """Find the ORM field clobber automatically with random bytecode schedules."""
    print(_SEP)
    print("Demo 2: Django ORM Field Clobber  (bytecode exploration + psycopg2 — automatic)")
    print(_SEP)
    print()
    print("  Generating random opcode-level schedules and running both")
    print("  handlers against real Postgres.  Checking whether both field")
    print("  updates persist after each interleaving.")
    print()

    _setup_table()

    # Persistent connections reused across all attempts (avoid connection churn).
    admin_conn = psycopg2.connect(_DSN)
    admin_conn.autocommit = True
    conn_a = psycopg2.connect(_DSN)
    conn_b = psycopg2.connect(_DSN)
    conn_a.autocommit = False
    conn_b.autocommit = False

    class _State:
        """Per-attempt state: reset the row, expose the reused connections."""

        def __init__(self) -> None:
            conn_a.rollback()
            conn_b.rollback()
            with admin_conn.cursor() as cur:
                cur.execute(
                    "UPDATE user_profiles SET name='Alice', email='alice@old.com', bio='Original bio' WHERE id = 1"
                )
            self.conn_a = conn_a
            self.conn_b = conn_b

    def _admin_txn(conn: psycopg2.extensions.connection) -> None:
        """Django-style: SELECT *, modify email, UPDATE SET all."""
        with conn.cursor() as cur:
            cur.execute("SELECT name, email, bio FROM user_profiles WHERE id = 1")
            name, email, bio = cur.fetchone()  # type: ignore[misc]
        email = "alice@new.com"
        with conn.cursor() as cur:
            cur.execute("UPDATE user_profiles SET name=%s, email=%s, bio=%s WHERE id = 1", (name, email, bio))
        conn.commit()

    def _user_txn(conn: psycopg2.extensions.connection) -> None:
        """Django-style: SELECT *, modify bio, UPDATE SET all."""
        with conn.cursor() as cur:
            cur.execute("SELECT name, email, bio FROM user_profiles WHERE id = 1")
            name, email, bio = cur.fetchone()  # type: ignore[misc]
        bio = "I love hiking"
        with conn.cursor() as cur:
            cur.execute("UPDATE user_profiles SET name=%s, email=%s, bio=%s WHERE id = 1", (name, email, bio))
        conn.commit()

    def _invariant(state: _State) -> bool:
        with admin_conn.cursor() as cur:
            cur.execute("SELECT email, bio FROM user_profiles WHERE id = 1")
            email, bio = cur.fetchone()  # type: ignore[misc]
        return email == "alice@new.com" and bio == "I love hiking"

    result = explore_interleavings(
        setup=_State,
        threads=[
            lambda s: _admin_txn(s.conn_a),
            lambda s: _user_txn(s.conn_b),
        ],
        invariant=_invariant,
        max_attempts=50,
        seed=42,
        detect_io=False,  # psycopg2 uses C-level sockets
        deadlock_timeout=15.0,
        timeout_per_run=60.0,
        reproduce_on_failure=5,
    )

    admin_conn.close()
    conn_a.close()
    conn_b.close()

    print(f"  property_holds    : {result.property_holds}")
    print(f"  attempts_explored : {result.num_explored}")
    if result.counterexample is not None:
        repro = ""
        if result.reproduction_attempts:
            repro = f"  ({result.reproduction_successes}/{result.reproduction_attempts} reproductions)"
        print(f"  counterexample found after {result.num_explored} attempt(s){repro}")
    print()
    if result.explanation:
        for line in result.explanation.splitlines():
            print("  " + line)
        print()
    if not result.property_holds:
        print("  FIELD CLOBBER confirmed via bytecode exploration.")
    else:
        print("  No field clobber found in the explored interleavings.")
    print()


# ============================================================================
# Demo 3: Naive threading — intermittent failure
# ============================================================================
#
# In production, two requests for the same user don't arrive at the exact
# same instant.  The race only manifests when the second request's SELECT
# lands inside the first request's SELECT→UPDATE window (typically <1ms on
# localhost).  We model realistic request arrival by staggering thread
# starts with a random 0–15ms offset, which gives roughly 10% collision
# rate on localhost — representative of moderate concurrent load on a
# Django app.
#
# If both threads started at the exact same instant (offset=0), the race
# would reproduce ~95% of the time because psycopg2 releases the GIL
# during I/O and both threads hit their SELECTs simultaneously.


def demo_naive_threading(trials: int = 500) -> None:
    """Show the intermittent nature of the race with plain threads + Postgres."""
    print(_SEP)
    print(f"Demo 3: Naive threading + psycopg2  ({trials} trials)")
    print(_SEP)
    print()
    print("  Running both handlers in plain threads against real Postgres.")
    print("  Threads start with a random 0–15ms offset to model realistic")
    print("  request arrival timing.  Counting how often the race manifests...")
    print()

    _setup_table()

    # Pre-create connections (simulates a connection pool, as in Django).
    conn_a = psycopg2.connect(_DSN)
    conn_b = psycopg2.connect(_DSN)
    failures = 0
    rng = random.Random(42)

    for _ in range(trials):
        conn_a.autocommit = True
        with conn_a.cursor() as cur:
            cur.execute(
                "UPDATE user_profiles SET name='Alice', email='alice@old.com', bio='Original bio' WHERE id = 1"
            )
        conn_a.autocommit = False
        conn_b.autocommit = False

        def admin() -> None:
            with conn_a.cursor() as cur:
                cur.execute("SELECT name, email, bio FROM user_profiles WHERE id = 1")
                name, email, bio = cur.fetchone()  # type: ignore[misc]
            email = "alice@new.com"
            with conn_a.cursor() as cur:
                cur.execute("UPDATE user_profiles SET name=%s, email=%s, bio=%s WHERE id = 1", (name, email, bio))
            conn_a.commit()

        def user() -> None:
            with conn_b.cursor() as cur:
                cur.execute("SELECT name, email, bio FROM user_profiles WHERE id = 1")
                name, email, bio = cur.fetchone()  # type: ignore[misc]
            bio = "I love hiking"
            with conn_b.cursor() as cur:
                cur.execute("UPDATE user_profiles SET name=%s, email=%s, bio=%s WHERE id = 1", (name, email, bio))
            conn_b.commit()

        t1 = threading.Thread(target=admin)
        t2 = threading.Thread(target=user)
        t1.start()
        # Random offset models realistic request arrival timing.
        time.sleep(rng.uniform(0, 0.015))
        t2.start()
        t1.join()
        t2.join()

        conn_a.autocommit = True
        with conn_a.cursor() as cur:
            cur.execute("SELECT email, bio FROM user_profiles WHERE id = 1")
            email, bio = cur.fetchone()  # type: ignore[misc]
        if email != "alice@new.com" or bio != "I love hiking":
            failures += 1

    conn_a.close()
    conn_b.close()

    rate = failures / trials * 100
    print(f"  Trials:   {trials}")
    print(f"  Failures: {failures}")
    print(f"  Rate:     {rate:.1f}%")
    print()
    if failures > 0:
        print(f"  The race manifested in {failures}/{trials} trials ({rate:.1f}%).")
        print("  frontrun makes it 100% reproducible.")
    else:
        print("  No race observed in this run.")
    print()


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    demo_trace_markers()
    demo_bytecode_exploration()
    demo_naive_threading()
