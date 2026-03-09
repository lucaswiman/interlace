# DPOR Scheduler Deadlock with PostgreSQL Row Locks

**TL;DR:** `SELECT FOR UPDATE` deadlocks the frontrun DPOR scheduler because the waiting
thread blocks at the C level (inside `libpq recv()`), invisible to the Python scheduler.
The DPOR `deadlock_timeout` fires instead of Postgres's deadlock detector, because it is
not a database-level deadlock.

**Workaround:** Set `lock_timeout` on each connection in the test setup. This makes Postgres
cancel the blocked `SELECT FOR UPDATE` with a `LockNotAvailable` error instead of waiting
forever, allowing the DPOR scheduler to regain control. See [Workaround](#workaround-lock_timeout-in-test-setup) below.

---

## Background: How the DPOR Scheduler Works

frontrun DPOR makes threads take turns one Python opcode at a time.
Each worker thread calls `report_and_wait()` after every opcode, which parks the thread
on a `threading.Condition` until the scheduler grants it a turn:

```python
# DporScheduler._report_and_wait (simplified)
while True:
    if self._current_thread == thread_id:
        # It's our turn — execute one opcode, then schedule next thread
        next_thread = self._schedule_next()
        self._current_thread = next_thread
        self._condition.notify_all()
        return True
    # Not our turn — park here until notified
    if not self._condition.wait(timeout=self.deadlock_timeout):
        raise TimeoutError("DPOR deadlock: ...")
```

The scheduler is **cooperative**: threads voluntarily yield after each opcode.
This works perfectly for Python-level operations and for cooperative locks
(`threading.Lock` patched by `patch_locks()`), where the lock-acquire call
yields to the scheduler rather than blocking at the OS level.

---

## What Happens with `SELECT FOR UPDATE`

### Step-by-step failure sequence

```
1. Scheduler grants Thread A its turn.

2. Thread A sends to Postgres:
     BEGIN;
     SELECT * FROM orders WHERE id=1 FOR UPDATE;
   Postgres acquires a row-level write lock on row id=1.
   Thread A's psycopg2 is now blocked in recv() at the C level,
   waiting for the SELECT response from Postgres.

3. The DPOR scheduler cannot see C-level blocking — it only intercepts
   Python opcodes. From the scheduler's perspective, Thread A is simply
   "running" (it has been granted the turn but hasn't yielded back yet).

4. The scheduler's deadlock_timeout eventually fires (default: 15 s),
   and it decides to schedule Thread B anyway, setting:
     _current_thread = Thread B

5. Thread B wakes up and sends to Postgres:
     BEGIN;
     SELECT * FROM orders WHERE id=1 FOR UPDATE;
   Postgres tries to acquire the same row lock — which Thread A holds.
   Postgres blocks Thread B's connection at the OS level.
   Thread B is now stuck in recv(), waiting for the lock.

6. Situation:
   - Thread A: blocked in C-level recv(), waiting for its SELECT response.
     It holds the Postgres row lock. Postgres has responded with the locked
     row, but Thread A's Python code is suspended — it hasn't read the
     response yet, so psycopg2 thinks the lock is still being negotiated.
   - Thread B: blocked in C-level recv(), waiting for the row lock
     held by Thread A.
   - DPOR scheduler: both threads are stuck in C; neither will ever call
     report_and_wait() again. The deadlock_timeout fires after 15 s.
```

### Why the deadlock timeout fires instead of Postgres's detector

The DPOR scheduler surfaces this as:

```
TimeoutError: DPOR deadlock: waiting for thread 0, current is 1
```

after `deadlock_timeout` seconds. This is **not** a Postgres deadlock error.

---

## Why Postgres's Deadlock Detector Doesn't Fire

Postgres detects deadlocks by scanning `pg_locks` for cycles in the
**connection-level waiter graph**. It would detect:

```
Connection A waits for lock held by Connection B
Connection B waits for lock held by Connection A  ← cycle → deadlock
```

But in the DPOR scenario there is **no cycle**:

```
Connection A holds row lock on id=1   (lock already granted)
Connection B waits for lock held by Connection A  ← simple chain, no cycle
```

Connection A is not waiting for anything from Postgres's perspective.
It holds the lock and has an open transaction. Postgres sees it as an
**idle-in-transaction** client — slow, but not deadlocked.

Postgres's deadlock detector (`DeadLockCheck` in `lock.c`) only fires
when two or more connections are **mutually waiting**. A single connection
holding a lock indefinitely looks like a slow client. Postgres will
eventually enforce `lock_timeout` or `idle_in_transaction_session_timeout`
if configured, but the default is no timeout.

### The two wait graphs are completely separate

| Layer | What is tracked | Who detects cycles |
|---|---|---|
| **Postgres** | `pg_locks`: DB connection → row lock waiter graph | Postgres deadlock detector (~1 s cycle check) |
| **DPOR scheduler** | `threading.Condition`: Python thread → `_current_thread` | `WaitForGraph` + fallback `deadlock_timeout` |

The DPOR scheduler suspends threads by parking them in `threading.Condition.wait()` —
a Python-level primitive. The parked thread's database *connection* remains open and
continues to hold any Postgres locks it acquired. Postgres has no visibility into the
Python scheduler state.

---

## Why Cooperative Locks Don't Have This Problem

When `patch_locks()` is active, `threading.Lock.acquire()` is replaced with a
cooperative version that yields to the DPOR scheduler instead of blocking at the OS level:

```
Thread A acquires patched Lock L:
  → DPOR records "Thread A holds L" in WaitForGraph

Thread B tries to acquire patched Lock L:
  → patched acquire() sees L is held, calls report_and_wait() to yield
  → DPOR records "Thread B waits for L held by A" in WaitForGraph
  → WaitForGraph detects no cycle; schedules Thread A to release L
  → Thread B is eventually unblocked by Thread A's release
```

PostgreSQL row locks go through `libpq` → OS socket → Postgres process.
The blocking happens inside `psycopg2`'s C extension (`libpq` `PQexec`),
below the Python GIL and below any Python hook. The DPOR scheduler cannot
intercept it, cannot yield cooperatively from inside it, and cannot see it
in the wait-for graph.

---

## Workaround: `lock_timeout` in Test Setup

Setting `lock_timeout` on each connection before the test prevents the indefinite block.
When Thread B tries `SELECT ... FOR UPDATE` and the row is locked, Postgres cancels it
after the timeout with `ERROR: canceling statement due to lock timeout` rather than
blocking forever. Thread B's C-level `recv()` returns with an error, Thread B returns
to Python, calls `report_and_wait()`, and the DPOR scheduler regains control.

### Django (psycopg2)

```python
from django.db import connections

def _set_lock_timeout(ms: int = 500) -> None:
    """Call at the start of _State.__init__ and each thread function."""
    for alias in connections:
        with connections[alias].cursor() as cur:
            cur.execute(f"SET lock_timeout = '{ms}ms'")
```

### SQLAlchemy

```python
from sqlalchemy import text

def _set_lock_timeout(conn, ms: int = 500) -> None:
    conn.execute(text(f"SET lock_timeout = '{ms}ms'"))
```

### Handling `LockNotAvailable` in thread functions

Thread functions need to catch `psycopg2.errors.LockNotAvailable`
(or `sqlalchemy.exc.OperationalError` wrapping it) and record the outcome:

```python
import psycopg2

def _thread_fn(state):
    try:
        with transaction.atomic():
            obj = MyModel.objects.select_for_update().get(pk=state.obj_id)
            # ... check and update ...
            state.results[idx] = "success"
    except psycopg2.errors.LockNotAvailable:
        state.results[idx] = "lock_timeout"
    except Exception as exc:
        state.results[idx] = f"error: {exc}"
```

The invariant should treat `"lock_timeout"` as the expected outcome for the
losing thread — i.e. `SELECT FOR UPDATE` is working correctly:

```python
def _invariant(state) -> bool:
    # With SELECT FOR UPDATE, at most one thread should succeed.
    # The other should get lock_timeout (expected) or error (unexpected).
    success_count = sum(1 for r in state.results if r == "success")
    return success_count <= 1
```

### What this workaround does and does not provide

| | `lock_timeout` workaround | Ideal (wire-protocol interception) |
|---|---|---|
| Prevents DPOR scheduler deadlock | ✓ | ✓ |
| Explores "Thread B waits, then gets lock" interleavings | ✗ | ✓ |
| Verifies fix prevents double-success | ✓ | ✓ |
| Thread B outcome | `LockNotAvailable` error | eventually succeeds |

The workaround converts a blocking wait into a fast failure, so the DPOR engine
explores "Thread B fails to acquire lock" interleavings rather than
"Thread B waits and eventually succeeds" interleavings. This is sufficient to
verify that a `SELECT FOR UPDATE` fix is correct (only one thread succeeds),
but it does not exhaustively explore all interleavings of the full two-thread
success path.

### `lock_timeout` vs `statement_timeout`

`lock_timeout` is the right setting here — it fires only when a statement is
**waiting for a lock**, not during normal query execution. `statement_timeout`
applies to the total statement execution time and can cancel legitimate slow
queries in the test setup.

---

## What Would Be Required for Full Support

To make `SELECT FOR UPDATE` fully interoperable with the DPOR scheduler (exploring
"both threads eventually complete" interleavings), frontrun would need to:

1. **Intercept the lock grant in the LD_PRELOAD library** — when `libpq` receives the
   `SELECT FOR UPDATE` response from Postgres (a `DataRow` message in the PostgreSQL
   wire protocol), record that this connection now holds a row lock on the specified row.

2. **Intercept the lock wait** — when `libpq` is about to block waiting for a row lock
   (Postgres responds with nothing until the lock is available), detect this condition
   before blocking and report it to the DPOR scheduler, allowing it to schedule another
   thread.

3. **Coordinate lock release** — `COMMIT` / `ROLLBACK` release all Postgres row locks;
   the LD_PRELOAD library would need to observe these and unblock waiting threads.

This requires parsing the PostgreSQL wire protocol in `crates/io/` to identify lock
grant, lock wait, and lock release messages — significantly more complex than the
current byte-level send/recv interception.

---

## Practical Implications for Tests

Three options in increasing order of DPOR coverage:

**Option 1 — `lock_timeout` workaround (recommended for fix verification):**
Set `lock_timeout` in the test setup. DPOR finds "lock_timeout" interleavings;
the invariant confirms at most one thread succeeds. Works today with no frontrun changes.

**Option 2 — Direct threading (probabilistic fix verification):**
Use `threading.Thread` + `threading.Barrier` for the fix path, DPOR for the bug path.
Not exhaustive, but sufficient for high-confidence fix confirmation.

```python
# Cannot use explore_dpor here without lock_timeout — would deadlock
# Use direct threading instead to verify the fix probabilistically
ITERATIONS = 10
double_approved = 0
for _ in range(ITERATIONS):
    # ... reset state ...
    t0 = threading.Thread(target=_make_thread_fn(0))
    t1 = threading.Thread(target=_make_thread_fn(1))
    t0.start(); t1.start()
    t0.join(timeout=10); t1.join(timeout=10)
    if results[0] == "approved" and results[1] == "approved":
        double_approved += 1
assert double_approved == 0
```

See `test_django_fsm_transition_race.py::TestFSMTransitionFixedWithSelectForUpdate`
for a working example.

**Option 3 — Wire-protocol interception in LD_PRELOAD (future work):**
Full DPOR exploration of both-threads-complete interleavings. Requires significant
work in `crates/io/` to parse the PostgreSQL wire protocol.
