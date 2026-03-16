# Frontrun Defects

Known issues with the frontrun library are tracked here.

## Defect #2: INSERT/DELETE operations don't generate SQL conflict arcs

**Status:** FIXED (March 2026) — phantom read detection via `:seq` resources

As of commit `80d8c85`, frontrun detects phantom reads using table-sequence
resources. SELECT reports a READ on `sql:<table>:seq`, INSERT and DELETE
report a WRITE on `sql:<table>:seq`. This creates conflict arcs that cause
DPOR to explore interleavings where SELECT→INSERT TOCTOU races can manifest.

UPDATE now also reports a READ on `sql:<table>:seq` (added by defect #6
fix). Using READ (not WRITE) avoids false write-write conflicts between
UPDATEs on different rows, while creating read-write conflict arcs with
INSERT's WRITE on `:seq` for phantom race detection.

## Defect #4: DPOR opcode tracer IndexError on certain bytecode patterns

**Status:** Fixed in 603fd7a218fc87f8c934b310bf844542c38777eb

**Symptom:** Tests crash with `IndexError: list index out of range` inside
`frontrun/frontrun/dpor.py:1401` (`_process_opcode`), specifically in the
shadow stack access `shadow.stack[-(argc - _pt_obj_idx)]`. The crash
occurs during DPOR thread scheduling when tracing certain Python bytecodes.

**Affected libraries:**
- `django-email-verification` — crashes on `setattr(user, "is_active", True)`
  inside the email callback
- `django-two-factor-auth` — crashes on `getattr(settings, 'OTP_TOTP_SYNC', True)`
  inside `django_otp`'s `verify_token()` method
- `django-tenants` — crashes inside `django-tenants`'s `postgresql_backend/base.py`
  database backend

**Root cause:** The DPOR bytecode tracer maintains a shadow stack to track
object references through Python opcodes. Certain opcode sequences (likely
involving `CALL_FUNCTION` with keyword arguments, `setattr`/`getattr` with
default values, or database backend metaclass operations) cause the shadow
stack to underflow. The tracer expects more items on the stack than are
present.

**Impact:** Tests for these three libraries cannot complete DPOR exploration.
The crash occurs before any interleaving result is produced, so no race
condition determination can be made.

**Workaround:** None currently. The affected tests should be marked with
`@pytest.mark.skip(reason="Blocked on frontrun defect #4 (opcode tracer IndexError)")`.

---

## Defect #3: CooperativeQueue missing `__class_getitem__` breaks psycopg v3

**Status:** Fixed in commit 603fd7a218fc87f8c934b310bf844542c38777eb

**Symptom:** Any test that imports psycopg v3 (e.g., Django tests using
`django.db.backends.postgresql` with psycopg v3 as the driver) fails during
collection with:

```
TypeError: type 'CooperativeQueue' is not subscriptable
```

**Root cause:** Frontrun's `CooperativeQueue` (`frontrun/_cooperative.py`)
monkey-patches `queue.Queue`. psycopg v3's `_acompat.py` subclasses
`queue.Queue[T]` using generic subscript syntax (`Queue[T]` as a base class).
Python 3.9+ added `__class_getitem__` to the real `queue.Queue` to support
this, but `CooperativeQueue` did not implement it.

**Impact:** Blocked ALL Django/SQL DPOR tests from running when psycopg v3
was installed, since the `TypeError` was raised during module collection
before any test code executed.

**Fix:** Added `__class_getitem__` to `CooperativeQueue` that returns `cls`,
matching the standard library behavior:

```python
def __class_getitem__(cls, item):
    return cls
```

---

## Defect #5: NondeterministicSQLError on INSERT without RETURNING clause

**Status:** Open

**Symptom:** Tests crash with `NondeterministicSQLError` during DPOR
exploration when a Django model INSERT does not use a `RETURNING` clause (or
the intercepted cursor cannot capture `lastrowid`). The error message is:

```
frontrun._sql_cursor.NondeterministicSQLError: INSERT on `<table>` could not
capture lastrowid; needs RETURNING clause or pre-allocated IDs.
```

**Affected libraries:**
- `dj-stripe` — INSERT on `djstripe_event`
- `django-allauth` — INSERTs on `account_emailaddress`, `auth_user`,
  `socialaccount_socialaccount`
- `django-hijack` — INSERT on `django_session`
- `django-organizations` — INSERTs on `organizations_organizationowner`,
  `organizations_organizationuser`

**Root cause:** Frontrun's SQL cursor interceptor (`_sql_cursor.py`) needs to
track auto-generated primary keys across DPOR interleavings to maintain
determinism. When an INSERT statement does not include a `RETURNING` clause
and the cursor does not expose `lastrowid` in a way frontrun can capture,
the interceptor raises `NondeterministicSQLError` because it cannot guarantee
that replayed interleavings will see the same IDs.

**Impact:** Tests for these libraries cannot complete DPOR exploration. The
error is raised during the first interleaving attempt, before any race
condition analysis can occur.

**Workaround:** Use models with explicitly pre-allocated primary keys (e.g.,
UUIDs) in test setup, or ensure the database backend uses `RETURNING id` in
INSERT statements (psycopg v3 + Django 4.2+ does this by default for
PostgreSQL).

---

## Defect #6: Row-lock arbitration prevents DPOR from detecting UPDATE-INSERT phantom races within transactions

**Status:** FIXED (March 2026)

**Symptom:** DPOR cannot detect races involving the pattern
SELECT→UPDATE→INSERT within a transaction (`autocommit=False` or
`@transaction.atomic`), where two concurrent UPDATEs both match 0 rows
and then both INSERTs create rows that would have matched the UPDATE's
WHERE clause. The same race IS detected when each statement runs with
`autocommit=True`.

**Affected libraries:**
- `django-tenants` — duplicate primary domain via TOCTOU in
  `DomainMixin.save()` (wrapped in `@transaction.atomic`)

**Reproduction:** `frontrun-bugs/tests/test_defect6_update_seq.py`

```python
# Thread A (inside a transaction):
cur.execute("SELECT EXISTS(... WHERE is_primary = TRUE)")   # -> FALSE
cur.execute("UPDATE t SET is_primary = FALSE WHERE is_primary = TRUE")  # 0 rows
cur.execute("INSERT INTO t (..., is_primary) VALUES (..., TRUE)")
conn.commit()

# Thread B (same, concurrent):
# ...same three statements then commit...
```

The race requires the interleaving where both UPDATEs execute before
either INSERT:

```
Thread A: SELECT (empty) → UPDATE (0 rows)
Thread B: SELECT (empty) → UPDATE (0 rows)
Thread A: INSERT (is_primary=TRUE)
Thread B: INSERT (is_primary=TRUE)
Thread A: COMMIT
Thread B: COMMIT
→ Two rows with is_primary=TRUE (invariant violated)
```

This interleaving is valid under PostgreSQL READ COMMITTED: UPDATEs
matching 0 rows acquire no row locks, so both can proceed without
blocking. Manually verified with two psycopg2 connections.

**Root cause (two parts):**

1. Frontrun's row-lock arbitration (`acquire_row_locks` in `dpor.py`)
   blocked Thread B's UPDATE until Thread A's transaction committed, even
   when the UPDATE matched 0 rows. In real PostgreSQL, an UPDATE matching
   0 rows acquires no row locks — there are no rows to lock.

2. UPDATE was excluded from `:seq` tracking (defect #2), so DPOR had no
   conflict arc between the UPDATE (which depends on which rows exist)
   and the INSERT (which changes which rows exist).

**Related:** Defect #2 (phantom read detection via `:seq` resources).

**Impact:** DPOR misses an entire class of real concurrency bugs:
check-then-act patterns where the "check" is an UPDATE (or SELECT
followed by UPDATE) that returns 0 rows, and the "act" is an INSERT.
These are common in Django's ORM (e.g., `queryset.update()` followed by
`model.save()`).

**Fix (two parts in `_sql_cursor.py`):**

1. **Release row locks for 0-row UPDATEs:** After `cursor.execute()`,
   check `cursor.rowcount`. If 0, release any row locks acquired for
   the UPDATE. This matches PostgreSQL semantics.

2. **Add UPDATE to `:seq` tracking:** UPDATEs now report a READ on
   `sql:<table>:seq`, creating conflict arcs with INSERT's WRITE on
   `:seq`. This gives DPOR a reason to schedule both UPDATEs before
   either INSERT. READ (not WRITE) avoids false write-write conflicts
   between UPDATEs on different rows.

**Tests:** `tests/test_defect6_update_insert_phantom.py`

---

## Defect #7: DPOR deadlock when redis-py `__del__` fires during `_process_opcode`

**Status:** Open

**Symptom:** DPOR threads deadlock during Redis-based tests. One thread hangs
inside `_process_opcode` → `_get_instructions` while Python's garbage collector
invokes `redis.client.Redis.__del__()`, which calls `self.close()` →
`connection_pool.disconnect()` → acquires `pool._lock`. Frontrun's cooperative
lock wrapper (`_cooperative.py`) intercepts this lock acquisition and calls
`_sync_reporter()`, which tries to acquire `scheduler._condition`. But the
current thread already holds `scheduler._condition` (it's inside
`_report_and_wait` → `_process_opcode`), causing a reentrancy deadlock.

**Affected libraries:**
- `flask-caching` — memoize test deadlocks during DPOR exploration
- `celery-redbeat` — both `_next_instance()` and `save()` race tests
  deadlock during DPOR cleanup. The race IS detected (invariant fails in
  the first explored interleaving) but `_unpatch_locks()` →
  `WaitForGraph.clear()` hangs because `_lock` is held by a GC-triggered
  `__del__` chain.

**Stack trace (simplified):**

```
dpor-0: report_and_wait → _process_opcode → _get_instructions
  → [GC triggers redis.Redis.__del__]
  → redis.client.close() → connection_pool.disconnect()
  → cooperative Lock.__exit__ → _sync_reporter
  → scheduler._condition.__enter__  ← DEADLOCK (already held)
```

Alternative manifestation (celery-redbeat):

```
dpor-0: cooperative.acquire() → graph.add_holding() → with self._lock (holds _lock)
  → [GC: redis.Redis.__del__]
  → connection_pool.disconnect() → cooperative.acquire()
  → graph.add_holding() → with self._lock  ← DEADLOCK (reentrant)
```

**Root cause:** Python's garbage collector can run at arbitrary points during
bytecode execution, including inside `_process_opcode` where the scheduler
condition lock is already held, or inside `WaitForGraph` methods where
`_lock` is held. When a redis-py client object is garbage collected, its
`__del__` method closes the connection pool, which acquires a lock.
Frontrun's cooperative lock wrapper reports this to the DPOR scheduler,
but neither the scheduler's condition lock nor `WaitForGraph._lock` is
reentrant.

**Impact:** Tests that create temporary `redis.Redis()` connections inside
thread functions may deadlock during DPOR exploration when the GC collects
these connections at an inopportune time. The `@cached` decorator test works
because it happens to avoid triggering GC during opcode processing, while the
`@memoize` test (which has a more complex code path with more temporary
objects) consistently deadlocks. celery-redbeat tests deadlock during DPOR
cleanup (`_unpatch_locks`) because redbeat's `get_redis()` creates Redis
client objects that are GC'd while `WaitForGraph._lock` is held.

**Workaround:** None reliably. The issue is timing-dependent based on when
Python's GC runs. Tests affected by this should be left as-is (they will
timeout).

---

## Defect #8: DPOR does not model Redis Lua script (EVALSHA) atomicity — false positive races

**Status:** Open

**Symptom:** DPOR reports race conditions (invariant violations) on Redis
keys accessed by Lua scripts, but reproduction always fails (0/10). The
counterexample interleaving found by DPOR cannot be reproduced because
Redis Lua scripts execute atomically (Redis is single-threaded and blocks
other commands during script execution).

**Affected libraries:**
- `throttled-py` — Sliding window and token bucket rate limiters use Lua
  scripts via `redis.register_script()` / `EVALSHA`. DPOR detects
  write-write conflicts on the rate limit keys and explores interleavings
  where two Lua scripts interleave, but this is impossible in real Redis.

**Root cause:** DPOR's Redis key-level analysis (intercepting
`execute_command()`) classifies each Redis command as a read or write on
specific keys. When a Lua script is invoked via EVALSHA, it is treated as
a single read+write on the affected keys. However, DPOR treats two
concurrent EVALSHA commands on the same key as conflicting operations that
can be interleaved, which is correct at the command level but incorrect at
the Redis execution level: Redis guarantees that Lua scripts are atomic
(no other commands can run between the script's Redis calls).

During counterexample exploration, DPOR schedules thread interleavings
where one thread's Lua script runs "between" another's Redis operations.
The counterexample interleaving appears to violate the invariant, but when
reproduced against real Redis, the Lua script atomicity ensures correct
behavior, resulting in 0/10 reproduction.

**Impact:** False positive race condition reports for any library that uses
Redis Lua scripts for atomicity. The detection phase finds a spurious
counterexample, but reproduction confirms it cannot occur. Tests that
expect Lua-script-protected operations to be safe will fail with
"Race condition found" but "Reproduced 0/10 times."

**Workaround:** Set `reproduce_on_failure=0` to skip reproduction, or
accept that the 0/10 reproduction rate indicates a false positive. Tests
affected by this should be left as-is (red) and documented as blocked on
this defect.

---

## Defect #9: Redis DPOR counterexamples fail to reproduce (0/10) for real races

**Status:** Open

**Symptom:** DPOR correctly detects real race conditions in Redis
GET-then-SET patterns, but reproduction against real Redis consistently
returns 0/10. All 28 reference Redis DPOR tests in
`test_integration_redis_dpor.py` use `reproduce_on_failure=0` to work
around this. External library tests confirm the same pattern: races
verified by DPOR (lost updates, TOCTOU, stampedes) all show 0/10
reproduction.

**Minimal reproduction:**

```python
class State:
    def __init__(self):
        r = redis.Redis(port=16399, decode_responses=True)
        r.set("counter", "0")
        r.close()

def increment(state):
    r = redis.Redis(port=16399, decode_responses=True)
    val = int(r.get("counter"))
    r.set("counter", str(val + 1))
    r.close()

def invariant(state):
    r = redis.Redis(port=16399, decode_responses=True)
    result = int(r.get("counter"))
    r.close()
    return result == 2

result = explore_dpor(
    setup=State,
    threads=[increment, increment],
    invariant=invariant,
    detect_io=True,
    reproduce_on_failure=10,
)
assert not result.property_holds          # Race detected ✓
assert result.reproduction_successes == 0  # But 0/10 reproduction ✗
```

DPOR finds the lost-update interleaving (Thread A: GET→SET, Thread B:
GET→SET, both reading "0") in ≤2 interleavings. But all 10 reproduction
attempts succeed with counter==2 (no race observed).

**Root cause:** During DPOR exploration, `patch_redis()` intercepts
`execute_command()` and injects scheduling points (`scheduler.pause()`)
around each Redis command. The DPOR engine uses these IO-level scheduling
points along with bytecode-level points to build the conflict graph and
explore interleavings.

During reproduction, `_reproduce_dpor_counterexample()` (dpor.py:2101)
disables IO-level scheduling:

1. `_set_preload_pipe_fd(-1)` disables C-level IO interception
2. `_ReplayEngine` is a no-op — `report_io_access()` does nothing
3. The replay scheduler enforces the schedule at bytecode granularity
   only, via `_ReplayDporScheduler._wait_for_turn()` at each opcode

The schedule recorded during exploration includes decisions at IO
scheduling points (Redis command boundaries). During replay, these IO
scheduling points no longer exist — the `patch_redis()` interceptor
still calls `_report_redis_access()`, but the no-op `_ReplayEngine`
ignores it. The bytecode-level scheduler replays the thread schedule,
but without IO-level synchronization, the Redis commands execute against
real Redis too fast for the interleaving to manifest. Both threads'
GET→SET sequences complete atomically from Redis's perspective.

Compare with SQL reproduction, which has a fallback: if the initial
replay fails, `_attempt(patch_sql_for_replay=True)` re-enables SQL
interception with row-lock arbitration. No equivalent fallback exists
for Redis.

**Affected libraries:** All Redis-based tests:
- `django-redis` — `get_or_set()` stampede (0/10)
- `walrus` — `Cache.cached()` stampede (0/10), `RedisDict` lost update (0/10)
- `Flask-Caching` — `@cached` stampede (0/10)
- `pottery` — `RedisDict` lost update (0/10), `RedisCounter.update()` (0/10)
- `channels_redis` — capacity overflow TOCTOU detected but untested with
  `reproduce_on_failure>0`

Note: Some races (django-redis `incr_version()` 10/10, walrus
`Model.save()` 10/10, channels_redis `send()` 10/10) DO reproduce. These
succeed because the thread functions contain enough Python bytecode between
Redis commands that the bytecode-level scheduler can force the needed
interleaving without Redis-specific scheduling points.

**Impact:** DPOR's Redis race detection is "detect-only" — it finds
races but cannot confirm them via reproduction. This reduces confidence:
users cannot distinguish between a real race with 0/10 reproduction
(this defect) and a false positive with 0/10 reproduction (defect #8).

**Fix:** Add a `patch_redis_for_replay` fallback in
`_reproduce_dpor_counterexample()`, analogous to `patch_sql_for_replay`.
During replay, re-enable Redis command interception with scheduling
points so the replay scheduler can enforce the exact interleaving at
Redis command boundaries, not just bytecode boundaries.

**Tests:** `tests/test_defect9_redis_reproduction.py`
