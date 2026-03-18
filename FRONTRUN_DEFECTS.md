# Frontrun Defects

Known issues with the frontrun library are tracked here.

## Defect #9: Redis DPOR counterexamples fail to reproduce (0/10) for real races

**Status:** FIXED (March 2026) — `patch_redis_for_replay` fallback re-enables Redis scheduling during replay

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

---

## Defect #10: Redis DPOR counterexamples still fail to reproduce (0/10) for memoize-style version-check patterns

**Status:** FIXED (March 2026) — `_intercept_pipeline_execute` now checks `_redis_replay_mode`

**Symptom:** After defect #9's `patch_redis_for_replay` fix, most Redis race
patterns now reproduce correctly. The original defect #10 report listed many
0/10 patterns (stampedes, multi-command RMW through libraries), but retesting
reveals that most of these now reproduce 10/10:

| Library | Pattern | Reproduction |
|---------|---------|--------------|
| `django-redis` | `get_or_set()` stampede | **10/10** (was 0/10) |
| `Flask-Caching` | `@cached` stampede | **10/10** (was 0/10) |
| `pottery` | `RedisDict` lost update | **10/10** (was 0/10) |
| `Flask-Caching` | `@memoize` stampede | **0/10** (still broken) |

The only remaining 0/10 pattern is Flask-Caching's `@memoize` decorator.

**Root cause of the residual `@memoize` 0/10:**

The `@memoize` decorator calls `_memoize_version()` which issues
`cache.get_many()` and conditionally `cache.set_many()` (via cachelib's
`RedisCache` backend using `MGET` and pipeline `SET`) to manage a version
key (`_memver`).  This creates a **data-dependent code path**:

- First call (version key absent): `get_many` → `set_many` (pipeline SET)
  → more bytecodes for pipeline creation/execution
- Subsequent calls (version exists): `get_many` only → fewer bytecodes

During DPOR exploration (`detect_io=True`), the schedule records scheduling
points at I/O boundaries only.  For the memoize race, the counterexample
schedule has ~11 entries at Redis command boundaries.

During replay (`detect_io=False` in `_run_dpor_schedule`), the bytecode
tracer adds scheduling points at every opcode.  The 11-entry I/O-level
schedule is consumed by the first 11 bytecodes of the Flask/decorator
overhead (before any Redis command executes).  The `_extend_schedule`
round-robin padding does not enforce the critical interleaving (both
threads' GETs before either SET) at the Redis command level.

The `@cached` decorator (which does NOT have version checking via
`get_many`/`set_many`) reproduces 10/10, confirming the issue is specific
to `_memoize_version`'s data-dependent branching.

Simple stampede patterns (GET → compute → SET with arbitrary amounts of
Python between Redis commands) reproduce 10/10, confirming the defect #9
fix works for direct Redis usage.

**Affected libraries (0/10 reproduction after defect #9 fix):**
- `Flask-Caching` — `@memoize` stampede (0/10)

**Previously affected, now fixed by defect #9 (10/10 reproduction):**
- `django-redis` — `get_or_set()` stampede
- `Flask-Caching` — `@cached` stampede
- `pottery` — `RedisDict` lost update

**Distinguishing from defect #8:** Defect #8 was about false positives from
Lua script atomicity not being modeled. Those are now fixed (EVALSHA commands
no longer generate conflicts). Defect #10 is about real races that DPOR
correctly detects but cannot reproduce. The DPOR counterexample trace shows
a valid interleaving, but the replay engine cannot force that interleaving
against real Redis.

**Minimal reproduction:** Cannot reproduce with raw Redis commands alone.
The 0/10 failure requires Flask-Caching's `@memoize` decorator (or similar
library code that creates data-dependent code paths with `get_many`/`set_many`).
The test includes a Flask-Caching `@memoize` case that demonstrates the 0/10
alongside simple stampede tests that pass 10/10.

**Tests:** `frontrun-bugs/tests/test_defect10_stampede_reproduction.py`

---

## Defect #11: DPOR cannot detect races between two separate lock objects in the same operation

**Status:** Open

**Symptom:** DPOR does not detect races where a compound operation touches
two *different* mutex-protected values in sequence, and a concurrent reader
reads both values between the two writes.

**Minimal reproduction (prometheus_client Summary):**

```python
# Thread A (observe):
with _count._lock:     # Lock 1
    _count._value += 1
# << race window here >>
with _sum._lock:       # Lock 2
    _sum._value += amount

# Thread B (read / collect):
with _count._lock:     # Lock 1
    c = _count._value  # sees _count incremented but ...
with _sum._lock:       # Lock 2
    s = _sum._value    # ... _sum not yet incremented → c > s
```

A stress test (100k iterations, 2 threads) confirms `count > sum` is
observable within ~1000 iterations.

**Root cause:** DPOR creates new scheduling points by detecting conflicts
between thread accesses to the *same* shared object. In the above pattern,
`_count._value` is written by Thread A and read by Thread B, and
`_sum._value` is written by Thread A and read by Thread B. These are two
different objects. DPOR sees:
- Thread A writes `_count._value` (conflict with Thread B's read of `_count._value`)
- Thread A writes `_sum._value` (conflict with Thread B's read of `_sum._value`)

But DPOR does not model the cross-object dependency: "Thread B's read of
`_sum._value` must happen BEFORE Thread A's write to `_sum._value`, even
though the read is interposed AFTER Thread A's write to `_count._value`."
This cross-object ordering constraint requires reasoning about the *gap
between two releases* of different locks, which is not a dependency DPOR
can derive from single-object conflict detection.

**Impact:** Any library that implements a compound update as two sequential
individually-locked operations will have its races missed by DPOR. The test
will pass (green) even when the race is real. The affected code pattern is:
```
with lockA: obj_a.update()   # op 1
with lockB: obj_b.update()   # op 2 (should be atomic with op 1)
```
where a concurrent reader reads `obj_a` and `obj_b` across the two operations.

**Affected libraries:**
- `prometheus_client` — `Summary.observe()` increments `_count` then `_sum`
  separately; a reader can see `_count > _sum`.
  See `libraries/prometheus_client/tests/test_summary_observe_race.py`.

**Workaround:** None currently. Manual stress testing can confirm the race
exists, but DPOR/`explore_interleavings` will not find it automatically.

---

## Defect #12: DPOR false positive — STORE_ATTR on distinct str-subclass instances treated as conflicting writes

**Status:** Open

**Symptom:** DPOR reports a "write-write conflict" when two threads each
create a fresh instance of a `str` subclass (e.g., `BoundEvent` in
python-statemachine) and both write to an instance attribute of the same name
(e.g., `instance.id = id`) in `__new__`. Each thread writes to its own
distinct object, but DPOR reports them as conflicting. Reproduction rate is
0/10, confirming the conflict is spurious.

**Minimal reproduction:**

```python
from statemachine import StateMachine, State
from frontrun.dpor import explore_dpor

class _State:
    def __init__(self):
        class CyclingMachine(StateMachine):
            s1 = State(initial=True)
            s2 = State()
            s3 = State()
            cycle = s1.to(s2) | s2.to(s3) | s3.to(s1)
        self.sm = CyclingMachine()

def thread_fn(s):
    s.sm.cycle()  # internally creates a fresh BoundEvent (str subclass) per call

r = explore_dpor(
    setup=_State,
    threads=[thread_fn, thread_fn],
    invariant=lambda s: True,
    detect_io=False,
)
# DPOR incorrectly reports: "Write-write conflict: threads 0 and 1 both wrote to id"
# at event.py:104 (instance.id = id in Event.__new__)
# Reproduction: 0/10
```

**Reproduction trace:**

```
Thread 0 | event.py:104              instance.id = id
         | [write BoundEvent.id]
Thread 1 | event.py:104              instance.id = id
         | [write BoundEvent.id]
Reproduced 0/10 times (0%)
```

**Root cause:** DPOR's bytecode tracer tracks attribute writes via
`STORE_ATTR` opcodes. For `str` subclass instances created via `__new__`,
the tracer appears to use the attribute *name* (`'id'`) as the conflict key
without reliably distinguishing the object identity. Two threads writing to
the same attribute name on different (freshly-created) objects are incorrectly
classified as a write-write conflict.

This likely occurs because `str` subclasses are immutable at the C level
(`tp_hash` is set), and the object identity used for conflict tracking is
derived from the interned string value rather than the instance address, or
the shadow stack does not correctly track the target object through
`super().__new__(cls, id)` in a str subclass.

**Impact:** Any library that creates fresh `str` subclass instances inside
thread functions (a common pattern for event/message types) will generate
false positive write-write conflicts when two threads both instantiate these
types with attributes of the same name. The false positive causes DPOR to
explore a spurious counterexample schedule, fails with 0/10 reproduction,
and makes the test red even for a correct (non-racy) library.

**Affected libraries:**
- `python-statemachine` — `BoundEvent` (a `str` subclass) is created fresh
  on every `sm.cycle()` call; DPOR falsely reports a race on `BoundEvent.id`.
  See `libraries/python-statemachine/tests/test_concurrent_transitions.py`.
- `django-cachalot` — `_unset_raw_connection()` in `monkey_patch.py` writes
  `compiler.connection.raw = False` before every SQL operation. DPOR falsely
  reports a write-write conflict on `DatabaseWrapper.raw` even though each
  thread has its own distinct `DatabaseWrapper` instance (confirmed by
  checking `id(connections['default'])` in each thread). The false positive
  occurs for all django-cachalot tests and prevents DPOR from exploring the
  real SQL/cache race (invalidate-before-write TOCTOU).
  See `libraries/django-cachalot/tests/test_cachalot_race.py`.

**Workaround:** None. Tests affected by this defect will be red with 0/10
reproduction rate.

---

## Defect #13: DPOR false positive — SQL row-level locking not modeled for FOR UPDATE SKIP LOCKED and single-statement DML

**Status:** Open

**Symptom:** DPOR reports race conditions (invariant violations) for SQL
patterns protected by PostgreSQL row-level locking, where the invariant
cannot actually be violated in real PostgreSQL. The reported interleavings
are impossible because PostgreSQL serializes concurrent access at the row
level within individual SQL statements.

**Minimal reproduction (pgmq-style dequeue):**

```python
# Thread A (concurrent dequeue):
conn_a.execute("""
    WITH cte AS (SELECT msg_id FROM queue WHERE vt <= now()
                 ORDER BY msg_id LIMIT 1 FOR UPDATE SKIP LOCKED)
    UPDATE queue SET vt = now() + '30s', read_ct = read_ct + 1
    FROM cte WHERE queue.msg_id = cte.msg_id
    RETURNING msg_id
""")

# Thread B (concurrent dequeue — same pattern):
conn_b.execute("""  # same SQL
""")
```

DPOR explores an interleaving where both Thread A and Thread B execute
their dequeue statements concurrently and both return the same `msg_id`,
claiming the at-most-once invariant is violated. The invariant check also
fails with 10/10 reproduction in DPOR's replay, but manual stress testing
with 100+ iterations shows this never occurs in real PostgreSQL.

Similarly for concurrent DELETEs:

```python
# Thread A:
conn_a.execute("DELETE FROM queue WHERE msg_id = :id RETURNING msg_id", id=1)
# Thread B:
conn_b.execute("DELETE FROM queue WHERE msg_id = :id RETURNING msg_id", id=1)
```

DPOR explores an interleaving where both succeed (both DELETE the same row),
but in real PostgreSQL the second DELETE serializes after the first commit
and finds the row already gone.

**Root cause:** DPOR's SQL interception model (``_sql_cursor.py``) treats
all SQL statements that touch the same table as potentially conflicting,
creating conflict arcs between them. This is correct for statements that
actually read-then-write in separate logical steps (e.g., SELECT + UPDATE
in separate statements). However, for:

1. `FOR UPDATE SKIP LOCKED` in a CTE: The row lock acquisition and the
   check for a locked row are atomic within a single PostgreSQL statement.
   DPOR cannot model this intra-statement atomicity and explores interleavings
   where Thread A's lock acquisition and Thread B's lock check are interleaved.

2. Single-statement DELETE: Two concurrent DELETEs of the same row serialize
   at the row level (the second blocks until the first commits, then returns
   nothing). DPOR doesn't model this blocking/serialization and explores the
   impossible interleaving where both succeed.

**Impact:** Any test that uses ``FOR UPDATE SKIP LOCKED`` (common in
message queue implementations, task queues, and job dispatchers) will
generate false positive race condition reports. The tests will be red
with 10/10 reproduction rate (DPOR replays the impossible schedule
against real PostgreSQL, where row locking causes the "violation" to
manifest differently than expected).

**Affected libraries:**
- `PGMQ` — `pgmq.read()` uses `FOR UPDATE SKIP LOCKED`; concurrent dequeue
  test shows false positive. Concurrent DELETE test also shows false positive.
  See `libraries/PGMQ/tests/test_pgmq_race.py`.

**Workaround:** None currently. Tests affected by this defect will be red
with 10/10 reproduction rate. Manual stress testing confirms the race cannot
occur in real PostgreSQL.

## Defect #11: GC destructor deadlock with Redis connection pool (CooperativeRLock missing reentrancy guard)

**Status:** FIXED (March 2026) — `_in_dpor_machinery()` guard added to `CooperativeRLock.release()`

**Symptom:** Tests that create `redis.Redis` objects stored in `_State` (and thus
GC'd between DPOR runs) time out after 120s with a deadlock in frontrun's
cooperative lock machinery. The deadlock occurs even when per-connection instances
are created in `_State.__init__()` rather than inside thread functions, because
the old `_State` objects are collected when a new `_State` is created for the next
interleaving.

**Stack trace (simplified):**

```
dpor-0: _report_and_wait → _process_opcode → _get_instructions
  → [GC triggers redis.Redis.__del__]
  → connection_pool.disconnect() → with self._lock (CooperativeRLock.__enter__)
  → body executes → CooperativeRLock.__exit__ → release()
  → self._report("lock_release") → reporter() → with scheduler._condition
  ← DEADLOCK (scheduler._condition already held by _report_and_wait)
```

**Root cause:** `_report_and_wait` (dpor.py:400) holds `scheduler._condition`
and sets `_scheduler_tls._in_dpor_machinery = True` before calling
`_process_opcode`. GC fires during `_get_instructions` inside `_process_opcode`.
The GC calls `redis.Redis.__del__` → `connection_pool.disconnect()` → `with
self._lock` (the pool's `threading.RLock`, patched to `CooperativeRLock`).

`CooperativeRLock.acquire()` checks `_in_dpor_machinery()` → True → falls back
to real blocking → succeeds. The `with` body executes. Then `CooperativeRLock.
__exit__` → `release()` → calls `self._report("lock_release")` WITHOUT checking
`_in_dpor_machinery()`. The `_report()` method then calls `reporter()` → `with
scheduler._condition` → DEADLOCK.

The fix for Defect #7 added a reentrancy guard to `CooperativeLock.release()`:
```python
if _in_dpor_machinery():
    self._lock.release()
    return
```
But `CooperativeRLock.release()` does NOT have this guard, only the `_report()`
method has the guard (which is too late — it sets the flag but immediately tries
to acquire `_condition`).

**Fix:** Add the same `_in_dpor_machinery()` guard to `CooperativeRLock.release()`
before the `self._report("lock_release")` call:
```python
def release(self) -> None:
    ...
    if self._count == 0:
        ...
        if _in_dpor_machinery():      # ADD THIS GUARD
            self._lock.release()
            return
        self._report("lock_release")
        self._lock.release()
```

**Affected libraries:**
- `redis-py` — all DPOR tests deadlock because redis-py's `ConnectionPool` uses
  `threading.RLock()` (patched to `CooperativeRLock`). When `_State` objects are
  collected between DPOR runs, their Redis connections trigger `__del__` →
  `disconnect()` → `with self._lock` cycle.
  See `libraries/redis-py/tests/test_redis_py_race.py`.

**Workaround:** None reliably. Calling `r.close()` explicitly before returning
from thread functions helps if the GC fires at the right time, but does not
guarantee the connection pool's RLock is released before `_process_opcode` is
called.

---

## Defect #14: Some SQL DPOR counterexamples fail to reproduce (0/10)

**Status:** Open

**Symptom:** DPOR correctly detects real SQL race conditions (lost updates,
write-write conflicts on the same table), but reproduction against real
PostgreSQL returns 0/10 for certain tests, even though the race patterns
are clearly valid and the DPOR counterexample interleavings are correct.
Many other SQL races reproduce at 10/10, so this is not a systemic SQL
reproduction issue — it appears to be specific to certain test/library
patterns.

**Affected libraries:**
- `dagster` — `handle_run_event()` lost update on `run_body` (0/10).
  The race is acknowledged by dagster's own source comments.
- `getpaid` — `CallbackDetailView.post()` concurrent payment callbacks (0/10)
- `rq` — scheduled job double-enqueue (0/10)

**Root cause:** Unknown. The `patch_sql_for_replay` fallback exists and
works for many SQL tests, but fails to force the needed interleaving
for these specific libraries. The issue may be related to the complexity
of the SQL operations involved (e.g., dagster's SQLAlchemy reflection
queries, multiple connections, or transaction boundaries).

**Impact:** These specific SQL races are "detect-only" — DPOR finds valid
counterexample interleavings but cannot confirm them. The race patterns
are real (verified by code inspection), but the 0/10 rate reduces
confidence.

**Workaround:** Rely on code inspection and the DPOR counterexample trace
to confirm the race is real.
