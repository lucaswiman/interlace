# Frontrun Defects

Known issues in the frontrun library discovered during concurrency bug hunting.

---

## #1 — Row-level predicate tracking misses cross-column conflicts (FIXED in frontrun#86)

**Status:** FIXED in commit b335c4a

**Severity:** HIGH — caused DPOR to miss real TOCTOU races

**Discovered:** 2026-03-11 during django-registration testing

**Fixed:** 2026-03-12 in frontrun commit b335c4a (PR #86)

**Description:**

When `detect_io=True` and row-level predicate extraction is active, DPOR
builds resource IDs that include the WHERE-clause predicates. A SELECT
filtered by one column and an UPDATE filtered by a different column produce
different resource IDs, even when they refer to the same row.

Example from Django ORM:

```sql
-- Thread A (or B): SELECT by username
SELECT ... FROM "auth_user" WHERE "auth_user"."username" = 'testuser' LIMIT 21
-- Resource ID: sql:auth_user:(('username', 'testuser'),)

-- Thread A (or B): UPDATE by id
UPDATE "auth_user" SET "is_active" = true WHERE "auth_user"."id" = 1
-- Resource ID: sql:auth_user:(('id', '1'),)
```

**Fix:** Frontrun now falls back to table-level resource IDs when different
predicates are used for the same table across threads. This correctly
detects cross-column conflicts.

**Verification (2026-03-12):** After the fix, the following tests now
successfully detect their race conditions:

| Library | Interleavings | Conflict type |
|---------|--------------|---------------|
| django-registration | 2 | write-write on `auth_user` |
| django-q2 | 2 | write-write on `django_q_task` |
| django-userena | 11 | lost update on `userena_userenasignup` |
| django-tenants | 4 | write-write on `test_domain` |

**Note:** django-constance was also blocked on this defect, but its test
invariant has a separate issue (see Test Issues section below) that prevents
race detection regardless of DPOR behavior.

---

## #2 — SQL parsing fails for DELETE and INSERT with %s placeholders (FIXED in frontrun#85)

**Status:** FIXED in commit fe0995e

**Severity:** CRITICAL — INSERTs and DELETEs were invisible to DPOR

**Discovered:** 2026-03-11 during django-friendship testing

**Fixed:** 2026-03-12 in frontrun commit fe0995e (PR #85)

**Description:**

`parse_sql_access()` failed to extract table names from `DELETE` and
`INSERT` statements that use `%s` parameter placeholders (Django's
`pyformat` paramstyle). The regex fast-path bailed on `IN (` and
`RETURNING` keywords, falling through to `_sqlglot_parse()`. sqlglot
could not parse `%s` as valid SQL, so parsing failed and returned empty
read/write table sets.

**Fix:** The regex fast-path now handles `RETURNING` for INSERT and `IN (`
for DELETE, and `%s` placeholders are pre-processed before passing to
sqlglot.

**Verification (2026-03-12):**

```python
from frontrun._sql_parsing import parse_sql_access

# DELETE: now works
parse_sql_access('DELETE FROM "t" WHERE "t"."id" IN (%s)')
# → SqlAccessResult(read_tables={'t'}, write_tables={'t'}, ...)

# INSERT: now works
parse_sql_access('INSERT INTO "t" ("a", "b") VALUES (%s, %s) RETURNING "t"."id"')
# → SqlAccessResult(read_tables=set(), write_tables={'t'}, ...)
```

After the fix, the following tests now successfully detect their races:

| Library | Interleavings | Conflict type |
|---------|--------------|---------------|
| django-friendship | 17 | double-accept on FriendshipRequest |
| django-hitcount | 11 | duplicate hit via concurrent hit_count() |
| django-tenants | 4 | (also required defect #1 fix) |

**Note:** django-rest-knox, authlib, and dj-stripe were also blocked on
this defect. Their tests now *run* (SQL parsing works), but DPOR does not
detect the races due to defect #4 (phantom read problem — see below).

---

## #3 — DPOR deadlock in pytest plugin (FIXED in frontrun#84)

**Status:** FIXED in commit 9fb5519

The frontrun pytest plugin used cooperative locks internally, which
caused deadlocks when the DPOR scheduler itself needed to synchronize.
Fixed by switching to real (non-cooperative) internal locks.

---

## #4 — DPOR does not detect phantom reads (SELECT + INSERT conflicts) (FIXED on frontrun branch claude/fix-phantom-read-detection-UvfGY)

**Status:** FIXED in commit 80d8c85 (branch `claude/fix-phantom-read-detection-UvfGY`)

**Severity:** HIGH — DPOR missed check-then-insert TOCTOU races

**Discovered:** 2026-03-12 during re-testing after defect #1/#2 fixes

**Fixed:** 2026-03-12 in frontrun commit 80d8c85

**Description:**

DPOR used row-level (or table-level fallback) conflict tracking. When
thread A performed a SELECT that returned existing rows (or checked a COUNT),
and thread B performed an INSERT that added a new row to the same table,
DPOR did not detect a conflict. The new row did not exist at the time of
thread A's SELECT, so there was no row-level read-write conflict.

**Fix:** Sequence-number tracking. Each table gets a virtual `:seq` resource:
- SELECT (pure-read) tables report READ on `sql:<table>:seq`
- INSERT tables report WRITE on `sql:<table>:seq`
- DELETE tables report WRITE on `sql:<table>:seq`
- UPDATE is excluded (doesn't change table membership)

This creates READ-WRITE conflicts between SELECT and INSERT/DELETE on the
same table, detecting phantom read races.

**Verification (2026-03-12):** After the fix, the following tests now
successfully detect their race conditions:

| Library | Interleavings | Conflict type |
|---------|--------------|---------------|
| django-rest-knox | 2 | write-write on `knox_authtoken:seq` |
| authlib | 2 | write-write on `test_authlib_authcode` (requires NullPool — see defect #5) |
| dj-stripe | 2 | write-write on `djstripe_event` (requires `lock_timeout`) |

**Note:** authlib uses SQLAlchemy (not Django), so it requires `NullPool`
to avoid defect #5 (connection pooling). dj-stripe requires `lock_timeout`
to work around defect #6 (cooperative scheduler deadlocks with PG row locks).

---

## #5 — SQL cursor patching invisible to pooled connections (OPEN)

**Severity:** MEDIUM — DPOR misses all SQL on connections created before `patch_sql()`

**Discovered:** 2026-03-12 during authlib testing after defect #4 fix

**Description:**

`patch_sql()` monkey-patches `psycopg2.connect` (and other drivers) to
inject a traced cursor factory. However, connections created *before*
`patch_sql()` is called — typically by SQLAlchemy's connection pool during
module import or fixture setup — retain their original (untraced) cursor
factory. When threads later obtain sessions from the pool, they reuse these
pre-existing connections, and their SQL operations are invisible to DPOR.

This only affects `explore_dpor()` with non-Django SQL libraries (SQLAlchemy,
Flask-SQLAlchemy, etc.). Django's `django_dpor` wrapper closes and reopens
connections per thread, which creates fresh connections through the patched
`psycopg2.connect`.

**Example:** The authlib test uses SQLAlchemy with a module-level engine.
The engine's connection pool creates connections during the `_pg_available`
fixture (before `explore_dpor` calls `patch_sql`). When thread functions
create sessions, they get pooled connections with untraced cursors. DPOR
sees `num_explored=1` — no SQL conflicts detected.

**Workaround:** Use `NullPool` for SQLAlchemy engines in DPOR tests:

```python
from sqlalchemy.pool import NullPool

_engine = create_engine(_DB_URL, poolclass=NullPool)
```

This forces every session to create a fresh connection through the patched
`psycopg2.connect`, making SQL operations visible to DPOR.

**Possible fix:** `patch_sql()` could also patch `psycopg2.extensions.cursor`
class methods directly (not just `connect`), so existing connections get traced
cursors retroactively. Alternatively, `explore_dpor` could invalidate
SQLAlchemy connection pools before starting exploration.

---

## #6 — DPOR cooperative scheduler deadlocks with PostgreSQL row locks (OPEN)

**Status:** OPEN

**Severity:** MEDIUM — DPOR hangs when exploring interleavings that cause PG row lock contention

**Discovered:** 2026-03-12 during dj-stripe testing after defect #4 fix

**Scope:** General `explore_dpor` bug — NOT Django-specific. Any `explore_dpor`
test where two threads contend on the same PG row lock will deadlock. The
`django_dpor` wrapper has a `lock_timeout` parameter that works around this,
but `explore_dpor` itself does not.

**Minimal reproduction:** `frontrun-bugs/tests/test_defect6_scheduler_deadlock.py`

```bash
# HANGS (demonstrates the bug):
PYTHONPATH=libraries .venv/bin/frontrun pytest \
  frontrun-bugs/tests/test_defect6_scheduler_deadlock.py::test_deadlock_without_lock_timeout \
  -v --timeout=30

# PASSES (demonstrates the workaround):
PYTHONPATH=libraries .venv/bin/frontrun pytest \
  frontrun-bugs/tests/test_defect6_scheduler_deadlock.py::test_workaround_with_lock_timeout \
  -v --timeout=30
```

**Description:**

DPOR's cooperative scheduler controls thread execution by suspending and
resuming threads at scheduling points (SQL statements, lock operations, etc.).
When a racy interleaving causes two threads to contend on the same PostgreSQL
row lock, the two scheduling systems deadlock:

1. Thread A executes INSERT (acquires PG row lock via UNIQUE constraint)
2. DPOR suspends thread A at the next scheduling point (e.g., `conn.commit()`)
3. DPOR resumes thread B
4. Thread B executes INSERT on the same row — PG blocks thread B in the kernel
   waiting for thread A's row lock
5. Thread B never reaches a DPOR scheduling point (blocked on PG socket read)
6. DPOR waits for thread B to yield — **deadlock**

Stack traces from the reproduction confirm this exactly:

```
# Thread frontrun-1: blocked in kernel waiting for PG row lock
  File "test_defect6_scheduler_deadlock.py", line 103, in _thread_fn
    cur.execute(                    # INSERT — PG blocks waiting for thread 0's lock

# Thread frontrun-0: blocked in DPOR scheduler waiting for its turn
  File "test_defect6_scheduler_deadlock.py", line 107, in _thread_fn
    conn.commit()                   # hits DPOR scheduling point
  File "frontrun/bytecode.py", line 263, in trace
    scheduler.wait_for_turn()       # DPOR waiting — but thread 1 never yields
```

This is inherent to the cooperative scheduling model: DPOR cannot preempt a
thread that is blocked inside a kernel-level wait (PostgreSQL's lock manager).
The irony is that this deadlock only occurs on the racy interleavings DPOR
is specifically trying to explore — the ones where both threads contend on
the same resource.

**Workaround:** Set PostgreSQL's `lock_timeout` so the blocked thread gets an
error instead of waiting forever. `django_dpor` has a `lock_timeout` parameter
for this; with raw `explore_dpor`, each thread must `SET lock_timeout` on its
own connection:

```python
# With django_dpor (has built-in lock_timeout support):
result = django_dpor(
    setup=_State,
    threads=[thread_fn_0, thread_fn_1],
    invariant=_invariant,
    lock_timeout=2000,  # milliseconds
)

# With raw explore_dpor (must SET lock_timeout manually per connection):
def _thread_fn(state):
    conn = psycopg2.connect(dsn)
    with conn.cursor() as cur:
        cur.execute("SET lock_timeout = '2s'")
        # ... rest of thread logic
```

**Possible fix:** `explore_dpor` should accept a `lock_timeout` parameter and
set it automatically on any PostgreSQL connections it detects (via the SQL
interception layer). Alternatively, DPOR could detect when a thread is blocked
on a PG row lock (e.g., by monitoring socket readability with a timeout) and
treat the block itself as evidence of a conflict, rather than waiting for the
thread to yield.
