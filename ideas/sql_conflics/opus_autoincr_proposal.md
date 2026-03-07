# Autoincrement / Sequence Non-Determinism in DPOR

## The Problem

When two threads concurrently INSERT into a table with an autoincrement
primary key (Postgres `SERIAL`/`IDENTITY`, SQLite `AUTOINCREMENT`, MySQL
`AUTO_INCREMENT`), the assigned IDs depend on execution order:

```
Interleaving 1:         Interleaving 2:
  Thread A INSERT -> 1    Thread B INSERT -> 1
  Thread B INSERT -> 2    Thread A INSERT -> 2
```

This creates three compounding problems for frontrun's DPOR:

1. **Downstream reference divergence.** Code that uses the returned ID
   (`cursor.lastrowid`, `RETURNING id`, SQLAlchemy's
   `inserted_primary_key`) will pass different values to subsequent
   queries in different interleavings. A child-table INSERT like
   `INSERT INTO orders (user_id) VALUES (<returned_id>)` refers to
   different rows depending on scheduling.

2. **Invariant fragility.** Users writing invariants need to reason
   about "the set of rows that exist" rather than specific ID values.
   `assert user.id == 1` breaks across interleavings. This is a
   usability trap.

3. **Row-level ObjectId instability.** After an INSERT, subsequent
   operations on the newly-created row get a row-level ObjectId
   containing the auto-assigned PK value. In a different interleaving,
   the "same" logical row has a different PK, so DPOR sees it as a
   *different* resource. This can cause both false positives (spurious
   conflicts between logically-independent operations that happen to
   reuse an ID) and false negatives (missed conflicts because the
   "same" row has different ObjectIds across interleavings).

Note that the current system handles the simplest cases fine: two bare
INSERTs to the same table always conflict at table-level (INSERT has no
WHERE clause, so row-level never kicks in). The problem is what happens
*after* the INSERT, when code uses the auto-generated ID.

---

## Good Ideas

### 1. Post-INSERT ID Capture with Logical Aliases

After a patched `cursor.execute()` runs an INSERT, inspect
`cursor.lastrowid` (or the `RETURNING` clause result) to learn the
assigned ID. Record a mapping from `(thread_id, table, insert_sequence_num)`
to the concrete ID. When subsequent operations reference that ID, the
DPOR reporting layer can translate to a *logical* resource ID like
`sql:users:logical_insert_0_thread_a` instead of `sql:users:(('id','42'))`.

Two threads that each INSERT and then UPDATE "their own row" would get
distinct logical ObjectIds regardless of the concrete ID values,
correctly marking them as independent.

**Pros:** No user-facing API changes. Captures the actual semantics
(each thread is working on "its own" newly-created row). Works with
existing DPOR engine -- just changes the string fed to
`_make_object_key`.

**Cons:** Requires heuristic matching of "this WHERE id=42 refers to
the row I just inserted." Breaks down when IDs are passed between
threads or stored in shared data structures. `lastrowid` semantics vary
across drivers (psycopg2 doesn't support it; you need RETURNING).

**Complexity:** ~200 lines in `_sql_cursor.py` + 30 tests. Moderate.

### 2. Sequence-as-Resource: Model the Counter Itself

Treat the autoincrement sequence as a separate DPOR resource. When an
INSERT targets a table with a known autoincrement column, report an
additional write access to `sql:seq:users_id` (or whatever the sequence
name is). Two concurrent INSERTs both write to the sequence resource,
creating a conflict. A SELECT that calls `currval()` or `lastval()`
reads the sequence resource.

This doesn't solve the downstream-reference problem, but it gives DPOR
an accurate model of the *actual* shared state (the sequence counter).
Combined with schema introspection (already partially implemented for
FK analysis), autoincrement columns can be detected at connection time.

**Pros:** Conceptually clean -- models the real shared state. Ensures
DPOR explores orderings around concurrent INSERTs even if table-level
analysis somehow misses the conflict. Pairs naturally with
`information_schema.columns` introspection.

**Cons:** Doesn't help with the downstream ID problem. May be
redundant -- two INSERTs to the same table already conflict at
table-level. Adds noise to conflict traces.

**Complexity:** ~50 lines + 10 tests. Low.

### 3. Document the Pattern: Explicit IDs in Test Fixtures

Instead of solving autoincrement at the library level, document that
concurrent-INSERT tests should pre-allocate IDs. The test setup assigns
IDs explicitly (or uses UUIDs generated in the setup function before
threads start), so the auto-generated values never matter:

```python
def setup():
    # Pre-create rows so threads UPDATE rather than INSERT
    with Session(engine) as session:
        session.add(User(id=1, name="alice", login_count=0))
        session.add(User(id=2, name="bob", login_count=0))
        session.commit()
    return State(alice_id=1, bob_id=2)

def thread_a(state):
    with Session(engine) as session:
        user = session.get(User, state.alice_id)
        ...
```

This sidesteps the entire problem. Most real concurrency bugs are in
the read-modify-write pattern on *existing* rows, not in the INSERT
itself. When the INSERT is the thing being tested, the user can use
UUIDs or explicit IDs.

**Pros:** Zero library complexity. Works today. Matches how most
production code is structured (INSERTs are rarely the concurrent
bottleneck). Makes tests easier to reason about.

**Cons:** Shifts burden to the user. Doesn't help users who genuinely
need to test concurrent INSERTs with auto-IDs. Could feel like a
cop-out.

**Complexity:** Documentation only.

### 4. RETURNING-Clause Injection for INSERT Statements

When the SQL parser detects an INSERT without a RETURNING clause on a
Postgres connection, automatically append `RETURNING <pk_column>` (the
PK column is known from schema introspection). The patched cursor
captures the returned ID and stores it in thread-local state. This gives
frontrun reliable access to the assigned ID without depending on
driver-specific `lastrowid` behavior.

The captured ID feeds into Idea 1 (logical aliases) or is simply
recorded for diagnostic purposes in the DPOR trace output ("Thread A
inserted users.id=42, Thread B inserted users.id=43").

**Pros:** Works uniformly across all Postgres drivers (psycopg2,
psycopg3, asyncpg). Doesn't change query semantics (RETURNING is
side-effect-free). Gives the user actionable info in failure traces.

**Cons:** Postgres-only (SQLite and MySQL have different mechanisms).
Modifying the user's SQL is invasive and could break edge cases (INSERT
... ON CONFLICT, INSERT ... SELECT). Needs careful handling of
executemany.

**Complexity:** ~80 lines + 20 tests. Moderate.

### 5. Invariant Helpers for Order-Independent Assertions

Provide utility functions that make it easy to write invariants that
don't depend on specific ID values:

```python
from frontrun.sql_helpers import row_set, row_count

def invariant(state):
    with Session(engine) as s:
        # "There should be exactly 2 users" -- doesn't care about IDs
        assert row_count(s, User) == 2
        # "The set of names should be {alice, bob}" -- order-independent
        assert row_set(s, User.name) == {"alice", "bob"}
        # "Total login_count across all users should be 2"
        assert sum(u.login_count for u in s.query(User).all()) == 2
```

This doesn't solve the DPOR-internal problem but makes the user-facing
problem (writing correct invariants) much easier.

**Pros:** Purely additive. Useful regardless of whether we solve the
internal problem. Helps users fall into the pit of success.

**Cons:** Doesn't address the ObjectId instability problem. Users still
need to understand *why* they can't use `assert user.id == 1`.

**Complexity:** ~50 lines + examples in docs. Low.

---

## Bad Ideas

### 1. Deterministic Sequence Mocking

Replace the database's autoincrement mechanism with a Python-side
counter that assigns IDs deterministically based on thread identity or
insertion order. E.g., thread A always gets odd IDs, thread B gets even.

**Why it's bad:** This fundamentally changes what's being tested. The
whole point of frontrun is to test real code against real databases.
Mocking the sequence means you're testing against a fake database that
doesn't exist in production. Subtle bugs that depend on ID ordering or
gaps (e.g., `ON CONFLICT DO UPDATE` with specific IDs, or code that
assumes IDs are monotonically increasing) would be invisible.
Additionally, database-side triggers, FK constraints, and CHECK
constraints all see the real ID, so you'd need to mock those too. It's
turtles all the way down.

### 2. Require Deterministic Test Setup with Explicit Starting Points

Force every test to include a `sequence_start` parameter that resets the
database sequence to a known value before each interleaving. Before each
DPOR exploration, run `ALTER SEQUENCE users_id_seq RESTART WITH 1000`
(or equivalent).

**Why it's bad:** Requires DDL permissions that test users may not have.
Doesn't work with SQLite (no ALTER SEQUENCE). Doesn't work with
connection pooling (the RESTART might not be visible to other
connections in the same pool). Creates ordering dependencies between
tests. Imposes a mental model where the user has to think about sequence
state, which is exactly the low-level detail they're trying to avoid.
This is the approach a Gemini-style "make the user handle it" design
would gravitate toward and it's hostile to anyone who just wants to
write `explore_dpor(setup, threads, invariant)` and have it work.

### 3. Global INSERT Serialization

Treat all INSERT statements as conflicting with all other INSERT
statements, regardless of table. Two threads doing `INSERT INTO users`
and `INSERT INTO logs` would be forced to serialize.

**Why it's bad:** Massively over-conservative. The whole point of
table-level conflict detection is to avoid exactly this. In a system
with N tables, this turns O(1) independent INSERTs into O(N^2) forced
interleavings. It would make DPOR unusable for any test that inserts
into multiple tables. And it doesn't even help with the core problem --
two INSERTs to the *same* table already conflict at table-level.

### 4. Rewrite All Queries to Use UUIDs Instead of Autoincrement

At the cursor patching layer, intercept CREATE TABLE statements and
rewrite autoincrement columns to UUID columns. Intercept INSERT
statements and inject `uuid.uuid4()` values. Rewrite all downstream
references.

**Why it's bad:** This is query rewriting on steroids. It changes the
schema, which breaks FK constraints, indexes, and any code that assumes
integer IDs. It changes the type of every PK column from int to UUID,
which breaks ORM model definitions, comparison operations, and API
contracts. It doesn't work for tables you didn't create (third-party
schemas). And UUIDs aren't even deterministic -- you'd still have the
same problem unless you also mock `uuid.uuid4()`, at which point you've
replaced one non-determinism with another.

### 5. Snapshot-and-Diff: Compare Database State Structurally

After each interleaving, dump the entire database state, normalize it
(sort rows by non-PK columns, strip auto-generated IDs), and compare
the normalized snapshots to detect anomalies. Two interleavings that
produce the same normalized state are considered equivalent.

**Why it's bad:** O(rows * tables) per interleaving, which is already
expensive, but the real killer is that it's semantically wrong.
Autoincrement IDs *matter* -- they're foreign keys in other tables,
they're returned to API callers, they're stored in caches. Stripping
them away means you can't detect bugs where the wrong ID propagates.
It also can't detect bugs where the *count* of rows is correct but the
*assignment* of data to IDs is wrong (e.g., Alice's data ends up under
Bob's ID). And it requires the invariant to be defined as "equivalent
DB state" rather than arbitrary Python predicates, which is far less
expressive than frontrun's current `invariant` callback.

### 6. Thread-Affine Sequence Ranges

Pre-partition the ID space: thread 0 gets IDs 1-1000, thread 1 gets
1001-2000, etc. Set this up via `ALTER SEQUENCE ... START WITH ...
INCREMENT BY ...` before each exploration.

**Why it's bad:** Same DDL permission issues as Bad Idea 2. Also
introduces artificial ID gaps that don't exist in production, which can
mask bugs that depend on contiguous IDs. Breaks if a thread does more
than 1000 inserts (or whatever the partition size is). Doesn't work with
more than a handful of threads. And the partitioned sequences themselves
have concurrency semantics -- on Postgres, `nextval()` is
non-transactional and never rolls back, so a failed INSERT still
consumes a sequence value. Mimicking this in a partitioned scheme
requires understanding the exact sequence semantics of each database,
which is a nightmare.

---

## Discussion

The honest assessment is that autoincrement non-determinism is a
**fundamental mismatch** between DPOR's model (same operations,
different orderings, compare outcomes) and database reality (ordering
changes the identity of created objects). DPOR assumes that the
*operations* are fixed and only the *schedule* varies. But with
autoincrement, the schedule changes the operations themselves --
specifically, it changes what ID subsequent operations reference.

The pragmatic path is probably a combination of:

- **Idea 3** (documentation + guidance) as the immediate default
- **Idea 5** (invariant helpers) to reduce friction
- **Idea 4** (RETURNING injection) for Postgres to capture IDs in traces
- **Idea 1** (logical aliases) as a longer-term investment if users
  actually hit the ObjectId instability problem in practice

The key insight is that most real concurrency bugs involving databases
are read-modify-write races on *existing* rows (the lost-update
pattern), not races between concurrent INSERTs. The autoincrement
problem is real but narrow. Solving it perfectly requires either
changing what DPOR means (comparing outcomes modulo ID renaming, which
is a research problem) or constraining the user's code (explicit IDs,
which is practical but unglamorous).
