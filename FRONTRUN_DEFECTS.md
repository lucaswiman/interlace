# Frontrun Defects

Known issues in the frontrun library discovered during concurrency bug hunting.

---

## #1 — Row-level predicate tracking misses cross-column conflicts (OPEN)

**Severity:** HIGH — causes DPOR to miss real TOCTOU races

**Discovered:** 2026-03-11 during django-registration testing

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

DPOR sees these as **different resources** because the predicate key tuples
differ (`username` vs `id`). It therefore does not detect a read-write
conflict between the SELECT and the UPDATE, and does not explore the
interleaving where both threads read before either writes.

**Impact:** Any TOCTOU race where the ORM SELECTs by one column (e.g.
username, email, slug) and UPDATEs by primary key (id) is invisible to
DPOR. This is the standard Django ORM pattern — `.get(username=...)` returns
an object, then `.save()` issues `UPDATE ... WHERE id = ...`.

**Reproduction:**

```python
# With django_dpor, this returns num_explored=2, property_holds=True
# when it should find the race (both threads activating the same user)
class _State:
    def __init__(self):
        User.objects.filter(username='testuser').delete()
        User.objects.create_user(username='testuser', is_active=False)
        self.results = [None, None]

def _make_fn(i):
    def fn(s):
        user = User.objects.get(username='testuser')  # SELECT WHERE username=...
        if user.is_active:
            s.results[i] = 'already_active'
            return
        user.is_active = True
        user.save()  # UPDATE WHERE id=...
        s.results[i] = 'activated'
    return fn
```

**Expected behavior:** DPOR should recognize that a SELECT on
`auth_user[username='testuser']` and an UPDATE on `auth_user[id='1']`
*may* conflict (they could refer to the same row), and explore the
interleaving where both SELECTs happen before either UPDATE.

**Possible fixes:**

1.  **Fallback to table-level:** Fall back to table-level resource IDs when
    different predicates (different sets of columns) are used for the same
    table across threads. Conservative but correct.
2.  **Alias Resolution (INSERT-tracker):** Use the INSERT-tracker's alias
    resolution to map `id=1` back to the row inserted with
    `username='testuser'`, unifying the resource IDs.
3.  **Mandatory Table-level Conflict:** Treat any read + write on the same
    table as conflicting regardless of row-level predicates.
4.  **SMT Solver (Z3):** Use an SMT solver to check if predicates can overlap
    given schema constraints (e.g., `id=1` and `username='testuser'` can
    overlap if both can be true for the same row). Precise but heavy.
5.  **DB-assisted Resolution (PK Query):** For each statement, query the DB
    (or use `EXPLAIN`) to resolve the predicates to a set of affected
    primary keys. Use PKs as the resource IDs. Reliable but slow.
6.  **Query Rewriting (RETURNING):** Rewrite SELECT/UPDATE to use
    `RETURNING id` (or equivalent) to get the exact affected rows at
    runtime and use those IDs for conflict detection.
7.  **Bloom Filter / Shadow Table:** Maintain an in-memory shadow table of
    which primary keys have been accessed. Requires a way to map every
    predicate to a PK (back to #5).
8.  **Tainted Model Tracking (ORM-level):** In `frontrun.contrib.django`,
    attach the original lookup predicate to the model instance. When
    `.save()` is called, report both the PK predicate AND the original
    lookup predicate to link the resources.
9.  **Conservative Column-Set Partitioning:** Partition row-level access by
    column set. If two operations use different sets of columns in their
    WHERE clauses on the same table, they are treated as conflicting at
    the table level.

**Workaround:** None available within the current frontrun API. The
row-level predicate feature cannot be disabled independently of
`detect_io`. Tests using `django_dpor` with standard ORM patterns
(get-by-natural-key, save-by-pk) will not find TOCTOU races.

---

## #2 — SQL parsing fails for DELETE and INSERT with %s placeholders (FIXED)

**Status:** FIXED in commit 21f97d7

**Severity:** CRITICAL — INSERTs and DELETEs are invisible to DPOR

**Discovered:** 2026-03-11 during django-friendship testing

**Description:**

`parse_sql_access()` failed to extract table names from `DELETE` and
`INSERT` statements that use `%s` parameter placeholders.

**Fix:**
1.  Modified `_regex_parse` to allow `RETURNING` (INSERT) and `IN (` (DELETE)
    to be handled by the fast-path regexes when no subqueries are present.
2.  Added pre-processing to `_sqlglot_parse` to replace `%s` and `%(name)s`
    placeholders with `?` before passing to the full parser, as sqlglot
    default dialect misinterprets `%` as a modulo operator.

---

## #3 — DPOR deadlock in pytest plugin (FIXED in frontrun#84)

**Status:** FIXED in commit 9fb5519

The frontrun pytest plugin used cooperative locks internally, which
caused deadlocks when the DPOR scheduler itself needed to synchronize.
Fixed by switching to real (non-cooperative) internal locks.
