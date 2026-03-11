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

1. Fall back to table-level resource IDs when different predicates are
   used for the same table across threads (conservative but correct).
2. Use the INSERT-tracker's alias resolution to map `id=1` back to the
   row inserted with `username='testuser'`, unifying the resource IDs.
3. Treat any read + write on the same table as conflicting regardless
   of row-level predicates (loses precision but guarantees soundness).

**Workaround:** None available within the current frontrun API. The
row-level predicate feature cannot be disabled independently of
`detect_io`. Tests using `django_dpor` with standard ORM patterns
(get-by-natural-key, save-by-pk) will not find TOCTOU races.

---

## #2 — SQL parsing fails for DELETE and INSERT with %s placeholders (OPEN)

**Severity:** CRITICAL — INSERTs and DELETEs are invisible to DPOR

**Discovered:** 2026-03-11 during django-friendship testing

**Description:**

`parse_sql_access()` fails to extract table names from `DELETE` and
`INSERT` statements that use `%s` parameter placeholders (Django's
`pyformat` paramstyle). The regex fast-path bails on `IN (` and
`RETURNING` keywords, falling through to `_sqlglot_parse()`. sqlglot
cannot parse `%s` as valid SQL, so parsing fails and returns empty
read/write table sets. DPOR never learns about these operations.

```python
from frontrun._sql_parsing import parse_sql_access

# DELETE: fails — returns empty tables
parse_sql_access('DELETE FROM "t" WHERE "t"."id" IN (%s)')
# → SqlAccessResult(read_tables=set(), write_tables=set(), ...)

# INSERT: fails — returns empty tables
parse_sql_access('INSERT INTO "t" ("a", "b") VALUES (%s, %s) RETURNING "t"."id"')
# → SqlAccessResult(read_tables=set(), write_tables=set(), ...)

# UPDATE: succeeds (regex handles it)
parse_sql_access('UPDATE "t" SET "a" = %s WHERE "t"."id" = %s')
# → SqlAccessResult(read_tables={'t'}, write_tables={'t'}, ...)

# SELECT: succeeds (regex handles it)
parse_sql_access('SELECT "t"."id" FROM "t" WHERE "t"."id" = %s')
# → SqlAccessResult(read_tables={'t'}, write_tables=set(), ...)
```

**Root cause:** In `_sql_parsing.py`, the regex fast-path at line 148-152
bails to the sqlglot full parser when it sees `IN (` (DELETE case) or
`RETURNING` (INSERT case). sqlglot then fails because `%s` is not valid
SQL syntax.

**Impact:** Any Django code that uses `DELETE ... WHERE id IN (%s)` or
`INSERT ... RETURNING id` is invisible to DPOR. Since Django generates
both patterns for standard ORM operations (`.delete()` and `.create()`),
DPOR cannot detect conflicts involving INSERTs or DELETEs. This makes
it impossible to find races involving:
- Double-INSERT (e.g., duplicate friend request creation)
- DELETE-then-SELECT (e.g., accepting a deleted request)
- INSERT-after-check TOCTOU patterns

**Possible fixes:**

1. In the regex fast-path, handle `RETURNING` for INSERT and `IN (` for
   DELETE before bailing to sqlglot. The table name is already captured
   by the INSERT/DELETE regex — just return it.
2. Pre-process `%s` placeholders (replace with `NULL` or `0`) before
   passing to sqlglot.
3. Make `_regex_parse` less conservative about bailing: `RETURNING` in
   an INSERT context doesn't require the full parser since the table
   is always `INSERT INTO <table>`.

---

## #3 — DPOR deadlock in pytest plugin (FIXED in frontrun#84)

**Status:** FIXED in commit 9fb5519

The frontrun pytest plugin used cooperative locks internally, which
caused deadlocks when the DPOR scheduler itself needed to synchronize.
Fixed by switching to real (non-cooperative) internal locks.
