# Frontrun Defects

Known issues in the frontrun library discovered during concurrency bug hunting.

---

## #1 — Row-level predicate tracking misses cross-column conflicts (FIXED)

**Status:** FIXED in commit [current]

**Severity:** HIGH — causes DPOR to miss real TOCTOU races

**Discovered:** 2026-03-11 during django-registration testing

**Description:**

When `detect_io=True` and row-level predicate extraction is active, DPOR
builds resource IDs that include the WHERE-clause predicates. A SELECT
filtered by one column and an UPDATE filtered by a different column produce
different resource IDs, even when they refer to the same row.

**Fix:**
Implemented **Conservative Column-Set Partitioning**.
1.  The first column set seen for each table in a DPOR session is designated
    as the "primary" column set (usually the PK).
2.  Every row-level access reports a "bridge" resource (`sql:<table>`).
3.  Accesses using the primary column set report a **READ** on the bridge.
4.  Accesses using any other column set (or table-level) report a **WRITE**
    on the bridge.
5.  This ensures that cross-column accesses properly conflict on the bridge
    resource, while preserving row-level independence for the primary column
    set (where both READ the bridge).

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
