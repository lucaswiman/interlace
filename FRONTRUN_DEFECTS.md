# Frontrun Defects

Known bugs, limitations, and footguns in frontrun's DPOR engine.

## 1. DPOR skips site-packages — no scheduling points in library code

**Status: FIXED** — `_intercept_execute()` now calls
`scheduler.report_and_wait(None, thread_id)` directly after
`_acquire_pending_row_locks()`, forcing a scheduling point at each SQL
operation without relying on `sys.settrace`. The `frame=None` path skips
`_process_opcode` but still flushes `pending_io` and runs scheduling.
Transaction atomicity is preserved: the existing `_in_transaction and not
_is_autobegin` early-return in `report_and_wait` skips scheduling inside
explicit transactions. Same fix applied to `_intercept_execute_async` and
`_intercept_asyncpg_execute`.

**Severity:** Critical — DPOR explores only 1 interleaving

`frontrun/_tracing.py` defines `_SKIP_DIRS` which includes all `site-packages`
directories. The `should_trace_file()` function returns `False` for any file in
these directories, meaning DPOR's `sys.settrace` callback never fires inside
Django ORM code (or any installed library).

Since SQL operations happen inside site-packages (e.g. `django/db/backends/`),
DPOR never sees scheduling points between SQL calls. It explores a single
serial interleaving and reports `num_explored=1` with `property_holds=True`
even when a race exists.

**Workaround:** Wrap `_intercept_execute` with a function defined in the test
file (user code, not site-packages) that performs a global variable access.
The `LOAD_GLOBAL`/`STORE_GLOBAL` bytecodes create scheduling points that DPOR
can intercept:

```python
import frontrun._sql_cursor as _sql_mod

_original_intercept = _sql_mod._intercept_execute
_sql_call_count = 0

def _traced_intercept(original_method, self, operation, parameters=None, **kwargs):
    global _sql_call_count
    _sql_call_count += 1
    return _original_intercept(original_method, self, operation, parameters, **kwargs)

# Install before running DPOR:
_sql_mod._intercept_execute = _traced_intercept
```

The `global _sql_call_count; _sql_call_count += 1` is essential — a simple
pass-through wrapper without global access does NOT generate enough bytecodes
for DPOR scheduling points.

**Proper fix:** frontrun should either (a) trace into site-packages selectively
for known SQL cursor modules, or (b) inject scheduling points directly in
`_intercept_execute` / `TracedCursor.execute()` rather than relying on
`sys.settrace`.

## 2. IndexError in shadow stack passthrough builtin handling

**Severity:** Intermittent crash

`dpor.py:1226` crashes with `IndexError: list index out of range` when
processing passthrough builtins:

```python
_pt_target = shadow.stack[-(argc - _pt_obj_idx)]
```

Triggered by `getattr()` calls in traced code. The shadow stack does not
correctly track arguments for certain builtin function calls.

**Workaround:** Avoid `getattr()` in code that runs under DPOR tracing.

## 3. `django_dpor()` helper not documented

**Status: FIXED** — Added "ORM helpers" section to `docs/dpor_guide.rst`
documenting both `django_dpor` and `sqlalchemy_dpor`. Added a note in
`docs/orm_race.rst` pointing readers to `sqlalchemy_dpor`. Added
`frontrun.contrib.django` and `frontrun.contrib.sqlalchemy` to
`docs/api_reference.rst`.

**Severity:** Usability

`frontrun.contrib.django.django_dpor()` is a convenience wrapper around
`explore_dpor()` that handles per-thread Django connection management
(`connections.close_all()`, lock timeouts, etc.). It is not mentioned in
`FINDING_SQL_RACE_CONDITIONS.md`. Users who follow the template must manually
manage connections, which is error-prone.

