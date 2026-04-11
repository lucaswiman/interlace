# Integrations and Detection: Remaining Work

This document consolidates remaining unfinished work from SQL conflict detection, Redis, and stateful resource detection layers. **Already implemented:** SQL cursor patching for sqlite3/psycopg2/pymysql/aiosqlite/asyncpg, table/row-level conflict detection, wire protocol parsing, Redis key-level detection, I/O detection layers (sys.setprofile, socket/file patching), LD_PRELOAD library.

## High Priority

### Autoincrement RETURNING Clause Injection (PostgreSQL)

**What:** For psycopg2 and psycopg3, `lastrowid` is unavailable after INSERT. Inject a RETURNING clause to capture the inserted row's ID explicitly.

**Why:** Currently raises `NondeterministicSQLError` when `warn_nondeterministic_sql=True`. With RETURNING, every INSERT gets an indexical alias like `sql:users:t0_ins0`, mapping concrete row IDs to these aliases for downstream conflict detection.

**Complexity:** Low. Modify `_sql_insert_tracker.py` to wrap INSERT statements lacking RETURNING and inject `RETURNING id` (or the appropriate PK column). Handle edge cases: multi-row inserts, composite PKs, explicit RETURNING clauses already present.

**Location:** `frontrun/_sql_insert_tracker.py`

---

## Medium Priority

### Cross-Table Foreign Key Analysis

**What:** Schema introspection to detect FK dependencies, e.g. `orders.user_id` → `users.id`. Currently `INSERT INTO orders (user_id, ...)` and `DELETE FROM users WHERE id = ?` are marked independent (different tables), but the FK creates a real conflict.

**Why:** More accurate conflict detection. Especially important for referential integrity bugs and cascade-delete scenarios.

**Complexity:** Medium (~150 lines + 25 tests).
- Query `information_schema.referential_constraints` on first connection to PostgreSQL/MySQL
- Build FK dependency graph: `{orders → users, shipments → orders}`
- At conflict detection: if Op1 touches T1 and Op2 touches T2 with T1 → T2 via FK, mark as dependent
- Manual FK registration via `frontrun/_schema.py` already exists; automatic introspection is the remaining piece

**Location:** `frontrun/_schema.py`, `frontrun/_sql_cursor.py`

---

## Low Priority / Long-Term

### Transaction Identity via Driver APIs

**What:** Use `cursor.connection.info.backend_pid` (psycopg2/psycopg3) or `conn.in_transaction` (SQLite) to track connection and transaction boundaries more reliably than wire-protocol parsing.

**Why:** Distinguishes independent connections from shared connections. Handles autocommit, savepoints, and driver-specific transaction semantics without C-level wire parsing.

**Complexity:** Low. Already accessible from `_sql_cursor.py`. Add to resource_id: `sql:{table}:conn={backend_pid}`. Replaces removed "shared socket" warning with per-connection tracking.

**Status:** Wire-protocol approach documented but deferred. Python driver APIs sufficient for >95% of real-world cases.

**Location:** `frontrun/_sql_cursor.py`, `frontrun/_io_detection.py`

---

### sys.addaudithook Integration (Layer 0)

**What:** Zero-config safety net using `sys.addaudithook` to intercept `socket.connect` and `open` events from C code before they even reach Python's socket/file layers.

**Why:** Catches I/O from C extensions that bypass Python's socket module (rare but possible), and provides a fallback for detection when other layers are disabled.

**Complexity:** Low (~20 lines). Already tested in `ideas/experiments/test_audit_hook.py`. Limitation: granularity is coarse (entire endpoint, not per-table); audit hooks can't be removed (must gate on test-run flag).

**Status:** Tested experimentally. Production integration deferred pending need for broader compatibility. Currently `sys.setprofile` + socket/file patching cover the practical cases.

**Location:** Could be added to `frontrun/_io_detection.py` as fallback layer

---

### sys.monitoring CALL Events (Layer 1.5, Python 3.12+)

**What:** Use PEP 669 `CALL` event type to detect calls to `.execute()`, `.send()`, `.write()` etc. without code rewriting.

**Why:** Lower overhead than `sys.settrace`-based detection. Coexists with existing INSTRUCTION events on same tool ID.

**Complexity:** Low (~30 lines). Add `CALL` to event bitmask, check callable name against `RESOURCE_METHOD_NAMES = {"execute", "send", "recv", "read", "write", "commit", "rollback"}`.

**Status:** Tested in `ideas/experiments/test_monitoring_c_call.py`. Python 3.13 confirmed INSTRUCTION + CALL events coexist on same tool. Not yet integrated into production code.

**Location:** `frontrun/_io_detection.py` (3.12+ only path)

---

### `__class__` Reassignment for Pure-Python Objects (Taint Propagation)

**What:** Intercept `PY_RETURN` events in DPOR's `handle_py_return()` callback. Swap the `__class__` of returned objects that look like resources (have `.execute()`, `.send()`, etc.) with instrumented subclasses that self-report method calls.

**Why:** Once swapped, objects auto-report resource accesses with zero tracing overhead for that object.

**Complexity:** Medium (~50 lines). Must handle: C extension types fail silently (fall back to `sys.setprofile`), duck-type check (`hasattr(retval, "execute")`), already-instrumented objects (avoid double-wrapping).

**Status:** Verified in `ideas/experiments/test_class_reassignment.py`. Works perfectly on pure-Python objects (SQLAlchemy Session, Connection). Fails on C extension types (sqlite3.Cursor, socket.socket).

**Location:** `frontrun/_dpor.py` (extend `handle_py_return` callback)

---

### `gc.get_referrers()` One-Shot Resource Discovery

**What:** When an audit hook or profile callback detects I/O to endpoint `(host, port)`, call `gc.get_referrers(sock)` once to walk up the reference chain: socket → DBConnection → ConnectionPool → Engine. Cache the mapping `id(engine) → endpoint_identity`.

**Why:** Maps low-level I/O objects to high-level resource owners, enabling more precise conflict reporting and diagnostics.

**Complexity:** Low (~40 lines). Key implementation detail: `gc.get_referrers()` returns `__dict__` dicts, not objects directly; must walk *through* dicts by checking `getattr(r, "__dict__", None) is current`.

**Status:** Verified in `ideas/experiments/test_gc_referrers.py`. Cold walk ~1.5–25ms (one-time cost at connection). Cache lookup ~1µs (1300x speedup).

**Location:** Could be added to `frontrun/_io_detection.py` for resource owner discovery

---

## Deferred / Experimental

### SQL Wire-Protocol Parsing (LD_PRELOAD Level)

**What:** Parse PostgreSQL `K` (BackendKeyData) and `Z` (ReadyForQuery) messages in `LD_PRELOAD` recv hooks to extract `backend_pid` and transaction boundaries.

**Why:** Enables transaction tracking and connection identity at C level for non-Python drivers (libpq FFI, etc.).

**Complexity:** Medium. ~10 lines for BackendKeyData (one-time). ~50 lines for ReadyForQuery (requires message framing per fd).

**Status:** Documented. Deferred pending actual use case of C-level direct libpq access (niche). Python driver APIs (`conn.info.backend_pid`, `cursor.connection`) sufficient for mainstream usage.

---

### Deterministic Record/Replay of External State (Layer 5)

**What:** Record all I/O operations on first run with their thread ID and vector clock. On subsequent runs, replay recorded responses rather than talking to real databases.

**Why:** Eliminates non-determinism from external state (DB is modified by first run, affects second run).

**Complexity:** High. Requires: message recording (socket/file), replay layer, mocking of I/O responses.

**Status:** Conceptual. Not implemented. Adds architectural complexity; most tests use isolated databases or transactions anyway. Reserved as a future option for integration tests.

---

### Import Hook Known-Library Registry (Layer 4)

**What:** On import, detect `sqlite3`, `psycopg2`, `redis`, `sqlalchemy`, etc. and auto-patch their resource methods.

**Why:** Higher precision than audit hooks or profile callbacks. You know exactly which methods are stateful.

**Complexity:** Medium (~100 lines of registry + patching logic). Requires maintenance as libraries evolve.

**Status:** Documented. Not implemented. Current monkey-patching in `_sql_cursor.py` and `_cooperative.py` is sufficient. Would be useful as a plugin system if users want to add custom libraries.

---

### One-Line Decorator Annotation (Layer 6)

**What:** `@frontrun.resource("db")` or `with frontrun.accessing("db"):` syntax to mark resource-accessing regions.

**Why:** Minimal config (one decorator) with perfect precision (user says exactly what's a resource).

**Complexity:** Low (~30 lines). Analogous to existing `# frontrun:` trace markers but for resource identity.

**Status:** Documented. Not implemented. Users already have SQL cursor patching + redis patching covering ~95% of cases. Could be added as opt-in if needed.

---

## Frame-Local Variable Poisoning (Narrow Use, Deprecated)

**What:** On `sys.settrace` "call" event, swap function arguments with proxy objects using `frame.f_locals` + `PyFrame_LocalsToFast` (3.13+) or `ctypes.pythonapi` (older).

**Why:** Catches C extension types that don't support `__class__` swapping (e.g. `sqlite3.Cursor`).

**Complexity:** Medium. Version-sensitive. Prefer `sys.setprofile` or `__class__` swapping.

**Status:** Documented as fallback. Not implemented. `sys.setprofile` (already in production) is simpler and more reliable for C extension detection.

---

## Testing & Validation

- **SQL tests:** `tests/test_sql_*.py` (sqlite3, psycopg2, asyncpg, etc.)
- **Resource detection experiments:** `ideas/experiments/test_*.py` (audit_hook, class_reassignment, gc_referrers, monitoring_c_call)
- **Integration tests:** `tests/test_integration_*.py` (require Redis, Postgres)

Run via `make test-3.14` or `make test-integration-3.14`.

---

## Layered Detection Summary

For reference, the complete detection stack (highest to lowest precision):

```
Layer 6: User annotations (@frontrun.resource, with accessing)
Layer 4: Known-library plugins (sqlalchemy, redis, psycopg2)
Layer 3: Duck-typing heuristics (sys.monitoring CALL → .execute(), .send())
Layer 2: Taint propagation (__class__ reassignment / proxy + gc.get_referrers)
Layer 1.5: sys.setprofile C_CALL events (already implemented)
Layer 1: Socket/file monkey-patching (already implemented)
Layer 0: sys.addaudithook (zero-config but coarse)
```

Currently deployed: **Layers 1, 1.5, and targeted Layer 2 (via cursor patching)**. Layers 0, 3, 4, 6 are documented and experimentally verified but not yet integrated.
