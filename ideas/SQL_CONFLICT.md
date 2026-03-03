# SQL Resource Conflict Detection for DPOR

## Problem Statement

Frontrun's DPOR engine currently detects conflicts at two granularities:

1. **Python object attribute access** — `LOAD_ATTR`/`STORE_ATTR` on `id(obj)`, tracked via the shadow stack.
2. **I/O endpoint** — `LD_PRELOAD` intercepts `send()`/`recv()` to the same `(host, port)`, reported as a single virtual object.

The endpoint-level granularity lumps *all* SQL operations to the same database into a single conflict domain. Two threads that touch completely independent tables (`INSERT INTO logs ...` vs. `SELECT * FROM users`) appear to conflict, forcing DPOR to explore interleavings that can never actually race. Conversely, it can't distinguish a read (`SELECT`) from a write (`UPDATE`) on the same table — everything is a "write to the socket."

**Goal:** Parse intercepted SQL to derive *table-level* and optionally *row-level* read/write sets, and feed those into the DPOR engine as fine-grained conflict objects. This eliminates spurious interleavings between independent tables and enables read/read independence within the same table.

---

## Prior Art & Ecosystem Survey

### SQL Parsing Libraries

| Library | Language | Strengths | Weaknesses |
|---------|----------|-----------|------------|
| [**sqlglot**](https://github.com/tobymao/sqlglot) | Python | Full optimizer pipeline: `qualify_columns`, `qualify_tables`, column lineage via `build_scope`. Handles 31 dialects. Schema-aware: given a schema dict, resolves `SELECT *` and maps unqualified columns to tables. Zero dependencies. | Pure Python — may be slow for per-query hot-path parsing. No built-in read/write classification (must infer from statement type). |
| [**sqlparse**](https://github.com/andialbrecht/sqlparse) | Python | Lightweight tokenizer/formatter. | No AST — only token streams. Table extraction is fragile ([issue #157](https://github.com/andialbrecht/sqlparse/issues/157)). Cannot resolve column→table without schema. Not suitable for semantic analysis. |
| [**sqlparser-rs**](https://github.com/apache/datafusion-sqlparser-rs) | Rust | Fast Pratt parser + recursive descent. Rich AST with `Visitor` trait and `visit_relations()` for table extraction. Used by DataFusion, Polars, GlueSQL. Source spans available since v0.53. | Syntax-only — no semantic analysis, no column qualification. Must build read/write set logic ourselves. |
| [**sqloxide**](https://github.com/wseaton/sqloxide) | Rust+Python | Python bindings for `sqlparser-rs` via PyO3. ~100x faster than `sqlparse`. | Thin wrapper — exposes AST as Python dicts, no semantic helpers. |

**Recommendation:** Use **sqlparser-rs** in the Rust DPOR engine for hot-path parsing (it's already a Rust crate, integrates naturally into `crates/dpor/`). Use **sqlglot** on the Python side for schema-aware analysis, test utilities, and any offline conflict reasoning. Use **sqloxide** if we need fast Python-side parsing without the full sqlglot optimizer.

### Formal Models of Transaction Conflicts

**Adya/Liskov/O'Neil (ICDE 2000)** — ["Generalized Isolation Level Definitions"](http://pmg.csail.mit.edu/papers/icde00.pdf): The gold-standard formalism. Defines three dependency types on a *Direct Serialization Graph* (DSG):

- **WR (write-read / read-dependency):** T2 reads a version written by T1.
- **WW (write-write / write-dependency):** T2 overwrites a version written by T1.
- **RW (read-write / anti-dependency):** T2 writes a version that T1 previously read.

Isolation levels are defined as *cycle prohibitions* on subsets of these edges:
- **Read Committed (PL-2):** No G1 cycles (dirty reads/writes).
- **Repeatable Read:** No G2-item cycles (non-repeatable reads).
- **Serializable:** No G2 cycles at all (full DSG acyclicity).

This maps directly onto DPOR's conflict model: a WR or WW dependency means "these two accesses conflict (at least one is a write)," and an RW anti-dependency means "a read depends on writes from other threads." This is exactly what `ObjectState::dependent_accesses` already computes.

**[Elle](https://github.com/jepsen-io/elle)** (Jepsen) — Black-box transactional safety checker. Constructs DSGs from observed histories and detects cycles. Implemented in Clojure. Key insight: Elle works on *list-append* semantics (each value is a list; appends are writes, reads return the list) so it can infer WW/WR/RW edges without instrumentation. We can't use Elle directly (it assumes key-value workloads), but its cycle-detection approach validates our DSG-based model.

**[IsoRel](https://dl.acm.org/doi/10.1145/3728953)** (ACM 2025) — Black-box isolation checker for *relational* DBMSs. Adds two auxiliary columns per table to track which transaction wrote each row and which rows each statement read. Constructs transaction dependency graphs from this instrumentation. Found 48 unique anomalies across MySQL, PostgreSQL, MariaDB, CockroachDB, and TiDB. **Directly relevant** — their SQL instrumentation approach could be adapted for frontrun.

**[TxCheck](https://www.usenix.org/system/files/osdi23-jiang.pdf)** (OSDI 2023) — SQL-level statement instrumentation for black-box isolation testing. Inserts auxiliary SQL to collect execution information. Found 56 bugs across TiDB, MySQL, MariaDB.

**[GRAIL](https://hal.science/hal-04886090v1/document)** (2024) — Uses graph databases to detect isolation violations as anti-patterns in dependency graphs. Expresses anomaly patterns as graph queries.

### Logic Programming & Constraint Solving

| Library | Potential Use |
|---------|--------------|
| [**kanren**](https://github.com/pythological/kanren) (Python miniKanren) | Encode conflict rules as relations. Given a set of SQL operations, derive which pairs conflict via relational search. Elegant for expressing "T1 writes table X ∧ T2 reads table X → conflict(T1, T2)." |
| [**python-constraint**](https://github.com/python-constraint/python-constraint) | CSP solver. Could model "find a schedule where no conflicting operations overlap" as a constraint satisfaction problem. More natural for *generating* conflict-free schedules than for *detecting* conflicts. |
| [**Alloy**](https://alloytools.org/) | Adya's isolation levels have been [modeled in Alloy](https://surfingcomplexity.blog/2024/11/18/reading-the-generalized-isolation-level-definitions-paper-with-alloy/). Could generate test cases from Alloy models. |
| [**z3**](https://github.com/Z3Prover/z3) (via z3-py) | SMT solver. Encode WHERE clauses as z3 formulas; two operations on the same table conflict iff their row predicates are *satisfiable simultaneously* (i.e., their row sets can overlap). This is the key insight for row-level conflict detection — see §Row-Level Refinement below. |

---

## Design: Three-Layer Conflict Model

```
┌──────────────────────────────────────────────────────────┐
│  Layer 3: Row-Level (predicate intersection via z3/SMT)  │  ← optional, highest precision
│  "UPDATE accounts WHERE id=1" vs "SELECT WHERE id=2"     │
│  → independent (disjoint predicates)                      │
├──────────────────────────────────────────────────────────┤
│  Layer 2: Table-Level (SQL parsing)                       │  ← primary target
│  "UPDATE accounts" vs "SELECT users"                      │
│  → independent (different tables)                         │
├──────────────────────────────────────────────────────────┤
│  Layer 1: Endpoint-Level (current LD_PRELOAD)             │  ← baseline, already works
│  send() to localhost:5432 vs send() to localhost:5432     │
│  → conflict (same endpoint)                               │
└──────────────────────────────────────────────────────────┘
```

Higher layers refine lower ones. If table-level parsing succeeds, it replaces endpoint-level for that operation. If row-level analysis is available, it further refines table-level. Fallback is always to the coarser level.

---

## Interception Architecture

### Option A: DBAPI Cursor Monkey-Patching (Recommended for MVP)

Patch `cursor.execute()` at the DBAPI level (PEP 249). Every compliant driver — psycopg2, sqlite3, pymysql, pg8000 — goes through this interface.

```python
# frontrun/_sql_detection.py

import functools
from frontrun._io_detection import _io_tls

_ORIGINAL_EXECUTE = {}  # driver_class -> original_execute

def _patched_execute(self, operation, parameters=None):
    """Intercept cursor.execute() to extract SQL read/write sets."""
    reporter = getattr(_io_tls, "sql_reporter", None)
    if reporter is not None:
        read_tables, write_tables = parse_sql_access(operation, parameters)
        for table in read_tables:
            reporter(f"sql:{table}", "read")
        for table in write_tables:
            reporter(f"sql:{table}", "write")

    original = _ORIGINAL_EXECUTE[type(self)]
    if parameters is not None:
        return original(self, operation, parameters)
    return original(self, operation)


def install_sql_detection():
    """Monkey-patch known DBAPI cursor classes."""
    import importlib

    TARGETS = {
        "psycopg2.extensions": "cursor",
        "psycopg.cursor": "Cursor",
        "sqlite3": "Cursor",
        "pymysql.cursors": "Cursor",
        "pg8000.native": "Connection",  # pg8000 uses conn.run()
    }

    for module_path, class_name in TARGETS.items():
        try:
            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name)
            _ORIGINAL_EXECUTE[cls] = cls.execute
            cls.execute = _patched_execute
        except (ImportError, AttributeError):
            pass  # driver not installed
```

**Advantages:**
- Works with any DBAPI-compliant driver.
- Sees the SQL text *and* parameters before they hit the wire.
- Naturally per-thread (each thread has its own cursor).
- Follows the established pattern from `_cooperative.py` (lock patching) and `_io_detection.py` (socket patching).

**Disadvantages:**
- Doesn't catch raw `connection.exec_driver_sql()` or driver-specific methods.
- C-extension cursors (psycopg2) may need `sys.setprofile` fallback for method interception.

### Option B: SQLAlchemy Event Hooks

For users already on SQLAlchemy, hook into the engine event system:

```python
from sqlalchemy import event

@event.listens_for(engine, "before_cursor_execute")
def _on_execute(conn, cursor, statement, parameters, context, executemany):
    read_tables, write_tables = parse_sql_access(statement, parameters)
    # report to DPOR engine...
```

This is cleaner but SQLAlchemy-specific. Use as a Layer 4 "known library plugin" alongside the generic DBAPI patching.

### Option C: Wire Protocol Parsing (Future)

Parse the PostgreSQL/MySQL wire protocol from the `LD_PRELOAD` `send()` buffer. The SQL text is embedded in the protocol messages:

- **PostgreSQL:** Simple Query message (`'Q'` + int32 length + SQL string).
- **MySQL:** `COM_QUERY` packet (command byte `0x03` + SQL string).

This catches *everything* including raw libpq calls, but is fragile (prepared statements, binary protocol, SSL) and dialect-specific. Reserve for a future enhancement.

---

## SQL Parsing: Extracting Read/Write Sets

### Statement Classification

| SQL Statement | Read Tables | Write Tables |
|--------------|-------------|--------------|
| `SELECT ... FROM t1 JOIN t2 ...` | {t1, t2} | {} |
| `INSERT INTO t1 SELECT ... FROM t2` | {t2} | {t1} |
| `UPDATE t1 SET ... WHERE ... (SELECT FROM t2)` | {t1, t2} | {t1} |
| `DELETE FROM t1 WHERE ... IN (SELECT FROM t2)` | {t1, t2} | {t1} |
| `CREATE TABLE t1 AS SELECT ... FROM t2` | {t2} | {t1} |
| `MERGE INTO t1 USING t2 ...` | {t2} | {t1} |

Note: `UPDATE` and `DELETE` *read* their target table (the `WHERE` clause scans rows) in addition to writing it. This matters for conflict detection — a concurrent `SELECT` on the same table has an RW anti-dependency with an `UPDATE`.

### Python-Side: sqlglot

```python
import sqlglot
from sqlglot import exp
from sqlglot.optimizer.qualify import qualify

def parse_sql_access(sql: str, schema: dict | None = None) -> tuple[set[str], set[str]]:
    """Extract (read_tables, write_tables) from a SQL statement."""
    try:
        ast = sqlglot.parse_one(sql)
    except sqlglot.errors.ParseError:
        return set(), set()  # unparseable → fall back to endpoint-level

    write_tables: set[str] = set()
    read_tables: set[str] = set()

    if isinstance(ast, (exp.Insert, exp.Create)):
        # Target table is written
        if ast.this and isinstance(ast.this, exp.Table):
            write_tables.add(ast.this.name)
        # Source tables (FROM/subquery) are read
        for table in ast.find_all(exp.Table):
            if table.name not in write_tables:
                read_tables.add(table.name)

    elif isinstance(ast, exp.Update):
        target = ast.this
        if isinstance(target, exp.Table):
            write_tables.add(target.name)
            read_tables.add(target.name)  # WHERE clause reads
        for table in ast.find_all(exp.Table):
            if table.name not in write_tables:
                read_tables.add(table.name)

    elif isinstance(ast, exp.Delete):
        target = ast.this
        if isinstance(target, exp.Table):
            write_tables.add(target.name)
            read_tables.add(target.name)
        for table in ast.find_all(exp.Table):
            if table.name not in write_tables:
                read_tables.add(table.name)

    elif isinstance(ast, exp.Select):
        for table in ast.find_all(exp.Table):
            read_tables.add(table.name)

    else:
        # DDL, GRANT, etc. — conservatively treat as write to all mentioned tables
        for table in ast.find_all(exp.Table):
            write_tables.add(table.name)

    return read_tables, write_tables
```

### Rust-Side: sqlparser-rs (in DPOR engine)

```rust
// crates/dpor/src/sql.rs

use sqlparser::ast::*;
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::Parser;

pub struct SqlAccess {
    pub read_tables: Vec<String>,
    pub write_tables: Vec<String>,
}

pub fn parse_sql_access(sql: &str) -> Option<SqlAccess> {
    let dialect = GenericDialect {};
    let stmts = Parser::parse_sql(&dialect, sql).ok()?;
    let stmt = stmts.into_iter().next()?;

    let mut read_tables = Vec::new();
    let mut write_tables = Vec::new();

    match &stmt {
        Statement::Query(q) => {
            collect_query_tables(q, &mut read_tables);
        }
        Statement::Insert(Insert { table_name, source, .. }) => {
            write_tables.push(table_name.to_string());
            if let Some(src) = source {
                collect_query_tables(src, &mut read_tables);
            }
        }
        Statement::Update { table, selection, from, .. } => {
            let name = table_factor_name(&table.relation);
            write_tables.push(name.clone());
            read_tables.push(name);  // WHERE reads the table
            if let Some(from) = from {
                // ... collect FROM tables
            }
        }
        Statement::Delete(Delete { from, selection, .. }) => {
            for table in from {
                let name = table_factor_name(&table.relation);
                write_tables.push(name.clone());
                read_tables.push(name);
            }
        }
        _ => return None,  // unknown → fall back to endpoint-level
    }

    Some(SqlAccess { read_tables, write_tables })
}
```

The Rust parser runs inside the DPOR engine, so parsing happens in the same process as conflict detection — no IPC overhead. Add `sqlparser = { version = "0.59", features = ["visitor"] }` to `crates/dpor/Cargo.toml`.

---

## Integration with DPOR Engine

### New Object ID Scheme

Currently, I/O objects use `hash(resource_key)` as the `ObjectId`. For SQL, derive finer-grained IDs:

```python
# Python side: report to Rust engine
def _sql_object_id(table: str, kind: str) -> int:
    """Derive a stable ObjectId for a SQL table access."""
    return hash(("sql", table)) & 0xFFFFFFFFFFFFFFFF
```

```rust
// Rust side: derive ObjectId
fn sql_object_id(table: &str) -> ObjectId {
    use std::hash::{Hash, Hasher};
    use std::collections::hash_map::DefaultHasher;
    let mut h = DefaultHasher::new();
    "sql".hash(&mut h);
    table.hash(&mut h);
    h.finish()
}
```

### Reporting Flow

```
Thread A calls cursor.execute("UPDATE accounts SET balance = balance - 100 WHERE id = 1")
  │
  ├─ _patched_execute() intercepts
  ├─ parse_sql_access() → read={"accounts"}, write={"accounts"}
  ├─ reporter("sql:accounts", "read")   → engine.process_io_access(exec, tid, hash("sql:accounts"), Read)
  └─ reporter("sql:accounts", "write")  → engine.process_io_access(exec, tid, hash("sql:accounts"), Write)

Thread B calls cursor.execute("SELECT * FROM users WHERE id = 42")
  │
  ├─ _patched_execute() intercepts
  ├─ parse_sql_access() → read={"users"}, write={}
  └─ reporter("sql:users", "read")      → engine.process_io_access(exec, tid, hash("sql:users"), Read)

Result: Thread A touches "sql:accounts", Thread B touches "sql:users"
        → INDEPENDENT, DPOR does NOT explore interleavings between them.
        (Currently: both touch "socket:localhost:5432" → CONFLICT, wasted exploration.)
```

### Suppressing Endpoint-Level Reports

When SQL-level detection succeeds, suppress the coarser endpoint-level report for the same `send()` call. The `_patched_execute` sets a thread-local flag; the `_io_detection` socket patch checks the flag and skips reporting if SQL-level already reported.

```python
# In _patched_execute:
_io_tls._sql_reported = True
try:
    return original(self, operation, parameters)
finally:
    _io_tls._sql_reported = False

# In _traced_send (socket patch):
if getattr(_io_tls, "_sql_reported", False):
    return _real_send(self, data, *args)  # skip endpoint-level report
```

---

## Row-Level Refinement via SMT (z3)

Table-level conflicts are conservative: two operations on the same table conflict even if they touch disjoint rows. For row-level precision, we can check whether their WHERE predicates can select overlapping rows.

### The Insight

Two SQL operations on the same table **definitely don't conflict** if their row predicates are *unsatisfiable* when conjoined:

```
Thread A: UPDATE accounts SET balance = 0 WHERE id = 1
Thread B: SELECT balance FROM accounts WHERE id = 2

Predicate A: id = 1
Predicate B: id = 2
Conjunction: id = 1 ∧ id = 2  →  UNSAT  →  NO CONFLICT
```

### Implementation with z3

```python
from z3 import Int, Solver, sat

def predicates_overlap(pred_a: dict, pred_b: dict, schema: dict) -> bool:
    """Check if two WHERE predicates can select overlapping rows.

    pred_a, pred_b: {"column": "op:value"} extracted from SQL WHERE.
    schema: {"column": "type"} for z3 variable typing.
    Returns True if overlap is possible (or if analysis fails).
    """
    solver = Solver()
    z3_vars = {}

    for col, typ in schema.items():
        if typ in ("int", "integer", "bigint"):
            z3_vars[col] = Int(col)
        else:
            return True  # can't model non-integer types easily → conservative

    for pred in (pred_a, pred_b):
        for col, constraint in pred.items():
            if col not in z3_vars:
                return True
            op, val = constraint.split(":", 1)
            v = z3_vars[col]
            if op == "eq":
                solver.add(v == int(val))
            elif op == "gt":
                solver.add(v > int(val))
            elif op == "lt":
                solver.add(v < int(val))
            elif op == "gte":
                solver.add(v >= int(val))
            elif op == "lte":
                solver.add(v <= int(val))

    return solver.check() == sat
```

### When z3 is Worth It

- **High-contention single-table workloads** where many threads touch the same table but different rows (e.g., partitioned account operations). Without row-level analysis, DPOR must explore O(n!) interleavings; with it, independent partitions are pruned.
- **Not worth it** for workloads that already touch different tables (table-level pruning suffices) or for trivial 2-thread scenarios.

### Alternative: Lightweight Predicate Intersection Without z3

For the common case of simple equality predicates on primary keys, we don't need an SMT solver:

```python
def simple_predicate_disjoint(pred_a: dict, pred_b: dict) -> bool:
    """Fast check: do two equality-predicate sets provably select disjoint rows?"""
    for col in pred_a:
        if col in pred_b:
            a_val = pred_a[col]
            b_val = pred_b[col]
            if a_val.startswith("eq:") and b_val.startswith("eq:"):
                if a_val != b_val:
                    return True  # Same column, different equality values → disjoint
    return False  # Can't prove disjoint → conservatively assume overlap
```

This handles the `WHERE id = 1` vs `WHERE id = 2` case without z3's overhead. Fall back to z3 for range predicates and complex expressions.

---

## Alternative / Complementary Approaches

### Approach: DSG Construction (Elle-style)

Instead of per-operation conflict detection, record all SQL operations with their timestamps and thread IDs, then construct a Direct Serialization Graph offline:

```python
@dataclass
class SqlOperation:
    thread_id: int
    timestamp: int  # vector clock position
    statement_type: str  # SELECT, UPDATE, INSERT, DELETE
    tables_read: set[str]
    tables_written: set[str]
    predicates: dict[str, str]  # column → "op:value"

def build_dsg(operations: list[SqlOperation]) -> nx.DiGraph:
    """Build Adya-style Direct Serialization Graph."""
    G = nx.DiGraph()
    for i, op_a in enumerate(operations):
        for j, op_b in enumerate(operations):
            if i >= j or op_a.thread_id == op_b.thread_id:
                continue
            # Check for WR, WW, RW dependencies
            common_written = op_a.tables_written & op_b.tables_written
            if common_written:
                G.add_edge(i, j, type="WW")
            wr = op_a.tables_written & op_b.tables_read
            if wr:
                G.add_edge(i, j, type="WR")
            rw = op_a.tables_read & op_b.tables_written
            if rw:
                G.add_edge(i, j, type="RW")
    return G
```

Cycles in the DSG indicate potential isolation anomalies. This is complementary to DPOR's online conflict detection — DPOR explores interleavings, DSG analysis characterizes *what kind* of anomaly was found.

### Approach: Datalog / Logic Programming

Express conflict rules declaratively with kanren or a Datalog engine:

```python
from kanren import run, var, eq, Relation, facts

writes = Relation()
reads = Relation()

# Populate from parsed SQL:
facts(writes, ("thread_a", "accounts"), ("thread_b", "accounts"))
facts(reads, ("thread_a", "accounts"), ("thread_b", "users"))

# Query: which thread pairs conflict?
t1, t2, table = var(), var(), var()
conflicts = run(0, (t1, t2, table),
    writes(t1, table),
    (reads(t2, table) | writes(t2, table)),
    neq(t1, t2))
```

This is appealing for complex conflict rules (e.g., "thread A writes table X *and* thread B reads table X *through a foreign key from table Y*") but overkill for simple table-level read/write set intersection. Reserve for future "semantic conflict" analysis.

### Approach: IsoRel-Style Auxiliary Column Instrumentation

Instead of parsing SQL from outside, *rewrite* SQL statements to add tracking:

```sql
-- Original:
UPDATE accounts SET balance = balance - 100 WHERE id = 1;

-- Instrumented:
UPDATE accounts SET balance = balance - 100, _txn_id = 'T42' WHERE id = 1;
-- Also: INSERT INTO _frontrun_audit (txn_id, table_name, op, row_id) VALUES ('T42', 'accounts', 'write', 1);
```

This is what IsoRel and TxCheck do. It gives *ground truth* about which rows were actually touched (not just which rows *could* be touched based on the WHERE clause). But it's invasive — it modifies the database schema and the SQL statements. Better suited for a dedicated "frontrun audit mode" than for transparent interception.

---

## Implementation Plan

### Phase 1: Table-Level SQL Conflict Detection (MVP)

1. **Add `sqlglot` dependency** to `pyproject.toml` (Python-side parsing).
2. **Add `sqlparser` crate** to `crates/dpor/Cargo.toml` (Rust-side parsing).
3. **Implement `_sql_detection.py`:**
   - Monkey-patch DBAPI `cursor.execute()` for known drivers.
   - Extract SQL text and parameters.
   - Parse with sqlglot → `(read_tables, write_tables)`.
   - Report to DPOR engine via existing `io_reporter` callback with `"sql:{table}"` resource keys.
4. **Implement `sql.rs` in `crates/dpor/`:**
   - `parse_sql_access(sql: &str) -> Option<SqlAccess>` using sqlparser-rs.
   - New `process_sql_access` method on `DporEngine` that calls `parse_sql_access` and routes to `process_io_access` with table-derived `ObjectId`s.
   - Alternatively, keep parsing on the Python side and just report `("sql:accounts", Read/Write)` pairs to the Rust engine — simpler, avoids duplicating parsing logic.
5. **Suppress endpoint-level reports** when SQL-level succeeds.
6. **Add integration tests** extending `test_integration_orm.py`:
   - Two threads updating *the same table* → DPOR finds the interleaving (as today).
   - Two threads updating *different tables* → DPOR explores only 1 execution (new!).
   - Mix of reads and writes on the same table → DPOR correctly identifies RW conflicts.

### Phase 2: SQLAlchemy Event Integration

7. **Hook `before_cursor_execute`** as a Layer 4 "known library plugin" for SQLAlchemy users.
8. **Schema-aware parsing** via sqlglot's `qualify_columns()` for resolving ambiguous column references.

### Phase 3: Row-Level Predicate Refinement

9. **Extract WHERE predicates** from the AST (sqlglot or sqlparser-rs).
10. **Simple equality check** for `WHERE pk = constant` (fast path, no z3).
11. **Optional z3 integration** for range predicates and complex expressions.
12. **New conflict check:** Same table, both operations have WHERE clauses, predicates provably disjoint → independent, skip interleaving.

### Phase 4: DSG Analysis & Anomaly Classification

13. **Record SQL operation history** during DPOR exploration.
14. **Build DSG** from recorded operations (Adya-style WR/WW/RW edges).
15. **Classify anomalies:** When DPOR finds a failing interleaving, report *which* isolation anomaly it corresponds to (dirty read, lost update, write skew, etc.).
16. **Output:** "This interleaving exhibits a G2-item anomaly (write skew) on table `accounts` — thread A read rows {1,2} while thread B updated row {1}."

---

## Open Questions

1. **Where to parse — Python or Rust?** Parsing in Python (sqlglot) is simpler to integrate with the monkey-patching layer. Parsing in Rust (sqlparser-rs) is faster and keeps everything in the DPOR engine. Could do both: Python-side for the MVP, Rust-side as an optimization when parsing becomes a bottleneck. Or: Python-side for schema-aware analysis, Rust-side for fast statement classification.

2. **Parameterized queries:** `cursor.execute("SELECT * FROM users WHERE id = %s", (42,))` — the SQL text contains `%s`, not `42`. For table-level detection this doesn't matter (we only need the table name). For row-level predicate analysis, we need to substitute parameters into the WHERE clause before analyzing. This is driver-specific (`%s` for psycopg2, `?` for sqlite3, `:name` for Oracle).

3. **Prepared statements:** Some drivers send `PREPARE` + `EXECUTE` separately. The SQL text is in `PREPARE`, the parameters in `EXECUTE`. Need to track the `PREPARE` → `EXECUTE` mapping per cursor/connection.

4. **ORM-generated SQL:** SQLAlchemy generates complex SQL with subqueries, CTEs, and joins. sqlglot handles these well; sqlparse does not. Validate against real SQLAlchemy output for common patterns (session.get, session.query, bulk operations).

5. **Schema discovery:** For column→table resolution, sqlglot needs the schema. Options: (a) user provides it, (b) query `information_schema` at test setup, (c) infer from `CREATE TABLE` statements if available. For table-level detection, schema isn't needed.

6. **Transaction boundaries:** The current model treats each SQL statement as an independent access. For transaction-level analysis (e.g., detecting that `BEGIN; SELECT; UPDATE; COMMIT` is an atomic unit), we'd need to track `BEGIN`/`COMMIT`/`ROLLBACK` boundaries and group statements into transactions.

---

## References

- Adya, Liskov, O'Neil. ["Generalized Isolation Level Definitions"](http://pmg.csail.mit.edu/papers/icde00.pdf). ICDE 2000.
- Berenson, Bernstein, Gray, Melton, O'Neil, O'Neil. ["A Critique of ANSI SQL Isolation Levels"](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-95-51.pdf). SIGMOD 1995.
- Kingsbury, Alvaro. [Elle: Inferring Isolation Anomalies](https://github.com/jepsen-io/elle). VLDB 2021.
- Cui et al. [IsoRel: Detecting Isolation Anomalies in Relational DBMSs](https://dl.acm.org/doi/10.1145/3728953). ACM 2025.
- Jiang et al. [TxCheck](https://www.usenix.org/system/files/osdi23-jiang.pdf). OSDI 2023.
- Mao. [sqlglot: Python SQL Parser and Transpiler](https://github.com/tobymao/sqlglot).
- Apache. [datafusion-sqlparser-rs](https://github.com/apache/datafusion-sqlparser-rs).
- Weston. [sqloxide: Python bindings for sqlparser-rs](https://github.com/wseaton/sqloxide).
- Pythological. [kanren: Logic programming in Python](https://github.com/pythological/kanren).
- de Moura, Bjørner. [Z3: An Efficient SMT Solver](https://github.com/Z3Prover/z3).
