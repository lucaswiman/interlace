# Correctness Argument

## Soundness (No missed bugs)

The system is **sound** (never claims independence when a conflict exists) because:

1. **Parse failure → fallback.** If `parse_sql_access` returns empty sets, the endpoint-level suppression doesn't activate, and the coarser socket-level conflict detection remains in effect.

2. **Conservative classification.** UPDATE/DELETE are classified as *both* read and write on their target table. DDL is classified as write to all mentioned tables. Unknown statements return empty sets.

3. **Table-level is a strict refinement.** Any two SQL operations that touch the same table get the same `ObjectId` and are correctly classified as Read or Write. The only operations that become independent are those on *different* tables — which genuinely cannot conflict at the SQL level.

4. **Row-level requires full PK.** Row-level ObjectIds are only used when both operations have complete primary key equality predicates, guaranteeing that different PK values select provably disjoint rows.

## Completeness (No false positives)

The system reduces false positives (spurious interleavings) but does not eliminate them entirely:

1. **Table-level is conservative.** Two operations on the same table with `WHERE id=1` and `WHERE id=2` are reported as conflicting (table-level), even though they touch different rows. This is fixed by Phase 2 row-level detection.

2. **Cross-table dependencies.** Foreign key relationships are invisible. Thread A inserts into `orders` (references `users.id`), Thread B deletes from `users` — these are classified as independent (different tables), but the FK constraint could cause a real conflict. This is a known limitation; fixing it requires schema-aware analysis.

---

## Known Limitations & TODOs

### TODO: SELECT FOR UPDATE / FOR SHARE Locking Not Modeled
**Soundness impact:** None (sound but overly pessimistic)
**Completeness impact:** False positives for lock-contended rows

Two concurrent operations with locking intent are reported as conflicting based on table/row access, but the actual serialization guarantee from `FOR UPDATE` is not modeled:
```sql
-- Thread A: SELECT * FROM users WHERE id = 1 FOR UPDATE → exclusive lock on row 1
-- Thread B: SELECT * FROM users WHERE id = 1 FOR UPDATE → blocks until A releases
```
DPOR reports conflict (correct conclusion), but for the wrong reason (read vs read, not lock vs lock).

**Mitigation:** Row-level predicate detection (Phase 2) ensures different rows are independent; same-row locking is conservative but correct.

**Future fix:** Algorithm 1a could extract `lock_intent` from SQL AST, allowing DPOR to track lock ownership explicitly.

---

### TODO: Advisory Lock Semantics Not Tracked
**Soundness impact:** None (falls back to endpoint-level)
**Completeness impact:** False positives for different advisory lock IDs

Advisory locks are function calls, not SQL DML:
```python
cursor.execute("SELECT pg_advisory_lock(?)", (lock_id,))
```
These are detected only at the socket level (LD_PRELOAD), not at the lock-ID level. Two threads acquiring different advisory lock IDs are still reported as conflicting because the system doesn't see the lock-ID parameter.

**Mitigation:** Wire protocol parsing (`crates/io/src/lib.rs`) attempts to extract SQL from PostgreSQL wire protocol, but function semantics are opaque.

**Future fix:** Extend wire protocol parser to recognize advisory lock function calls and extract lock IDs, enabling per-lock-ID suppression.

---

### TODO: Transaction Boundaries Ignored
**Soundness impact:** None (sound)
**Completeness impact:** Explosion of search space

DPOR treats each SQL statement independently, missing the atomicity guarantee of transactions:
```python
# Thread A:
cursor.execute("BEGIN")
cursor.execute("SELECT * FROM accounts WHERE id = 1")
cursor.execute("UPDATE accounts SET balance = ...")
cursor.execute("COMMIT")
```
DPOR explores all interleavings of the 4 statements. In reality, the entire transaction is atomic, and only commit/rollback points are observable to other threads.

**Mitigation:** For read-committed isolation, this is conservative but correct (interleavings within a transaction don't cause anomalies).

**Future fix (Phase 4):** Track transaction boundaries (BEGIN/COMMIT/ROLLBACK) and group SQL operations into transaction-level ObjectIds.

---

### TODO: Stored Procedures Opaque
**Soundness impact:** None (falls back to endpoint-level)
**Completeness impact:** False positives for sproc-internal operations

Stored procedures are treated as opaque socket I/O:
```sql
CALL sp_update_user(1, 'new_name');  -- internal SQL is not parsed
```
All threads calling any stored procedure are reported as conflicting (endpoint-level detection).

**Mitigation:** Rare in modern Python ORMs; most code uses direct SQL.

**Future fix:** Intercept sproc definitions and parse their body SQL; cache table access sets for each sproc signature.
