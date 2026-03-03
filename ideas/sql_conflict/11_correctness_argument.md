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
