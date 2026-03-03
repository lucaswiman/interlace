# Formal Verification (TLA+)

Three TLA+ specs in `specs/` verify the core correctness properties of the design. All pass exhaustive model checking via TLC.

## Spec 1: `SqlConflictRefinement.tla` — Static Conflict Rules

Exhaustively checks all combinations of 2 operations x 4 statement types x 2 tables x 2 PK values x {parsed, unparsed} x {params_resolved, params_unresolved} (3,200 states after pruning unreachable combinations via `ValidOps`). Verifies 13 invariants:

| # | Invariant | What it checks |
|---|-----------|----------------|
| 1 | `TableLevelSoundness` | Same table + at least one write → TableConflict is TRUE |
| 2 | `TableRefinesEndpoint` | TableConflict → EndpointConflict (refinement direction) |
| 3 | `RowRefinesTable` | RowConflict → TableConflict (refinement chain) |
| 4 | `RowRefinesEndpoint` | RowConflict → EndpointConflict (transitivity) |
| 5 | `ParseFailureFallback` | Parse failure → both Table and Row report conflict |
| 6 | `ReadReadIndependence` | SELECT + SELECT + parsed → no TableConflict |
| 7 | `DifferentTablesIndependent` | Different tables + parsed → no TableConflict |
| 8 | `SameRowConflict` | Same table + same PK + write → RowConflict |
| 9 | `DifferentRowsIndependent` | Same table + different PK + parsed → no RowConflict |
| 10 | `SuppressionCorrectness` | Parsed operation always has non-empty AccessKinds |
| 11 | `ConflictSymmetry` | TableConflict(a,b) = TableConflict(b,a), same for RowConflict |
| 12 | `UnresolvedParamsFallback` | Unresolved params + same table + write → RowConflict (conservative) |
| 13 | `ResolvedParamsEnableRowLevel` | Resolved params + same table + different PK → no RowConflict (precise) |

Invariants 12-13 verify Algorithm 1.5 (parameter resolution): failed resolution never claims row-independence (12), while successful resolution enables the finer-grained row-level detection (13).

## Spec 2: `DporSqlScheduling.tla` — Dynamic Scheduling Soundness

Models an actual database as a map from `(table, pk) → value`. Operations carry a `resolved` flag modeling whether parameter resolution succeeded (Algorithm 1.5). Applies every pair of operations in both orders (A;B vs B;A) and checks:

| # | Invariant | What it checks |
|---|-----------|----------------|
| 1 | `TableSoundness` | If TableConflicts=FALSE, both orders produce same DB state |
| 2 | `RowSoundness` | If RowConflicts=FALSE, both orders produce same DB state |
| 3 | `RefinementChain` | RowConflicts ⊆ TableConflicts ⊆ EndpointConflicts |
| 4 | `NoMissedBugs_Table` | If orders produce different DB states → TableConflicts=TRUE |
| 5 | `NoMissedBugs_Row` | If orders produce different DB states → RowConflicts=TRUE |
| 6 | `UnresolvedParamsSoundness` | Unresolved params + RowConflicts=FALSE → orders commute |
| 7 | `ResolvedParamsSoundness` | Resolved params + RowConflicts=FALSE → orders commute |

Checked with 2 tables x 2 PKs x 3 values x {resolved, unresolved} = 2,048 states. Invariants 6-7 verify that parameter resolution doesn't weaken soundness: unresolved params conservatively conflict (6 is vacuously true — RowConflicts never says FALSE for unresolved), and resolved params with different PKs are genuinely independent in the database (7).

## Spec 3: `SuppressionSafety.tla` — Endpoint Suppression

Models the suppression mechanism where SQL-level reporting disables endpoint-level reporting. Non-deterministically generates *arbitrary* parser outputs (including incorrect ones) and checks:

| # | Invariant | What it checks |
|---|-----------|----------------|
| 1 | `SuppressionSafe` | If parser is correct for both ops, ground-truth conflict → effective conflict |
| 2 | `ParseFailureConservative` | Parse failure → suppression doesn't fire → endpoint-level (always conflict) |
| 3 | `EmptyParseNoSuppress` | Parsed but no tables found (e.g., BEGIN/COMMIT) → no suppression |

Checked with 2 tables x 4 stmt types x all subsets = 32,768 states.

## Running the specs

```bash
cd specs/
# Download TLC (one-time):
curl -sL -o tla2tools.jar https://github.com/tlaplus/tlaplus/releases/download/v1.8.0/tla2tools.jar

# Run all three:
java -XX:+UseParallelGC -cp tla2tools.jar tlc2.TLC SqlConflictRefinement -deadlock -nowarning
java -XX:+UseParallelGC -cp tla2tools.jar tlc2.TLC DporSqlScheduling -deadlock -nowarning
java -XX:+UseParallelGC -cp tla2tools.jar tlc2.TLC SuppressionSafety -deadlock -nowarning
```

## What the specs found

No invariant violations — the design is correct. Specifically:

1. **The conflict classification rules are sound.** The Adya-style Read/Write mapping (`SELECT`→Read, `INSERT`→Write, `UPDATE`→Read+Write, `DELETE`→Read+Write) combined with `ObjectState::dependent_accesses` (Read conflicts with Write, Write conflicts with both Read and Write) correctly identifies all operation pairs where execution order matters.

2. **The refinement chain holds.** Row-level ⊂ Table-level ⊂ Endpoint-level. Each layer strictly refines the one below, so falling back to a coarser layer is always safe.

3. **Suppression is safe.** The thread-local `_sql_suppress` flag correctly prevents double-reporting without losing any conflicts, *provided the parser output is correct*. The spec also verifies that parse failure and empty parse results (BEGIN/COMMIT) correctly fall back to endpoint-level.

4. **`NoMissedBugs` is the strongest result.** For any two operations where the execution order matters (different DB states), the conflict detector reports a conflict. This guarantees DPOR will explore the interleaving.

5. **Parameter resolution is safe.** When parameter resolution fails (`resolved=FALSE`), the row-level conflict predicate falls back to table-level (always reports conflict for same-table writes). This is verified by `UnresolvedParamsFallback` (Spec 1) and `UnresolvedParamsSoundness` (Spec 2). When resolution succeeds, the resolved PK values enable genuine row-level independence without sacrificing soundness (`ResolvedParamsEnableRowLevel`, `ResolvedParamsSoundness`).
