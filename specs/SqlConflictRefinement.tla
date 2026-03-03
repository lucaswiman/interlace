---- MODULE SqlConflictRefinement ----
(***************************************************************************
 TLA+ specification for verifying SQL conflict detection refinement.

 Models two conflict detection strategies:
   1. ENDPOINT-LEVEL: All SQL to the same (host,port) is one object.
      Every pair of operations conflicts (conservative).
   2. TABLE-LEVEL: Parse SQL to extract table names.  Operations on
      different tables are independent; same-table uses Read/Write rules.

 The core property: TABLE-LEVEL is a SOUND REFINEMENT of ENDPOINT-LEVEL.
 If table-level says "independent" for two operations, then swapping their
 order cannot change the observable result — so DPOR may safely skip that
 interleaving without missing bugs.

 We also model:
   - Suppression: when SQL parse succeeds, endpoint report is suppressed
   - Parse failure fallback: unparseable SQL falls back to endpoint-level
   - Row-level refinement: same table, different PK → independent
   - Parameter resolution: parameterized queries (WHERE id = %s) require
     placeholder substitution before predicate extraction.  When resolution
     fails, row-level falls back to table-level (pk treated as unknown).

 The model checks these invariants across all possible combinations of
 2 threads × {SELECT, INSERT, UPDATE, DELETE} × {table_a, table_b} ×
 {pk=1, pk=2, pk=none} × {parse_success, parse_failure} ×
 {params_resolved, params_unresolved}.
 ***************************************************************************)

EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS
    Threads,       \* e.g. {1, 2}
    Tables,        \* e.g. {"accounts", "users"}
    PKValues,      \* e.g. {1, 2}  (primary key values for row-level)
    StmtTypes      \* e.g. {"SELECT", "INSERT", "UPDATE", "DELETE"}

(* --------------------------------------------------------------------------
   SQL Statement Model
   -------------------------------------------------------------------------- *)

\* A SQL operation issued by a thread.
\* read_tables:  set of tables read (FROM, JOIN, WHERE scan)
\* write_tables: set of tables written (INSERT INTO, UPDATE, DELETE FROM)
\* pk:           primary key predicate value, or 0 if no predicate / unresolved
\* parsed:       TRUE if SQL parsing succeeded
\* param_resolved: TRUE if parameter placeholders were successfully
\*                 substituted with actual values (Algorithm 1.5).
\*                 When FALSE, pk is forced to 0 (row-level falls back
\*                 to table-level).  Covers: parameterized queries with
\*                 failed resolution, queries with no parameters (trivially
\*                 resolved), and queries with literal WHERE values.
SqlOps == [
    stmt_type      : StmtTypes,
    table          : Tables,
    pk             : PKValues \union {0},    \* 0 = no PK predicate (full scan)
    parsed         : BOOLEAN,
    param_resolved : BOOLEAN
]

\* Reachable operations: constrain impossible combinations.
\*   - Can't resolve params if parsing failed
\*   - Can't have a concrete PK if params weren't resolved
ValidOps == { o \in SqlOps :
    /\ (~o.parsed => ~o.param_resolved)
    /\ (~o.param_resolved => o.pk = 0)
}

(* --------------------------------------------------------------------------
   Access Kind Classification
   --------------------------------------------------------------------------
   Maps SQL statement type to (read_tables, write_tables) exactly as
   Algorithm 1 in SQL_CONFLICT.md does.
   - SELECT: reads target table
   - INSERT: writes target table
   - UPDATE: reads AND writes target table (WHERE clause scans rows)
   - DELETE: reads AND writes target table
   -------------------------------------------------------------------------- *)

ReadsTable(op) ==
    op.stmt_type \in {"SELECT", "UPDATE", "DELETE"}

WritesTable(op) ==
    op.stmt_type \in {"INSERT", "UPDATE", "DELETE"}

\* The access kind(s) an operation performs on its table.
\* Returns a set: one of {"read"}, {"write"}, or {"read", "write"}.
AccessKinds(op) ==
    LET r == IF ReadsTable(op) THEN {"read"} ELSE {}
        w == IF WritesTable(op) THEN {"write"} ELSE {}
    IN r \union w

(* --------------------------------------------------------------------------
   Conflict Detection: Endpoint-Level (Abstract / Current System)
   --------------------------------------------------------------------------
   All operations go to the same database endpoint.
   ANY two operations by different threads are treated as conflicting
   (both reported as "write to socket:host:port").
   -------------------------------------------------------------------------- *)

EndpointConflict(op1, op2) ==
    \* All operations to the same DB conflict at endpoint level.
    \* This is the current (conservative) behavior.
    TRUE

(* --------------------------------------------------------------------------
   Conflict Detection: Table-Level (Refined System)
   --------------------------------------------------------------------------
   Two operations conflict iff they touch the same table AND at least one
   is a write.  This is the Adya Read/Write dependency model:
     Read-Read   = independent
     Read-Write  = conflict (WR dependency)
     Write-Read  = conflict (RW anti-dependency)
     Write-Write = conflict (WW dependency)
   -------------------------------------------------------------------------- *)

\* Do two operations touch any table in common?
SameTable(op1, op2) == op1.table = op2.table

\* At least one operation writes.
AtLeastOneWrite(op1, op2) ==
    WritesTable(op1) \/ WritesTable(op2)

TableConflict(op1, op2) ==
    IF ~op1.parsed \/ ~op2.parsed
    THEN
        \* Parse failure: fall back to endpoint-level (conservative)
        EndpointConflict(op1, op2)
    ELSE
        \* Both parsed: use table-level rules
        SameTable(op1, op2) /\ AtLeastOneWrite(op1, op2)

(* --------------------------------------------------------------------------
   Conflict Detection: Row-Level (Further Refinement)
   --------------------------------------------------------------------------
   Same table, but different primary key values → independent.
   Only applicable when BOTH operations have a PK predicate (pk /= 0).
   -------------------------------------------------------------------------- *)

RowConflict(op1, op2) ==
    IF ~op1.parsed \/ ~op2.parsed
    THEN
        \* Parse failure: fall back to endpoint-level
        EndpointConflict(op1, op2)
    ELSE IF ~SameTable(op1, op2)
    THEN
        \* Different tables: independent
        FALSE
    ELSE IF ~AtLeastOneWrite(op1, op2)
    THEN
        \* Same table, both reads: independent
        FALSE
    ELSE IF op1.pk = 0 \/ op2.pk = 0
    THEN
        \* At least one has no PK predicate → table-level fallback
        TRUE
    ELSE
        \* Both have PK predicates: conflict iff same PK value
        op1.pk = op2.pk

(* --------------------------------------------------------------------------
   State Machine
   --------------------------------------------------------------------------
   Non-deterministically choose operations for two threads, then check
   that the refinement properties hold.
   -------------------------------------------------------------------------- *)

VARIABLES op1, op2, checked

vars == <<op1, op2, checked>>

Init ==
    /\ op1 \in ValidOps
    /\ op2 \in ValidOps
    /\ checked = FALSE

\* Single step: check properties and mark done.
Check ==
    /\ checked = FALSE
    /\ checked' = TRUE
    /\ UNCHANGED <<op1, op2>>

Next == Check

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

(* --------------------------------------------------------------------------
   INVARIANTS — The Properties We Verify
   -------------------------------------------------------------------------- *)

\* INVARIANT 1: Table-level is a sound refinement of "true conflict"
\*
\* Meaning: if table-level says "no conflict" (independent), then the
\* operations genuinely access different resources.  We can't state this
\* directly (we don't have a ground truth oracle), but we CAN state:
\*
\* Table-level NEVER says "independent" when the operations touch the
\* same table with at least one write.  I.e., table-level correctly
\* identifies all same-table read-write/write-write conflicts.
\*
\* This is trivially true by construction, but the model checker verifies
\* there's no edge case in the AccessKinds logic.

TableLevelSoundness ==
    (SameTable(op1, op2) /\ AtLeastOneWrite(op1, op2) /\ op1.parsed /\ op2.parsed)
    => TableConflict(op1, op2)

\* INVARIANT 2: Table-level is a refinement of endpoint-level.
\*
\* If table-level says "conflict", endpoint-level must also say "conflict".
\* Contrapositive: if endpoint says "independent", table must also say
\* "independent".  Since endpoint ALWAYS says "conflict", this is vacuously
\* true, but we verify it anyway to catch any future logic changes.

TableRefinesEndpoint ==
    TableConflict(op1, op2) => EndpointConflict(op1, op2)

\* INVARIANT 3: Row-level is a refinement of table-level.
\*
\* If row-level says "conflict", table-level must also say "conflict".
\* Equivalently: if table-level says "independent", row-level must too.

RowRefinesTable ==
    RowConflict(op1, op2) => TableConflict(op1, op2)

\* INVARIANT 4: Row-level is a refinement of endpoint-level (transitivity).

RowRefinesEndpoint ==
    RowConflict(op1, op2) => EndpointConflict(op1, op2)

\* INVARIANT 5: Parse failure is conservative.
\*
\* If either operation fails to parse, table-level and row-level both
\* fall back to endpoint-level (which always reports conflict).

ParseFailureFallback ==
    (~op1.parsed \/ ~op2.parsed)
    => (TableConflict(op1, op2) /\ RowConflict(op1, op2))

\* INVARIANT 6: Read-Read independence.
\*
\* Two SELECTs on the same table with successful parsing must be
\* independent at table-level and row-level.

ReadReadIndependence ==
    (op1.stmt_type = "SELECT" /\ op2.stmt_type = "SELECT"
     /\ op1.parsed /\ op2.parsed)
    => (~TableConflict(op1, op2))

\* INVARIANT 7: Different tables are independent (when parsing succeeds).

DifferentTablesIndependent ==
    (op1.table /= op2.table /\ op1.parsed /\ op2.parsed)
    => (~TableConflict(op1, op2))

\* INVARIANT 8: Same table, same PK, at least one write → row-level conflict.

SameRowConflict ==
    (SameTable(op1, op2) /\ op1.pk /= 0 /\ op2.pk /= 0
     /\ op1.pk = op2.pk /\ AtLeastOneWrite(op1, op2)
     /\ op1.parsed /\ op2.parsed)
    => RowConflict(op1, op2)

\* INVARIANT 9: Same table, different PK, both have PK → row-level independent.

DifferentRowsIndependent ==
    (SameTable(op1, op2) /\ op1.pk /= 0 /\ op2.pk /= 0
     /\ op1.pk /= op2.pk /\ op1.parsed /\ op2.parsed)
    => (~RowConflict(op1, op2))

\* INVARIANT 10: Suppression correctness.
\*
\* The suppression flag (_sql_suppress) is set iff parsing succeeded AND
\* at least one table was found.  Endpoint reports are skipped only when
\* suppression is active.  This means: if suppression is active, SQL-level
\* MUST have reported something (at least one table).
\*
\* Model this as: parsed=TRUE implies AccessKinds is non-empty for all
\* statement types in our classification.

SuppressionCorrectness ==
    op1.parsed => (AccessKinds(op1) /= {})

(* --------------------------------------------------------------------------
   Conflict Symmetry — operations from thread 1 vs thread 2 are
   interchangeable.  This verifies our conflict functions are symmetric.
   -------------------------------------------------------------------------- *)

ConflictSymmetry ==
    /\ TableConflict(op1, op2) = TableConflict(op2, op1)
    /\ RowConflict(op1, op2)   = RowConflict(op2, op1)

(* --------------------------------------------------------------------------
   Parameter Resolution Invariants (Algorithm 1.5)
   -------------------------------------------------------------------------- *)

\* INVARIANT 12: Unresolved parameters are conservative.
\*
\* When parameter resolution fails for either operation, row-level
\* detection MUST NOT claim row-independence.  It falls back to table-level:
\* same table + at least one write → conflict, regardless of PK values.
\*
\* This guarantees that parameterized queries like
\*   cursor.execute("UPDATE accounts SET balance = %s WHERE id = %s", (100, 1))
\* where resolution fails (e.g., unsupported paramstyle) are handled
\* conservatively — DPOR will explore the interleaving.

UnresolvedParamsFallback ==
    (op1.parsed /\ op2.parsed
     /\ (~op1.param_resolved \/ ~op2.param_resolved)
     /\ SameTable(op1, op2) /\ AtLeastOneWrite(op1, op2))
    => RowConflict(op1, op2)

\* INVARIANT 13: Resolved parameters enable row-level precision.
\*
\* When BOTH operations are parsed AND parameters are resolved,
\* AND they target different PKs on the same table, row-level
\* correctly reports them as independent (even if both write).
\* This is the payoff of parameter resolution.

ResolvedParamsEnableRowLevel ==
    (op1.parsed /\ op2.parsed
     /\ op1.param_resolved /\ op2.param_resolved
     /\ SameTable(op1, op2)
     /\ op1.pk /= 0 /\ op2.pk /= 0
     /\ op1.pk /= op2.pk)
    => (~RowConflict(op1, op2))

(* --------------------------------------------------------------------------
   Exhaustive check: combine all invariants.
   -------------------------------------------------------------------------- *)

AllInvariants ==
    /\ TableLevelSoundness
    /\ TableRefinesEndpoint
    /\ RowRefinesTable
    /\ RowRefinesEndpoint
    /\ ParseFailureFallback
    /\ ReadReadIndependence
    /\ DifferentTablesIndependent
    /\ SameRowConflict
    /\ DifferentRowsIndependent
    /\ SuppressionCorrectness
    /\ ConflictSymmetry
    /\ UnresolvedParamsFallback
    /\ ResolvedParamsEnableRowLevel

====
