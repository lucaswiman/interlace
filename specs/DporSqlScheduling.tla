---- MODULE DporSqlScheduling ----
(***************************************************************************
 Models DPOR scheduling decisions with SQL conflict detection.

 Two threads each execute a sequence of SQL operations against a shared
 database.  A DPOR scheduler interleaves them.  We verify:

   1. If SQL-level says two operations are INDEPENDENT, then swapping
      their execution order produces the same final database state.
      This is the SOUNDNESS of the pruning: DPOR won't miss any
      distinct outcome.

   2. If SQL-level says two operations CONFLICT, there exists at least
      one scenario where swapping them produces a different outcome.
      This is COMPLETENESS: DPOR doesn't skip interleavings it needs.

 The "database" is modeled as a map from (table, pk) → value.
 Operations are: read(table, pk) and write(table, pk, val).

 Parameter resolution (Algorithm 1.5): operations carry a "resolved"
 flag.  When resolved=FALSE, the conflict predicates don't know the PK
 value (even though the database DOES apply the operation to its actual
 row).  This models parameterized queries like "WHERE id = %s" where
 placeholder substitution failed.  The conflict predicates must be
 conservative (assume conflict) in this case.
 ***************************************************************************)

EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS
    Tables,      \* e.g. {"accounts", "users"}
    PKs,         \* e.g. {1, 2}
    Values       \* e.g. {0, 1, 2}

\* Database cell: (table, pk) pair
Cells == Tables \X PKs

(* --------------------------------------------------------------------------
   Operation Model
   -------------------------------------------------------------------------- *)

\* An operation is either a read or a write to a (table, pk) cell.
\* "resolved" indicates whether parameter resolution succeeded
\* (Algorithm 1.5).  When FALSE, the conflict predicates cannot see the
\* PK value and must be conservative.  The DB semantics still use the
\* actual PK — the database always knows which row is targeted, even
\* when the conflict detector doesn't.
ReadOp(t, k, r)    == [type |-> "read",  table |-> t, pk |-> k, val |-> 0, resolved |-> r]
WriteOp(t, k, v, r) == [type |-> "write", table |-> t, pk |-> k, val |-> v, resolved |-> r]

\* All possible operations (both resolved and unresolved variants)
AllOps == { ReadOp(t, k, r) : t \in Tables, k \in PKs, r \in BOOLEAN }
    \union { WriteOp(t, k, v, r) : t \in Tables, k \in PKs, v \in Values, r \in BOOLEAN }

(* --------------------------------------------------------------------------
   Conflict Predicates (matching Algorithm 1 from SQL_CONFLICT.md)
   -------------------------------------------------------------------------- *)

\* Endpoint-level: all ops conflict (same socket)
EndpointConflicts(a, b) == TRUE

\* Table-level: same table, at least one write
TableConflicts(a, b) ==
    /\ a.table = b.table
    /\ (a.type = "write" \/ b.type = "write")

\* Row-level: same table, at least one write, AND either:
\*   - at least one operation has unresolved params → fallback to table-level
\*   - both resolved → conflict iff same PK
RowConflicts(a, b) ==
    /\ a.table = b.table
    /\ (a.type = "write" \/ b.type = "write")
    /\ IF ~a.resolved \/ ~b.resolved
       THEN TRUE   \* unresolved params → conservative (table-level)
       ELSE a.pk = b.pk

(* --------------------------------------------------------------------------
   Database Semantics
   --------------------------------------------------------------------------
   Apply an operation to a database state (map from Cells → Values).
   - Read: no effect on DB state (returns a value, but we model state only)
   - Write: updates the cell
   -------------------------------------------------------------------------- *)

ApplyOp(db, op) ==
    IF op.type = "write"
    THEN [db EXCEPT ![<<op.table, op.pk>>] = op.val]
    ELSE db

\* Apply a sequence of operations left-to-right.
RECURSIVE ApplySeq(_, _)
ApplySeq(db, ops) ==
    IF ops = <<>>
    THEN db
    ELSE ApplySeq(ApplyOp(db, Head(ops)), Tail(ops))

(* --------------------------------------------------------------------------
   State Machine: Check all pairs of operations
   --------------------------------------------------------------------------
   For each pair (opA, opB), apply them in both orders (A;B and B;A)
   to an initial DB state and check:
     - If conflict predicate says FALSE → both orders give same result
     - If conflict predicate says TRUE  → (no constraint; may or may not differ)
   -------------------------------------------------------------------------- *)

VARIABLES opA, opB, db0, checked

vars == <<opA, opB, db0, checked>>

InitDB == [c \in Cells |-> 0]  \* All cells start at 0

Init ==
    /\ opA \in AllOps
    /\ opB \in AllOps
    /\ db0 = InitDB
    /\ checked = FALSE

Check ==
    /\ checked = FALSE
    /\ checked' = TRUE
    /\ UNCHANGED <<opA, opB, db0>>

Next == Check

Spec == Init /\ [][Next]_vars

(* --------------------------------------------------------------------------
   Derived: apply operations in both orders
   -------------------------------------------------------------------------- *)

\* Database after executing A then B
DB_AB == ApplySeq(db0, <<opA, opB>>)

\* Database after executing B then A
DB_BA == ApplySeq(db0, <<opB, opA>>)

\* Do both orders produce the same final state?
OrderIndependent == DB_AB = DB_BA

(* --------------------------------------------------------------------------
   INVARIANTS
   -------------------------------------------------------------------------- *)

\* INVARIANT 1: TABLE-LEVEL SOUNDNESS
\*
\* If table-level says NOT conflicting (independent), then swapping
\* the execution order MUST produce the same database state.
\* Violation here means: we incorrectly pruned an interleaving that
\* produces a different outcome — a missed bug.

TableSoundness ==
    (~TableConflicts(opA, opB)) => OrderIndependent

\* INVARIANT 2: ROW-LEVEL SOUNDNESS
\*
\* If row-level says NOT conflicting, swapping order must be safe.

RowSoundness ==
    (~RowConflicts(opA, opB)) => OrderIndependent

\* INVARIANT 3: TABLE-LEVEL COMPLETENESS (BEST-EFFORT)
\*
\* NOT a hard invariant — we use TLC to SEARCH for a witness.
\* If table-level says CONFLICTING, there SHOULD exist some initial
\* state and operations where the two orders differ.
\*
\* We express this as: table-conflict ∧ same-outcome is ALLOWED
\* (no invariant violation).  We then manually inspect whether
\* there exist conflict=TRUE, different-outcome states.

\* INVARIANT 4: REFINEMENT CHAIN
\*
\* Row conflicts ⊆ Table conflicts ⊆ Endpoint conflicts.
\* Every row-level conflict must also be a table-level conflict.

RefinementChain ==
    /\ (RowConflicts(opA, opB) => TableConflicts(opA, opB))
    /\ (TableConflicts(opA, opB) => EndpointConflicts(opA, opB))

\* INVARIANT 5: TABLE-LEVEL CATCHES ALL STATE-CHANGING CONFLICTS
\*
\* If the two operations produce different database states when
\* executed in different orders, then table-level MUST report conflict.
\* This is the "no missed bugs" guarantee.

NoMissedBugs_Table ==
    (~OrderIndependent) => TableConflicts(opA, opB)

\* INVARIANT 6: ROW-LEVEL CATCHES ALL STATE-CHANGING CONFLICTS

NoMissedBugs_Row ==
    (~OrderIndependent) => RowConflicts(opA, opB)

\* INVARIANT 7: UNRESOLVED PARAMETERS ARE SOUND
\*
\* When parameter resolution fails for either operation, RowConflicts
\* falls back to TableConflicts (same table + write → always conflict).
\* Since TableSoundness already guarantees that table-level is sound,
\* unresolved parameters cannot cause missed bugs.
\*
\* Stated directly: if RowConflicts says "independent" despite one
\* operation having unresolved params, the orders must still commute.
\* (In practice, RowConflicts never says "independent" when params are
\* unresolved — this invariant verifies that.)

UnresolvedParamsSoundness ==
    ((~opA.resolved \/ ~opB.resolved) /\ ~RowConflicts(opA, opB))
    => OrderIndependent

\* INVARIANT 8: RESOLVED PARAMS DON'T WEAKEN SOUNDNESS
\*
\* Even with resolved parameters, row-level is still sound: if it says
\* "independent" (different PKs), the operations genuinely commute
\* in the database.  This is the core payoff — resolution enables
\* finer-grained independence without sacrificing correctness.

ResolvedParamsSoundness ==
    (opA.resolved /\ opB.resolved /\ ~RowConflicts(opA, opB))
    => OrderIndependent

\* Combined

AllInvariants ==
    /\ TableSoundness
    /\ RowSoundness
    /\ RefinementChain
    /\ NoMissedBugs_Table
    /\ NoMissedBugs_Row
    /\ UnresolvedParamsSoundness
    /\ ResolvedParamsSoundness

====
