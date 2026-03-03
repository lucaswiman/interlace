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
ReadOp(t, k)    == [type |-> "read",  table |-> t, pk |-> k, val |-> 0]
WriteOp(t, k, v) == [type |-> "write", table |-> t, pk |-> k, val |-> v]

\* All possible operations
AllOps == { ReadOp(t, k) : t \in Tables, k \in PKs }
    \union { WriteOp(t, k, v) : t \in Tables, k \in PKs, v \in Values }

(* --------------------------------------------------------------------------
   Conflict Predicates (matching Algorithm 1 from SQL_CONFLICT.md)
   -------------------------------------------------------------------------- *)

\* Endpoint-level: all ops conflict (same socket)
EndpointConflicts(a, b) == TRUE

\* Table-level: same table, at least one write
TableConflicts(a, b) ==
    /\ a.table = b.table
    /\ (a.type = "write" \/ b.type = "write")

\* Row-level: same table, same pk, at least one write
RowConflicts(a, b) ==
    /\ a.table = b.table
    /\ a.pk = b.pk
    /\ (a.type = "write" \/ b.type = "write")

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

\* Combined

AllInvariants ==
    /\ TableSoundness
    /\ RowSoundness
    /\ RefinementChain
    /\ NoMissedBugs_Table
    /\ NoMissedBugs_Row

====
