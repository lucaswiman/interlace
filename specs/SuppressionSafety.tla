---- MODULE SuppressionSafety ----
(***************************************************************************
 Verifies the endpoint-suppression mechanism is safe.

 When cursor.execute() fires:
   1. parse_sql_access(sql) runs
   2. If it returns non-empty tables, SQL-level reports are emitted AND
      _sql_suppress is set to True
   3. The original execute() runs (which triggers socket send/recv)
   4. _report_socket_io checks _sql_suppress → skips endpoint report
   5. _sql_suppress is reset to False in finally block

 The danger: if SQL parsing returns tables but MISCLASSIFIES them
 (e.g., reports only read when it should also report write), the
 endpoint-level report is suppressed and the correct write-conflict
 is lost.  DPOR would then incorrectly skip an interleaving.

 This spec models the suppression decision and verifies:
   - If suppression fires, SQL-level must have reported AT LEAST as many
     conflicts as endpoint-level would have.
   - Specifically: SQL-level reports must be a SUPERSET of what the
     endpoint-level would classify as conflicting accesses.

 We model this as: for any operation, endpoint says "write to socket".
 SQL-level says some combination of reads/writes to specific tables.
 Suppression is safe iff: for any SECOND operation by another thread,
 if endpoint would say "conflict", SQL-level also says "conflict".
 ***************************************************************************)

EXTENDS Integers, FiniteSets, TLC

CONSTANTS
    Tables,
    StmtTypes

(* --------------------------------------------------------------------------
   An operation and what the parser produces
   -------------------------------------------------------------------------- *)

\* What the SQL parser returns for an operation
ParserOutputs == [
    tables_read  : SUBSET Tables,
    tables_write : SUBSET Tables,
    parsed       : BOOLEAN
]

\* Two operations and their parse results
VARIABLES op1_stmt, op1_parse, op2_stmt, op2_parse, suppress1, suppress2, checked

vars == <<op1_stmt, op1_parse, op2_stmt, op2_parse, suppress1, suppress2, checked>>

(* --------------------------------------------------------------------------
   Correct parse output for each statement type (ground truth)
   --------------------------------------------------------------------------
   This encodes Algorithm 1's rules.  If the parser is correct, its output
   must match these rules.  We check what happens when it does/doesn't.
   -------------------------------------------------------------------------- *)

\* For a given statement type and table, what's the correct classification?
CorrectReads(stmt, table) ==
    IF stmt \in {"SELECT", "UPDATE", "DELETE"} THEN {table} ELSE {}

CorrectWrites(stmt, table) ==
    IF stmt \in {"INSERT", "UPDATE", "DELETE"} THEN {table} ELSE {}

(* --------------------------------------------------------------------------
   Conflict detection at each level
   -------------------------------------------------------------------------- *)

\* Endpoint level: all operations conflict (same socket, all writes)
EndpointConflict == TRUE

\* SQL level: conflict iff any table has (at least one write, overlapping)
SqlConflict(parse1, parse2) ==
    \E t \in Tables :
        /\ (t \in parse1.tables_read \/ t \in parse1.tables_write)
        /\ (t \in parse2.tables_read \/ t \in parse2.tables_write)
        /\ (t \in parse1.tables_write \/ t \in parse2.tables_write)

(* --------------------------------------------------------------------------
   Suppression logic
   -------------------------------------------------------------------------- *)

\* Suppression fires when parse succeeded and found at least one table
ShouldSuppress(parse) ==
    parse.parsed /\ (parse.tables_read \union parse.tables_write) /= {}

(* --------------------------------------------------------------------------
   State Machine
   -------------------------------------------------------------------------- *)

Init ==
    \* Non-deterministically choose two statement types
    /\ op1_stmt \in StmtTypes
    /\ op2_stmt \in StmtTypes
    \* Non-deterministically choose parse outputs
    \* The parser output may be correct OR incorrect (to find bugs)
    /\ op1_parse \in ParserOutputs
    /\ op2_parse \in ParserOutputs
    \* Suppression state
    /\ suppress1 = ShouldSuppress(op1_parse)
    /\ suppress2 = ShouldSuppress(op2_parse)
    /\ checked = FALSE

Check ==
    /\ checked = FALSE
    /\ checked' = TRUE
    /\ UNCHANGED <<op1_stmt, op1_parse, op2_stmt, op2_parse, suppress1, suppress2>>

Next == Check

Spec == Init /\ [][Next]_vars

(* --------------------------------------------------------------------------
   The effective conflict detection
   --------------------------------------------------------------------------
   If both operations suppress endpoint, SQL-level decides.
   If either doesn't suppress, endpoint-level decides (always conflict).
   -------------------------------------------------------------------------- *)

EffectiveConflict ==
    IF suppress1 /\ suppress2
    THEN SqlConflict(op1_parse, op2_parse)
    ELSE EndpointConflict

(* --------------------------------------------------------------------------
   INVARIANTS
   -------------------------------------------------------------------------- *)

\* INVARIANT 1: SUPPRESSION SAFETY (with correct parser)
\*
\* IF the parser correctly classifies both operations, THEN effective
\* conflict detection is at least as conservative as a hypothetical
\* "ground truth" detector that checks same-table + at-least-one-write.
\*
\* "Correct" means: parse succeeded, and the output matches the
\* statement type's expected read/write classification.

\* Helper: is this parse output correct for the given statement+table?
IsCorrectParse(parse, stmt) ==
    \E t \in Tables :
        /\ parse.parsed
        /\ parse.tables_read  = CorrectReads(stmt, t)
        /\ parse.tables_write = CorrectWrites(stmt, t)

\* Ground truth: two operations on specific tables
\* We can only state ground truth when we know the tables.
\* Extract the table from a correct parse:
ParseTable(parse) ==
    CHOOSE t \in Tables :
        t \in (parse.tables_read \union parse.tables_write)

GroundTruthConflict(parse1, parse2) ==
    LET t1 == ParseTable(parse1)
        t2 == ParseTable(parse2)
    IN t1 = t2 /\ (parse1.tables_write /= {} \/ parse2.tables_write /= {})

\* The main safety property: if the parser is correct for both operations,
\* then the effective conflict detection correctly identifies all ground-truth
\* conflicts.  I.e., ground truth conflict → effective conflict.
SuppressionSafe ==
    (IsCorrectParse(op1_parse, op1_stmt) /\ IsCorrectParse(op2_parse, op2_stmt))
    => (GroundTruthConflict(op1_parse, op2_parse) => EffectiveConflict)

\* INVARIANT 2: PARSE FAILURE IS CONSERVATIVE
\*
\* If either parse fails, suppression doesn't fire, and endpoint-level
\* (which always says conflict) takes over.
ParseFailureConservative ==
    (~op1_parse.parsed \/ ~op2_parse.parsed)
    => (~suppress1 \/ ~suppress2)
    \* Note: if EITHER doesn't suppress, EffectiveConflict = TRUE

\* INVARIANT 3: EMPTY PARSE IS NOT SUPPRESSED
\*
\* If parsing succeeds but finds no tables (e.g., "BEGIN" or "COMMIT"),
\* suppression must NOT fire (otherwise we'd lose the endpoint report).
EmptyParseNoSuppress ==
    (op1_parse.parsed /\ op1_parse.tables_read = {} /\ op1_parse.tables_write = {})
    => ~suppress1

\* Combined
AllInvariants ==
    /\ SuppressionSafe
    /\ ParseFailureConservative
    /\ EmptyParseNoSuppress

====
