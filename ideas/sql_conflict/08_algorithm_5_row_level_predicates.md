# Algorithm 5: Row-Level Predicate Extraction (Phase 2)

Extract WHERE clause predicates from the parsed SQL and encode them as part of the ObjectId, enabling two operations on the same table but different rows to be independent.

## 5a. Predicate Extraction via sqlglot

Extracts two predicate types from AND-conjuncts in WHERE clauses:

```python
from sqlglot import exp

@dataclass(frozen=True)
class EqualityPredicate:
    """A simple WHERE column = literal predicate."""
    column: str
    value: str  # string representation of the literal

@dataclass(frozen=True)
class InListPredicate:
    """A WHERE column IN (v1, v2, ...) predicate."""
    column: str
    values: frozenset[str]

Predicate = EqualityPredicate | InListPredicate

def extract_equality_predicates(sql: str) -> list[Predicate]:
    """Extract equality and IN-list predicates from a SQL WHERE clause.

    Extracts conjuncts (ANDed together) of these forms:
    - column = literal → EqualityPredicate
    - column IN (lit1, lit2, ...) → InListPredicate

    Returns empty list for OR, BETWEEN, subqueries, function calls, etc.
    IN-lists with non-literal values or exceeding _MAX_IN_LIST_SIZE (32)
    are skipped (falls back to table-level).
    """
```

## 5b. Predicate Expansion for Resource IDs

IN-list predicates must be expanded into per-row resource IDs for the DPOR scheduler. The scheduler uses ObjectId equality for conflict detection, so each row an operation touches needs its own report.

```python
def expand_predicate_rows(
    preds: list[Predicate],
) -> list[list[EqualityPredicate]] | None:
    """Expand IN-list predicates into per-row equality predicate sets.

    WHERE id = 1           → [[(id, 1)]]
    WHERE id IN (1, 2)     → [[(id, 1)], [(id, 2)]]
    WHERE id IN (1, 2) AND region = 'us'
        → [[(region, us), (id, 1)], [(region, us), (id, 2)]]

    Returns None if expansion would exceed _MAX_EXPANSION (64),
    in which case the caller falls back to table-level.
    """
```

In `_intercept_execute`, each expanded row generates a separate `reporter()` call:

```python
pred_rows: list[list[Any]] = [[]]  # default: table-level (one report, no predicates)
if preds:
    expanded = expand_predicate_rows(preds)
    if expanded is not None:
        pred_rows = expanded

for row_preds in pred_rows:
    res_id = _sql_resource_id(table, row_preds)
    reporter(res_id, kind)
```

## 5c. Soundness: When Row-Level is Safe

Row-level ObjectIds are sound (no missed conflicts) when:

1. The WHERE clause is a conjunction of equalities/IN-lists on the *primary key* columns.
2. Both operations have complete primary key predicates.
3. IN-lists are expanded into per-row reports (each value gets its own ObjectId).

If either operation has no WHERE clause (full table scan), range predicates, OR conditions, or predicates on non-key columns, we fall back to table-level (conservative, correct).

**Decision function:**

```python
def can_use_row_level(
    table: str,
    predicates_a: list[Predicate],
    predicates_b: list[Predicate],
    pk_columns: set[str] | None,  # from schema, or None if unknown
) -> bool:
    """Can we safely use row-level ObjectIds for these two operations?"""
    if not predicates_a or not predicates_b:
        return False  # one has no WHERE → full table scan
    if pk_columns is None:
        return False  # unknown schema → conservative

    cols_a = {p.column for p in predicates_a}
    cols_b = {p.column for p in predicates_b}

    # Both must have predicates on ALL primary key columns
    return pk_columns <= cols_a and pk_columns <= cols_b
```

## 5d. Row-Level Disjointness (No z3 Needed)

When `can_use_row_level` is true, two operations are independent iff their value sets for any shared PK column are disjoint:

```python
def pk_predicates_disjoint(
    preds_a: list[Predicate],
    preds_b: list[Predicate],
) -> bool:
    """Are two sets of PK predicates provably disjoint?

    Handles both EqualityPredicate (single value) and InListPredicate (set).
    True if any shared column's value sets are disjoint.

    (id=1)              vs (id=2)              → True
    (id IN (1,2,3))     vs (id IN (4,5,6))     → True
    (id IN (1,2,3))     vs (id=2)              → False (2 overlaps)
    (id=1)              vs (id=1)              → False (same row)
    """
    def _values_set(preds):
        m = {}
        for p in preds:
            if isinstance(p, EqualityPredicate):
                m.setdefault(p.column, set()).add(p.value)
            elif isinstance(p, InListPredicate):
                m.setdefault(p.column, set()).update(p.values)
        return m

    a_map = _values_set(preds_a)
    b_map = _values_set(preds_b)
    for col in a_map:
        if col in b_map and a_map[col].isdisjoint(b_map[col]):
            return True
    return False
```

This is an O(k) check where k = number of PK columns. No SMT solver needed — equality lookups and IN-lists are handled by simple set disjointness. See [13_phased_implementation.md#design-note-why-not-z3smt-for-row-level-conflicts](13_phased_implementation.md#design-note-why-not-z3smt-for-row-level-conflicts) for the full rationale on why z3 is not used.
