# Algorithm 5: Row-Level Predicate Extraction (Phase 2)

Extract WHERE clause predicates from the parsed SQL and encode them as part of the ObjectId, enabling two operations on the same table but different rows to be independent.

## 5a. Predicate Extraction via sqlglot

```python
from sqlglot import exp

@dataclass(frozen=True)
class EqualityPredicate:
    """A simple WHERE column = literal predicate."""
    column: str
    value: str  # string representation of the literal

def extract_equality_predicates(sql: str) -> list[EqualityPredicate]:
    """Extract simple equality predicates from a SQL WHERE clause.

    Only extracts conjuncts of the form `column = literal` (ANDed together).
    Returns empty list for OR, IN, BETWEEN, subqueries, function calls, etc.
    """
    import sqlglot
    try:
        ast = sqlglot.parse_one(sql)
    except sqlglot.errors.ParseError:
        return []

    where = ast.find(exp.Where)
    if where is None:
        return []

    predicates: list[EqualityPredicate] = []
    predicate_expr = where.this

    # Flatten ANDs into individual conjuncts
    conjuncts: list[exp.Expression]
    if isinstance(predicate_expr, exp.And):
        conjuncts = list(predicate_expr.flatten())
    else:
        conjuncts = [predicate_expr]

    for conjunct in conjuncts:
        if not isinstance(conjunct, exp.EQ):
            continue  # skip non-equality predicates
        left, right = conjunct.this, conjunct.expression
        # Normalize: column on left, literal on right
        if isinstance(left, exp.Column) and isinstance(right, exp.Literal):
            predicates.append(EqualityPredicate(left.name, right.this))
        elif isinstance(right, exp.Column) and isinstance(left, exp.Literal):
            predicates.append(EqualityPredicate(right.name, left.this))

    return predicates
```

## 5b. Row-Level ObjectId

When predicates are available, derive a finer-grained ObjectId that encodes the table *and* the row predicate:

```python
def sql_row_object_key(table: str, predicates: list[EqualityPredicate]) -> int:
    """Derive ObjectId from table + WHERE equality predicates.

    If predicates are present, the key includes them — so
    "UPDATE accounts WHERE id=1" and "SELECT accounts WHERE id=2"
    get different ObjectIds and are independent.

    If no predicates, falls back to table-level key.
    """
    if not predicates:
        # No predicates → table-level granularity
        return _make_object_key(hash(f"sql:{table}"), f"sql:{table}")

    # Sort predicates for deterministic hashing
    pred_key = tuple(sorted((p.column, p.value) for p in predicates))
    resource_id = f"sql:{table}:{pred_key}"
    return _make_object_key(hash(resource_id), resource_id)
```

## 5c. Soundness: When Row-Level is Safe

Row-level ObjectIds are sound (no missed conflicts) when:

1. The WHERE clause is a conjunction of equalities on the *primary key* columns.
2. Both operations have complete primary key predicates.

If either operation has no WHERE clause (full table scan), range predicates, OR conditions, or predicates on non-key columns, we must fall back to table-level (conservative, correct).

**Decision function:**

```python
def can_use_row_level(
    table: str,
    predicates_a: list[EqualityPredicate],
    predicates_b: list[EqualityPredicate],
    pk_columns: set[str] | None,  # from schema, or None if unknown
) -> bool:
    """Can we safely use row-level ObjectIds for these two operations?"""
    if not predicates_a or not predicates_b:
        return False  # one has no WHERE → full table scan
    if pk_columns is None:
        return False  # unknown schema → conservative

    cols_a = {p.column for p in predicates_a}
    cols_b = {p.column for p in predicates_b}

    # Both must have equality predicates on ALL primary key columns
    return pk_columns <= cols_a and pk_columns <= cols_b
```

## 5d. Row-Level Disjointness (No z3 Needed for Equalities)

When `can_use_row_level` is true, two operations are independent iff their PK predicates select different rows:

```python
def pk_predicates_disjoint(
    preds_a: list[EqualityPredicate],
    preds_b: list[EqualityPredicate],
) -> bool:
    """Are two sets of PK equality predicates provably disjoint?

    True if any shared column has different values.
    E.g., (id=1) vs (id=2) → True (disjoint).
         (id=1, region='us') vs (id=1, region='eu') → True.
         (id=1) vs (id=1) → False (same row).
    """
    a_map = {p.column: p.value for p in preds_a}
    b_map = {p.column: p.value for p in preds_b}
    for col in a_map:
        if col in b_map and a_map[col] != b_map[col]:
            return True
    return False
```

This is an O(k) check where k = number of PK columns. No SMT solver needed. z3 is only needed for range predicates (`WHERE id > 10` vs `WHERE id < 5`), which can be deferred to a later phase.
