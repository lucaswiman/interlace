"""Row-level predicate extraction for SQL conflict detection.

Extracts WHERE clause equality predicates from SQL statements and provides
functions to determine if two operations on the same table target disjoint
rows (enabling finer-grained conflict detection than table-level).

Requires sqlglot for SQL parsing; returns empty results if unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EqualityPredicate:
    """A simple WHERE column = literal predicate."""

    column: str
    value: str  # string representation of the literal


def extract_equality_predicates(sql: str) -> list[EqualityPredicate]:
    """Extract simple equality predicates from a SQL WHERE clause.

    Only extracts conjuncts of the form ``column = literal`` (ANDed together).
    Returns empty list for OR, IN, BETWEEN, subqueries, function calls, etc.
    """
    try:
        import sqlglot  # type: ignore[import-untyped]
        from sqlglot import exp  # type: ignore[import-untyped]
    except ImportError:
        return []

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
            continue  # skip non-equality predicates (OR, IN, BETWEEN, etc.)
        left, right = conjunct.this, conjunct.expression
        # Normalize: column on left, literal on right
        if isinstance(left, exp.Column) and isinstance(right, exp.Literal):
            predicates.append(EqualityPredicate(left.name, right.this))
        elif isinstance(right, exp.Column) and isinstance(left, exp.Literal):
            predicates.append(EqualityPredicate(right.name, left.this))

    return predicates


def can_use_row_level(
    table: str,
    predicates_a: list[EqualityPredicate],
    predicates_b: list[EqualityPredicate],
    pk_columns: set[str] | None,
) -> bool:
    """Can we safely use row-level ObjectIds for these two operations?

    Row-level is sound when both operations have equality predicates on
    ALL primary key columns.  Falls back to table-level otherwise.
    """
    if not predicates_a or not predicates_b:
        return False  # one has no WHERE → full table scan
    if pk_columns is None:
        return False  # unknown schema → conservative

    cols_a = {p.column for p in predicates_a}
    cols_b = {p.column for p in predicates_b}

    # Both must have equality predicates on ALL primary key columns
    return pk_columns <= cols_a and pk_columns <= cols_b


def pk_predicates_disjoint(
    preds_a: list[EqualityPredicate],
    preds_b: list[EqualityPredicate],
) -> bool:
    """Are two sets of PK equality predicates provably disjoint?

    True if any shared column has different values.
    E.g., ``(id=1)`` vs ``(id=2)`` → True (disjoint).
         ``(id=1, region='us')`` vs ``(id=1, region='eu')`` → True.
         ``(id=1)`` vs ``(id=1)`` → False (same row).
    """
    a_map = {p.column: p.value for p in preds_a}
    b_map = {p.column: p.value for p in preds_b}
    for col in a_map:
        if col in b_map and a_map[col] != b_map[col]:
            return True
    return False
