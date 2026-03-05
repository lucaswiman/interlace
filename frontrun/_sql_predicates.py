"""Row-level predicate extraction for SQL conflict detection.

Extracts WHERE clause equality predicates and IN-list predicates from SQL
statements. Provides functions to determine if two operations on the same
table target disjoint rows (enabling finer-grained conflict detection than
table-level).

Requires sqlglot for SQL parsing; returns empty results if unavailable.

Design note (z3/SMT): We use simple value comparison and set disjointness
rather than an SMT solver like z3 because:
  1. ORM SQL is overwhelmingly equality lookups (WHERE id = ?) and IN-lists
     (WHERE id IN (?, ?, ?)) — the 95% case needs no solver.
  2. z3 adds ~50ms per check vs nanoseconds for set operations, which matters
     on the hot path (every SQL execute call).
  3. z3-solver is a ~200MB dependency, heavy for a testing library.
  4. Encoding SQL types (VARCHAR, DECIMAL, timestamps, NULL three-valued logic)
     into z3 sorts is error-prone and a maintenance burden.
  5. The fallback is safe: unhandled predicates (ranges, OR, subqueries) fall
     back to table-level conflict detection, which is conservative but sound.
"""

from __future__ import annotations

from dataclasses import dataclass

# Maximum number of values in an IN-list before falling back to table-level.
_MAX_IN_LIST_SIZE = 32

# Maximum total expansion (cross-product of all IN-lists) before falling back.
_MAX_EXPANSION = 64


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
    - ``column = literal`` → :class:`EqualityPredicate`
    - ``column IN (lit1, lit2, ...)`` → :class:`InListPredicate`

    Returns empty list for OR, BETWEEN, subqueries, function calls, etc.
    IN-lists with non-literal values or exceeding :data:`_MAX_IN_LIST_SIZE`
    are skipped (falls back to table-level).
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

    predicates: list[Predicate] = []
    predicate_expr = where.this

    # Flatten ANDs into individual conjuncts
    conjuncts: list[exp.Expression]
    if isinstance(predicate_expr, exp.And):
        conjuncts = list(predicate_expr.flatten())
    else:
        conjuncts = [predicate_expr]

    for conjunct in conjuncts:
        if isinstance(conjunct, exp.EQ):
            left, right = conjunct.this, conjunct.expression
            # Normalize: column on left, literal on right
            if isinstance(left, exp.Column) and isinstance(right, exp.Literal):
                predicates.append(EqualityPredicate(left.name, right.this))
            elif isinstance(right, exp.Column) and isinstance(left, exp.Literal):
                predicates.append(EqualityPredicate(right.name, left.this))
        elif isinstance(conjunct, exp.In):
            col = conjunct.this
            if not isinstance(col, exp.Column):
                continue
            expressions = conjunct.expressions
            if not expressions or len(expressions) > _MAX_IN_LIST_SIZE:
                continue
            values: set[str] = set()
            all_literals = True
            for expr in expressions:
                if isinstance(expr, exp.Literal):
                    values.add(expr.this)
                else:
                    all_literals = False
                    break
            if all_literals and values:
                predicates.append(InListPredicate(col.name, frozenset(values)))

    return predicates


def expand_predicate_rows(preds: list[Predicate]) -> list[list[EqualityPredicate]] | None:
    """Expand IN-list predicates into per-row equality predicate sets.

    Returns a list of predicate combinations, each representing one row.
    Each combination is suitable for passing to ``_sql_resource_id()``.

    Returns ``None`` if expansion would exceed :data:`_MAX_EXPANSION`
    (caller should fall back to table-level).

    Examples::

        WHERE id = 1           → [[(id, 1)]]
        WHERE id IN (1, 2)     → [[(id, 1)], [(id, 2)]]
        WHERE id IN (1, 2) AND region = 'us'
            → [[(region, us), (id, 1)], [(region, us), (id, 2)]]
    """
    equalities = [p for p in preds if isinstance(p, EqualityPredicate)]
    in_lists = [p for p in preds if isinstance(p, InListPredicate)]

    if not in_lists:
        return [equalities] if equalities else None

    # Check expansion size before computing cross product
    total = 1
    for il in in_lists:
        total *= len(il.values)
        if total > _MAX_EXPANSION:
            return None

    # Generate cross product of IN-list values
    from itertools import product

    value_lists = [sorted(il.values) for il in in_lists]
    columns = [il.column for il in in_lists]

    rows: list[list[EqualityPredicate]] = []
    for combo in product(*value_lists):
        row = list(equalities)
        for col, val in zip(columns, combo):
            row.append(EqualityPredicate(col, val))
        rows.append(row)

    return rows


def can_use_row_level(
    table: str,
    predicates_a: list[Predicate],
    predicates_b: list[Predicate],
    pk_columns: set[str] | None,
) -> bool:
    """Can we safely use row-level ObjectIds for these two operations?

    Row-level is sound when both operations have equality or IN-list
    predicates on ALL primary key columns.  Falls back to table-level
    otherwise.
    """
    if not predicates_a or not predicates_b:
        return False  # one has no WHERE → full table scan
    if pk_columns is None:
        return False  # unknown schema → conservative

    cols_a = {p.column for p in predicates_a}
    cols_b = {p.column for p in predicates_b}

    # Both must have predicates on ALL primary key columns
    return pk_columns <= cols_a and pk_columns <= cols_b


def pk_predicates_disjoint(
    preds_a: list[Predicate],
    preds_b: list[Predicate],
) -> bool:
    """Are two sets of PK predicates provably disjoint?

    Handles both equality predicates (single value) and IN-list predicates
    (set of values). True if any shared column's value sets are disjoint.

    E.g., ``(id=1)`` vs ``(id=2)`` → True.
         ``(id IN (1,2,3))`` vs ``(id IN (4,5,6))`` → True.
         ``(id IN (1,2,3))`` vs ``(id=2)`` → False (2 overlaps).
         ``(id=1)`` vs ``(id=1)`` → False (same row).
    """

    def _values_set(preds: list[Predicate]) -> dict[str, set[str]]:
        m: dict[str, set[str]] = {}
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
