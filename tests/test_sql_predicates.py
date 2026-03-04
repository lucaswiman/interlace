"""Tests for SQL row-level predicate extraction (Algorithm 5).

Tests cover:
- EqualityPredicate dataclass
- extract_equality_predicates() — WHERE clause parsing
- can_use_row_level() — decision function
- pk_predicates_disjoint() — row disjointness check
- End-to-end: parameter resolution → predicate extraction
"""

from __future__ import annotations

import pytest

sqlglot = pytest.importorskip("sqlglot")

from frontrun._sql_predicates import (
    EqualityPredicate,
    can_use_row_level,
    extract_equality_predicates,
    pk_predicates_disjoint,
)

# ---------------------------------------------------------------------------
# extract_equality_predicates
# ---------------------------------------------------------------------------


class TestExtractEqualityPredicates:
    def test_simple_equality(self):
        preds = extract_equality_predicates("SELECT * FROM users WHERE id = 42")
        assert preds == [EqualityPredicate("id", "42")]

    def test_string_literal(self):
        preds = extract_equality_predicates("SELECT * FROM users WHERE name = 'alice'")
        assert preds == [EqualityPredicate("name", "alice")]

    def test_compound_and(self):
        preds = extract_equality_predicates("UPDATE t SET x = 1 WHERE id = 1 AND region = 'us'")
        assert EqualityPredicate("id", "1") in preds
        assert EqualityPredicate("region", "us") in preds
        assert len(preds) == 2

    def test_triple_and(self):
        preds = extract_equality_predicates("SELECT * FROM t WHERE a = 1 AND b = 2 AND c = 3")
        assert len(preds) == 3
        assert EqualityPredicate("a", "1") in preds
        assert EqualityPredicate("b", "2") in preds
        assert EqualityPredicate("c", "3") in preds

    def test_or_returns_empty(self):
        """OR predicates are not conjunctive — return empty."""
        preds = extract_equality_predicates("SELECT * FROM t WHERE id = 1 OR id = 2")
        assert preds == []

    def test_no_where_clause(self):
        preds = extract_equality_predicates("SELECT * FROM users")
        assert preds == []

    def test_insert_no_where(self):
        preds = extract_equality_predicates("INSERT INTO t (id) VALUES (1)")
        assert preds == []

    def test_inequality_skipped(self):
        """Non-equality predicates (>, <, !=) are skipped."""
        preds = extract_equality_predicates("SELECT * FROM t WHERE id > 5")
        assert preds == []

    def test_in_clause_skipped(self):
        preds = extract_equality_predicates("SELECT * FROM t WHERE id IN (1, 2, 3)")
        assert preds == []

    def test_between_skipped(self):
        preds = extract_equality_predicates("SELECT * FROM t WHERE id BETWEEN 1 AND 10")
        assert preds == []

    def test_like_skipped(self):
        preds = extract_equality_predicates("SELECT * FROM t WHERE name LIKE '%alice%'")
        assert preds == []

    def test_subquery_skipped(self):
        preds = extract_equality_predicates("SELECT * FROM t WHERE id = (SELECT MAX(id) FROM t)")
        # The subquery's = is not column = literal, so empty
        assert preds == []

    def test_mixed_equality_and_range(self):
        """Only equality predicates extracted, range predicates skipped."""
        preds = extract_equality_predicates("SELECT * FROM t WHERE id = 42 AND age > 18")
        assert preds == [EqualityPredicate("id", "42")]

    def test_update_where(self):
        preds = extract_equality_predicates("UPDATE accounts SET balance = 0 WHERE id = 7")
        assert EqualityPredicate("id", "7") in preds

    def test_delete_where(self):
        preds = extract_equality_predicates("DELETE FROM sessions WHERE session_id = 'abc123'")
        assert EqualityPredicate("session_id", "abc123") in preds

    def test_reversed_operands(self):
        """Handles literal = column (reversed order)."""
        preds = extract_equality_predicates("SELECT * FROM t WHERE 42 = id")
        assert preds == [EqualityPredicate("id", "42")]

    def test_parse_error_returns_empty(self):
        preds = extract_equality_predicates("NOT VALID SQL {{{")
        assert preds == []

    def test_function_call_in_predicate_skipped(self):
        """Function calls like NOW() are not literals."""
        preds = extract_equality_predicates("SELECT * FROM t WHERE created_at = NOW()")
        assert preds == []

    def test_null_comparison(self):
        """IS NULL is not an equality — should be empty."""
        preds = extract_equality_predicates("SELECT * FROM t WHERE deleted_at IS NULL")
        assert preds == []

    def test_qualified_column(self):
        """Table-qualified column (t.id) should extract just the column name."""
        preds = extract_equality_predicates("SELECT * FROM t WHERE t.id = 42")
        assert preds == [EqualityPredicate("id", "42")]


# ---------------------------------------------------------------------------
# can_use_row_level
# ---------------------------------------------------------------------------


class TestCanUseRowLevel:
    def test_both_have_pk_predicates(self):
        preds_a = [EqualityPredicate("id", "1")]
        preds_b = [EqualityPredicate("id", "2")]
        assert can_use_row_level("users", preds_a, preds_b, {"id"})

    def test_empty_predicates_a(self):
        preds_b = [EqualityPredicate("id", "1")]
        assert not can_use_row_level("users", [], preds_b, {"id"})

    def test_empty_predicates_b(self):
        preds_a = [EqualityPredicate("id", "1")]
        assert not can_use_row_level("users", preds_a, [], {"id"})

    def test_unknown_pk(self):
        preds_a = [EqualityPredicate("id", "1")]
        preds_b = [EqualityPredicate("id", "2")]
        assert not can_use_row_level("users", preds_a, preds_b, None)

    def test_missing_pk_column_a(self):
        """Predicates don't cover all PK columns."""
        preds_a = [EqualityPredicate("name", "alice")]
        preds_b = [EqualityPredicate("id", "2")]
        assert not can_use_row_level("users", preds_a, preds_b, {"id"})

    def test_composite_pk_both_complete(self):
        preds_a = [EqualityPredicate("id", "1"), EqualityPredicate("region", "us")]
        preds_b = [EqualityPredicate("id", "2"), EqualityPredicate("region", "eu")]
        assert can_use_row_level("users", preds_a, preds_b, {"id", "region"})

    def test_composite_pk_one_incomplete(self):
        preds_a = [EqualityPredicate("id", "1")]
        preds_b = [EqualityPredicate("id", "2"), EqualityPredicate("region", "eu")]
        assert not can_use_row_level("users", preds_a, preds_b, {"id", "region"})

    def test_extra_predicates_ok(self):
        """Extra non-PK predicates don't prevent row-level."""
        preds_a = [EqualityPredicate("id", "1"), EqualityPredicate("name", "alice")]
        preds_b = [EqualityPredicate("id", "2"), EqualityPredicate("name", "bob")]
        assert can_use_row_level("users", preds_a, preds_b, {"id"})


# ---------------------------------------------------------------------------
# pk_predicates_disjoint
# ---------------------------------------------------------------------------


class TestPkPredicatesDisjoint:
    def test_different_pk_values(self):
        a = [EqualityPredicate("id", "1")]
        b = [EqualityPredicate("id", "2")]
        assert pk_predicates_disjoint(a, b)

    def test_same_pk_values(self):
        a = [EqualityPredicate("id", "1")]
        b = [EqualityPredicate("id", "1")]
        assert not pk_predicates_disjoint(a, b)

    def test_composite_pk_one_differs(self):
        a = [EqualityPredicate("id", "1"), EqualityPredicate("region", "us")]
        b = [EqualityPredicate("id", "1"), EqualityPredicate("region", "eu")]
        assert pk_predicates_disjoint(a, b)

    def test_composite_pk_all_same(self):
        a = [EqualityPredicate("id", "1"), EqualityPredicate("region", "us")]
        b = [EqualityPredicate("id", "1"), EqualityPredicate("region", "us")]
        assert not pk_predicates_disjoint(a, b)

    def test_no_shared_columns(self):
        """Different column sets → can't prove disjoint."""
        a = [EqualityPredicate("id", "1")]
        b = [EqualityPredicate("name", "alice")]
        assert not pk_predicates_disjoint(a, b)

    def test_subset_columns_differ(self):
        a = [EqualityPredicate("id", "1"), EqualityPredicate("region", "us")]
        b = [EqualityPredicate("id", "2")]
        assert pk_predicates_disjoint(a, b)

    def test_empty_a(self):
        a: list[EqualityPredicate] = []
        b = [EqualityPredicate("id", "1")]
        assert not pk_predicates_disjoint(a, b)

    def test_empty_b(self):
        a = [EqualityPredicate("id", "1")]
        b: list[EqualityPredicate] = []
        assert not pk_predicates_disjoint(a, b)

    def test_both_empty(self):
        assert not pk_predicates_disjoint([], [])


# ---------------------------------------------------------------------------
# End-to-end: parameter resolution → predicate extraction
# ---------------------------------------------------------------------------


class TestParameterizedPredicateExtraction:
    """End-to-end: resolve params → extract predicates."""

    def test_qmark_select(self):
        from frontrun._sql_params import resolve_parameters

        sql = "SELECT * FROM users WHERE id = ?"
        resolved = resolve_parameters(sql, (42,), "qmark")
        preds = extract_equality_predicates(resolved)
        assert preds == [EqualityPredicate("id", "42")]

    def test_format_update(self):
        from frontrun._sql_params import resolve_parameters

        sql = "UPDATE accounts SET balance = balance - %s WHERE id = %s"
        resolved = resolve_parameters(sql, (100, 7), "format")
        preds = extract_equality_predicates(resolved)
        assert EqualityPredicate("id", "7") in preds

    def test_named_compound(self):
        from frontrun._sql_params import resolve_parameters

        sql = "SELECT * FROM t WHERE id = :id AND region = :region"
        resolved = resolve_parameters(sql, {"id": 1, "region": "us"}, "named")
        preds = extract_equality_predicates(resolved)
        assert EqualityPredicate("id", "1") in preds
        assert EqualityPredicate("region", "us") in preds

    def test_pyformat_orm_style(self):
        from frontrun._sql_params import resolve_parameters

        sql = "SELECT users.id FROM users WHERE users.id = %s"
        resolved = resolve_parameters(sql, (42,), "pyformat")
        preds = extract_equality_predicates(resolved)
        assert EqualityPredicate("id", "42") in preds
