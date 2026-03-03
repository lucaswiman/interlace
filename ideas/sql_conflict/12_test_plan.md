# Test Plan

## Unit Tests (`tests/test_sql_detection.py`)

```python
class TestRegexParse:
    def test_select(self):
        r, w = _regex_parse("SELECT id, name FROM users WHERE id = 1")
        assert r == {"users"} and w == set()

    def test_insert(self):
        r, w = _regex_parse("INSERT INTO orders (user_id, amount) VALUES (1, 100)")
        assert r == set() and w == {"orders"}

    def test_insert_select(self):
        r, w = _regex_parse("INSERT INTO archive SELECT * FROM orders")
        assert r == {"orders"} and w == {"archive"}

    def test_update(self):
        r, w = _regex_parse("UPDATE accounts SET balance = balance - 100 WHERE id = 1")
        assert r == {"accounts"} and w == {"accounts"}

    def test_delete(self):
        r, w = _regex_parse("DELETE FROM sessions WHERE expires_at < NOW()")
        assert r == {"sessions"} and w == {"sessions"}

    def test_join(self):
        r, w = _regex_parse("SELECT u.name, o.total FROM users u JOIN orders o ON u.id = o.user_id")
        assert r == {"users", "orders"} and w == set()

    def test_cte_falls_through(self):
        assert _regex_parse("WITH cte AS (SELECT 1) SELECT * FROM cte") is None

    def test_quoted_identifiers(self):
        r, w = _regex_parse('SELECT * FROM "My Table"')
        assert r == {"My Table"} and w == set()

    def test_schema_qualified(self):
        r, w = _regex_parse("SELECT * FROM public.users")
        assert r == {"users"} and w == set()


class TestSqlglotParse:
    def test_cte(self):
        r, w = _sqlglot_parse("WITH active AS (SELECT * FROM users WHERE active) SELECT * FROM active")
        assert "users" in r

    def test_subquery_in_where(self):
        r, w = _sqlglot_parse("UPDATE accounts SET status = 'closed' WHERE id IN (SELECT account_id FROM expired)")
        assert "expired" in r and "accounts" in w

    def test_update_from(self):
        r, w = _sqlglot_parse("UPDATE t1 SET t1.col = t2.col FROM t1 JOIN t2 ON t1.id = t2.id")
        assert "t2" in r and "t1" in w


class TestCombined:
    def test_simple_uses_regex(self):
        """Simple statements should not require sqlglot."""
        r, w = parse_sql_access("SELECT * FROM users")
        assert r == {"users"}

    def test_complex_uses_sqlglot(self):
        r, w = parse_sql_access("WITH x AS (SELECT 1) SELECT * FROM x JOIN y ON x.id = y.id")
        assert "y" in r

    def test_unparseable_returns_empty(self):
        r, w = parse_sql_access("GIBBERISH NOT SQL")
        assert r == set() and w == set()


class TestParameterResolution:
    def test_qmark(self):
        resolved = resolve_parameters(
            "SELECT * FROM users WHERE id = ? AND name = ?",
            (42, "alice"), "qmark",
        )
        assert resolved == "SELECT * FROM users WHERE id = 42 AND name = 'alice'"

    def test_numeric(self):
        resolved = resolve_parameters(
            "SELECT * FROM users WHERE id = :1 AND name = :2",
            (42, "alice"), "numeric",
        )
        assert resolved == "SELECT * FROM users WHERE id = 42 AND name = 'alice'"

    def test_named(self):
        resolved = resolve_parameters(
            "SELECT * FROM users WHERE id = :id AND name = :name",
            {"id": 42, "name": "alice"}, "named",
        )
        assert resolved == "SELECT * FROM users WHERE id = 42 AND name = 'alice'"

    def test_format(self):
        resolved = resolve_parameters(
            "SELECT * FROM users WHERE id = %s AND name = %s",
            (42, "alice"), "format",
        )
        assert resolved == "SELECT * FROM users WHERE id = 42 AND name = 'alice'"

    def test_pyformat_with_dict(self):
        resolved = resolve_parameters(
            "SELECT * FROM users WHERE id = %(id)s AND name = %(name)s",
            {"id": 42, "name": "alice"}, "pyformat",
        )
        assert resolved == "SELECT * FROM users WHERE id = 42 AND name = 'alice'"

    def test_pyformat_with_tuple_falls_back_to_format(self):
        """psycopg2 declares pyformat but commonly uses %s with tuples."""
        resolved = resolve_parameters(
            "SELECT * FROM users WHERE id = %s",
            (42,), "pyformat",
        )
        assert resolved == "SELECT * FROM users WHERE id = 42"

    def test_none_becomes_null(self):
        resolved = resolve_parameters(
            "UPDATE t SET x = ? WHERE id = ?",
            (None, 1), "qmark",
        )
        assert "NULL" in resolved

    def test_bool_values(self):
        resolved = resolve_parameters(
            "SELECT * FROM t WHERE active = ?",
            (True,), "qmark",
        )
        assert "TRUE" in resolved

    def test_string_with_quotes(self):
        resolved = resolve_parameters(
            "SELECT * FROM t WHERE name = ?",
            ("O'Brien",), "qmark",
        )
        assert "O''Brien" in resolved  # SQL-escaped single quote

    def test_no_params_returns_unchanged(self):
        sql = "SELECT * FROM users WHERE id = 1"
        assert resolve_parameters(sql, None, "qmark") is sql

    def test_missing_param_returns_unchanged(self):
        """If param lookup fails, return original SQL (table-level fallback)."""
        sql = "SELECT * FROM users WHERE id = :missing"
        resolved = resolve_parameters(sql, {"other": 1}, "named")
        assert resolved == sql

    def test_pg_cast_not_matched_by_named(self):
        """:: type cast should not be treated as a :name placeholder."""
        resolved = resolve_parameters(
            "SELECT id::text FROM users WHERE id = :id",
            {"id": 42}, "named",
        )
        assert "::text" in resolved  # cast preserved
        assert "42" in resolved       # param resolved

    def test_escaped_percent_not_matched(self):
        """%%s should not be treated as a %s placeholder."""
        resolved = resolve_parameters(
            "SELECT '%%s' FROM users WHERE id = %s",
            (42,), "format",
        )
        assert "%%s" in resolved  # escape preserved
        assert "42" in resolved    # param resolved


class TestParameterizedPredicateExtraction:
    """End-to-end: resolve params → extract predicates."""

    def test_qmark_select(self):
        sql = "SELECT * FROM users WHERE id = ?"
        resolved = resolve_parameters(sql, (42,), "qmark")
        preds = extract_equality_predicates(resolved)
        assert preds == [EqualityPredicate("id", "42")]

    def test_format_update(self):
        sql = "UPDATE accounts SET balance = balance - %s WHERE id = %s"
        resolved = resolve_parameters(sql, (100, 7), "format")
        preds = extract_equality_predicates(resolved)
        assert EqualityPredicate("id", "7") in preds

    def test_named_compound(self):
        sql = "SELECT * FROM t WHERE id = :id AND region = :region"
        resolved = resolve_parameters(sql, {"id": 1, "region": "us"}, "named")
        preds = extract_equality_predicates(resolved)
        assert EqualityPredicate("id", "1") in preds
        assert EqualityPredicate("region", "us") in preds

    def test_pyformat_orm_style(self):
        """Typical ORM: pyformat paramstyle, %s placeholder, tuple params."""
        sql = "SELECT users.id FROM users WHERE users.id = %s"
        resolved = resolve_parameters(sql, (42,), "pyformat")
        preds = extract_equality_predicates(resolved)
        assert EqualityPredicate("id", "42") in preds


class TestPredicateExtraction:
    def test_simple_equality(self):
        preds = extract_equality_predicates("SELECT * FROM users WHERE id = 42")
        assert preds == [EqualityPredicate("id", "42")]

    def test_compound_and(self):
        preds = extract_equality_predicates("UPDATE t SET x = 1 WHERE id = 1 AND region = 'us'")
        assert EqualityPredicate("id", "1") in preds
        assert EqualityPredicate("region", "us") in preds

    def test_or_returns_empty(self):
        preds = extract_equality_predicates("SELECT * FROM t WHERE id = 1 OR id = 2")
        assert preds == []  # OR is not a conjunction


class TestPredicateDisjointness:
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
```

## Integration Tests (`tests/test_integration_orm.py`)

```python
class TestOrmSqlConflictDetection:
    """Verify that SQL-level conflict detection reduces DPOR exploration."""

    def test_different_tables_independent(self, engine):
        """Two threads writing different tables: DPOR should explore 1 execution."""
        # Thread A: INSERT INTO table_a
        # Thread B: INSERT INTO table_b
        # With SQL detection: independent → 1 execution
        # Without: same socket → many executions

    def test_same_table_write_write_conflict(self, engine):
        """Two threads writing same table: DPOR should find the conflict."""
        # Thread A: UPDATE accounts SET balance = balance - 100 WHERE id = 1
        # Thread B: UPDATE accounts SET balance = balance + 50 WHERE id = 1
        # Both write "accounts" → conflict → DPOR explores interleavings

    def test_same_table_read_read_independent(self, engine):
        """Two threads reading same table: DPOR should explore 1 execution."""
        # Thread A: SELECT * FROM users WHERE id = 1
        # Thread B: SELECT * FROM users WHERE id = 2
        # Both read "users" → no conflict → 1 execution

    def test_read_write_conflict(self, engine):
        """One reader, one writer on same table: DPOR should find conflict."""
        # Thread A: SELECT balance FROM accounts WHERE id = 1
        # Thread B: UPDATE accounts SET balance = 0 WHERE id = 1
        # RW anti-dependency → conflict
```
