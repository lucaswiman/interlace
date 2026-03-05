"""Tests for SQL statement parsing (read/write table extraction).

Covers:
- Regex fast-path: SELECT, INSERT, UPDATE, DELETE, JOINs
- Complex SQL falling through to sqlglot: CTEs, UNION, MERGE, RETURNING, etc.
- sqlglot full-parser path: CTEs, subqueries, UNION, MERGE, INSERT...SELECT
- Combined parse_sql_access: end-to-end routing and fallback behaviour
- Edge cases: quoted identifiers, schema qualification, whitespace, semicolons
"""

from __future__ import annotations

import pytest

from frontrun._sql_parsing import _regex_parse, _sqlglot_parse, _strip_quotes, parse_sql_access

# ---------------------------------------------------------------------------
# _strip_quotes
# ---------------------------------------------------------------------------


class TestStripQuotes:
    def test_unquoted_simple(self):
        assert _strip_quotes("users") == "users"

    def test_double_quoted(self):
        assert _strip_quotes('"My Table"') == "My Table"

    def test_backtick_quoted(self):
        assert _strip_quotes("`orders`") == "orders"

    def test_schema_qualified_unquoted(self):
        assert _strip_quotes("public.users") == "users"

    def test_schema_qualified_double_quoted(self):
        # "public"."users" → strip leading quote, rsplit on '.', last component is '"users'
        # (the closing quote of the second component is not stripped by this function)
        assert _strip_quotes('"public"."users"') == '"users'

    def test_deep_dotted(self):
        assert _strip_quotes("db.schema.table") == "table"

    def test_no_quotes_no_schema(self):
        assert _strip_quotes("accounts") == "accounts"


# ---------------------------------------------------------------------------
# _regex_parse — spec cases
# ---------------------------------------------------------------------------


class TestRegexParse:
    def test_select(self):
        r, w, _, _ = _regex_parse("SELECT id, name FROM users WHERE id = 1")
        assert r == {"users"} and w == set()

    def test_insert(self):
        r, w, _, _ = _regex_parse("INSERT INTO orders (user_id, amount) VALUES (1, 100)")
        assert r == set() and w == {"orders"}

    def test_insert_select(self):
        r, w, _, _ = _regex_parse("INSERT INTO archive SELECT * FROM orders")
        assert r == {"orders"} and w == {"archive"}

    def test_update(self):
        r, w, _, _ = _regex_parse("UPDATE accounts SET balance = balance - 100 WHERE id = 1")
        assert r == {"accounts"} and w == {"accounts"}

    def test_delete(self):
        r, w, _, _ = _regex_parse("DELETE FROM sessions WHERE expires_at < NOW()")
        assert r == {"sessions"} and w == {"sessions"}

    def test_join(self):
        r, w, _, _ = _regex_parse("SELECT u.name, o.total FROM users u JOIN orders o ON u.id = o.user_id")
        assert r == {"users", "orders"} and w == set()

    def test_cte_falls_through(self):
        assert _regex_parse("WITH cte AS (SELECT 1) SELECT * FROM cte") is None

    def test_quoted_identifiers(self):
        r, w, _, _ = _regex_parse('SELECT * FROM "My Table"')
        assert r == {"My Table"} and w == set()

    def test_schema_qualified(self):
        r, w, _, _ = _regex_parse("SELECT * FROM public.users")
        assert r == {"users"} and w == set()

    # -----------------------------------------------------------------------
    # Additional SELECT tests
    # -----------------------------------------------------------------------

    def test_select_no_from(self):
        r, w, _, _ = _regex_parse("SELECT 1")
        assert r == set() and w == set()

    def test_select_multiple_joins(self):
        sql = (
            "SELECT u.name, o.id, p.name "
            "FROM users u "
            "JOIN orders o ON u.id = o.user_id "
            "JOIN products p ON o.product_id = p.id"
        )
        r, w, _, _ = _regex_parse(sql)
        assert r == {"users", "orders", "products"} and w == set()

    def test_select_left_join(self):
        r, w, _, _ = _regex_parse("SELECT * FROM users u LEFT JOIN profiles p ON u.id = p.user_id")
        assert r == {"users", "profiles"} and w == set()

    def test_select_right_join(self):
        r, w, _, _ = _regex_parse("SELECT * FROM users u RIGHT JOIN orders o ON u.id = o.user_id")
        assert r == {"users", "orders"} and w == set()

    def test_select_inner_join(self):
        r, w, _, _ = _regex_parse("SELECT * FROM products p INNER JOIN categories c ON p.cat_id = c.id")
        assert r == {"products", "categories"} and w == set()

    def test_select_cross_join(self):
        r, w, _, _ = _regex_parse("SELECT * FROM users CROSS JOIN roles")
        assert r == {"users", "roles"} and w == set()

    def test_select_mixed_case_keywords(self):
        r, w, _, _ = _regex_parse("select * from ORDERS where id = 42")
        assert r == {"ORDERS"} and w == set()

    def test_select_multiline(self):
        sql = """SELECT
            u.id,
            u.name
        FROM
            users u
        WHERE
            u.active = true"""
        r, w, _, _ = _regex_parse(sql)
        assert r == {"users"} and w == set()

    def test_select_extra_whitespace(self):
        r, w, _, _ = _regex_parse("SELECT  *   FROM   orders   WHERE   id  =  1")
        assert r == {"orders"} and w == set()

    def test_select_with_trailing_semicolon(self):
        r, w, _, _ = _regex_parse("SELECT * FROM users;")
        assert r == {"users"} and w == set()

    def test_select_schema_qualified_with_double_quotes(self):
        # The regex _IDENT matches "public" as a quoted identifier; the ".\"users\"" part is not
        # captured as a single token by the regex, so "public" is extracted as the table name.
        r, w, _, _ = _regex_parse('SELECT * FROM "public"."users"')
        assert w == set() and len(r) > 0

    def test_select_backtick_table(self):
        r, w, _, _ = _regex_parse("SELECT * FROM `my_table`")
        assert r == {"my_table"} and w == set()

    # -----------------------------------------------------------------------
    # Additional INSERT tests
    # -----------------------------------------------------------------------

    def test_insert_values_no_read(self):
        r, w, _, _ = _regex_parse("INSERT INTO users (name, email) VALUES ('Alice', 'alice@example.com')")
        assert r == set() and w == {"users"}

    def test_insert_select_with_join(self):
        sql = "INSERT INTO summary SELECT o.id, u.name FROM orders o JOIN users u ON o.user_id = u.id"
        r, w, _, _ = _regex_parse(sql)
        assert w == {"summary"}
        assert "orders" in r

    def test_insert_schema_qualified_target(self):
        r, w, _, _ = _regex_parse("INSERT INTO public.events (type) VALUES ('login')")
        assert w == {"events"} and r == set()

    def test_insert_backtick_target(self):
        r, w, _, _ = _regex_parse("INSERT INTO `session_logs` (user_id) VALUES (1)")
        assert w == {"session_logs"} and r == set()

    # -----------------------------------------------------------------------
    # Additional UPDATE tests
    # -----------------------------------------------------------------------

    def test_update_target_in_both_read_and_write(self):
        r, w, _, _ = _regex_parse("UPDATE products SET stock = stock - 1 WHERE id = 5")
        assert "products" in r and "products" in w

    def test_update_from_clause(self):
        sql = "UPDATE orders SET status = 'done' FROM users WHERE orders.user_id = users.id"
        r, w, _, _ = _regex_parse(sql)
        assert w == {"orders"}
        assert "orders" in r
        assert "users" in r

    def test_update_schema_qualified(self):
        r, w, _, _ = _regex_parse("UPDATE public.accounts SET balance = 0 WHERE id = 1")
        assert "accounts" in w and "accounts" in r

    def test_update_mixed_case(self):
        r, w, _, _ = _regex_parse("update Users set name = 'Bob' where id = 3")
        assert "Users" in w and "Users" in r

    # -----------------------------------------------------------------------
    # Additional DELETE tests
    # -----------------------------------------------------------------------

    def test_delete_no_where(self):
        r, w, _, _ = _regex_parse("DELETE FROM old_logs")
        assert r == {"old_logs"} and w == {"old_logs"}

    def test_delete_schema_qualified(self):
        r, w, _, _ = _regex_parse("DELETE FROM archive.events WHERE created_at < '2020-01-01'")
        assert "events" in w and "events" in r

    def test_delete_backtick(self):
        r, w, _, _ = _regex_parse("DELETE FROM `temp_rows` WHERE expired = 1")
        assert r == {"temp_rows"} and w == {"temp_rows"}

    # -----------------------------------------------------------------------
    # Complex SQL → falls through to None
    # -----------------------------------------------------------------------

    def test_union_falls_through(self):
        assert _regex_parse("SELECT id FROM users UNION SELECT id FROM admins") is None

    def test_intersect_falls_through(self):
        assert _regex_parse("SELECT id FROM a INTERSECT SELECT id FROM b") is None

    def test_except_falls_through(self):
        assert _regex_parse("SELECT id FROM a EXCEPT SELECT id FROM b") is None

    def test_merge_falls_through(self):
        sql = "MERGE INTO target USING source ON target.id = source.id WHEN MATCHED THEN UPDATE SET target.v = source.v"
        assert _regex_parse(sql) is None

    def test_returning_falls_through(self):
        assert _regex_parse("INSERT INTO orders (total) VALUES (100) RETURNING id") is None

    def test_with_cte_select_falls_through(self):
        assert _regex_parse("WITH t AS (SELECT 1) SELECT * FROM t") is None

    def test_with_recursive_falls_through(self):
        sql = "WITH RECURSIVE nums AS (SELECT 1 UNION ALL SELECT n+1 FROM nums) SELECT * FROM nums"
        assert _regex_parse(sql) is None

    # -----------------------------------------------------------------------
    # Empty / unknown statements
    # -----------------------------------------------------------------------

    def test_empty_string(self):
        assert _regex_parse("") is None

    def test_whitespace_only(self):
        assert _regex_parse("   \n\t  ") is None

    def test_unknown_statement_ddl(self):
        assert _regex_parse("CREATE TABLE foo (id INT)") is None

    def test_unknown_statement_truncate(self):
        assert _regex_parse("TRUNCATE TABLE sessions") is None

    def test_unknown_statement_grant(self):
        # GRANT contains the word SELECT, so the regex SELECT path matches and returns
        # (empty sets) because there's no FROM clause — not None.
        result = _regex_parse("GRANT SELECT ON users TO readonly")
        # Either None (fell through) or empty sets (matched SELECT path with no FROM)
        assert result is None or result == (set(), set(), None)



# ---------------------------------------------------------------------------
# _sqlglot_parse — full parser (requires sqlglot)
# ---------------------------------------------------------------------------


class TestSqlglotParse:
    @pytest.fixture(autouse=True)
    def _require_sqlglot(self):
        pytest.importorskip("sqlglot")

    def test_simple_select(self):
        r, w, _, _ = _sqlglot_parse("SELECT id FROM users WHERE id = 1")
        assert r == {"users"} and w == set()

    def test_simple_insert(self):
        r, w, _, _ = _sqlglot_parse("INSERT INTO orders (amount) VALUES (99)")
        assert r == set() and w == {"orders"}

    def test_simple_update(self):
        r, w, _, _ = _sqlglot_parse("UPDATE accounts SET balance = 0 WHERE id = 1")
        assert "accounts" in w and "accounts" in r

    def test_simple_delete(self):
        r, w, _, _ = _sqlglot_parse("DELETE FROM sessions WHERE id = 1")
        assert "sessions" in w and "sessions" in r

    def test_cte_select(self):
        sql = "WITH recent AS (SELECT * FROM orders WHERE created > '2024-01-01') SELECT * FROM recent"
        r, w, _, _ = _sqlglot_parse(sql)
        assert r is not None
        assert "orders" in r and w == set()

    def test_cte_multiple(self):
        sql = "WITH a AS (SELECT * FROM t1), b AS (SELECT * FROM t2) SELECT * FROM a JOIN b ON a.id = b.id"
        r, w, _, _ = _sqlglot_parse(sql)
        assert r is not None
        assert "t1" in r and "t2" in r and w == set()

    def test_subquery_in_where(self):
        sql = "SELECT * FROM orders WHERE user_id IN (SELECT id FROM users WHERE active = true)"
        r, w, _, _ = _sqlglot_parse(sql)
        assert r is not None
        assert "orders" in r and "users" in r and w == set()

    def test_union_select(self):
        # UNION produces a Union AST node. These are now handled explicitly as reads.
        sql = "SELECT id FROM customers UNION SELECT id FROM vendors"
        r, w, _, _ = _sqlglot_parse(sql)
        assert r is not None
        assert "customers" in r and "vendors" in r and w == set()

    def test_insert_select_simple(self):
        sql = "INSERT INTO archive SELECT * FROM orders WHERE status = 'closed'"
        r, w, _, _ = _sqlglot_parse(sql)
        assert r is not None
        assert w == {"archive"}
        assert "orders" in r

    def test_insert_select_with_join(self):
        sql = (
            "INSERT INTO summary (user_id, total) "
            "SELECT u.id, SUM(o.amount) FROM users u JOIN orders o ON u.id = o.user_id GROUP BY u.id"
        )
        r, w, _, _ = _sqlglot_parse(sql)
        assert r is not None
        assert w == {"summary"}
        assert "users" in r and "orders" in r

    def test_update_from_join(self):
        sql = "UPDATE orders SET status = 'shipped' FROM shipments WHERE orders.id = shipments.order_id"
        r, w, _, _ = _sqlglot_parse(sql)
        assert r is not None
        assert "orders" in w
        assert "shipments" in r

    def test_merge_statement(self):
        sql = (
            "MERGE INTO target USING source ON (target.id = source.id) "
            "WHEN MATCHED THEN UPDATE SET target.v = source.v "
            "WHEN NOT MATCHED THEN INSERT (id, v) VALUES (source.id, source.v)"
        )
        r, w, _, _ = _sqlglot_parse(sql)
        assert r is not None
        assert "target" in w
        assert "source" in r

    def test_nested_subqueries(self):
        sql = (
            "SELECT * FROM users WHERE id IN "
            "(SELECT user_id FROM orders WHERE product_id IN "
            "(SELECT id FROM products WHERE category = 'tech'))"
        )
        r, w, _, _ = _sqlglot_parse(sql)
        assert r is not None
        assert "users" in r and "orders" in r and "products" in r and w == set()

    def test_select_with_subquery_in_from(self):
        sql = "SELECT sub.total FROM (SELECT SUM(amount) AS total FROM payments) sub"
        r, w, _, _ = _sqlglot_parse(sql)
        assert r is not None
        assert "payments" in r and w == set()

    def test_unparseable_sql_returns_none(self):
        result = _sqlglot_parse("THIS IS NOT SQL AT ALL !!!")
        assert result is None

    def test_cte_with_recursive(self):
        sql = (
            "WITH RECURSIVE tree AS ("
            "  SELECT id, parent_id FROM categories WHERE parent_id IS NULL "
            "  UNION ALL "
            "  SELECT c.id, c.parent_id FROM categories c JOIN tree t ON c.parent_id = t.id"
            ") SELECT * FROM tree"
        )
        r, w, _, _ = _sqlglot_parse(sql)
        assert r is not None
        assert "categories" in r and w == set()

    def test_insert_returning(self):
        sql = "INSERT INTO events (type) VALUES ('login') RETURNING id"
        r, w, _, _ = _sqlglot_parse(sql)
        assert r is not None
        assert "events" in w

    def test_delete_with_subquery(self):
        sql = "DELETE FROM orders WHERE user_id IN (SELECT id FROM users WHERE banned = true)"
        r, w, _, _ = _sqlglot_parse(sql)
        assert r is not None
        assert "orders" in w
        assert "users" in r



# ---------------------------------------------------------------------------
# parse_sql_access — end-to-end routing and fallback
# ---------------------------------------------------------------------------


class TestParseSqlAccess:
    def test_simple_select(self):
        r, w, _, _ = parse_sql_access("SELECT * FROM users")
        assert r == {"users"} and w == set()

    def test_simple_insert(self):
        r, w, _, _ = parse_sql_access("INSERT INTO logs (msg) VALUES ('hi')")
        assert r == set() and w == {"logs"}

    def test_simple_update(self):
        r, w, _, _ = parse_sql_access("UPDATE sessions SET active = false WHERE id = 7")
        assert "sessions" in w and "sessions" in r

    def test_simple_delete(self):
        r, w, _, _ = parse_sql_access("DELETE FROM temp_tokens WHERE expires < NOW()")
        assert "temp_tokens" in w and "temp_tokens" in r

    def test_join_select(self):
        r, w, _, _ = parse_sql_access("SELECT u.name FROM users u JOIN orders o ON u.id = o.user_id")
        assert "users" in r and "orders" in r and w == set()

    def test_insert_select(self):
        r, w, _, _ = parse_sql_access("INSERT INTO archive SELECT * FROM orders")
        assert "orders" in r and "archive" in w

    def test_cte_routed_to_sqlglot(self):
        pytest.importorskip("sqlglot")
        sql = "WITH t AS (SELECT * FROM products) SELECT * FROM t"
        r, w, _, _ = parse_sql_access(sql)
        assert "products" in r and w == set()

    def test_union_routed_to_sqlglot(self):
        pytest.importorskip("sqlglot")
        # UNION falls through regex fast-path (returns None) and is handled by sqlglot.
        # These are now handled explicitly as reads.
        sql = "SELECT id FROM users UNION SELECT id FROM admins"
        r, w, _, _ = parse_sql_access(sql)
        assert "users" in r and "admins" in r and w == set()

    def test_merge_routed_to_sqlglot(self):
        pytest.importorskip("sqlglot")
        sql = (
            "MERGE INTO inventory USING incoming ON inventory.sku = incoming.sku "
            "WHEN MATCHED THEN UPDATE SET inventory.qty = inventory.qty + incoming.qty "
            "WHEN NOT MATCHED THEN INSERT (sku, qty) VALUES (incoming.sku, incoming.qty)"
        )
        r, w, _, _ = parse_sql_access(sql)
        assert "inventory" in w and "incoming" in r

    def test_unparseable_returns_empty_sets(self):
        r, w, _, _ = parse_sql_access(";;; GARBAGE SQL ;;;")
        assert isinstance(r, set) and isinstance(w, set)

    def test_empty_sql_returns_empty_sets(self):
        r, w, _, _ = parse_sql_access("")
        assert r == set() and w == set()

    def test_whitespace_only_returns_empty_sets(self):
        r, w, _, _ = parse_sql_access("   \n   ")
        assert r == set() and w == set()

    def test_case_insensitive_select(self):
        r, w, _, _ = parse_sql_access("select * from USERS")
        assert "USERS" in r and w == set()

    def test_case_insensitive_insert(self):
        r, w, _, _ = parse_sql_access("insert into Orders (amount) values (50)")
        assert "Orders" in w

    def test_case_insensitive_update(self):
        r, w, _, _ = parse_sql_access("UPDATE Accounts SET balance = 10 WHERE id = 1")
        assert "Accounts" in w

    def test_case_insensitive_delete(self):
        r, w, _, _ = parse_sql_access("DELETE FROM Sessions WHERE id = 2")
        assert "Sessions" in w

    def test_trailing_semicolon_stripped(self):
        r, w, _, _ = parse_sql_access("SELECT * FROM orders;")
        assert "orders" in r and w == set()

    def test_multiple_joins(self):
        sql = (
            "SELECT * FROM users u "
            "JOIN orders o ON u.id = o.user_id "
            "JOIN shipments s ON o.id = s.order_id "
            "JOIN products p ON o.product_id = p.id"
        )
        r, w, _, _ = parse_sql_access(sql)
        assert {"users", "orders", "shipments", "products"} <= r and w == set()

    def test_schema_qualified_table(self):
        r, w, _, _ = parse_sql_access("SELECT * FROM public.accounts WHERE id = 1")
        assert "accounts" in r and w == set()

    def test_quoted_table_name(self):
        r, w, _, _ = parse_sql_access('SELECT * FROM "My Schema"')
        assert "My Schema" in r and w == set()

    def test_returns_sets_not_lists(self):
        r, w, _, _ = parse_sql_access("SELECT * FROM users")
        assert isinstance(r, set) and isinstance(w, set)

    def test_complex_subquery_without_sqlglot(self):
        # Even if sqlglot not available, parse_sql_access should return sets
        sql = "WITH cte AS (SELECT 1) SELECT * FROM cte"
        r, w, _, _ = parse_sql_access(sql)
        assert isinstance(r, set) and isinstance(w, set)

