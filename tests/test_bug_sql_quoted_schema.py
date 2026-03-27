"""Test that SQL parsing correctly handles quoted schema-qualified table names.

Bug: The _IDENT regex in _sql_parsing.py matches quoted identifiers as
"[^"]+" which captures "public" but NOT "public"."users". For the SQL
statement SELECT * FROM "public"."users", the regex captures only the
schema name "public", and _strip_quotes returns "public" instead of
the table name "users".

This causes incorrect conflict detection: two queries accessing the
same table via different schemas would appear to access different tables,
or a query on "public"."users" would be classified as accessing table
"public" instead of "users".
"""

from frontrun._sql_parsing import parse_sql_access


def test_quoted_schema_qualified_select():
    """SELECT from a quoted schema-qualified table should extract the table name."""
    result = parse_sql_access('SELECT * FROM "public"."users" WHERE id = 1')
    assert "users" in result.read_tables, (
        f"Expected 'users' in read_tables for quoted schema.table, "
        f"but got {result.read_tables}. The regex likely only captured "
        f"the schema name 'public'."
    )


def test_quoted_schema_qualified_insert():
    """INSERT into a quoted schema-qualified table should extract the table name."""
    result = parse_sql_access('INSERT INTO "public"."orders" (id) VALUES (1)')
    assert "orders" in result.write_tables, (
        f"Expected 'orders' in write_tables for quoted schema.table, but got {result.write_tables}."
    )


def test_quoted_schema_qualified_update():
    """UPDATE on a quoted schema-qualified table should extract the table name."""
    result = parse_sql_access('UPDATE "public"."accounts" SET balance = 100 WHERE id = 1')
    assert "accounts" in result.write_tables, (
        f"Expected 'accounts' in write_tables for quoted schema.table, but got {result.write_tables}."
    )


def test_quoted_schema_qualified_delete():
    """DELETE from a quoted schema-qualified table should extract the table name."""
    result = parse_sql_access('DELETE FROM "public"."orders" WHERE id = 1')
    assert "orders" in result.write_tables, (
        f"Expected 'orders' in write_tables for quoted schema.table, but got {result.write_tables}."
    )


def test_unquoted_schema_qualified_works():
    """Unquoted schema.table should already work correctly."""
    result = parse_sql_access("SELECT * FROM public.users WHERE id = 1")
    assert "users" in result.read_tables, (
        f"Expected 'users' in read_tables for unquoted schema.table, but got {result.read_tables}."
    )


def test_backtick_schema_qualified():
    """Backtick-quoted schema.table should extract the table name."""
    result = parse_sql_access("SELECT * FROM `myschema`.`users` WHERE id = 1")
    assert "users" in result.read_tables, (
        f"Expected 'users' in read_tables for backtick schema.table, but got {result.read_tables}."
    )
