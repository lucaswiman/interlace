
import pytest
from unittest.mock import Mock, patch
from frontrun._sql_cursor import _intercept_execute, _io_tls
from frontrun._schema import Schema, ForeignKey, register_schema

# We need to mock get_io_reporter to capture reports
@pytest.fixture
def reporter():
    m = Mock()
    with patch("frontrun._sql_cursor.get_io_reporter", return_value=m):
        yield m

@pytest.fixture(autouse=True)
def clean_tls():
    # Clear TLS state
    if hasattr(_io_tls, "_in_transaction"): delattr(_io_tls, "_in_transaction")
    if hasattr(_io_tls, "_tx_buffer"): delattr(_io_tls, "_tx_buffer")
    if hasattr(_io_tls, "_tx_savepoints"): delattr(_io_tls, "_tx_savepoints")
    if hasattr(_io_tls, "_sql_suppress"): delattr(_io_tls, "_sql_suppress")

def test_fk_dependency_table_level(reporter):
    """Test that writing to a child table implies reading the parent table."""
    # Define Schema: orders.user_id -> users.id
    schema = Schema()
    schema.add_foreign_key(ForeignKey("orders", "user_id", "users", "id"))
    register_schema(schema)

    class MockCursor:
        def execute(self, op, params=None): pass

    cursor = MockCursor()
    
    # INSERT into orders should report write(orders) AND read(users)
    _intercept_execute(cursor.execute, cursor, "INSERT INTO orders (user_id) VALUES (1)")
    
    # Check reports
    calls = [c.args for c in reporter.call_args_list]
    # We expect ('sql:orders', 'write') and ('sql:users', 'read')
    # Note: might be row-level if implemented, but checking for presence of users read
    
    resources = {args[0] for args in calls}
    tables = {r.split(":")[1] for r in resources}
    
    assert "orders" in tables
    assert "users" in tables
    
    # Verify kinds
    users_reports = [args for args in calls if "users" in args[0]]
    assert any(kind == "read" for _, kind in users_reports)

def test_fk_dependency_row_level(reporter):
    """Test that FK dependency preserves row-level granularity."""
    # Schema: orders.user_id -> users.id
    schema = Schema()
    schema.add_foreign_key(ForeignKey("orders", "user_id", "users", "id"))
    register_schema(schema)

    class MockCursor:
        def execute(self, op, params=None): pass
    cursor = MockCursor()

    # INSERT INTO orders (user_id) VALUES (42)
    # Should report:
    #   write sql:orders:user_id=42 (or similar, depending on extraction)
    #   read  sql:users:id=42  <-- The implicit read
    
    _intercept_execute(cursor.execute, cursor, "INSERT INTO orders (user_id) VALUES (42)")
    
    calls = [c.args for c in reporter.call_args_list]
    
    # Check for users read with predicate id=42
    # The resource ID format is sql:table:((col, val),)
    # We need to match roughly
    
    users_reads = [res for res, kind in calls if "sql:users" in res and kind == "read"]
    assert len(users_reads) > 0
    
    # The predicate for users should be id=42
    # We don't assert exact string format as it might change, but check content
    assert "id" in users_reads[0]
    assert "42" in users_reads[0]

