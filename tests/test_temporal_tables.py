
import pytest
from frontrun._sql_parsing import parse_sql_access

class TestTemporalTables:
    """Tests for temporal/system-versioned table support.

    Expected: FOR SYSTEM_TIME clauses should be extracted with temporal bounds.
    """

    def test_for_system_time_as_of(self):
        """FOR SYSTEM_TIME AS OF should extract temporal predicate."""
        sql = "SELECT * FROM users FOR SYSTEM_TIME AS OF '2024-01-01' WHERE id = 1"
        r, w, lock, tx, temporal = parse_sql_access(sql)
        
        assert r == {"users"}
        assert w == set()
        assert temporal == {"users": "AS OF '2024-01-01'"}

    def test_for_system_time_between(self):
        """FOR SYSTEM_TIME BETWEEN should extract temporal range."""
        sql = "SELECT * FROM accounts FOR SYSTEM_TIME BETWEEN '2024-01-01' AND '2024-01-31'"
        r, w, lock, tx, temporal = parse_sql_access(sql)
        assert r == {"accounts"}
        # sqlglot formats BETWEEN as a tuple of literals
        assert temporal == {"accounts": "BETWEEN ('2024-01-01', '2024-01-31')"}

    def test_join_with_temporal(self):
        """One table temporal, one regular."""
        # Use a join syntax that tsql (the fallback) understands well
        sql = "SELECT * FROM users FOR SYSTEM_TIME AS OF '2024-01-01' u JOIN orders o ON u.id = o.user_id"
        r, w, lock, tx, temporal = parse_sql_access(sql)
        assert "users" in r
        assert "orders" in r
        assert temporal.get("users") == "AS OF '2024-01-01'"
        assert "orders" not in temporal
