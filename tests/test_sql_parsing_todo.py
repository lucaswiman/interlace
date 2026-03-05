"""
Failing tests documenting SQL parsing TODOs and incomplete features.

Each test documents expected behavior that is not yet implemented.
These tests serve as:
1. Documentation of intended behavior
2. Acceptance criteria for TODO implementation
3. Regression tests once features are complete

When implementing a TODO, change @pytest.mark.xfail to @pytest.mark.skip
and update the test to assert correct behavior.

Reference: ideas/sql_conflict/03_algorithm_1_sql_parsing.md#known-limitations--todos
"""

from __future__ import annotations

import pytest

from frontrun._sql_parsing import parse_sql_access


# =============================================================================
# TODO: SELECT FOR UPDATE / FOR SHARE Locking Semantics
# =============================================================================
# Issue: Parser extracts tables but not locking intent
# Phase: 5
# Effort: ~50 lines + 20 tests
# Reference: ideas/sql_conflict/03_algorithm_1_sql_parsing.md#todo-select-for-update--for-share-locking-semantics


class TestSelectForUpdateTodo:
    """Tests for SELECT FOR UPDATE / FOR SHARE locking semantics.

    Expected: parse_sql_access() should return lock_intent alongside read/write tables.
    Current: Treated as regular SELECT (read-only).
    """

    @pytest.mark.xfail(reason="SELECT FOR UPDATE lock intent not yet modeled")
    def test_select_for_update_extracts_lock_intent(self):
        """SELECT FOR UPDATE should indicate exclusive lock."""
        sql = "SELECT * FROM users WHERE id = 1 FOR UPDATE"
        # Expected API (not yet implemented):
        # read, write, lock_intent = parse_sql_access_with_locks(sql)
        # assert read == {"users"}
        # assert write == set()
        # assert lock_intent == "UPDATE"  # exclusive lock
        r, w = parse_sql_access(sql)
        assert r == {"users"} and w == set()
        # TODO: assert has lock_intent attribute

    @pytest.mark.xfail(reason="SELECT FOR SHARE lock intent not yet modeled")
    def test_select_for_share_extracts_lock_intent(self):
        """SELECT FOR SHARE should indicate shared lock."""
        sql = "SELECT * FROM accounts WHERE user_id = ? FOR SHARE"
        # Expected: lock_intent == "SHARE"
        r, w = parse_sql_access(sql)
        assert r == {"accounts"} and w == set()
        # TODO: assert has lock_intent attribute == "SHARE"

    @pytest.mark.xfail(reason="SELECT FOR UPDATE NOWAIT not yet supported")
    def test_select_for_update_nowait(self):
        """SELECT FOR UPDATE NOWAIT should parse lock with no-wait semantics."""
        sql = "SELECT * FROM orders WHERE id = ? FOR UPDATE NOWAIT"
        r, w = parse_sql_access(sql)
        assert r == {"orders"} and w == set()
        # TODO: assert lock_intent == "UPDATE" with nowait=True

    @pytest.mark.xfail(reason="SELECT FOR UPDATE SKIP LOCKED not yet supported")
    def test_select_for_update_skip_locked(self):
        """SELECT FOR UPDATE SKIP LOCKED should parse with skip semantics."""
        sql = "SELECT * FROM inventory WHERE product_id IN (...) FOR UPDATE SKIP LOCKED"
        r, w = parse_sql_access(sql)
        assert r == {"inventory"} and w == set()
        # TODO: assert lock_intent == "UPDATE" with skip_locked=True


# =============================================================================
# TODO: LOCK TABLE Statement Support
# =============================================================================
# Issue: LOCK TABLE DDL not parsed
# Phase: 5
# Effort: ~30 lines + 15 tests
# Reference: ideas/sql_conflict/03_algorithm_1_sql_parsing.md#todo-lock-table-statement-support


class TestLockTableTodo:
    """Tests for LOCK TABLE statement parsing.

    Expected: parse_sql_access() should recognize LOCK TABLE and extract lock mode.
    Current: Falls back to endpoint-level (conservative "all tables write").
    """

    @pytest.mark.xfail(reason="LOCK TABLE statement not yet parsed")
    def test_lock_table_exclusive(self):
        """LOCK TABLE ... IN EXCLUSIVE MODE should be recognized."""
        sql = "LOCK TABLE users IN EXCLUSIVE MODE"
        # Expected API:
        # read, write, lock_type = parse_sql_access_with_locks(sql)
        # assert read == set()
        # assert write == set()
        # assert lock_type == ("users", "EXCLUSIVE")
        r, w = parse_sql_access(sql)
        # Currently falls back to endpoint-level
        # TODO: assert recognizes lock statement

    @pytest.mark.xfail(reason="LOCK TABLE SHARE mode not yet parsed")
    def test_lock_table_share(self):
        """LOCK TABLE ... IN SHARE MODE should be recognized."""
        sql = "LOCK TABLE orders IN SHARE MODE"
        r, w = parse_sql_access(sql)
        # TODO: assert lock extracted with mode "SHARE"

    @pytest.mark.xfail(reason="LOCK TABLE IN ROW EXCLUSIVE MODE not yet parsed")
    def test_lock_table_row_exclusive(self):
        """LOCK TABLE ... IN ROW EXCLUSIVE MODE should be recognized."""
        sql = "LOCK TABLE inventory IN ROW EXCLUSIVE MODE"
        r, w = parse_sql_access(sql)
        # TODO: assert lock mode == "ROW EXCLUSIVE"

    @pytest.mark.xfail(reason="Multiple LOCK TABLE statements not yet parsed")
    def test_lock_multiple_tables(self):
        """Multiple LOCK TABLE statements should be parsed in sequence."""
        sqls = [
            "LOCK TABLE users IN EXCLUSIVE MODE",
            "LOCK TABLE orders IN SHARE MODE",
        ]
        # Expected: parse and track all locks
        for sql in sqls:
            parse_sql_access(sql)
        # TODO: assert lock tracking


# =============================================================================
# TODO: Advisory Lock Detection
# =============================================================================
# Issue: Advisory lock function calls not parsed (socket-level only)
# Phase: 5
# Effort: ~100 lines (Rust) + 30 tests
# Reference: ideas/sql_conflict/03_algorithm_1_sql_parsing.md#todo-advisory-locks-postgresql-mysql


class TestAdvisoryLocksTodo:
    """Tests for PostgreSQL advisory lock function recognition.

    Expected: parse_sql_access() should extract lock IDs from advisory lock calls.
    Current: Treated as opaque function calls (socket-level detection only).

    Note: These are function calls, not DML. Detection requires wire protocol parsing.
    """

    @pytest.mark.xfail(reason="Advisory lock ID not extracted from function call")
    def test_pg_advisory_lock_id_extraction(self):
        """pg_advisory_lock(id) should extract lock ID."""
        sql = "SELECT pg_advisory_lock(?)"
        # Expected API:
        # read, write, advisory_locks = parse_sql_access_with_locks(sql)
        # assert advisory_locks == {12345}  # when params=(12345,)
        r, w = parse_sql_access(sql)
        # Currently no lock ID extraction
        # TODO: extract lock ID from function

    @pytest.mark.xfail(reason="Advisory xact lock (transaction-scoped) not modeled")
    def test_pg_advisory_xact_lock(self):
        """pg_advisory_xact_lock(id) should indicate transaction-scoped lock."""
        sql = "SELECT pg_advisory_xact_lock(?)"
        # Expected: xact_lock=True in lock metadata
        # (vs pg_advisory_lock which is session-scoped)
        r, w = parse_sql_access(sql)
        # TODO: distinguish transaction vs session scope

    @pytest.mark.xfail(reason="Advisory shared lock not distinguished")
    def test_pg_advisory_shared_lock(self):
        """pg_advisory_shared_lock(id) should indicate shared lock intent."""
        sql = "SELECT pg_advisory_shared_lock(?)"
        r, w = parse_sql_access(sql)
        # TODO: assert shared lock mode

    @pytest.mark.xfail(reason="MySQL GET_LOCK not yet parsed")
    def test_mysql_get_lock(self):
        """MySQL GET_LOCK(name, timeout) should be recognized."""
        sql = "SELECT GET_LOCK(?, ?)"
        r, w = parse_sql_access(sql)
        # TODO: extract lock name from MySQL function

    @pytest.mark.xfail(reason="Advisory locks in DO block not extracted")
    def test_pg_advisory_lock_in_do_block(self):
        """Advisory locks in DO blocks should be detected."""
        sql = """
        DO $$
        BEGIN
            PERFORM pg_advisory_lock(1);
            UPDATE accounts SET balance = balance - 100 WHERE id = 1;
            PERFORM pg_advisory_unlock(1);
        END $$
        """
        r, w = parse_sql_access(sql)
        # TODO: extract lock IDs from PL/pgSQL


# =============================================================================
# TODO: UNION / INTERSECT / EXCEPT Optimization
# =============================================================================
# Issue: Set operations treated as "all tables write" (conservative)
# Phase: 5
# Effort: ~20 lines + 8 tests
# Reference: ideas/sql_conflict/03_algorithm_1_sql_parsing.md#todo-union-handling-overly-conservative


class TestUnionOptimizationTodo:
    """Tests for optimized UNION/INTERSECT/EXCEPT handling.

    Expected: Recognize as read-only compositions (all branches are reads).
    Current: Conservative fallback treats all tables as writes.
    """

    def test_union_select_should_be_reads_not_writes(self):
        """UNION of two SELECT should extract all tables as reads."""
        sql = "SELECT id FROM users UNION SELECT id FROM archived_users"
        r, w = parse_sql_access(sql)
        # Expected: r == {"users", "archived_users"}, w == set()
        assert "users" in r and "archived_users" in r
        assert w == set(), "UNION reads should not be classified as writes"

    def test_intersect_should_be_reads(self):
        """INTERSECT of two SELECT should extract reads only."""
        sql = "SELECT id FROM users INTERSECT SELECT id FROM admins"
        r, w = parse_sql_access(sql)
        # Expected: both → reads
        assert "users" in r and "admins" in r
        assert w == set()

    def test_except_should_be_reads(self):
        """EXCEPT of two SELECT should extract reads only."""
        sql = "SELECT id FROM all_users EXCEPT SELECT id FROM banned_users"
        r, w = parse_sql_access(sql)
        assert "all_users" in r and "banned_users" in r
        assert w == set()

    def test_union_all_should_be_reads(self):
        """UNION ALL (without deduplication) should still be read-only."""
        sql = "SELECT * FROM orders UNION ALL SELECT * FROM archived_orders"
        r, w = parse_sql_access(sql)
        assert "orders" in r and "archived_orders" in r
        assert w == set()

    def test_insert_union_target_is_write(self):
        """INSERT with UNION source should correctly identify target as write."""
        sql = "INSERT INTO summary SELECT * FROM users UNION SELECT * FROM archived_users"
        r, w = parse_sql_access(sql)
        # Expected: summary → write, users + archived_users → read
        assert w == {"summary"}
        assert "users" in r and "archived_users" in r


# =============================================================================
# TODO: Cross-Table Foreign Key Dependencies
# =============================================================================
# Issue: FK relationships not parsed (schema introspection needed)
# Phase: 6
# Effort: ~150 lines + 25 tests
# Reference: ideas/sql_conflict/03_algorithm_1_sql_parsing.md#todo-cross-table-foreign-key-dependencies


class TestForeignKeyDependenciesTodo:
    """Tests for FK dependency detection.

    Expected: Parser should recognize FK relationships and mark dependent operations as conflicting.
    Current: Cross-table operations on different tables are independent (false negatives).

    Note: Requires schema introspection (information_schema queries).
    """

    @pytest.mark.xfail(reason="FK dependencies not detected")
    def test_fk_insert_delete_detected(self):
        """INSERT into child table + DELETE from parent should be dependent via FK."""
        # Scenario: orders.user_id → users.id (FK constraint)
        insert_sql = "INSERT INTO orders (user_id, amount) VALUES (?, ?)"
        delete_sql = "DELETE FROM users WHERE id = ?"

        r_insert, w_insert = parse_sql_access(insert_sql)
        r_delete, w_delete = parse_sql_access(delete_sql)

        # Currently: independent (different tables)
        # Expected: should be marked dependent via FK
        assert w_insert == {"orders"}
        assert w_delete == {"users"}
        # TODO: add FK detection that marks as dependent

    @pytest.mark.xfail(reason="FK chain dependencies not detected")
    def test_fk_chain_dependencies(self):
        """Chain of FK dependencies should be recognized."""
        # shipments → orders → users (chain of FKs)
        r1, w1 = parse_sql_access("DELETE FROM users WHERE id = ?")
        r2, w2 = parse_sql_access("DELETE FROM orders WHERE user_id = ?")
        r3, w3 = parse_sql_access("DELETE FROM shipments WHERE order_id = ?")

        # Currently: all independent
        # Expected: detect transitive dependencies
        # TODO: implement transitive FK detection

    @pytest.mark.xfail(reason="Self-referential FK not detected")
    def test_self_referential_fk(self):
        """Self-referential FK (e.g., manager_id → id in same table) should be detected."""
        sql = "UPDATE employees SET manager_id = ? WHERE id = ?"
        r, w = parse_sql_access(sql)
        # Same table, same column involved; should recognize self-reference
        # TODO: detect self-referential constraints


# =============================================================================
# TODO: Transaction Boundaries
# =============================================================================
# Issue: Statement-level granularity misses transaction atomicity
# Phase: 6
# Effort: ~80 lines + 20 tests
# Reference: ideas/sql_conflict/03_algorithm_1_sql_parsing.md#todo-transaction-boundaries-not-tracked


class TestTransactionBoundariesTodo:
    """Tests for transaction boundary tracking.

    Expected: BEGIN/COMMIT should group SQL operations; atomicity modeled.
    Current: Each statement treated independently (search space explosion).

    Note: Requires tracking state across multiple cursor.execute() calls.
    """

    @pytest.mark.xfail(reason="Transaction boundaries not grouped")
    def test_transaction_grouping_begin_commit(self):
        """Operations between BEGIN and COMMIT should be grouped atomically."""
        begin_sql = "BEGIN"
        select_sql = "SELECT * FROM accounts WHERE id = 1"
        update_sql = "UPDATE accounts SET balance = balance - 100 WHERE id = 1"
        commit_sql = "COMMIT"

        # Expected: parse_sql_access should track transaction state
        # and group these operations into a single "transaction" ObjectId
        # Currently: treated as 4 independent operations
        for sql in [begin_sql, select_sql, update_sql, commit_sql]:
            parse_sql_access(sql)
        # TODO: implement transaction grouping

    @pytest.mark.xfail(reason="Savepoints not tracked")
    def test_savepoint_tracking(self):
        """Savepoint creation should be tracked separately from transaction."""
        sqls = [
            "BEGIN",
            "UPDATE accounts SET balance = balance - 100 WHERE id = 1",
            "SAVEPOINT sp1",
            "UPDATE accounts SET balance = balance - 50 WHERE id = 2",
            "ROLLBACK TO SAVEPOINT sp1",
            "COMMIT",
        ]
        # Expected: recognize savepoint as sub-transaction boundary
        for sql in sqls:
            parse_sql_access(sql)
        # TODO: distinguish savepoint scope

    @pytest.mark.xfail(reason="ROLLBACK not recognized as transaction end")
    def test_rollback_transaction_boundary(self):
        """ROLLBACK should mark transaction boundary and discard operations."""
        sqls = [
            "BEGIN",
            "DELETE FROM sensitive_data WHERE id = 1",
            "ROLLBACK",
        ]
        for sql in sqls:
            parse_sql_access(sql)
        # TODO: track rollback semantics


# =============================================================================
# TODO: Temporal Tables & System Versioning
# =============================================================================
# Issue: FOR SYSTEM_TIME clauses not parsed
# Phase: 7
# Effort: ~40 lines + 10 tests
# Priority: Very Low (rare; specialized SQL)
# Reference: ideas/sql_conflict/03_algorithm_1_sql_parsing.md#todo-temporal-tables--system-versioning


class TestTemporalTablesTodo:
    """Tests for temporal/system-versioned table support.

    Expected: FOR SYSTEM_TIME clauses should be extracted with temporal bounds.
    Current: Treated as regular table access (false positives on historical queries).

    Note: PostgreSQL, MySQL 8.0+, SQL Server support temporal tables.
    """

    @pytest.mark.xfail(reason="FOR SYSTEM_TIME AS OF not parsed")
    def test_for_system_time_as_of(self):
        """FOR SYSTEM_TIME AS OF should extract temporal predicate."""
        sql = "SELECT * FROM users FOR SYSTEM_TIME AS OF '2024-01-01' WHERE id = 1"
        r, w = parse_sql_access(sql)
        # Expected: extract temporal bounds for conflict analysis
        # (historical read shouldn't conflict with current writes)
        assert r == {"users"}
        # TODO: assert temporal predicate extracted

    @pytest.mark.xfail(reason="FOR SYSTEM_TIME BETWEEN not parsed")
    def test_for_system_time_between(self):
        """FOR SYSTEM_TIME BETWEEN should extract temporal range."""
        sql = "SELECT * FROM accounts FOR SYSTEM_TIME BETWEEN '2024-01-01' AND '2024-01-31'"
        r, w = parse_sql_access(sql)
        # Expected: temporal range extracted
        assert r == {"accounts"}
        # TODO: assert range predicate

    @pytest.mark.xfail(reason="System-versioned INSERT not recognized")
    def test_system_versioned_table_insert(self):
        """INSERT into system-versioned table should be recognized."""
        sql = "INSERT INTO audit_log (event, valid_from) VALUES (?, NOW())"
        r, w = parse_sql_access(sql)
        # System-versioned tables auto-manage time dimension
        assert w == {"audit_log"}
        # TODO: mark as system-versioned


# =============================================================================
# TODO: Generated & Computed Columns
# =============================================================================
# Issue: Generated/computed columns not marked
# Phase: 7
# Effort: ~30 lines + 5 tests
# Priority: Very Low (informational; minimal impact)
# Reference: ideas/sql_conflict/03_algorithm_1_sql_parsing.md#todo-generatedcomputed-columns


class TestGeneratedColumnsTodo:
    """Tests for generated/computed column handling.

    Expected: Generated columns should be marked; excluded from row predicates.
    Current: Treated as user-settable columns (may be referenced in WHERE).

    Note: Minimal impact; mostly informational metadata.
    """

    @pytest.mark.xfail(reason="Generated columns not recognized in schema")
    def test_generated_column_excluded_from_predicates(self):
        """Generated columns should not appear in row-level predicates."""
        # Table: orders (id PK, user_id, amount, total GENERATED AS (amount * tax_rate))
        sql = "SELECT * FROM orders WHERE id = ? AND total > ?"
        r, w = parse_sql_access(sql)
        # Expected: recognize 'total' as computed (read-only)
        assert r == {"orders"}
        # TODO: mark total as generated

    @pytest.mark.xfail(reason="Cannot set generated column in UPDATE")
    def test_generated_column_not_writable(self):
        """UPDATE should not allow setting generated columns."""
        sql = "UPDATE orders SET total = 100 WHERE id = 1"  # total is generated!
        r, w = parse_sql_access(sql)
        # This is a malformed query (should error at DB)
        # Parser should recognize total is computed
        # TODO: validate generated columns


# =============================================================================
# TODO: Window Functions & Partitioning
# =============================================================================
# Issue: PARTITION BY semantics not modeled
# Phase: 7
# Effort: ~20 lines + 3 tests
# Priority: Very Low (rare; conservative fallback sufficient)
# Reference: ideas/sql_conflict/03_algorithm_1_sql_parsing.md#todo-window-functions--partitioning


class TestWindowFunctionsTodo:
    """Tests for window function semantics.

    Expected: Window functions should be recognized; partitioning should be marked.
    Current: Treated as regular table reads (conservative but correct).

    Note: Window functions implicitly reference all rows in partition.
    """

    @pytest.mark.xfail(reason="Window function partitioning not modeled")
    def test_window_function_partition_by(self):
        """PARTITION BY should indicate row interdependencies within partition."""
        sql = """
        SELECT id, salary,
               RANK() OVER (PARTITION BY dept_id ORDER BY salary DESC) AS rank
        FROM employees
        """
        r, w = parse_sql_access(sql)
        assert r == {"employees"}
        # TODO: mark that all rows in same dept_id are interdependent

    @pytest.mark.xfail(reason="Window function frame not recognized")
    def test_window_function_frame(self):
        """Window FRAME specification should be recognized."""
        sql = """
        SELECT id, salary,
               AVG(salary) OVER (ORDER BY salary ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING) AS rolling_avg
        FROM employees
        """
        r, w = parse_sql_access(sql)
        assert r == {"employees"}
        # TODO: mark window frame


# =============================================================================
# TODO: Prepared Statements & Caching
# =============================================================================
# Issue: ObjectId caching could miss conflicts with different parameters
# Phase: 5
# Effort: ~10 lines + 3 tests
# Priority: Medium (correctness issue if caching added)
# Reference: ideas/sql_conflict/03_algorithm_1_sql_parsing.md#todo-prepared-statements--caching


class TestPreparedStatementCachingTodo:
    """Tests for prepared statement parameter handling.

    Expected: Same prepared statement with different parameters → different ObjectIds.
    Current: Parameters resolved at execute-time (correct, but no caching).

    Note: Only relevant if future optimization adds ObjectId caching.
    """

    @pytest.mark.xfail(reason="Prepared statement parameter cache not yet tested")
    def test_prepared_statement_different_params_independent(self):
        """Same prepared stmt with different parameters should have different ObjectIds."""
        # Simulate: stmt = cursor.prepare("SELECT * FROM users WHERE id = ? FOR UPDATE")
        sql = "SELECT * FROM users WHERE id = ? FOR UPDATE"

        # Execute with param 1
        params_1 = (1,)
        r1, w1 = parse_sql_access(sql)  # params not passed in this API

        # Execute with param 2
        params_2 = (2,)
        r2, w2 = parse_sql_access(sql)

        # Both parse same way currently; need param-aware API
        # Expected: row-level ObjectIds differ
        # Lock on row 1 ≠ lock on row 2
        # TODO: ensure parameter resolution is fresh per execute


# =============================================================================
# TODO: Stored Procedures & Dynamic SQL
# =============================================================================
# Issue: Stored procedures opaque (dynamic SQL not introspectable)
# Phase: 7
# Effort: ~200 lines + 40 tests
# Priority: Very Low (rare in modern Python ORMs)
# Reference: ideas/sql_conflict/03_algorithm_1_sql_parsing.md#todo-stored-procedures--dynamic-sql


class TestStoredProceduresTodo:
    """Tests for stored procedure introspection.

    Expected: Stored procedure definitions should be parsed; body SQL extracted.
    Current: CALL statements treated as opaque socket I/O (endpoint-level only).

    Note: Very low priority; most Python ORMs use direct SQL.
    """

    @pytest.mark.xfail(reason="Stored procedure not introspected")
    def test_stored_procedure_call_introspection(self):
        """CALL sp_name should resolve to table access of procedure body."""
        # Assuming: CREATE PROCEDURE sp_update_user(p_id INT, p_name VARCHAR) AS
        #   UPDATE users SET name = p_name WHERE id = p_id;
        sql = "CALL sp_update_user(?, ?)"
        r, w = parse_sql_access(sql)
        # Currently: opaque (endpoint-level)
        # Expected: introspect procedure; return users → write
        # TODO: build procedure → table mapping

    @pytest.mark.xfail(reason="Dynamic SQL in DO block not parsed")
    def test_do_block_dynamic_sql(self):
        """DO blocks with dynamic SQL should extract table access."""
        sql = """
        DO $$
        DECLARE
            v_table_name TEXT := 'orders';
        BEGIN
            EXECUTE 'DELETE FROM ' || v_table_name || ' WHERE status = ''cancelled''';
        END $$
        """
        r, w = parse_sql_access(sql)
        # Currently: unknown
        # Expected: recognize 'orders' from concatenation (hard problem!)
        # TODO: dynamic SQL analysis


# =============================================================================
# TODO: Multi-Dialect SQL Differences
# =============================================================================
# Issue: Dialect-specific syntax differences
# Phase: 5-7
# Effort: ~0 lines (sqlglot handles most)
# Priority: Low (sqlglot already covers 30+ dialects)
# Reference: ideas/sql_conflict/03_algorithm_1_sql_parsing.md#todo-multi-dialect-sql-differences


class TestMultiDialectTodo:
    """Tests for database dialect support.

    Note: sqlglot already handles most dialects. These tests verify edge cases.
    """

    def test_mysql_insert_or_replace(self):
        """MySQL INSERT ... ON DUPLICATE KEY UPDATE should be recognized."""
        sql = "INSERT INTO users (id, name) VALUES (?, ?) ON DUPLICATE KEY UPDATE name = VALUES(name)"
        r, w = parse_sql_access(sql)
        assert w == {"users"}
        # Expected: recognize as INSERT with possible UPDATE (writes only)

    def test_sqlite_insert_or_replace(self):
        """SQLite INSERT OR REPLACE should be recognized."""
        sql = "INSERT OR REPLACE INTO accounts (id, balance) VALUES (?, ?)"
        r, w = parse_sql_access(sql)
        assert w == {"accounts"}

    def test_postgres_on_conflict(self):
        """PostgreSQL INSERT ... ON CONFLICT should be recognized."""
        sql = "INSERT INTO users (id, name) VALUES (?, ?) ON CONFLICT (id) DO UPDATE SET name = ?"
        r, w = parse_sql_access(sql)
        assert w == {"users"}
        # Expected: recognize as write (upsert)


# =============================================================================
# YAML-Based Test Cases (Formula for Easy Extension)
# =============================================================================
# This format makes it easy to add new test cases without writing Python boilerplate.
# Each entry: (sql, params, expected_read_tables, expected_write_tables, expected_features)


YAML_TODO_TEST_CASES = """
# Format: (sql, params, expected_read, expected_write, features_todo)

# SELECT FOR UPDATE cases
- sql: "SELECT * FROM users WHERE id = ? FOR UPDATE"
  params: (1,)
  expected_read: {users}
  expected_write: {}
  features_todo:
    - lock_intent: UPDATE  # exclusive lock
    - row_id: 1

- sql: "SELECT * FROM accounts WHERE balance > ? FOR SHARE"
  params: (100,)
  expected_read: {accounts}
  expected_write: {}
  features_todo:
    - lock_intent: SHARE  # shared lock
    - lock_mode: SHARE

# LOCK TABLE cases
- sql: "LOCK TABLE orders IN EXCLUSIVE MODE"
  params: ()
  expected_read: {}
  expected_write: {}
  features_todo:
    - lock_statement: (orders, EXCLUSIVE)

- sql: "LOCK TABLE inventory IN ROW SHARE MODE"
  params: ()
  expected_read: {}
  expected_write: {}
  features_todo:
    - lock_statement: (inventory, ROW SHARE)

# Advisory lock cases
- sql: "SELECT pg_advisory_lock(?)"
  params: (12345,)
  expected_read: {}
  expected_write: {}
  features_todo:
    - advisory_lock_id: 12345
    - lock_scope: session

- sql: "SELECT pg_advisory_xact_lock(?)"
  params: (99999,)
  expected_read: {}
  expected_write: {}
  features_todo:
    - advisory_lock_id: 99999
    - lock_scope: transaction

# UNION optimization cases
- sql: "SELECT id FROM users UNION SELECT id FROM archived_users"
  params: ()
  expected_read: {users, archived_users}
  expected_write: {}
  features_todo:
    - union_type: read_only
    - is_not_write: true

- sql: "INSERT INTO summary SELECT * FROM orders UNION SELECT * FROM archived_orders"
  params: ()
  expected_read: {orders, archived_orders}
  expected_write: {summary}
  features_todo:
    - insert_union_target: summary
    - union_sources_are_reads: true

# Transaction boundary cases
- sql: "BEGIN"
  params: ()
  expected_read: {}
  expected_write: {}
  features_todo:
    - transaction_start: true
    - tx_id: null  # assigned dynamically

- sql: "COMMIT"
  params: ()
  expected_read: {}
  expected_write: {}
  features_todo:
    - transaction_end: COMMIT
    - tx_id: null

# Temporal table cases
- sql: "SELECT * FROM users FOR SYSTEM_TIME AS OF '2024-01-01' WHERE id = ?"
  params: (1,)
  expected_read: {users}
  expected_write: {}
  features_todo:
    - temporal_predicate: "AS OF '2024-01-01'"
    - historical_read: true

# Window function cases
- sql: "SELECT id, ROW_NUMBER() OVER (PARTITION BY dept_id ORDER BY salary DESC) FROM employees"
  params: ()
  expected_read: {employees}
  expected_write: {}
  features_todo:
    - window_function: true
    - partition_column: dept_id
    - interdependent_rows: dept_id
"""


@pytest.mark.xfail(reason="YAML test cases not yet implemented")
class TestYamlTodoFormula:
    """Template for converting YAML test cases to pytest parametrization.

    To implement: parse YAML, parametrize tests, execute with assertions.
    This keeps test definitions concise and separates data from logic.
    """

    def test_yaml_cases_template(self):
        """Template for parametrized YAML-based tests."""
        # In a real implementation:
        # 1. Parse YAML_TODO_TEST_CASES
        # 2. For each case, call parse_sql_access(sql, params)
        # 3. Assert against expected_read, expected_write
        # 4. Check features_todo (once implemented)
        pass


# =============================================================================
# ADDITIONAL HIGH-PRIORITY GAPS (Not in original 27 TODOs)
# =============================================================================
# Based on: ideas/sql_conflict/ADDITIONAL_SQL_TEST_GAPS.md


class TestCorrelatedSubqueriesTodo:
    """Tests for correlated subqueries with outer table references.

    Expected: Parser should recognize implicit join dependency between outer and inner tables.
    Current: Treated as independent subquery (may miss conflicts).

    Priority: HIGH (common in production; affects conflict semantics)
    Phase: 6
    Effort: ~20 lines + 5 tests
    """

    @pytest.mark.xfail(reason="Correlated subqueries not recognized")
    def test_correlated_subquery_in_where(self):
        """WHERE with correlated subquery should mark outer table dependency."""
        sql = """
        SELECT * FROM users u
        WHERE balance > (SELECT AVG(balance) FROM accounts WHERE user_id = u.id)
        """
        r, w = parse_sql_access(sql)
        assert "users" in r and "accounts" in r
        # TODO: mark correlation dependency (users ← accounts)

    @pytest.mark.xfail(reason="Correlated subquery in SELECT list not recognized")
    def test_correlated_subquery_in_select_list(self):
        """SELECT clause with correlated subquery should mark dependency."""
        sql = """
        SELECT u.id, u.name,
               (SELECT COUNT(*) FROM orders WHERE user_id = u.id) AS order_count
        FROM users u
        """
        r, w = parse_sql_access(sql)
        assert "users" in r and "orders" in r
        # TODO: mark correlation in SELECT clause


class TestCaseExpressionsTodo:
    """Tests for CASE expressions affecting predicates.

    Expected: CASE expressions in WHERE/SET should be recognized.
    Current: Treated as part of WHERE/SET without special handling.

    Priority: HIGH (common in conditional logic)
    Phase: 6
    Effort: ~15 lines + 3 tests
    """

    @pytest.mark.xfail(reason="CASE in WHERE not semantically analyzed")
    def test_case_in_where_clause(self):
        """CASE expression in WHERE should be recognized."""
        sql = """
        SELECT * FROM orders
        WHERE CASE WHEN status = 'pending' THEN amount > 100
                   ELSE amount > 500 END
        """
        r, w = parse_sql_access(sql)
        assert r == {"orders"}
        # TODO: mark CASE expression presence

    @pytest.mark.xfail(reason="CASE in UPDATE SET not recognized")
    def test_case_in_update_set(self):
        """CASE expression in UPDATE SET should be recognized."""
        sql = """
        UPDATE accounts SET balance = balance +
          CASE WHEN type = 'premium' THEN 50 ELSE 10 END
        WHERE id = ?
        """
        r, w = parse_sql_access(sql)
        assert "accounts" in w and "accounts" in r


class TestExistsNotExistsTodo:
    """Tests for EXISTS / NOT EXISTS subqueries.

    Expected: Parser should recognize existence checks as implicit joins.
    Current: Subquery tables extracted but correlation not modeled.

    Priority: HIGH (existence checks create dependencies)
    Phase: 6
    Effort: ~15 lines + 4 tests
    """

    @pytest.mark.xfail(reason="EXISTS subquery correlation not recognized")
    def test_exists_subquery(self):
        """EXISTS with correlated subquery should mark dependency."""
        sql = """
        SELECT * FROM users u
        WHERE EXISTS (SELECT 1 FROM orders WHERE user_id = u.id)
        """
        r, w = parse_sql_access(sql)
        assert "users" in r and "orders" in r
        # TODO: mark EXISTS correlation

    @pytest.mark.xfail(reason="NOT EXISTS subquery correlation not recognized")
    def test_not_exists_subquery(self):
        """NOT EXISTS should mark existence-based dependency."""
        sql = """
        DELETE FROM accounts
        WHERE NOT EXISTS (SELECT 1 FROM transactions WHERE account_id = accounts.id)
        """
        r, w = parse_sql_access(sql)
        assert "accounts" in w
        assert "transactions" in r


class TestMultipleRowInsertTodo:
    """Tests for multiple-row INSERT statements.

    Expected: Multiple VALUES rows should be recognized as separate logical inserts.
    Current: Treated as single bulk insert (correct but loses row granularity).

    Priority: HIGH (common bulk operation)
    Phase: 6
    Effort: ~10 lines + 3 tests
    """

    @pytest.mark.xfail(reason="Multi-row INSERT not split into logical rows")
    def test_multiple_row_insert_values(self):
        """INSERT with multiple VALUES rows should be recognized."""
        sql = """
        INSERT INTO users (name, email) VALUES
          ('Alice', 'a@x'),
          ('Bob', 'b@x'),
          ('Carol', 'c@x')
        """
        r, w = parse_sql_access(sql)
        assert w == {"users"}
        # TODO: track row count or list of rows inserted


class TestDistinctTodo:
    """Tests for DISTINCT and DISTINCT ON semantics.

    Expected: DISTINCT should be recognized (affects result set).
    Current: Treated transparently (correct but loses semantic info).

    Priority: HIGH (affects which rows are returned)
    Phase: 6
    Effort: ~10 lines + 2 tests
    """

    @pytest.mark.xfail(reason="DISTINCT ON (PostgreSQL) not recognized")
    def test_distinct_on(self):
        """PostgreSQL DISTINCT ON should be recognized."""
        sql = """
        SELECT DISTINCT ON (user_id) * FROM events
        ORDER BY user_id, created_at DESC
        """
        r, w = parse_sql_access(sql)
        assert r == {"events"}
        # TODO: mark DISTINCT ON semantics


class TestSelfJoinsTodo:
    """Tests for self-joins and circular dependencies.

    Expected: Same table in multiple roles should be marked with explicit join.
    Current: Table extracted once (loses join information).

    Priority: HIGH (circular dependencies)
    Phase: 6
    Effort: ~15 lines + 3 tests
    """

    @pytest.mark.xfail(reason="Self-join not recognized as circular dependency")
    def test_self_join_employees(self):
        """Self-join for hierarchy (employees + managers) should be marked."""
        sql = """
        SELECT a.id, a.name, b.id AS manager_id
        FROM employees a
        JOIN employees b ON a.manager_id = b.id
        """
        r, w = parse_sql_access(sql)
        assert r == {"employees"}
        # TODO: mark as self-join with dependency

    @pytest.mark.xfail(reason="Self-referential FK in UPDATE not marked")
    def test_self_referential_update(self):
        """UPDATE on table with self-referential FK should be marked."""
        sql = """
        UPDATE categories SET parent_id = ? WHERE id = ?
        """
        r, w = parse_sql_access(sql)
        assert "categories" in r and "categories" in w
        # TODO: mark as self-referential


class TestLimitOffsetTodo:
    """Tests for LIMIT/OFFSET affecting row selection.

    Expected: LIMIT/OFFSET should be recognized for DELETE/UPDATE.
    Current: Treated as semantic metadata (correct but imprecise).

    Priority: HIGH (affects which rows are written)
    Phase: 6
    Effort: ~20 lines + 4 tests
    """

    @pytest.mark.xfail(reason="DELETE with LIMIT not recognized")
    def test_delete_with_limit(self):
        """DELETE with LIMIT should indicate limited write scope."""
        sql = """
        DELETE FROM sessions ORDER BY created_at LIMIT 10
        """
        r, w = parse_sql_access(sql)
        assert "sessions" in w and "sessions" in r
        # TODO: track LIMIT as write scope limiter

    @pytest.mark.xfail(reason="SELECT with LIMIT/OFFSET not tracked")
    def test_select_with_limit_offset(self):
        """SELECT with LIMIT/OFFSET should be recognized."""
        sql = """
        SELECT * FROM orders ORDER BY id LIMIT 20 OFFSET 100
        """
        r, w = parse_sql_access(sql)
        assert r == {"orders"}
        # TODO: track pagination parameters


class TestOuterJoinWhereSemanticsToodo:
    """Tests for WHERE after OUTER JOIN (changes join type semantics).

    Expected: Parser should recognize that WHERE after LEFT JOIN can change semantics to INNER.
    Current: Treated as standard LEFT JOIN with WHERE.

    Priority: HIGH (subtle but important semantic change)
    Phase: 6
    Effort: ~20 lines + 3 tests
    """

    @pytest.mark.xfail(reason="LEFT JOIN with WHERE semantics not distinguished")
    def test_left_join_with_where_on_outer_table(self):
        """WHERE on right table after LEFT JOIN converts to INNER JOIN semantics."""
        sql = """
        SELECT * FROM orders o
        LEFT JOIN users u ON o.user_id = u.id
        WHERE u.id IS NOT NULL
        """
        r, w = parse_sql_access(sql)
        # Should recognize that WHERE u.id IS NOT NULL makes this effectively INNER
        assert r == {"orders", "users"}
        # TODO: mark that WHERE changed join type


class TestLateralJoinsTodo:
    """Tests for PostgreSQL LATERAL joins (dependent subqueries).

    Expected: LATERAL subqueries should be marked as correlated.
    Current: Would fall through to endpoint-level or be misclassified.

    Priority: MEDIUM-HIGH (PostgreSQL specific)
    Phase: 6
    Effort: ~15 lines + 2 tests
    """

    @pytest.mark.xfail(reason="LATERAL joins not recognized as correlated")
    def test_lateral_join_with_correlation(self):
        """LATERAL subquery should be marked as dependent on outer table."""
        sql = """
        SELECT * FROM users u,
        LATERAL (SELECT * FROM orders WHERE user_id = u.id LIMIT 1) o
        """
        r, w = parse_sql_access(sql)
        assert "users" in r and "orders" in r
        # TODO: mark LATERAL correlation


class TestUpsertEdgeCasesTodo:
    """Tests for advanced UPSERT (INSERT ... ON CONFLICT) scenarios.

    Expected: CONFLICT clauses with WHERE should be recognized for precision.
    Current: Conservative "always write".

    Priority: MEDIUM-HIGH (conditional writes)
    Phase: 6
    Effort: ~25 lines + 6 tests
    """

    @pytest.mark.xfail(reason="ON CONFLICT DO UPDATE with WHERE not optimized")
    def test_insert_on_conflict_do_update_with_where(self):
        """ON CONFLICT DO UPDATE with WHERE should limit write scope."""
        sql = """
        INSERT INTO users (id, name) VALUES (?, ?)
        ON CONFLICT (id) DO UPDATE SET name = ? WHERE is_active = true
        """
        r, w = parse_sql_access(sql)
        assert w == {"users"}
        # TODO: mark conditional update scope

    @pytest.mark.xfail(reason="ON CONFLICT DO NOTHING optimization missing")
    def test_insert_on_conflict_do_nothing(self):
        """ON CONFLICT DO NOTHING is actually only INSERT (no UPDATE)."""
        sql = """
        INSERT INTO unique_tokens (token, user_id) VALUES (?, ?)
        ON CONFLICT (token) DO NOTHING
        """
        r, w = parse_sql_access(sql)
        assert w == {"unique_tokens"}
        # TODO: mark DO NOTHING as read-only conflict check


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
