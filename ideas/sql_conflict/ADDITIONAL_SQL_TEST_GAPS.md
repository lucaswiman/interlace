"""
Comprehensive SQL test case gaps analysis.

This document identifies additional SQL features and edge cases not yet covered
in test_sql_parsing_todo.py. Organized by priority and database dialect.
"""

# =============================================================================
# HIGH PRIORITY: Common in Production Applications
# =============================================================================

HIGH_PRIORITY_GAPS = """

1. CORRELATED SUBQUERIES (High priority - semantic dependency)
   Example:
   - SELECT * FROM users u WHERE balance > (SELECT AVG(balance) FROM accounts WHERE user_id = u.id)
   Issue: Subquery references outer table (u) - implicit join dependency
   Impact: Should mark both tables as conflicting (row interdependency)
   Effort: ~20 lines + 5 tests
   Phase: 5

2. CASE EXPRESSIONS (High priority - conditional logic affects predicates)
   Examples:
   - SELECT * FROM orders WHERE CASE WHEN status = 'pending' THEN amount > 100 ELSE amount > 500 END
   - UPDATE accounts SET balance = balance + CASE WHEN type = 'premium' THEN 50 ELSE 10 END
   Issue: Conditional logic in WHERE/SET not modeled
   Impact: Could affect row-level predicate extraction (conservative fallback OK)
   Effort: ~15 lines + 3 tests
   Phase: 6

3. EXISTS / NOT EXISTS (High priority - existence predicates)
   Examples:
   - SELECT * FROM users WHERE EXISTS (SELECT 1 FROM orders WHERE user_id = users.id)
   - DELETE FROM accounts WHERE NOT EXISTS (SELECT 1 FROM transactions WHERE account_id = accounts.id)
   Issue: Existence checks create implicit dependencies
   Impact: Should mark as correlated; affects conflict detection
   Effort: ~15 lines + 4 tests
   Phase: 6

4. MULTIPLE ROW INSERT (High priority - bulk operations)
   Examples:
   - INSERT INTO users (name, email) VALUES ('Alice', 'a@x'), ('Bob', 'b@x'), ('Carol', 'c@x')
   - INSERT INTO orders VALUES (1, 100), (2, 200), (3, 300)
   Issue: Currently parsed as single operation; actually multiple inserts
   Impact: Should track as multiple row insertions (if row-level tracking)
   Effort: ~10 lines + 3 tests
   Phase: 6

5. GROUP BY (High priority - aggregation semantics)
   Examples:
   - SELECT user_id, COUNT(*) FROM orders GROUP BY user_id
   - SELECT dept, AVG(salary) FROM employees GROUP BY dept HAVING AVG(salary) > 50000
   Issue: Aggregation doesn't affect which rows are read, but affects result set
   Impact: Currently tables extracted correctly; GROUP BY is semantic metadata only
   Effort: ~5 lines + 2 tests
   Phase: 5 (low effort, informational)

6. DISTINCT (High priority - affects row set semantics)
   Examples:
   - SELECT DISTINCT user_id FROM transactions
   - SELECT DISTINCT ON (user_id) * FROM events ORDER BY user_id, created_at DESC
   Issue: DISTINCT affects which rows are returned; PostgreSQL DISTINCT ON is table-specific
   Impact: Currently safe (conservative); could optimize for no-dependency cases
   Effort: ~10 lines + 2 tests
   Phase: 6 (optimization)

7. SELF-JOINS (High priority - circular dependencies)
   Examples:
   - SELECT a.id, b.id FROM employees a JOIN employees b ON a.manager_id = b.id
   - UPDATE categories SET parent_id = ? WHERE id IN (SELECT id FROM categories WHERE parent_id IS NULL)
   Issue: Same table read/written twice - circular dependency potential
   Impact: Should mark as same table with explicit join dependency
   Effort: ~15 lines + 3 tests
   Phase: 6

8. LIMIT / OFFSET (High priority - result set size)
   Examples:
   - SELECT * FROM orders ORDER BY id LIMIT 10 OFFSET 20
   - DELETE FROM sessions ORDER BY created_at LIMIT 10
   Issue: LIMIT/OFFSET affects which rows are returned/deleted
   Impact: For DELETE, limits which rows are written; affects conflict analysis
   Effort: ~20 lines + 4 tests
   Phase: 6

9. UPSERT EDGE CASES (Medium-high - PostgreSQL/SQLite specifics)
   PostgreSQL:
   - INSERT INTO users VALUES (?, ?) ON CONFLICT (id) DO UPDATE SET name = ? WHERE is_active
   - INSERT INTO orders VALUES (?, ?) ON CONFLICT DO NOTHING

   SQLite:
   - INSERT INTO accounts VALUES (?, ?) ON CONFLICT(id) DO UPDATE SET balance = balance + ?
   - INSERT INTO inventory VALUES (?, ?) ON CONFLICT REPLACE

   Issue: CONFLICT clauses create conditional writes; WHERE on DO UPDATE limits impact
   Impact: Currently conservative (all tables write); could optimize for precision
   Effort: ~25 lines + 6 tests
   Phase: 6

10. OUTER JOIN WHERE SEMANTICS (Medium-high - subtle semantic issue)
    Example:
    - SELECT * FROM orders o FULL OUTER JOIN users u ON o.user_id = u.id WHERE u.id IS NOT NULL
    Issue: WHERE after LEFT/FULL JOIN converts it to INNER JOIN semantics
    Impact: Changes which tables are truly read; could affect conflict detection
    Effort: ~20 lines + 3 tests
    Phase: 6

"""

# =============================================================================
# MEDIUM PRIORITY: Important Edge Cases & PostgreSQL Specifics
# =============================================================================

MEDIUM_PRIORITY_GAPS = """

1. LATERAL JOINS (PostgreSQL - dependent subqueries)
   Example:
   - SELECT * FROM users u, LATERAL (SELECT * FROM orders WHERE user_id = u.id LIMIT 1) o
   Issue: LATERAL makes subquery depend on outer query (implicit correlation)
   Impact: Should mark as correlated; affects conflict detection
   Effort: ~15 lines + 2 tests
   Phase: 6

2. PREPARED STATEMENTS - PREPARE/EXECUTE (PostgreSQL specific)
   Different from parameterized queries:
   - PREPARE stmt AS SELECT * FROM users WHERE id = $1
   - EXECUTE stmt(1)
   Issue: Two-phase process; statement cached server-side
   Impact: Parser sees PREPARE with template $1, then EXECUTE with actual params
   Effort: ~20 lines + 3 tests
   Phase: 5

3. LISTEN / NOTIFY (PostgreSQL pub/sub)
   Examples:
   - LISTEN channel_name
   - NOTIFY channel_name, 'message'
   Issue: Asynchronous messaging; not traditional I/O conflict
   Impact: Currently socket-level; could be modeled as pub/sub channels
   Effort: ~15 lines + 2 tests
   Phase: 7 (low priority)

4. COPY (PostgreSQL bulk loading)
   Examples:
   - COPY users FROM STDIN WITH (FORMAT csv)
   - COPY accounts TO STDOUT
   - COPY (SELECT * FROM users WHERE id > ?) TO 'file.csv'
   Issue: COPY is bulk I/O with query-like semantics
   Impact: Currently treated as socket I/O; could extract table + read/write mode
   Effort: ~20 lines + 3 tests
   Phase: 6

5. JSON OPERATIONS (PostgreSQL JSON operators)
   Examples:
   - SELECT * FROM data WHERE payload ->> 'field' = 'value'
   - SELECT * FROM logs WHERE metadata @> '{"level":"error"}'
   - UPDATE events SET metadata = jsonb_set(metadata, '{tags}', '[1,2,3]')
   Issue: JSON path operations are column-level; affect predicate extraction
   Impact: Currently safe (conservative table-level); could optimize
   Effort: ~20 lines + 4 tests
   Phase: 7 (low priority)

6. ARRAY OPERATIONS (PostgreSQL arrays)
   Examples:
   - SELECT * FROM inventory WHERE tags && ARRAY['urgent', 'high-priority']
   - SELECT * FROM collections WHERE members @> ARRAY[?]
   - UPDATE catalogs SET categories = array_append(categories, ?)
   Issue: Array operators on specific columns
   Impact: Column-level operations; safe with table-level detection
   Effort: ~15 lines + 3 tests
   Phase: 7 (low priority)

7. RANGE TYPES (PostgreSQL range operations)
   Examples:
   - SELECT * FROM events WHERE time_window && '[2024-01-01, 2024-02-01)'
   - SELECT * FROM bookings WHERE slot <@ daterange(?, ?)
   Issue: Range containment/overlap operators
   Impact: Affects which rows match; could be relevant for predicate analysis
   Effort: ~20 lines + 3 tests
   Phase: 7 (low priority)

8. UNLOGGED TABLES (PostgreSQL - special table type)
   Examples:
   - CREATE UNLOGGED TABLE temp_data (id INT, payload TEXT)
   - INSERT INTO temp_data VALUES (?, ?)
   Issue: Unlogged tables have different durability semantics
   Impact: Could affect concurrency assumptions; marked as writes but non-durable
   Effort: ~10 lines + 2 tests
   Phase: 7 (informational)

9. PARTITIONED TABLES (PostgreSQL table inheritance)
   Examples:
   - SELECT * FROM logs_2024 UNION ALL SELECT * FROM logs_2025
   - INSERT INTO measurements VALUES (?, ?) [auto-routed to partition]
   - UPDATE accounts SET balance = balance + ? WHERE account_id IN (SELECT account_id FROM partitions...)
   Issue: Logical table composed of physical partitions
   Impact: Should recognize partition routing; affects multi-table analysis
   Effort: ~25 lines + 4 tests
   Phase: 6

10. MATERIALIZED CTEs vs Regular (PostgreSQL optimization)
    Example:
    - WITH MATERIALIZED cte AS (SELECT * FROM large_table) SELECT * FROM cte, cte c2
    Issue: MATERIALIZED forces intermediate result; affects execution plan
    Impact: Currently safe (conservative); semantic difference in isolation
    Effort: ~10 lines + 2 tests
    Phase: 7 (optimization hint)

11. ATTACH/DETACH DATABASE (SQLite - dynamic loading)
    Examples:
    - ATTACH DATABASE 'other.db' AS other_db
    - SELECT * FROM other_db.users
    - DETACH DATABASE other_db
    Issue: Database scope changes at runtime
    Impact: Cross-database transactions create dependencies
    Effort: ~20 lines + 3 tests
    Phase: 6

12. SQLite FTS (Full-Text Search)
    Examples:
    - CREATE VIRTUAL TABLE docs USING fts5(title, content)
    - SELECT * FROM docs WHERE docs MATCH 'search_query'
    Issue: Virtual table with special query syntax
    Impact: Currently would fail to parse (falls back to endpoint-level)
    Effort: ~20 lines + 2 tests
    Phase: 7 (low priority)

13. SQLite JSON1 Extension
    Examples:
    - SELECT json_extract(data, '$.field') FROM records
    - UPDATE records SET data = json_set(data, '$.flag', true)
    Issue: JSON functions on columns
    Impact: Column-level ops; safe with table-level detection
    Effort: ~15 lines + 2 tests
    Phase: 7 (low priority)

"""

# =============================================================================
# MEDIUM-LOW PRIORITY: Cross-Database Edge Cases
# =============================================================================

MEDIUM_LOW_PRIORITY_GAPS = """

1. COLLATE CLAUSE (affects comparisons)
   Example:
   - SELECT * FROM names WHERE name COLLATE utf8_unicode_ci = ?
   Issue: Collation affects comparison semantics
   Impact: Safe with table-level; informational only
   Effort: ~10 lines + 2 tests
   Phase: 7

2. CAST / TYPE CONVERSIONS (affects predicate matching)
   Examples:
   - SELECT * FROM users WHERE CAST(id AS TEXT) = ?
   - SELECT * FROM events WHERE (created_at AT TIME ZONE 'UTC')::DATE = ?
   Issue: Type conversions affect value comparisons
   Impact: Safe with table-level; complex for row-level predicates
   Effort: ~15 lines + 3 tests
   Phase: 7

3. SCALAR FUNCTIONS IN WHERE (COALESCE, NULLIF, etc.)
   Examples:
   - SELECT * FROM accounts WHERE COALESCE(parent_id, id) = ?
   - SELECT * FROM events WHERE NULLIF(status, 'deleted') IS NOT NULL
   Issue: Functions wrap predicates
   Impact: Safe with table-level; could affect row-level predicate extraction
   Effort: ~20 lines + 4 tests
   Phase: 7

4. STRING FUNCTIONS IN WHERE (CONCAT, SUBSTRING, etc.)
   Examples:
   - SELECT * FROM users WHERE CONCAT(first_name, ' ', last_name) LIKE ?
   - SELECT * FROM emails WHERE SUBSTRING(address, 1, 1) = ?
   Issue: String ops on columns
   Impact: Safe with table-level; edge case for predicates
   Effort: ~20 lines + 4 tests
   Phase: 7

5. AGGREGATE FILTERS (PostgreSQL specific)
   Example:
   - SELECT COUNT(*) FILTER (WHERE status = 'active') FROM users
   Issue: Conditional aggregation
   Impact: Affects result set semantics; currently safe
   Effort: ~10 lines + 1 test
   Phase: 7 (informational)

6. NULLS FIRST / NULLS LAST (affects ORDER BY)
   Example:
   - SELECT * FROM orders ORDER BY created_at DESC NULLS FIRST
   Issue: Affects sort order but not data access
   Impact: Safe; informational only
   Effort: ~5 lines + 1 test
   Phase: 7

7. OFFSET WITHOUT LIMIT (unusual)
   Example:
   - SELECT * FROM records OFFSET 100  [no LIMIT]
   Issue: Unusual but valid; offsets result set
   Impact: Safe; edge case handling
   Effort: ~5 lines + 1 test
   Phase: 7

8. CROSS JOIN edge cases (all combinations)
   Example:
   - SELECT * FROM users CROSS JOIN roles CROSS JOIN permissions
   Issue: Cartesian product of multiple tables
   Impact: Currently safe (all tables extracted as reads)
   Effort: ~5 lines + 1 test
   Phase: 7

9. NATURAL JOIN (implicit column matching)
   Example:
   - SELECT * FROM users NATURAL JOIN profiles
   Issue: Auto-matches columns with same name
   Impact: Implicit join condition; currently safe
   Effort: ~10 lines + 2 tests
   Phase: 7

10. USING CLAUSE vs ON CLAUSE (equivalent but different syntax)
    Examples:
    - SELECT * FROM orders JOIN users USING (user_id)  [equivalent to ON orders.user_id = users.user_id]
    Issue: USING is syntactic sugar
    Impact: Currently safe; just different notation
    Effort: ~10 lines + 2 tests
    Phase: 7

11. DEFAULT VALUES in INSERT (uses defaults instead of VALUES)
    Example:
    - INSERT INTO logs DEFAULT VALUES
    Issue: No explicit values; uses column defaults
    Impact: Still a write; currently safe
    Effort: ~5 lines + 1 test
    Phase: 7

12. VALUES clause (standalone multi-row literal)
    Example:
    - VALUES (1, 2), (3, 4), (5, 6)  [used in UNION, etc.]
    Issue: Literal table construction
    Impact: Can appear in complex queries; currently safe
    Effort: ~10 lines + 2 tests
    Phase: 7

13. TABLE statement (PostgreSQL 8.4+, shorthand for SELECT *)
    Example:
    - TABLE users  [equivalent to SELECT * FROM users]
    Issue: Shorthand syntax not always recognized
    Impact: Would currently fall through to endpoint-level
    Effort: ~10 lines + 1 test
    Phase: 6

14. ONLY modifier (PostgreSQL table inheritance)
    Example:
    - SELECT * FROM ONLY parent_table  [exclude child tables]
    Issue: Table inheritance semantics
    Impact: Affects which physical tables are accessed
    Effort: ~10 lines + 1 test
    Phase: 7

15. FILTER clause in aggregate functions (PostgreSQL)
    Example:
    - SELECT status, COUNT(*) FILTER (WHERE active = true) FROM users GROUP BY status
    Issue: Conditional aggregation
    Impact: Safe; informational
    Effort: ~5 lines + 1 test
    Phase: 7

"""

# =============================================================================
# LOW PRIORITY: Rare/Specialist Features
# =============================================================================

LOW_PRIORITY_GAPS = """

1. WINDOW FRAME SPECIFICATIONS (ROWS/RANGE/GROUPS BETWEEN)
   Example:
   - SELECT SUM(amount) OVER (ORDER BY date ROWS BETWEEN 7 PRECEDING AND CURRENT ROW)
   Issue: Complex window semantics
   Impact: Currently marked as window function (safe); frame detail not extracted
   Effort: ~15 lines + 2 tests
   Phase: 8 (very low priority)

2. OVER() WITH MULTIPLE PARTITIONS (complex windowing)
   Example:
   - SELECT id, ROW_NUMBER() OVER w1, RANK() OVER w2 FROM employees WINDOW w1 AS (PARTITION BY ...), w2 AS (...)
   Issue: Multiple window definitions
   Impact: Currently safe (table-level)
   Effort: ~10 lines + 1 test
   Phase: 8

3. RECURSIVE CTEs - More Complex Cases
   Example:
   - WITH RECURSIVE tree AS (
       SELECT id, parent_id, 1 AS level FROM categories WHERE parent_id IS NULL
       UNION ALL
       SELECT c.id, c.parent_id, t.level + 1 FROM categories c JOIN tree t ON c.parent_id = t.id
     ) SELECT * FROM tree
   Issue: Mutual recursion, depth limits, etc.
   Impact: Currently basic support; edge cases not tested
   Effort: ~15 lines + 3 tests
   Phase: 8

4. CUSTOM TYPES (PostgreSQL user-defined types)
   Example:
   - CREATE TYPE address AS (street VARCHAR, city VARCHAR)
   - SELECT * FROM users WHERE home_address.city = 'NYC'
   Issue: Custom type semantics
   Impact: Safe with table-level; edge case
   Effort: ~10 lines + 1 test
   Phase: 8

5. CONSTRAINT TRIGGERS (PostgreSQL, deferred validation)
   Issue: Deferred constraint checking affects isolation semantics
   Impact: Rare; advanced feature
   Effort: ~10 lines + 1 test
   Phase: 8

6. RULE system (PostgreSQL, deprecated)
   Example:
   - CREATE RULE rule_name AS ON SELECT TO view_name DO INSTEAD SELECT ...
   Issue: Query rewriting via rules (deprecated in favor of triggers)
   Impact: Very rare; deprecated
   Effort: ~5 lines + 1 test
   Phase: 8

7. INSTEAD OF triggers (views with triggers)
   Example:
   - CREATE TRIGGER instead_of_delete INSTEAD OF DELETE ON view_name
   Issue: DML on views with trigger-based routing
   Impact: Affects which base tables are actually written
   Effort: ~15 lines + 2 tests
   Phase: 8

8. PRAGMA statements (SQLite configuration)
   Example:
   - PRAGMA journal_mode = WAL
   - PRAGMA foreign_keys = ON
   Issue: Configuration, not data access
   Impact: No conflict implications; ignored safely
   Effort: ~5 lines + 1 test
   Phase: 8

9. ANALYZE statement (statistics)
   Example:
   - ANALYZE table_name
   Issue: Statistics gathering; not data access
   Impact: Ignored safely; no conflict implications
   Effort: ~5 lines + 1 test
   Phase: 8

10. VACUUM statement (maintenance)
    Example:
    - VACUUM table_name
    - VACUUM (ANALYZE, VERBOSE)
    Issue: Maintenance operations
    Impact: Could block writes; model as table-level exclusive lock?
    Effort: ~10 lines + 2 tests
    Phase: 8

11. COPY with subquery (complex COPY)
    Example:
    - COPY (SELECT * FROM huge_table WHERE status = 'done') TO 'export.csv'
    Issue: COPY with query source
    Impact: Should extract table access from subquery
    Effort: ~10 lines + 1 test
    Phase: 7

12. MULTI-STATEMENT transaction edge cases
    Example:
    - BEGIN; SELECT ... FOR UPDATE; UPDATE ...; COMMIT;
    Issue: Complex transaction with locking
    Impact: Already deferred to Phase 6; testing edge cases
    Effort: ~10 lines + 2 tests
    Phase: 6

13. Savepoint edge cases (nested savepoints)
    Example:
    - SAVEPOINT sp1; INSERT ...; SAVEPOINT sp2; UPDATE ...; ROLLBACK TO sp2; RELEASE sp1;
    Issue: Complex savepoint nesting
    Impact: Should handle nested savepoint scope
    Effort: ~15 lines + 2 tests
    Phase: 6

14. Dynamically constructed queries (danger zone)
    Example:
    - EXECUTE 'SELECT * FROM ' || table_name || ' WHERE id = ' || param_id
    Issue: String concatenation for SQL construction
    Impact: Impossible to statically parse; runtime-dependent
    Effort: N/A (not solvable at parse time)
    Phase: N/A (limitation)

15. Comments in SQL (should be stripped)
    Example:
    - SELECT * FROM users -- inline comment
    - SELECT /* block comment */ * FROM orders
    Issue: Comments should be ignored
    Impact: Currently should work (comment handling in sqlglot)
    Effort: ~5 lines + 2 tests
    Phase: 5 (verify)

"""

# =============================================================================
# Summary Statistics
# =============================================================================

SUMMARY = """

TOTAL ADDITIONAL GAPS: ~95 test cases
- High Priority: 10 categories (multiple tests each) → ~30 tests
- Medium Priority: 13 categories → ~40 tests
- Medium-Low Priority: 15 categories → ~20 tests
- Low Priority: 15 categories → ~5 tests

EFFORT ESTIMATE:
- High Priority: ~180 lines + ~30 tests (most important)
- Medium Priority: ~200 lines + ~35 tests
- Medium-Low Priority: ~150 lines + ~25 tests
- Low Priority: ~100 lines + ~5 tests
- TOTAL: ~630 lines + ~95 tests (~2-3 weeks effort for comprehensive coverage)

PRIORITY FOR NEXT PHASE:
1. Correlated subqueries (affects semantics)
2. CASE expressions (conditional logic)
3. EXISTS / NOT EXISTS (implicit joins)
4. Multiple row INSERT (bulk operations)
5. DISTINCT (result set semantics)
6. Self-joins (circular dependencies)
7. LIMIT/OFFSET (row selection)
8. Outer join WHERE semantics (subtle semantic change)
9. LATERAL joins (PostgreSQL dependency)
10. UPSERT edge cases (conditional writes)

These 10 should be added to test_sql_parsing_todo.py before high-priority features
from the original list are fully implemented.
"""

if __name__ == "__main__":
    print("SQL PARSING TODO GAPS ANALYSIS")
    print("=" * 80)
    print(HIGH_PRIORITY_GAPS)
    print("\n")
    print(MEDIUM_PRIORITY_GAPS)
    print("\n")
    print(MEDIUM_LOW_PRIORITY_GAPS)
    print("\n")
    print(LOW_PRIORITY_GAPS)
    print("\n")
    print(SUMMARY)
