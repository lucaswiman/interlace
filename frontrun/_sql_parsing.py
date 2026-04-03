"""SQL statement parsing for read/write table extraction.

Provides ``parse_sql_access(sql)`` which returns a :class:`SqlAccessResult`
for conflict detection. Uses a regex fast-path for simple statements and falls
back to sqlglot for complex SQL (CTEs, subqueries, UNION, etc.).
"""

from __future__ import annotations

import enum
import re
from dataclasses import dataclass
from typing import Any, NamedTuple

# ---------------------------------------------------------------------------
# Typed enums for lock intent and transaction operations
# ---------------------------------------------------------------------------


class LockIntent(enum.Enum):
    """Lock mode extracted from SQL statements (FOR UPDATE, FOR SHARE, LOCK TABLE)."""

    UPDATE = "UPDATE"
    SHARE = "SHARE"
    UPDATE_SKIP_LOCKED = "UPDATE_SKIP_LOCKED"


class TxOp(enum.Enum):
    """Simple transaction control operations."""

    BEGIN = "BEGIN"
    COMMIT = "COMMIT"
    ROLLBACK = "ROLLBACK"


@dataclass(frozen=True)
class Savepoint:
    """SAVEPOINT <name> operation."""

    name: str


@dataclass(frozen=True)
class RollbackTo:
    """ROLLBACK TO SAVEPOINT <name> operation."""

    name: str


@dataclass(frozen=True)
class Release:
    """RELEASE SAVEPOINT <name> operation."""

    name: str


TxControl = TxOp | Savepoint | RollbackTo | Release


class SqlAccessResult(NamedTuple):
    """Result of parsing a SQL statement for read/write table extraction."""

    read_tables: set[str]
    write_tables: set[str]
    lock_intent: LockIntent | None
    tx_op: TxControl | None
    temporal_clauses: dict[str, str] | None
    ast: Any | None = None  # Pre-parsed sqlglot AST (when available from _sqlglot_parse)
    delete_tables: set[str] | None = None  # Tables targeted by DELETE (for phantom read detection)


_EMPTY = SqlAccessResult(set(), set(), None, None, None, None, None)


# Precompiled patterns for constructs not handled by sqlglot:
# transaction control (including savepoints and SET AUTOCOMMIT),
# LOCK TABLE (lock intent), COPY, and PREPARE/EXECUTE/DEALLOCATE.
_IDENT = r'(?:"[^"]+"(?:\."[^"]+")?|`[^`]+`(?:\.`[^`]+`)?|[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)'
_WS = r"[\s\n]+"

_RE_LOCK_TABLE = re.compile(rf"\bLOCK{_WS}TABLE{_WS}({_IDENT})", re.I)
_RE_LOCK_MODE = re.compile(rf"\bIN{_WS}(.+){_WS}MODE\b", re.I)
_RE_TX_BEGIN = re.compile(r"^\s*(BEGIN|START\s+TRANSACTION|SET\s+AUTOCOMMIT\s*=\s*0)\b", re.I)
_RE_TX_COMMIT = re.compile(r"^\s*(COMMIT|END|SET\s+AUTOCOMMIT\s*=\s*1)\b", re.I)
_RE_TX_ROLLBACK = re.compile(r"^\s*ROLLBACK\b", re.I)
_RE_TX_SAVEPOINT = re.compile(r"^\s*SAVEPOINT\s+(\w+)\b", re.I)
_RE_TX_RELEASE = re.compile(r"^\s*RELEASE\s+(SAVEPOINT\s+)?(\w+)\b", re.I)
_RE_TX_ROLLBACK_TO = re.compile(r"^\s*ROLLBACK\s+TO\s+(SAVEPOINT\s+)?(\w+)\b", re.I)
# COPY: COPY table FROM/TO ...
_RE_COPY = re.compile(rf"^\s*COPY{_WS}({_IDENT})", re.I)
_RE_COPY_DIR = re.compile(r"\b(FROM|TO)\b", re.I)
# PREPARE / EXECUTE (PostgreSQL server-side prepared statements)
_RE_PREPARE = re.compile(rf"^\s*PREPARE{_WS}(\w+)", re.I)
_RE_PREPARE_AS = re.compile(r"\bAS\b", re.I)
_RE_EXECUTE_STMT = re.compile(rf"^\s*EXECUTE{_WS}(\w+)", re.I)
_RE_DEALLOCATE = re.compile(rf"^\s*DEALLOCATE{_WS}", re.I)


def _strip_quotes(name: str) -> str:
    """Remove surrounding quotes/backticks and extract table from schema.table.

    Handles quoted schema-qualified names like "public"."users" and
    `myschema`.`users` by splitting on "." or `.` boundaries first,
    then stripping quotes from the last component.
    """
    # Split on dot boundaries between quoted identifiers: "schema"."table"
    # or `schema`.`table`.  Also handles unquoted schema.table.
    if '"."' in name:
        parts = name.split('"."')
        last = parts[-1]
    elif "`.`" in name:
        parts = name.split("`.`")
        last = parts[-1]
    elif '".`' in name or '`."' in name:
        # Mixed quoting — unlikely but handle gracefully
        last = name.rsplit(".", 1)[-1]
    else:
        # Unquoted or single quoted identifier
        if name.startswith(('"', "`")):
            name = name[1:-1]
        return name.rsplit(".", 1)[-1]

    # Strip remaining quotes from the last component
    if last.startswith(('"', "`")):
        last = last[1:]
    if last.endswith(('"', "`")):
        last = last[:-1]
    return last


def _regex_parse(sql: str) -> SqlAccessResult | None:
    """Handle SQL constructs not covered by sqlglot.

    Covers: transaction control (including savepoints and SET AUTOCOMMIT),
    LOCK TABLE (for lock intent), COPY, and PREPARE/EXECUTE/DEALLOCATE.
    Returns None for everything else, which falls through to sqlglot.
    """
    stripped = sql.strip().rstrip(";").strip()

    # Multi-statement → let sqlglot handle
    if ";" in stripped:
        return None

    # Transaction control — check before other patterns
    if _RE_TX_BEGIN.search(stripped):
        return SqlAccessResult(set(), set(), None, TxOp.BEGIN, None)
    if _RE_TX_COMMIT.search(stripped):
        return SqlAccessResult(set(), set(), None, TxOp.COMMIT, None)
    if _RE_TX_ROLLBACK.search(stripped):
        m = _RE_TX_ROLLBACK_TO.search(stripped)
        if m:
            return SqlAccessResult(set(), set(), None, RollbackTo(m.group(2)), None)
        return SqlAccessResult(set(), set(), None, TxOp.ROLLBACK, None)
    m = _RE_TX_SAVEPOINT.search(stripped)
    if m:
        return SqlAccessResult(set(), set(), None, Savepoint(m.group(1)), None)
    m = _RE_TX_RELEASE.search(stripped)
    if m:
        return SqlAccessResult(set(), set(), None, Release(m.group(2)), None)

    # LOCK TABLE — sqlglot parses it as DDL but doesn't set lock_intent
    m_lock = _RE_LOCK_TABLE.search(stripped)
    if m_lock:
        tbl = _strip_quotes(m_lock.group(1))
        lock_intent: LockIntent = LockIntent.UPDATE
        m_mode = _RE_LOCK_MODE.search(stripped)
        if m_mode:
            mode = m_mode.group(1).upper()
            if "SHARE" in mode and "EXCLUSIVE" not in mode:
                lock_intent = LockIntent.SHARE
        return SqlAccessResult(set(), {tbl}, lock_intent, None, None)

    # COPY table FROM/TO — PostgreSQL bulk I/O (sqlglot doesn't handle)
    m_copy = _RE_COPY.search(stripped)
    if m_copy:
        tbl = _strip_quotes(m_copy.group(1))
        # COPY (subquery) — has parenthesis right after COPY, bail to sqlglot
        after_copy = stripped[m_copy.start() :].lstrip()
        if after_copy.upper().startswith("COPY") and "(" in after_copy.split()[1:2]:
            return None
        m_dir = _RE_COPY_DIR.search(stripped, m_copy.end())
        if m_dir and m_dir.group(1).upper() == "TO":
            return SqlAccessResult({tbl}, set(), None, None, None)
        else:
            return SqlAccessResult(set(), {tbl}, None, None, None)

    # DEALLOCATE [PREPARE] stmt_name — no table access
    if _RE_DEALLOCATE.search(stripped):
        return SqlAccessResult(set(), set(), None, None, None)

    # PREPARE stmt AS <sql> — parse the inner SQL directly
    m_prepare = _RE_PREPARE.search(stripped)
    if m_prepare:
        m_as = _RE_PREPARE_AS.search(stripped, m_prepare.end())
        if m_as:
            inner_sql = stripped[m_as.end() :].strip()
            if inner_sql:
                inner = _regex_parse(inner_sql) or _sqlglot_parse(inner_sql)
                if inner is not None:
                    return inner
        return SqlAccessResult(set(), set(), None, None, None)

    # EXECUTE stmt_name [(params)] — opaque without prepared stmt registry
    m_exec = _RE_EXECUTE_STMT.search(stripped)
    if m_exec:
        return SqlAccessResult(set(), set(), None, None, None)

    return None  # fall through to sqlglot


def _merge_lock_intent(a: LockIntent | None, b: LockIntent | None) -> LockIntent | None:
    """Merge two lock intents, preferring UPDATE > UPDATE_SKIP_LOCKED > SHARE."""
    if a is LockIntent.UPDATE or b is LockIntent.UPDATE:
        return LockIntent.UPDATE
    if a is LockIntent.UPDATE_SKIP_LOCKED or b is LockIntent.UPDATE_SKIP_LOCKED:
        return LockIntent.UPDATE_SKIP_LOCKED
    if a is LockIntent.SHARE or b is LockIntent.SHARE:
        return LockIntent.SHARE
    return None


def _sqlglot_parse(sql: str) -> SqlAccessResult | None:
    """Full parser: handles CTEs, subqueries, UNION, MERGE, etc.

    Returns a single merged :class:`SqlAccessResult` for all statements.
    For multi-statement SQL, reads/writes are merged and the last tx_op wins.
    The ``ast`` field is populated with the first parsed AST (for single-statement
    SQL, this is the only AST; callers can use it to avoid re-parsing).
    """
    try:
        import sqlglot  # type: ignore[import-untyped]
        from sqlglot import errors as sqlglot_errors  # type: ignore[import-untyped]
        from sqlglot import exp  # type: ignore[import-untyped]
    except ImportError:
        return None

    # Pre-process pyformat parameter placeholders (%s, %(name)s) which
    # sqlglot default dialect chokes on (misinterprets % as modulo).
    if "%" in sql:
        sql = re.sub(r"%(?:\(\w+\))?s", "?", sql)

    try:
        expressions = sqlglot.parse(sql)
    except sqlglot_errors.ParseError:
        expressions = None

    # Fallback dialects: mysql handles backtick identifiers and MySQL-specific syntax
    # (ON DUPLICATE KEY UPDATE, etc.); tsql handles FOR SYSTEM_TIME.
    # Also re-parse with mysql when backticks are present even if default succeeded,
    # since the default dialect may misparse backtick-quoted identifiers.
    sql_upper = sql.upper()
    if not expressions or "`" in sql:
        try:
            mysql_exprs = sqlglot.parse(sql, read="mysql")
            if mysql_exprs:
                expressions = mysql_exprs
        except sqlglot_errors.ParseError:
            pass
    if not expressions and "FOR SYSTEM_TIME" in sql_upper:
        try:
            expressions = sqlglot.parse(sql, read="tsql")
        except sqlglot_errors.ParseError:
            pass
    if not expressions:
        return None  # unparseable → fall back to endpoint-level

    if not expressions:
        return None

    all_write: set[str] = set()
    all_read: set[str] = set()
    all_delete: set[str] = set()
    all_lock_intent: LockIntent | None = None
    all_tx_op: TxControl | None = None
    all_temporal: dict[str, str] | None = None
    first_ast: Any | None = None

    for ast in expressions:
        if ast is None:
            continue

        if first_ast is None:
            first_ast = ast

        write: set[str] = set()
        read: set[str] = set()
        lock_intent: LockIntent | None = None
        tx_op: TxControl | None = None

        # Transaction control
        if isinstance(ast, exp.Transaction):
            tx_op = TxOp.BEGIN
        elif isinstance(ast, exp.Commit):
            tx_op = TxOp.COMMIT
        elif isinstance(ast, exp.Rollback):
            tx_op = TxOp.ROLLBACK
        else:
            # Extract lock intent from SELECT (including inside CTEs)
            def _extract_lock_intent_from_select(select_node: exp.Expression) -> LockIntent | None:
                """Extract lock intent from a SELECT, checking FOR UPDATE SKIP LOCKED."""
                lock_node = select_node.find(exp.Lock)
                if not lock_node:
                    return None
                if lock_node.args.get("update"):
                    # wait=False means SKIP LOCKED in sqlglot
                    if lock_node.args.get("wait") is False:
                        return LockIntent.UPDATE_SKIP_LOCKED
                    return LockIntent.UPDATE
                # Not update → share lock
                intent = LockIntent.SHARE
                kind_val = lock_node.args.get("kind")
                if kind_val:
                    kind_upper = str(kind_val).upper()
                    if "UPDATE" in kind_upper:
                        intent = LockIntent.UPDATE
                    elif "SHARE" in kind_upper:
                        intent = LockIntent.SHARE
                return intent

            if isinstance(ast, exp.Select):
                lock_intent = _extract_lock_intent_from_select(ast)

            # Also extract lock intent from CTEs (e.g. WITH cte AS (SELECT ... FOR UPDATE SKIP LOCKED))
            for cte_node in ast.find_all(exp.CTE):
                cte_intent = _extract_lock_intent_from_select(cte_node.this)
                if cte_intent is not None:
                    lock_intent = _merge_lock_intent(lock_intent, cte_intent)

            # Advisory locks (PostgreSQL, MySQL)
            for call in ast.find_all(exp.Anonymous):
                name = call.this.lower()
                if name in (
                    "pg_advisory_lock",
                    "pg_advisory_xact_lock",
                    "pg_advisory_lock_shared",
                    "pg_advisory_xact_lock_shared",
                    "get_lock",
                ):
                    # Extract lock ID/name if it's a literal
                    if call.expressions:
                        lock_ids: list[str] = []
                        # For get_lock, the second argument is a timeout, not part of the ID
                        args_to_use = call.expressions[:1] if name == "get_lock" else call.expressions
                        for arg in args_to_use:
                            if isinstance(arg, exp.Literal):
                                lock_ids.append(str(arg.this))
                            else:
                                lock_ids.append("?")
                        lock_id_str = ":".join(lock_ids)
                        write.add(f"advisory_lock:{lock_id_str}")
                        if "shared" in name:
                            lock_intent = _merge_lock_intent(lock_intent, LockIntent.SHARE)
                        else:
                            lock_intent = _merge_lock_intent(lock_intent, LockIntent.UPDATE)

            # Shared table visitor logic
            for t in ast.find_all(exp.Table):
                # Check for system versioning (FOR SYSTEM_TIME)
                version = t.find(exp.Version)
                if version:
                    clause = str(version)
                    # Standardize: sqlglot often translates to "FOR TIMESTAMP" in its internal representation
                    clause = clause.replace("FOR TIMESTAMP ", "FOR SYSTEM_TIME ")
                    # Extract only the predicate part
                    clause = clause.replace("FOR SYSTEM_TIME ", "").strip()
                    if all_temporal is None:
                        all_temporal = {}
                    all_temporal[t.name] = clause

            if isinstance(ast, exp.Insert):
                tbl = ast.find(exp.Table)
                if tbl:
                    write.add(tbl.name)
                # Source tables (everything after the target)
                if ast.expression:  # the SELECT source
                    for t in ast.expression.find_all(exp.Table):
                        read.add(t.name)
            elif isinstance(ast, (exp.Update, exp.Delete)):
                tbl = ast.this
                if isinstance(tbl, exp.Table):
                    write.add(tbl.name)
                    read.add(tbl.name)
                    if isinstance(ast, exp.Delete):
                        all_delete.add(tbl.name)
                for t in ast.find_all(exp.Table):
                    if t.name not in write:
                        read.add(t.name)
            elif isinstance(ast, exp.Select):
                for t in ast.find_all(exp.Table):
                    read.add(t.name)
            elif isinstance(ast, (exp.Union, exp.Intersect, exp.Except)):
                for t in ast.find_all(exp.Table):
                    read.add(t.name)
            elif isinstance(ast, exp.Merge):
                target = ast.this
                if isinstance(target, exp.Table):
                    write.add(target.name)
                    read.add(target.name)
                # All non-target tables are read sources
                for t in ast.find_all(exp.Table):
                    if t.name not in write:
                        read.add(t.name)
            else:
                # DDL, GRANT, etc. — conservatively treat as write
                for t in ast.find_all(exp.Table):
                    write.add(t.name)

        all_read.update(read)
        all_write.update(write)
        all_lock_intent = _merge_lock_intent(all_lock_intent, lock_intent)
        if tx_op:
            all_tx_op = tx_op  # Take the last tx_op

    # For single-statement SQL, attach AST so callers can avoid re-parsing
    result_ast = first_ast if len([e for e in expressions if e is not None]) == 1 else None
    return SqlAccessResult(
        all_read,
        all_write,
        all_lock_intent,
        all_tx_op,
        all_temporal,
        result_ast,
        all_delete if all_delete else None,
    )


def parse_sql_access(sql: str) -> SqlAccessResult:
    """Extract table access info from a SQL statement.

    Uses regex fast-path for simple statements, falls back to sqlglot
    for complex SQL. Returns empty sets if parsing fails entirely
    (endpoint-level I/O detection remains as fallback).
    """
    # Fast path: covers ~90% of ORM-generated SQL
    result = _regex_parse(sql)
    if result is not None:
        return result

    # Full parser for complex SQL
    result = _sqlglot_parse(sql)
    if result is not None:
        return result

    # Parse failure: return empty sets → endpoint-level fallback
    return _EMPTY
