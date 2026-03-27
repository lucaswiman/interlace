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


# Precompiled patterns — matches the leading keyword + first table name.
# Handles optional schema qualification (schema.table) and quoted identifiers.
_IDENT = r'(?:"[^"]+"(?:\."[^"]+")?|`[^`]+`(?:\.`[^`]+`)?|[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)'
_WS = r"[\s\n]+"

_RE_SELECT = re.compile(r"\bSELECT\b", re.I)
_RE_INSERT = re.compile(rf"\bINSERT{_WS}INTO{_WS}({_IDENT})", re.I)
_RE_UPDATE = re.compile(rf"\bUPDATE{_WS}({_IDENT}){_WS}SET\b", re.I)
_RE_DELETE = re.compile(rf"\bDELETE{_WS}FROM{_WS}({_IDENT})", re.I)
_RE_FROM = re.compile(rf"\bFROM{_WS}({_IDENT})", re.I)
_RE_JOIN = re.compile(rf"\bJOIN{_WS}({_IDENT})", re.I)
_RE_LITERAL = re.compile(r"'[^']*'")
_RE_FOR_UPDATE_SKIP_LOCKED = re.compile(r"\bFOR" + _WS + r"UPDATE" + _WS + r"SKIP" + _WS + r"LOCKED\b", re.I)
_RE_FOR_UPDATE = re.compile(r"\bFOR" + _WS + r"UPDATE\b", re.I)
_RE_FOR_SHARE = re.compile(r"\bFOR" + _WS + r"SHARE\b", re.I)
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
    """Fast-path: extract tables from simple single-statement SQL.

    Returns a :class:`SqlAccessResult` or None if the SQL is too complex
    (subqueries, CTEs, UNION, MERGE) and needs the full parser.
    """
    stripped = sql.strip().rstrip(";").strip()

    # Quick multi-statement check (before tx control, avoids regex cost)
    # Only strip trailing semicolons; interior semicolons indicate multiple statements.
    if ";" in stripped:
        return None

    # Transaction control - check before other patterns (no literal stripping needed)
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

    # Strip literals to avoid false positives (e.g. " FROM " inside a string)
    no_literals = _RE_LITERAL.sub(" ", stripped)
    upper_no_literals = no_literals.upper()

    # Bail to full parser for complex SQL
    if any(kw in upper_no_literals for kw in ("WITH ", "UNION", "INTERSECT", "EXCEPT", "MERGE", "EXISTS")):
        return None

    # FOR SYSTEM_TIME is complex for regex, bail to sqlglot
    if "SYSTEM_TIME" in upper_no_literals:
        return None

    if "USING" in upper_no_literals and "DELETE" in upper_no_literals:
        return None

    if " FROM " in upper_no_literals:
        # Check for multiple tables in FROM clause (comma-separated)
        # Using a conservative split to avoid complicated regex
        from_after = upper_no_literals.split(" FROM ", 1)[1]
        for end_kw in (" WHERE ", " GROUP BY ", " ORDER BY ", " LIMIT ", " FOR "):
            from_after = from_after.split(end_kw, 1)[0]
        if "," in from_after:
            return None

    # Subqueries in DELETE or UPDATE (SELECT appearing after the first keyword)
    if (
        not any(stripped.lower().startswith(kw) for kw in ("select", "insert", "prepare"))
        and "SELECT" in upper_no_literals
    ):
        return None

    # Function calls (advisory locks etc) — bail to sqlglot
    if "(" in stripped and not stripped.lower().startswith("insert"):
        if any(kw in stripped.lower() for kw in ("pg_advisory", "get_lock")):
            return None

    read: set[str] = set()
    write: set[str] = set()
    lock_intent: LockIntent | None = None

    m_lock = _RE_LOCK_TABLE.search(no_literals)
    if m_lock:
        tbl = _strip_quotes(m_lock.group(1))
        # Treat LOCK TABLE as an exclusive write by default for safety.
        # This ensures it conflicts with all other accesses.
        lock_intent = LockIntent.UPDATE
        m_mode = _RE_LOCK_MODE.search(no_literals)
        if m_mode:
            mode = m_mode.group(1).upper()
            if "SHARE" in mode and "EXCLUSIVE" not in mode:
                lock_intent = LockIntent.SHARE
        write.add(tbl)
        return SqlAccessResult(read, write, lock_intent, None, None)

    if _RE_FOR_UPDATE_SKIP_LOCKED.search(no_literals):
        lock_intent = LockIntent.UPDATE_SKIP_LOCKED
    elif _RE_FOR_UPDATE.search(no_literals):
        lock_intent = LockIntent.UPDATE
    elif _RE_FOR_SHARE.search(no_literals):
        lock_intent = LockIntent.SHARE

    m_insert = _RE_INSERT.search(no_literals)
    if m_insert:
        write.add(_strip_quotes(m_insert.group(1)))
        # Source tables in INSERT ... SELECT ... FROM
        for m in _RE_FROM.finditer(no_literals, m_insert.end()):
            read.add(_strip_quotes(m.group(1)))
        return SqlAccessResult(read, write, lock_intent, None, None)

    m_update = _RE_UPDATE.search(no_literals)
    if m_update:
        tbl = _strip_quotes(m_update.group(1))
        write.add(tbl)
        read.add(tbl)  # WHERE reads the target
        # Subquery tables in FROM/JOIN (UPDATE ... FROM ... syntax)
        for m in _RE_FROM.finditer(no_literals, m_update.end()):
            t = _strip_quotes(m.group(1))
            if t not in write:
                read.add(t)
        return SqlAccessResult(read, write, lock_intent, None, None)

    m_delete = _RE_DELETE.search(no_literals)
    if m_delete:
        tbl = _strip_quotes(m_delete.group(1))
        write.add(tbl)
        read.add(tbl)
        return SqlAccessResult(read, write, lock_intent, None, None, None, {tbl})

    if _RE_SELECT.search(no_literals):
        for m in _RE_FROM.finditer(no_literals):
            read.add(_strip_quotes(m.group(1)))
        for m in _RE_JOIN.finditer(no_literals):
            read.add(_strip_quotes(m.group(1)))
        return SqlAccessResult(read, write, lock_intent, None, None)

    # COPY table FROM/TO — PostgreSQL bulk I/O
    m_copy = _RE_COPY.search(no_literals)
    if m_copy:
        tbl = _strip_quotes(m_copy.group(1))
        # COPY (subquery) — has parenthesis right after COPY, bail to sqlglot
        after_copy = no_literals[m_copy.start() :].lstrip()
        if after_copy.upper().startswith("COPY") and "(" in after_copy.split()[1:2]:
            return None
        m_dir = _RE_COPY_DIR.search(no_literals, m_copy.end())
        if m_dir and m_dir.group(1).upper() == "TO":
            read.add(tbl)
        else:
            write.add(tbl)
        return SqlAccessResult(read, write, None, None, None)

    # DEALLOCATE [PREPARE] stmt_name — no table access
    if _RE_DEALLOCATE.search(stripped):
        return SqlAccessResult(set(), set(), None, None, None)

    # PREPARE stmt AS <sql> — parse the inner SQL
    m_prepare = _RE_PREPARE.search(stripped)
    if m_prepare:
        m_as = _RE_PREPARE_AS.search(stripped, m_prepare.end())
        if m_as:
            inner_sql = stripped[m_as.end() :].strip()
            if inner_sql:
                inner = _regex_parse(inner_sql)
                if inner is not None:
                    return inner
                return None  # let sqlglot handle the inner SQL
        return SqlAccessResult(set(), set(), None, None, None)

    # EXECUTE stmt_name [(params)] — opaque, cannot resolve table without prepared stmt registry
    m_exec = _RE_EXECUTE_STMT.search(stripped)
    if m_exec:
        return SqlAccessResult(set(), set(), None, None, None)

    return None  # unknown statement type → fall through


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
        # Fallback for FOR SYSTEM_TIME if default dialect fails
        if "FOR SYSTEM_TIME" in sql.upper():
            try:
                expressions = sqlglot.parse(sql, read="tsql")
            except sqlglot_errors.ParseError:
                return None
        else:
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
                        if t.name not in write:
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
