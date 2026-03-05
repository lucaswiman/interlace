"""SQL statement parsing for read/write table extraction.

Provides ``parse_sql_access(sql)`` which returns ``(read_tables, write_tables, lock_intent)``
for conflict detection. Uses a regex fast-path for simple statements and falls
back to sqlglot for complex SQL (CTEs, subqueries, UNION, etc.).
"""

from __future__ import annotations

import re

# Precompiled patterns — matches the leading keyword + first table name.
# Handles optional schema qualification (schema.table) and quoted identifiers.
_IDENT = r'(?:"[^"]+"|`[^`]+`|[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)'
_WS = r"[\s\n]+"

_RE_SELECT = re.compile(r"\bSELECT\b", re.I)
_RE_INSERT = re.compile(rf"\bINSERT{_WS}INTO{_WS}({_IDENT})", re.I)
_RE_UPDATE = re.compile(rf"\bUPDATE{_WS}({_IDENT}){_WS}SET\b", re.I)
_RE_DELETE = re.compile(rf"\bDELETE{_WS}FROM{_WS}({_IDENT})", re.I)
_RE_FROM = re.compile(rf"\bFROM{_WS}({_IDENT})", re.I)
_RE_JOIN = re.compile(rf"\bJOIN{_WS}({_IDENT})", re.I)
_RE_LITERAL = re.compile(r"'[^']*'")
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


def _strip_quotes(name: str) -> str:
    """Remove surrounding quotes/backticks and extract table from schema.table."""
    if name.startswith(('"', "`")):
        name = name[1:-1]
    # Take last component: "public"."users" → users
    return name.rsplit(".", 1)[-1]


def _regex_parse(sql: str) -> tuple[set[str], set[str], str | None, str | None] | None:
    """Fast-path: extract tables from simple single-statement SQL.

    Returns (read_tables, write_tables, lock_intent, tx_op) or None if the SQL
    is too complex (subqueries, CTEs, UNION, MERGE) and needs the full parser.
    """
    stripped = sql.strip().rstrip(";").strip()
    upper = stripped.upper()

    # Transaction control - check before other patterns
    if _RE_TX_BEGIN.search(stripped):
        return set(), set(), None, "BEGIN"
    if _RE_TX_COMMIT.search(stripped):
        return set(), set(), None, "COMMIT"
    if _RE_TX_ROLLBACK.search(stripped):
        m = _RE_TX_ROLLBACK_TO.search(stripped)
        if m:
            return set(), set(), None, f"ROLLBACK_TO:{m.group(2)}"
        return set(), set(), None, "ROLLBACK"
    if _RE_TX_SAVEPOINT.search(stripped):
        m = _RE_TX_SAVEPOINT.search(stripped)
        return set(), set(), None, f"SAVEPOINT:{m.group(1)}"
    if _RE_TX_RELEASE.search(stripped):
        m = _RE_TX_RELEASE.search(stripped)
        return set(), set(), None, f"RELEASE:{m.group(2)}"

    # Bail to full parser for complex SQL
    if any(kw in upper for kw in (
        "WITH ", "UNION", "INTERSECT", "EXCEPT", "MERGE", "RETURNING",
        "EXISTS", "IN ("
    )):
        return None

    # Subqueries in DELETE or UPDATE (SELECT appearing after the first keyword)
    if not any(stripped.lower().startswith(kw) for kw in ("select", "insert")) and "SELECT" in upper:
        return None

    # Function calls (advisory locks etc) — bail to sqlglot
    if "(" in stripped and not stripped.lower().startswith("insert"):
        if any(kw in stripped.lower() for kw in ("pg_advisory", "get_lock")):
            return None

    # Strip literals to avoid false positives (e.g. " FROM " inside a string)
    no_literals = _RE_LITERAL.sub(" ", stripped)

    read: set[str] = set()
    write: set[str] = set()
    lock_intent: str | None = None

    m_lock = _RE_LOCK_TABLE.search(no_literals)
    if m_lock:
        tbl = _strip_quotes(m_lock.group(1))
        # Treat LOCK TABLE as an exclusive write by default for safety.
        # This ensures it conflicts with all other accesses.
        lock_intent = "UPDATE"
        m_mode = _RE_LOCK_MODE.search(no_literals)
        if m_mode:
            mode = m_mode.group(1).upper()
            if "SHARE" in mode and "EXCLUSIVE" not in mode:
                lock_intent = "SHARE"
        write.add(tbl)
        return read, write, lock_intent, None

    if _RE_FOR_UPDATE.search(no_literals):
        lock_intent = "UPDATE"
    elif _RE_FOR_SHARE.search(no_literals):
        lock_intent = "SHARE"

    m_insert = _RE_INSERT.search(no_literals)
    if m_insert:
        write.add(_strip_quotes(m_insert.group(1)))
        # Source tables in INSERT ... SELECT ... FROM
        for m in _RE_FROM.finditer(no_literals, m_insert.end()):
            read.add(_strip_quotes(m.group(1)))
        return read, write, lock_intent, None

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
        return read, write, lock_intent, None

    m_delete = _RE_DELETE.search(no_literals)
    if m_delete:
        tbl = _strip_quotes(m_delete.group(1))
        write.add(tbl)
        read.add(tbl)
        return read, write, lock_intent, None

    if _RE_SELECT.search(no_literals):
        for m in _RE_FROM.finditer(no_literals):
            read.add(_strip_quotes(m.group(1)))
        for m in _RE_JOIN.finditer(no_literals):
            read.add(_strip_quotes(m.group(1)))
        return read, write, lock_intent, None

    return None  # unknown statement type → fall through


def _sqlglot_parse(sql: str) -> tuple[set[str], set[str], str | None, str | None] | None:
    """Full parser: handles CTEs, subqueries, UNION, MERGE, etc."""
    try:
        import sqlglot  # type: ignore[import-untyped]
        from sqlglot import exp  # type: ignore[import-untyped]
    except ImportError:
        return None

    try:
        ast = sqlglot.parse_one(sql)
    except sqlglot.errors.ParseError:
        return None  # unparseable → fall back to endpoint-level

    write: set[str] = set()
    read: set[str] = set()
    lock_intent: str | None = None
    tx_op: str | None = None

    # Transaction control
    if isinstance(ast, exp.Transaction):
        return read, write, lock_intent, "BEGIN"
    if isinstance(ast, exp.Commit):
        return read, write, lock_intent, "COMMIT"
    if isinstance(ast, exp.Rollback):
        return read, write, lock_intent, "ROLLBACK"

    # Extract lock intent from SELECT
    if isinstance(ast, exp.Select):
        lock = ast.find(exp.Lock)
        if lock:
            if lock.args.get("update"):
                lock_intent = "UPDATE"
            elif lock.args.get("share"):
                lock_intent = "SHARE"
            else:
                kind = lock.args.get("kind")
                if kind:
                    kind_upper = str(kind).upper()
                    if "UPDATE" in kind_upper:
                        lock_intent = "UPDATE"
                    elif "SHARE" in kind_upper:
                        lock_intent = "SHARE"

    # Advisory locks (PostgreSQL, MySQL)
    for call in ast.find_all(exp.Anonymous):
        name = call.this.lower()
        if name in ("pg_advisory_lock", "pg_advisory_xact_lock",
                    "pg_advisory_lock_shared", "pg_advisory_xact_lock_shared",
                    "get_lock"):
            # Extract lock ID/name if it's a literal
            if call.expressions:
                arg = call.expressions[0]
                if isinstance(arg, exp.Literal):
                    lock_id = arg.this
                else:
                    lock_id = "?"
                write.add(f"advisory_lock:{lock_id}")
                if "shared" in name:
                    lock_intent = "SHARE"
                else:
                    lock_intent = "UPDATE"

    if isinstance(ast, exp.Insert):
        tbl = ast.find(exp.Table)
        if tbl:
            write.add(tbl.name)
        # Source tables (everything after the target)
        if ast.expression:  # the SELECT source
            for t in ast.expression.find_all(exp.Table):
                if t.name not in write:
                    read.add(t.name)
        return read, write, lock_intent, None

    if isinstance(ast, exp.Update):
        tbl = ast.this
        if isinstance(tbl, exp.Table):
            write.add(tbl.name)
            read.add(tbl.name)
        for t in ast.find_all(exp.Table):
            if t.name not in write:
                read.add(t.name)
        return read, write, lock_intent, None

    if isinstance(ast, exp.Delete):
        tbl = ast.this
        if isinstance(tbl, exp.Table):
            write.add(tbl.name)
            read.add(tbl.name)
        for t in ast.find_all(exp.Table):
            if t.name not in write:
                read.add(t.name)
        return read, write, lock_intent, None

    if isinstance(ast, exp.Select):
        for t in ast.find_all(exp.Table):
            read.add(t.name)
        return read, write, lock_intent, None

    if isinstance(ast, (exp.Union, exp.Intersect, exp.Except)):
        for t in ast.find_all(exp.Table):
            read.add(t.name)
        return read, write, lock_intent, None

    if isinstance(ast, exp.Merge):
        target = ast.this
        if isinstance(target, exp.Table):
            write.add(target.name)
            read.add(target.name)
        # All non-target tables are read sources
        for t in ast.find_all(exp.Table):
            if t.name not in write:
                read.add(t.name)
        return read, write, lock_intent, None

    # DDL, GRANT, etc. — conservatively treat as write
    for t in ast.find_all(exp.Table):
        write.add(t.name)
    return read, write, lock_intent, None


def parse_sql_access(sql: str) -> tuple[set[str], set[str], str | None, str | None]:
    """Extract (read_tables, write_tables, lock_intent, tx_op) from a SQL statement.

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
    return set(), set(), None, None
