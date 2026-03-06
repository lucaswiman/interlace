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

    no_literals = _RE_LITERAL.sub(" ", stripped)
    if ";" in no_literals:
        return None
    upper_no_literals = no_literals.upper()

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
    m = _RE_TX_SAVEPOINT.search(stripped)
    if m:
        return set(), set(), None, f"SAVEPOINT:{m.group(1)}"
    m = _RE_TX_RELEASE.search(stripped)
    if m:
        return set(), set(), None, f"RELEASE:{m.group(2)}"

    # Bail to full parser for complex SQL
    if any(kw in upper_no_literals for kw in (
        "WITH ", "UNION", "INTERSECT", "EXCEPT", "MERGE", "RETURNING",
        "EXISTS", "IN ("
    )):
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
    if not any(stripped.lower().startswith(kw) for kw in ("select", "insert")) and "SELECT" in upper_no_literals:
        return None

    # Function calls (advisory locks etc) — bail to sqlglot
    if "(" in stripped and not stripped.lower().startswith("insert"):
        if any(kw in stripped.lower() for kw in ("pg_advisory", "get_lock")):
            return None

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
        expressions = sqlglot.parse(sql)
    except sqlglot.errors.ParseError:
        return None  # unparseable → fall back to endpoint-level

    if not expressions:
        return None

    all_write: set[str] = set()
    all_read: set[str] = set()
    all_lock_intent: str | None = None
    all_tx_op: str | None = None

    def _merge_lock_intent(a: str | None, b: str | None) -> str | None:
        if a == "UPDATE" or b == "UPDATE":
            return "UPDATE"
        if a == "SHARE" or b == "SHARE":
            return "SHARE"
        return None

    for ast in expressions:
        if ast is None:
            continue
            
        write: set[str] = set()
        read: set[str] = set()
        lock_intent: str | None = None
        tx_op: str | None = None

        # Transaction control
        if isinstance(ast, exp.Transaction):
            tx_op = "BEGIN"
        elif isinstance(ast, exp.Commit):
            tx_op = "COMMIT"
        elif isinstance(ast, exp.Rollback):
            tx_op = "ROLLBACK"
        else:
            # Extract lock intent from SELECT
            if isinstance(ast, exp.Select):
                lock = ast.find(exp.Lock)
                if lock:
                    if lock.args.get("update"):
                        lock_intent = "UPDATE"
                    else:
                        # sqlglot might not have "share" key, but if it's a Lock and not update,
                        # it's usually a share lock in dialects that support it.
                        lock_intent = "SHARE"
                        # Extra check for some versions/dialects
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
                        lock_ids: list[str] = []
                        for arg in call.expressions:
                            if isinstance(arg, exp.Literal):
                                lock_ids.append(str(arg.this))
                            else:
                                lock_ids.append("?")
                        lock_id_str = ":".join(lock_ids)
                        write.add(f"advisory_lock:{lock_id_str}")
                        if "shared" in name:
                            lock_intent = _merge_lock_intent(lock_intent, "SHARE")
                        else:
                            lock_intent = _merge_lock_intent(lock_intent, "UPDATE")

            if isinstance(ast, exp.Insert):
                tbl = ast.find(exp.Table)
                if tbl:
                    write.add(tbl.name)
                # Source tables (everything after the target)
                if ast.expression:  # the SELECT source
                    for t in ast.expression.find_all(exp.Table):
                        if t.name not in write:
                            read.add(t.name)
            elif isinstance(ast, exp.Update):
                tbl = ast.this
                if isinstance(tbl, exp.Table):
                    write.add(tbl.name)
                    read.add(tbl.name)
                for t in ast.find_all(exp.Table):
                    if t.name not in write:
                        read.add(t.name)
            elif isinstance(ast, exp.Delete):
                tbl = ast.this
                if isinstance(tbl, exp.Table):
                    write.add(tbl.name)
                    read.add(tbl.name)
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
            all_tx_op = tx_op  # Take the last tx_op or any? usually only one makes sense

    return all_read, all_write, all_lock_intent, all_tx_op


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
