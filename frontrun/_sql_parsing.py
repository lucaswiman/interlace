"""SQL statement parsing for read/write table extraction.

Provides ``parse_sql_access(sql)`` which returns ``(read_tables, write_tables)``
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


def _strip_quotes(name: str) -> str:
    """Remove surrounding quotes/backticks and extract table from schema.table."""
    if name.startswith(('"', "`")):
        name = name[1:-1]
    # Take last component: "public"."users" → users
    return name.rsplit(".", 1)[-1]


def _regex_parse(sql: str) -> tuple[set[str], set[str]] | None:
    """Fast-path: extract tables from simple single-statement SQL.

    Returns (read_tables, write_tables) or None if the SQL is too complex
    (subqueries, CTEs, UNION, MERGE) and needs the full parser.
    """
    stripped = sql.strip().rstrip(";").strip()

    # Bail to full parser for complex SQL
    upper = stripped.upper()
    if any(kw in upper for kw in ("WITH ", "UNION", "INTERSECT", "EXCEPT", "MERGE", "RETURNING")):
        return None

    read: set[str] = set()
    write: set[str] = set()

    m_insert = _RE_INSERT.search(stripped)
    if m_insert:
        write.add(_strip_quotes(m_insert.group(1)))
        # Source tables in INSERT ... SELECT ... FROM
        for m in _RE_FROM.finditer(stripped, m_insert.end()):
            read.add(_strip_quotes(m.group(1)))
        return read, write

    m_update = _RE_UPDATE.search(stripped)
    if m_update:
        tbl = _strip_quotes(m_update.group(1))
        write.add(tbl)
        read.add(tbl)  # WHERE reads the target
        # Subquery tables in FROM/JOIN (UPDATE ... FROM ... syntax)
        for m in _RE_FROM.finditer(stripped, m_update.end()):
            t = _strip_quotes(m.group(1))
            if t not in write:
                read.add(t)
        return read, write

    m_delete = _RE_DELETE.search(stripped)
    if m_delete:
        tbl = _strip_quotes(m_delete.group(1))
        write.add(tbl)
        read.add(tbl)
        return read, write

    if _RE_SELECT.search(stripped):
        for m in _RE_FROM.finditer(stripped):
            read.add(_strip_quotes(m.group(1)))
        for m in _RE_JOIN.finditer(stripped):
            read.add(_strip_quotes(m.group(1)))
        return read, write

    return None  # unknown statement type → fall through


def _sqlglot_parse(sql: str) -> tuple[set[str], set[str]] | None:
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

    if isinstance(ast, exp.Insert):
        tbl = ast.find(exp.Table)
        if tbl:
            write.add(tbl.name)
        # Source tables (everything after the target)
        if ast.expression:  # the SELECT source
            for t in ast.expression.find_all(exp.Table):
                if t.name not in write:
                    read.add(t.name)
        return read, write

    if isinstance(ast, exp.Update):
        tbl = ast.this
        if isinstance(tbl, exp.Table):
            write.add(tbl.name)
            read.add(tbl.name)
        for t in ast.find_all(exp.Table):
            if t.name not in write:
                read.add(t.name)
        return read, write

    if isinstance(ast, exp.Delete):
        tbl = ast.this
        if isinstance(tbl, exp.Table):
            write.add(tbl.name)
            read.add(tbl.name)
        for t in ast.find_all(exp.Table):
            if t.name not in write:
                read.add(t.name)
        return read, write

    if isinstance(ast, exp.Select):
        for t in ast.find_all(exp.Table):
            read.add(t.name)
        return read, write

    if isinstance(ast, (exp.Union, exp.Intersect, exp.Except)):
        for t in ast.find_all(exp.Table):
            read.add(t.name)
        return read, write

    if isinstance(ast, exp.Merge):
        target = ast.this
        if isinstance(target, exp.Table):
            write.add(target.name)
            read.add(target.name)
        # All non-target tables are read sources
        for t in ast.find_all(exp.Table):
            if t.name not in write:
                read.add(t.name)
        return read, write

    # DDL, GRANT, etc. — conservatively treat as write
    for t in ast.find_all(exp.Table):
        write.add(t.name)
    return read, write


def parse_sql_access(sql: str) -> tuple[set[str], set[str]]:
    """Extract (read_tables, write_tables) from a SQL statement.

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
    return set(), set()
