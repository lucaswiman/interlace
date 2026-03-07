"""Async DBAPI cursor monkey-patching for SQL-level conflict detection.

Async counterpart to ``_sql_cursor.py``.  Intercepts ``cursor.execute()``
and ``cursor.executemany()`` on async database drivers to extract
table-level read/write sets from SQL statements.

Supported drivers:

* **aiosqlite** — async wrapper around sqlite3
* **psycopg.AsyncCursor** — psycopg3 async mode
* **aiomysql** — async MySQL driver
* **asyncpg** — PostgreSQL async driver (connection-level methods)

The SQL parsing, resource reporting, and transaction grouping logic is
shared with the sync module via ``_report_sql_access``.
"""

from __future__ import annotations

import importlib
from typing import Any

from frontrun._sql_cursor import _RE_IS_INSERT, _capture_insert_id, _report_sql_access, _suppress_endpoint_io

# ---------------------------------------------------------------------------
# Async interception
# ---------------------------------------------------------------------------


async def _intercept_execute_async(
    original_method: Any,
    self: Any,
    operation: Any,
    parameters: Any = None,
    *,
    is_executemany: bool = False,
    paramstyle: str = "format",
) -> Any:
    """Async version of ``_intercept_execute``.

    Parses *operation*, reports table accesses via the shared
    ``_report_sql_access`` helper, then ``await``s the original async method.
    """
    is_insert = isinstance(operation, str) and _RE_IS_INSERT.match(operation) is not None
    reported = _report_sql_access(operation, parameters, is_executemany=is_executemany, paramstyle=paramstyle)

    if reported:
        with _suppress_endpoint_io():
            if parameters is not None:
                result = await original_method(self, operation, parameters)
            else:
                result = await original_method(self, operation)
    else:
        if parameters is not None:
            result = await original_method(self, operation, parameters)
        else:
            result = await original_method(self, operation)

    # Post-INSERT: capture lastrowid and record indexical alias
    if is_insert and not is_executemany and reported:
        _capture_insert_id(self, operation)

    return result


async def _intercept_asyncpg_execute(
    original_method: Any,
    self: Any,
    operation: Any,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Intercept asyncpg connection methods (execute, fetch, fetchrow, fetchval).

    asyncpg uses ``$1``-style positional parameters passed as ``*args``, not
    a single parameters collection.  We report at table level (no parameter
    resolution for asyncpg's binary protocol parameters).
    """
    reported = _report_sql_access(operation, None, is_executemany=False, paramstyle="dollar")

    if reported:
        with _suppress_endpoint_io():
            return await original_method(self, operation, *args, **kwargs)
    return await original_method(self, operation, *args, **kwargs)


# ---------------------------------------------------------------------------
# Global patching state
# ---------------------------------------------------------------------------

_sql_async_patched = False
_ASYNC_PATCHES: list[tuple[Any, str, Any]] = []
_ASYNC_ORIGINAL_METHODS: dict[tuple[type, str], Any] = {}


# ---------------------------------------------------------------------------
# aiosqlite patching
# ---------------------------------------------------------------------------


def _patch_aiosqlite() -> None:
    """Patch aiosqlite.Cursor and aiosqlite.Connection execute/executemany."""
    try:
        import aiosqlite  # type: ignore[import-untyped]
    except ImportError:
        return

    # Patch both Cursor and Connection — users commonly use conn.execute()
    for target_cls in (aiosqlite.Cursor, aiosqlite.Connection):
        for method_name in ("execute", "executemany"):
            original = getattr(target_cls, method_name, None)
            if original is None:
                continue
            key = (target_cls, method_name)
            if key in _ASYNC_ORIGINAL_METHODS:
                continue
            _ASYNC_ORIGINAL_METHODS[key] = original

            _is_executemany = method_name == "executemany"

            def _make_patched(orig: Any, is_em: bool) -> Any:
                async def _patched(self: Any, sql: Any, parameters: Any = None, /) -> Any:
                    return await _intercept_execute_async(
                        orig, self, sql, parameters, is_executemany=is_em, paramstyle="qmark"
                    )

                _patched.__name__ = orig.__name__
                _patched.__qualname__ = orig.__qualname__
                return _patched

            setattr(target_cls, method_name, _make_patched(original, _is_executemany))
            _ASYNC_PATCHES.append((target_cls, method_name, original))


# ---------------------------------------------------------------------------
# psycopg AsyncCursor patching
# ---------------------------------------------------------------------------


def _patch_psycopg_async() -> None:
    """Patch psycopg.AsyncCursor.execute and executemany."""
    try:
        import psycopg  # type: ignore[import-untyped]
    except ImportError:
        return

    cursor_cls = getattr(psycopg, "AsyncCursor", None)
    if cursor_cls is None:
        return

    for method_name in ("execute", "executemany"):
        original = getattr(cursor_cls, method_name, None)
        if original is None:
            continue
        key = (cursor_cls, method_name)
        if key in _ASYNC_ORIGINAL_METHODS:
            continue
        _ASYNC_ORIGINAL_METHODS[key] = original

        _is_executemany = method_name == "executemany"

        def _make_patched(orig: Any, is_em: bool) -> Any:
            async def _patched(self: Any, query: Any, params: Any = None, /, **kwargs: Any) -> Any:
                reported = _report_sql_access(query, params, is_executemany=is_em, paramstyle="format")
                if reported:
                    with _suppress_endpoint_io():
                        return await orig(self, query, params, **kwargs)
                return await orig(self, query, params, **kwargs)

            _patched.__name__ = orig.__name__
            _patched.__qualname__ = orig.__qualname__
            return _patched

        setattr(cursor_cls, method_name, _make_patched(original, _is_executemany))
        _ASYNC_PATCHES.append((cursor_cls, method_name, original))


# ---------------------------------------------------------------------------
# aiomysql patching
# ---------------------------------------------------------------------------


def _patch_aiomysql() -> None:
    """Patch aiomysql.Cursor.execute and executemany."""
    try:
        mod = importlib.import_module("aiomysql.cursors")
    except ImportError:
        return

    cursor_cls = getattr(mod, "Cursor", None)
    if cursor_cls is None:
        return

    for method_name in ("execute", "executemany"):
        original = getattr(cursor_cls, method_name, None)
        if original is None:
            continue
        key = (cursor_cls, method_name)
        if key in _ASYNC_ORIGINAL_METHODS:
            continue
        _ASYNC_ORIGINAL_METHODS[key] = original

        _is_executemany = method_name == "executemany"

        def _make_patched(orig: Any, is_em: bool) -> Any:
            async def _patched(self: Any, query: Any, args: Any = None, *extra: Any, **kwargs: Any) -> Any:
                reported = _report_sql_access(query, args, is_executemany=is_em, paramstyle="pyformat")
                if reported:
                    with _suppress_endpoint_io():
                        return await orig(self, query, args, *extra, **kwargs)
                return await orig(self, query, args, *extra, **kwargs)

            _patched.__name__ = orig.__name__
            _patched.__qualname__ = orig.__qualname__
            return _patched

        setattr(cursor_cls, method_name, _make_patched(original, _is_executemany))
        _ASYNC_PATCHES.append((cursor_cls, method_name, original))


# ---------------------------------------------------------------------------
# asyncpg patching
# ---------------------------------------------------------------------------


def _patch_asyncpg() -> None:
    """Patch asyncpg.Connection query methods (execute, fetch, fetchrow, fetchval, executemany)."""
    try:
        import asyncpg  # type: ignore[import-untyped]
    except ImportError:
        return

    conn_cls = asyncpg.Connection

    # asyncpg methods all take (query, *args) — no separate params arg.
    # execute returns command tag, fetch/fetchrow/fetchval return results.
    for method_name in ("execute", "fetch", "fetchrow", "fetchval"):
        original = getattr(conn_cls, method_name, None)
        if original is None:
            continue
        key = (conn_cls, method_name)
        if key in _ASYNC_ORIGINAL_METHODS:
            continue
        _ASYNC_ORIGINAL_METHODS[key] = original

        def _make_patched(orig: Any) -> Any:
            async def _patched(self: Any, query: Any, *args: Any, **kwargs: Any) -> Any:
                return await _intercept_asyncpg_execute(orig, self, query, *args, **kwargs)

            _patched.__name__ = orig.__name__
            _patched.__qualname__ = orig.__qualname__
            return _patched

        setattr(conn_cls, method_name, _make_patched(original))
        _ASYNC_PATCHES.append((conn_cls, method_name, original))

    # executemany on asyncpg takes (command, args) where args is a list of tuples
    orig_em = getattr(conn_cls, "executemany", None)
    if orig_em is not None:
        key = (conn_cls, "executemany")
        if key not in _ASYNC_ORIGINAL_METHODS:
            _ASYNC_ORIGINAL_METHODS[key] = orig_em

            async def _patched_executemany(self: Any, command: Any, args: Any, **kwargs: Any) -> Any:
                reported = _report_sql_access(command, None, is_executemany=True, paramstyle="dollar")
                if reported:
                    with _suppress_endpoint_io():
                        return await orig_em(self, command, args, **kwargs)
                return await orig_em(self, command, args, **kwargs)

            _patched_executemany.__name__ = "executemany"
            _patched_executemany.__qualname__ = orig_em.__qualname__
            setattr(conn_cls, "executemany", _patched_executemany)
            _ASYNC_PATCHES.append((conn_cls, "executemany", orig_em))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def patch_sql_async() -> None:
    """Monkey-patch async DBAPI cursor.execute() for known drivers."""
    global _sql_async_patched  # noqa: PLW0603
    if _sql_async_patched:
        return

    _patch_aiosqlite()
    _patch_psycopg_async()
    _patch_aiomysql()
    _patch_asyncpg()

    _sql_async_patched = True


def unpatch_sql_async() -> None:
    """Restore original async DBAPI cursor methods."""
    global _sql_async_patched  # noqa: PLW0603
    if not _sql_async_patched:
        return
    for obj, attr, original in _ASYNC_PATCHES:
        setattr(obj, attr, original)
    _ASYNC_PATCHES.clear()
    _ASYNC_ORIGINAL_METHODS.clear()
    _sql_async_patched = False
