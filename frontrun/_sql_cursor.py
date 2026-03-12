"""DBAPI cursor monkey-patching for SQL-level conflict detection.

Intercepts ``cursor.execute()`` and ``cursor.executemany()`` calls to
extract table-level read/write sets from SQL statements.  Reports each
table as a separate resource to the I/O reporter, suppressing the
coarser endpoint-level socket I/O reports.

Follows the same monkey-patching pattern as ``_io_detection.py``.

Implementation note: C-extension cursor types (sqlite3.Cursor, psycopg2
cursor) are immutable and cannot be patched directly via ``setattr``.
Instead, we patch the ``connect()`` function of each driver module to
inject a traced connection/cursor factory subclass.  For pure-Python
drivers like pymysql, direct class patching is used as a fallback.
"""

from __future__ import annotations

import contextlib
import importlib
import re
import sqlite3
import threading
from collections.abc import Generator
from typing import Any

from frontrun import _real_threading as _rt
from frontrun._io_detection import _io_tls, get_io_reporter
from frontrun._schema import get_schema
from frontrun._sql_insert_tracker import record_insert, resolve_alias
from frontrun._sql_parsing import (
    LockIntent,
    Release,
    RollbackTo,
    Savepoint,
    TxOp,
    parse_sql_access,
)

# Try to import row-level predicate helpers.  These are always present in the
# same package, but guard with try/except for robustness.
try:
    from frontrun._sql_params import resolve_parameters
    from frontrun._sql_predicates import (
        EqualityPredicate,  # pyright: ignore[reportAssignmentType]
        extract_row_level_access,
    )
except ImportError:

    def resolve_parameters(sql: str, parameters: Any, paramstyle: str) -> str:  # type: ignore[misc]
        return sql

    def extract_row_level_access(sql: str, *, ast: Any | None = None) -> list[list[Any]] | None:  # type: ignore[misc]
        return None

    class EqualityPredicate:  # type: ignore[no-redef]
        def __init__(self, column: str, value: str):
            self.column = column
            self.value = value


# ---------------------------------------------------------------------------
# INSERT detection regex (used by _intercept_execute for post-INSERT capture).
# Reuses the same pattern as _sql_parsing._RE_INSERT to extract the table name
# in a single match, avoiding a redundant parse_sql_access call.
# ---------------------------------------------------------------------------

_RE_INSERT_TABLE = re.compile(r"^\s*INSERT\s+INTO\s+[`\"\[]?(\w+)", re.I)


# ---------------------------------------------------------------------------
# Suppression infrastructure
# ---------------------------------------------------------------------------

# OS thread IDs currently inside a patched execute call.
# The LD_PRELOAD bridge listener checks this to skip endpoint-level reports.
_suppress_tids: set[int] = set()
_suppress_lock = _rt.lock()  # Real lock (not cooperative)


@contextlib.contextmanager
def _suppress_endpoint_io() -> Generator[None, None, None]:
    """Temporarily suppress endpoint-level I/O for the current thread."""
    tid = threading.get_native_id()
    _io_tls._sql_suppress = True
    with _suppress_lock:
        _suppress_tids.add(tid)
    try:
        yield
    finally:
        with _suppress_lock:
            _suppress_tids.discard(tid)
        _io_tls._sql_suppress = False


def is_tid_suppressed(tid: int) -> bool:
    """Check if a thread ID is currently suppressed (for LD_PRELOAD bridge)."""
    with _suppress_lock:
        return tid in _suppress_tids


# ---------------------------------------------------------------------------
# Core interception logic
# ---------------------------------------------------------------------------


def _report_or_buffer(reporter: Any, res_id: str, kind: str, *, force_immediate: bool = False) -> None:
    """Report a SQL access immediately, or buffer it if inside a transaction.

    When ``force_immediate=True`` the access is reported right away even
    inside a transaction (used for SELECT FOR UPDATE to let the DPOR engine
    learn about write-intent conflicts before C-level blocking can occur).
    Transaction atomicity is preserved because the DPOR scheduler still
    skips yielding inside transactions.

    Autobegin transactions (``_is_autobegin=True``) are NOT buffered: with
    READ COMMITTED isolation (PostgreSQL default), individual statements are
    visible to other transactions, so DPOR must see each access point to
    explore interleavings.  Row-lock tracking still works because
    ``_in_transaction`` is True.
    """
    in_tx = getattr(_io_tls, "_in_transaction", False)
    is_autobegin = getattr(_io_tls, "_is_autobegin", False)
    if in_tx and not force_immediate and not is_autobegin:
        if not hasattr(_io_tls, "_tx_buffer"):
            _io_tls._tx_buffer = []
        _io_tls._tx_buffer.append((res_id, kind))
    else:
        reporter(res_id, kind)

    # Track resources that need row-lock arbitration (SELECT FOR UPDATE).
    # Only when inside a transaction — outside a tx the DB releases the lock
    # immediately so there's no blocking risk.
    if force_immediate and in_tx:
        pending = getattr(_io_tls, "_pending_row_locks", None)
        if pending is None:
            pending = []
            _io_tls._pending_row_locks = pending
        pending.append(res_id)


def _detect_autobegin(cursor: Any) -> None:
    """Set ``_in_transaction`` if the connection is in autobegin mode.

    DB-API drivers like psycopg2 default to ``autocommit=False``, which
    means the first statement implicitly starts a transaction at the
    C/driver level — no explicit ``BEGIN`` flows through
    ``cursor.execute()``.  We detect this by checking the cursor's
    connection: if ``autocommit`` is not ``True`` and we haven't already
    seen a ``BEGIN``, we treat the connection as having an implicit
    transaction.

    This is best-effort: if the connection doesn't expose ``autocommit``
    (e.g. sqlite3), we leave ``_in_transaction`` unchanged and fall back
    to statement-level tracking.
    """
    if getattr(_io_tls, "_in_transaction", False):
        return  # already in a transaction
    conn = getattr(cursor, "connection", None)
    if conn is None:
        return
    autocommit = getattr(conn, "autocommit", None)
    if autocommit is None:
        return  # driver doesn't expose autocommit — can't detect
    if not autocommit:
        # autocommit=False → autobegin: implicit transaction is active.
        # Set _is_autobegin so _report_or_buffer reports accesses
        # immediately (READ COMMITTED doesn't buffer) while still
        # tracking row locks via the _in_transaction flag.
        _io_tls._in_transaction = True
        _io_tls._is_autobegin = True
        _io_tls._tx_buffer = []
        _io_tls._tx_savepoints = {}


def _get_dpor_context() -> tuple[Any, int] | None:
    """Return (scheduler, thread_id) if DPOR is active, else ``None``."""
    from frontrun._io_detection import get_dpor_scheduler, get_dpor_thread_id

    scheduler = get_dpor_scheduler()
    if scheduler is None:
        return None
    thread_id = get_dpor_thread_id()
    if thread_id is None:
        return None
    return scheduler, thread_id


def _acquire_pending_row_locks() -> None:
    """Drain pending row-lock resources from TLS and acquire them on the scheduler."""
    lock_resources = getattr(_io_tls, "_pending_row_locks", None)
    if lock_resources:
        _io_tls._pending_row_locks = []
        ctx = _get_dpor_context()
        if ctx is not None:
            ctx[0].acquire_row_locks(ctx[1], lock_resources)


def _release_dpor_row_locks() -> None:
    """Release any DPOR row locks held by the current thread."""
    ctx = _get_dpor_context()
    if ctx is not None:
        ctx[0].release_row_locks(ctx[1])


# Global to track primary column set per table for cross-column conflict detection.
# Initialized to the first column set seen for each table in the current session.
_table_primary_colset: dict[str, tuple[str, ...]] = {}


def _get_primary_colset(table: str, colset: tuple[str, ...]) -> tuple[str, ...]:
    """Return the primary column set for a table, initializing it if necessary."""
    return _table_primary_colset.setdefault(table, colset)


def _sql_resource_id(table: str, predicates: list[Any], temporal: str | None = None) -> str:
    """Build a resource ID from table name and optional predicates."""
    if temporal:
        table = f"{table}:history:{temporal}"
    if not predicates:
        return f"sql:{table}"
    pred_key = tuple(sorted((p.column, p.value) for p in predicates))
    return f"sql:{table}:{pred_key}"


def _report_sql_access(
    operation: Any,
    parameters: Any = None,
    *,
    is_executemany: bool = False,
    paramstyle: str = "format",
) -> bool:
    """Parse SQL and report table accesses to the per-thread reporter.

    Returns ``True`` if any SQL-level reporting was performed (which means
    endpoint-level I/O should be suppressed for the subsequent DB call).

    This helper is shared by both sync ``_intercept_execute`` and async
    ``_intercept_execute_async``.
    """
    reporter = get_io_reporter()
    reported = False

    if reporter is not None and isinstance(operation, str):
        access = parse_sql_access(operation)

        # 1. Handle Transaction Control Operations
        if access.tx_op is not None:
            reported = True  # Suppress endpoint I/O for TX control too
            tx = access.tx_op
            if tx is TxOp.BEGIN:
                _io_tls._in_transaction = True
                _io_tls._is_autobegin = False
                _io_tls._tx_buffer = []
                _io_tls._tx_savepoints = {}
            elif tx is TxOp.COMMIT:
                _io_tls._in_transaction = False
                _io_tls._is_autobegin = False
                buffer = getattr(_io_tls, "_tx_buffer", [])
                for res_id, kind in buffer:
                    reporter(res_id, kind)
                _io_tls._tx_buffer = []
                _io_tls._tx_savepoints = {}
                _release_dpor_row_locks()
            elif tx is TxOp.ROLLBACK:
                _io_tls._in_transaction = False
                _io_tls._is_autobegin = False
                _io_tls._tx_buffer = []
                _io_tls._tx_savepoints = {}
                _release_dpor_row_locks()
            elif isinstance(tx, Savepoint):
                savepoints = getattr(_io_tls, "_tx_savepoints", {})
                buffer = getattr(_io_tls, "_tx_buffer", [])
                savepoints[tx.name] = len(buffer)
                _io_tls._tx_savepoints = savepoints
            elif isinstance(tx, RollbackTo):
                savepoints = getattr(_io_tls, "_tx_savepoints", {})
                if tx.name in savepoints:
                    idx = savepoints[tx.name]
                    buffer = getattr(_io_tls, "_tx_buffer", [])
                    _io_tls._tx_buffer = buffer[:idx]
            elif isinstance(tx, Release):  # pyright: ignore[reportUnnecessaryIsInstance]
                savepoints = getattr(_io_tls, "_tx_savepoints", {})
                savepoints.pop(tx.name, None)

        # 2. Handle Data Access Operations
        if access.read_tables or access.write_tables:
            reported = True
            all_tables = access.read_tables | access.write_tables

            # Row-level predicate extraction (WHERE equality, IN-lists, and INSERT VALUES)
            # Reuses the pre-parsed AST from parse_sql_access when available (avoids
            # a second sqlglot.parse_one call — Refactor 3).
            pred_rows: list[list[Any]] = [[]]  # default: one report, no predicates (table-level)
            has_row_level = False
            if len(all_tables) == 1 and not is_executemany:
                if parameters is not None:
                    resolved = resolve_parameters(operation, parameters, paramstyle)
                    rows = extract_row_level_access(resolved)
                else:
                    rows = extract_row_level_access(operation, ast=access.ast)
                if rows is not None:
                    pred_rows = rows
                    has_row_level = True

            lock_update = access.lock_intent is LockIntent.UPDATE

            # Track which tables already reported their bridge resource in this op.
            reported_bridges: set[str] = set()

            def report_or_buffer(table: str, kind: str, rows: list[list[Any]]) -> None:
                temporal = access.temporal_clauses.get(table) if access.temporal_clauses else None

                # Conservative Column-Set Partitioning (Defect #1):
                # When using row-level predicates, we must ensure that accesses using
                # DIFFERENT sets of columns (e.g. SELECT by username vs UPDATE by id)
                # properly conflict. We do this by reporting a "bridge" resource (sql:<table>)
                # for every row-level access.
                #
                # To preserve row-level benefits for the most common column set (usually the PK),
                # we designate the first column set seen for each table as "primary".
                # - Primary colset accesses report a READ on the bridge resource.
                # - Non-primary colset accesses report a WRITE on the bridge resource.
                # - Table-level accesses report their actual kind (READ/WRITE) on the bridge.
                #
                # Result:
                # - Primary vs Primary: both READ bridge -> NO conflict on bridge. Row-level works!
                # - Primary vs Non-primary: READ vs WRITE bridge -> CONFLICT. Correct.
                # - Non-primary vs Non-primary: WRITE vs WRITE bridge -> CONFLICT (table-level).
                if table not in reported_bridges:
                    if rows and rows[0]:
                        colset = tuple(sorted(p.column for p in rows[0]))
                        primary = _get_primary_colset(table, colset)

                        if colset == primary:
                            # Primary colset uses READ on the bridge to preserve
                            # row-level independence for the common access pattern.
                            bridge_kind = "read"
                        else:
                            # Non-primary colset conflicts conservatively at table scope.
                            bridge_kind = "write"

                        _report_or_buffer(reporter, f"sql:{table}", bridge_kind, force_immediate=lock_update)
                        reported_bridges.add(table)
                    elif not rows or not rows[0]:
                        # Table-level access
                        _report_or_buffer(reporter, f"sql:{table}", kind, force_immediate=lock_update)
                        reported_bridges.add(table)

                for row_preds in rows:
                    # Check if any predicate value matches a captured INSERT ID
                    alias = None
                    for pred in row_preds:
                        if isinstance(pred, EqualityPredicate):
                            alias = resolve_alias(table, pred.value)
                            if alias is not None:
                                break
                    res_id = alias if alias is not None else _sql_resource_id(table, row_preds, temporal)
                    _report_or_buffer(reporter, res_id, kind, force_immediate=lock_update)

            # Report explicit reads
            for table in access.read_tables:
                # SELECT FOR UPDATE is both read and write to create conflicts.
                # SHARE locks are treated as reads (they don't block other shares).
                kind = "write" if lock_update else "read"
                report_or_buffer(table, kind, pred_rows)

            # Report implicit reads from Foreign Key dependencies
            schema = get_schema()
            for table in access.write_tables:
                fks = schema.get_fks(table)
                for fk in fks:
                    # Determine predicates for the referenced table
                    ref_pred_rows: list[list[Any]] = [[]]  # default table-level

                    # If we have row-level predicates for the write table
                    if has_row_level:
                        mapped_rows = []
                        for row in pred_rows:
                            # Check if row has the FK column
                            fk_val = None
                            for pred in row:
                                if isinstance(pred, EqualityPredicate) and pred.column == fk.column:
                                    fk_val = pred.value
                                    break

                            if fk_val is not None:
                                mapped_rows.append([EqualityPredicate(fk.ref_column, fk_val)])
                            else:
                                # If any row is missing the FK value, we must fall back to table-level
                                mapped_rows = [[]]
                                break
                        ref_pred_rows = mapped_rows

                    report_or_buffer(fk.ref_table, "read", ref_pred_rows)

            # Report writes
            for table in access.write_tables:
                report_or_buffer(table, "write", pred_rows)

    return reported


def _capture_insert_id(cursor: Any, table: str) -> None:
    """Capture lastrowid after INSERT and report indexical alias + sequence resource."""
    reporter = get_io_reporter()
    if reporter is None:
        return

    lastrowid = getattr(cursor, "lastrowid", None)

    alias = record_insert(table, lastrowid)

    # Report the logical alias and shared sequence resource as writes
    _report_or_buffer(reporter, alias, "write")
    _report_or_buffer(reporter, f"sql:{table}:seq", "write")


def _intercept_execute(
    original_method: Any,
    self: Any,
    operation: Any,
    parameters: Any = None,
    *,
    is_executemany: bool = False,
    paramstyle: str = "format",
) -> Any:
    """Intercept a single execute/executemany call.

    Parses *operation*, reports table accesses to the per-thread reporter,
    activates suppression, then delegates to *original_method*.

    When *is_executemany* is False and the query touches exactly one table,
    resolves *parameters* and extracts row-level predicates (equality and
    IN-lists) so that
    the reported resource ID is finer-grained than plain ``sql:<table>``.

    Implements transaction grouping: when a transaction is active (BEGIN
    detected), I/O reports are buffered in TLS and only flushed when COMMIT is
    called.  ROLLBACK clears the buffer.  SAVEPOINTs and ROLLBACK TO SAVEPOINT
    are supported via buffer truncation.
    """
    insert_match = _RE_INSERT_TABLE.match(operation) if isinstance(operation, str) else None

    # Detect autobegin: most DB-API drivers (psycopg2, pymysql) default to
    # autocommit=False, meaning the first statement implicitly starts a
    # transaction at the C/driver level without sending an explicit BEGIN
    # through cursor.execute().  We detect this and set _in_transaction so
    # that (a) accesses are buffered atomically, (b) the DPOR scheduler
    # treats the transaction as an atomic block, and (c) row locks are
    # tracked for deadlock detection.
    #
    # We skip this when connection.autocommit is True (each statement is
    # its own transaction, locks released immediately) or when we've
    # already seen an explicit BEGIN.
    _detect_autobegin(self)

    reported = _report_sql_access(operation, parameters, is_executemany=is_executemany, paramstyle=paramstyle)

    # Block if another DPOR thread holds a conflicting row lock
    _acquire_pending_row_locks()

    # Force a DPOR scheduling point so the engine can interleave between
    # SQL operations.  Without this, all code inside frontrun/ is skipped
    # by the tracer, so pending_io is never flushed between back-to-back
    # SQL calls.
    if reported:
        _dpor_ctx = _get_dpor_context()
        if _dpor_ctx is not None:
            _dpor_ctx[0].report_and_wait(None, _dpor_ctx[1])

    if reported:
        with _suppress_endpoint_io():
            if parameters is not None:
                result = original_method(self, operation, parameters)
            else:
                result = original_method(self, operation)
    else:
        if parameters is not None:
            result = original_method(self, operation, parameters)
        else:
            result = original_method(self, operation)

    # Post-INSERT: capture lastrowid and record indexical alias
    if insert_match is not None and not is_executemany and reported:
        _capture_insert_id(self, insert_match.group(1))

    return result


# ---------------------------------------------------------------------------
# Traced cursor/connection subclasses (created dynamically per driver)
# ---------------------------------------------------------------------------


def _make_traced_cursor_class(base_cursor_cls: type, paramstyle: str = "format") -> type:
    """Return a subclass of *base_cursor_cls* that intercepts execute calls.

    The original methods are looked up from ``_ORIGINAL_METHODS`` at call time
    rather than captured at class creation.  This allows tests to swap out the
    stored original (e.g. to install a spy) and have the traced cursor pick up
    the new value transparently.

    *paramstyle* is the PEP 249 paramstyle for the driver (e.g. ``"qmark"``
    for sqlite3, ``"pyformat"`` for psycopg2, ``"format"`` for pymysql).
    It is stored as a class attribute so that ``_intercept_execute`` can
    resolve parameters before extracting row-level predicates.
    """

    _execute_key = (base_cursor_cls, "execute")
    _executemany_key = (base_cursor_cls, "executemany")
    _paramstyle = paramstyle

    class TracedCursor(base_cursor_cls):  # type: ignore[valid-type]
        _cursor_paramstyle: str = _paramstyle

        def execute(self, operation: Any, parameters: Any = None, /, **kwargs: Any) -> Any:  # type: ignore[override]
            original = _ORIGINAL_METHODS.get(_execute_key, base_cursor_cls.execute)
            return _intercept_execute(
                original, self, operation, parameters, is_executemany=False, paramstyle=self._cursor_paramstyle
            )

        def executemany(self, operation: Any, parameters: Any = None, /, **kwargs: Any) -> Any:  # type: ignore[override]
            original = _ORIGINAL_METHODS.get(_executemany_key, base_cursor_cls.executemany)
            return _intercept_execute(
                original, self, operation, parameters, is_executemany=True, paramstyle=self._cursor_paramstyle
            )

    TracedCursor.__name__ = f"Traced{base_cursor_cls.__name__}"
    TracedCursor.__qualname__ = f"Traced{base_cursor_cls.__qualname__}"
    return TracedCursor


def _make_traced_sqlite3_connection_class() -> type:
    """Return a sqlite3.Connection subclass whose cursor() uses TracedCursor.

    On Python 3.14+, ``Connection.execute()`` creates cursors in C without
    calling ``self.cursor()``, so we must also override ``execute`` and
    ``executemany`` to route through the traced cursor.
    """
    _traced_cursor_cls = _make_traced_cursor_class(sqlite3.Cursor, paramstyle="qmark")

    class TracedConnection(sqlite3.Connection):
        def cursor(self, factory: type = _traced_cursor_cls) -> sqlite3.Cursor:  # type: ignore[override]
            return super().cursor(factory)

        def execute(self, sql: Any, parameters: Any = (), /) -> sqlite3.Cursor:  # type: ignore[override]
            cur = self.cursor()
            cur.execute(sql, parameters)
            return cur

        def executemany(self, sql: Any, parameters: Any = (), /) -> sqlite3.Cursor:  # type: ignore[override]
            cur = self.cursor()
            cur.executemany(sql, parameters)
            return cur

    TracedConnection.__name__ = "TracedConnection"
    TracedConnection.__qualname__ = "TracedConnection"
    return TracedConnection


# ---------------------------------------------------------------------------
# Global patching state
# ---------------------------------------------------------------------------

_sql_patched = False

# Stores (module, attribute_name, original_value) for each patched site
_PATCHES: list[tuple[Any, str, Any]] = []

# Expose a dict-like view keyed by (class, method_name) for test introspection.
# For the factory-based approach we store the original connect function here.
_ORIGINAL_METHODS: dict[tuple[type, str], Any] = {}


# ---------------------------------------------------------------------------
# sqlite3 patching
# ---------------------------------------------------------------------------


def _patch_sqlite3() -> None:
    """Patch sqlite3.connect to inject TracedConnection factory."""
    orig_connect = sqlite3.connect
    traced_conn_cls = _make_traced_sqlite3_connection_class()

    def patched_connect(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        if "factory" not in kwargs:
            kwargs["factory"] = traced_conn_cls
        return orig_connect(*args, **kwargs)

    sqlite3.connect = patched_connect  # type: ignore[assignment]
    _PATCHES.append((sqlite3, "connect", orig_connect))
    # Expose for tests via _ORIGINAL_METHODS — key by (cursor_class, method_name)
    _ORIGINAL_METHODS[(sqlite3.Cursor, "execute")] = sqlite3.Cursor.execute
    _ORIGINAL_METHODS[(sqlite3.Cursor, "executemany")] = sqlite3.Cursor.executemany


# ---------------------------------------------------------------------------
# Generic Python-class patching (for pure-Python drivers)
# ---------------------------------------------------------------------------


def _patch_class_methods(cls: type, paramstyle: str) -> None:
    """Directly patch execute/executemany on a Python cursor class."""
    for method_name in ("execute", "executemany"):
        original = getattr(cls, method_name, None)
        if original is None:
            continue
        key = (cls, method_name)
        if key in _ORIGINAL_METHODS:
            continue
        _ORIGINAL_METHODS[key] = original

        def _make_patched(orig: Any, mname: str, ps: str) -> Any:
            _is_executemany = mname == "executemany"

            def _patched(self: Any, operation: Any, parameters: Any = None, *args: Any, **kwargs: Any) -> Any:
                return _intercept_execute(
                    orig, self, operation, parameters, is_executemany=_is_executemany, paramstyle=ps
                )

            _patched.__name__ = mname
            return _patched

        setattr(cls, method_name, _make_patched(original, method_name, paramstyle))
        _PATCHES.append((cls, method_name, original))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Drivers to attempt patching via direct class method replacement (pure Python)
_PYTHON_CURSOR_TARGETS: list[tuple[str, str, str]] = [
    ("pymysql.cursors", "Cursor", "pymysql"),
]


def patch_sql() -> None:
    """Monkey-patch DBAPI cursor.execute() for known drivers."""
    global _sql_patched  # noqa: PLW0603
    if _sql_patched:
        return

    # sqlite3 requires factory-injection approach
    _patch_sqlite3()

    # Pure-Python drivers can be patched directly
    for module_path, class_name, paramstyle_module in _PYTHON_CURSOR_TARGETS:
        try:
            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name)
            driver_mod = importlib.import_module(paramstyle_module)
            paramstyle = getattr(driver_mod, "paramstyle", "format")
            _patch_class_methods(cls, paramstyle)
        except (ImportError, AttributeError):
            pass  # driver not installed — skip silently

    def _make_patched_connect(orig: Any, default_cursor_cls: type, paramstyle: str) -> Any:
        # Cache traced subclasses by (caller-provided cursor_factory class) to avoid
        # creating a new class on every connect() call.
        _cache: dict[type, type] = {}

        def patched_connect(*args: Any, **kwargs: Any) -> Any:
            # Wrap whatever cursor_factory the caller already set (e.g. Django's Cursor),
            # rather than using setdefault, which is a no-op when the caller set it first.
            user_factory = kwargs.get("cursor_factory", default_cursor_cls)
            if user_factory not in _cache:
                _cache[user_factory] = _make_traced_cursor_class(user_factory, paramstyle=paramstyle)
            kwargs["cursor_factory"] = _cache[user_factory]
            return orig(*args, **kwargs)

        return patched_connect

    # psycopg2: patch via cursor_factory injection into connect()
    try:
        import psycopg2 as _pg2mod  # type: ignore[import-untyped]
        import psycopg2.extensions as _pg2ext  # type: ignore[import-untyped]

        orig_cursor_cls = _pg2ext.cursor
        orig_connect = _pg2mod.connect
        setattr(_pg2mod, "connect", _make_patched_connect(orig_connect, orig_cursor_cls, paramstyle="pyformat"))
        _PATCHES.append((_pg2mod, "connect", orig_connect))
        _ORIGINAL_METHODS[(orig_cursor_cls, "execute")] = orig_cursor_cls.execute
        _ORIGINAL_METHODS[(orig_cursor_cls, "executemany")] = orig_cursor_cls.executemany
    except (ImportError, AttributeError):
        pass

    # psycopg (v3): patch via cursor_factory injection into connect()
    try:
        import psycopg as _pg3mod  # type: ignore[import-untyped]

        orig_cursor_cls = _pg3mod.Cursor
        orig_connect = _pg3mod.connect
        # Psycopg 3 uses 'format' as default paramstyle (client-side)
        setattr(_pg3mod, "connect", _make_patched_connect(orig_connect, orig_cursor_cls, paramstyle="format"))
        _PATCHES.append((_pg3mod, "connect", orig_connect))
        _ORIGINAL_METHODS[(orig_cursor_cls, "execute")] = orig_cursor_cls.execute
        _ORIGINAL_METHODS[(orig_cursor_cls, "executemany")] = orig_cursor_cls.executemany
    except (ImportError, AttributeError):
        pass

    _sql_patched = True


def reset_connection_state() -> None:
    """Clear per-thread SQL transaction state.

    Call this when a connection is returned to a connection pool to prevent
    stale ``_in_transaction`` / ``_tx_buffer`` / ``_tx_savepoints`` state
    from leaking across logical sessions.  Safe to call even when no
    transaction is active (it's a no-op in that case).
    """
    for attr in ("_in_transaction", "_tx_buffer", "_tx_savepoints", "_pending_row_locks"):
        if hasattr(_io_tls, attr):
            delattr(_io_tls, attr)


def clear_sql_metadata() -> None:
    """Reset all global SQL resource tracking metadata.

    Call this between DPOR exploration sessions to ensure test isolation.
    """
    _table_primary_colset.clear()


def unpatch_sql() -> None:
    """Restore original DBAPI cursor methods and connect functions."""
    global _sql_patched  # noqa: PLW0603
    if not _sql_patched:
        return
    for obj, attr, original in _PATCHES:
        setattr(obj, attr, original)
    _PATCHES.clear()
    _ORIGINAL_METHODS.clear()
    _sql_patched = False
