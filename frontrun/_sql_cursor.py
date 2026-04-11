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
import hashlib
import importlib
import os
import re
import sqlite3
import sys
import threading
import time
from collections.abc import Generator
from typing import Any

from frontrun import _real_threading as _rt
from frontrun._io_detection import _io_tls, get_io_reporter
from frontrun._io_detection import get_dpor_context as _get_dpor_context
from frontrun._patching import patch_method, restore_patches, wrap_method_metadata
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
from frontrun._sql_patch_registry import CONNECT_FACTORY_TARGETS, PYTHON_CURSOR_TARGETS
from frontrun._trace_format import build_call_chain
from frontrun._tracing import should_trace_file as _should_trace_file

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

_RE_INSERT_TABLE = re.compile(
    r"^\s*INSERT\s+(?:OR\s+\w+\s+|IGNORE\s+)?INTO\s+(?:[`\"\[]?\w+[`\"\]]?\s*\.\s*)?[`\"\[]?(\w+)", re.I
)
_RE_UPDATE_TABLE = re.compile(r"^\s*UPDATE\s+[`\"\[]?(\w+)", re.I)


# ---------------------------------------------------------------------------
# Suppression infrastructure
# ---------------------------------------------------------------------------

# OS thread IDs currently inside a patched execute call.
# The LD_PRELOAD bridge listener checks this to skip endpoint-level reports.
_suppress_tids: set[int] = set()
_suppress_lock = _rt.lock()  # Real lock (not cooperative)
_DB_SCOPE_ATTR = "_frontrun_db_scope"
_CONNECTION_DB_SCOPES: dict[int, str] = {}
_ACTIVE_SQL_IO_CONTEXTS: dict[int, tuple[str | None, list[str] | None]] = {}


def _warm_sql_parsers() -> None:
    """Load optional SQL parsing dependencies before managed threads start.

    The first row-level SQL statement may lazily import ``sqlglot`` and a large
    set of helper modules. If that happens inside a DPOR-managed worker thread,
    the preload bridge observes a burst of unrelated file I/O and exploration
    can spend its small preemption budget on import noise instead of the user
    race. Warm the parser stack once on the main thread when patching SQL.

    Importantly, this also forces ``sqlglot.dialects.Dialect`` to be imported
    eagerly.  The ``sqlglot.dialects`` module uses a module-level
    ``_import_lock = threading.RLock()`` to guard lazy dialect loading.  If
    this lock is created after cooperative lock patching replaces
    ``threading.RLock``, it becomes a :class:`CooperativeRLock` which can
    deadlock when acquired outside a DPOR scheduler context (e.g. during
    counterexample reproduction).  Warming here — before ``patch_locks()`` —
    ensures the lock is a real ``RLock`` and all lazy imports are resolved.
    """
    try:
        extract_row_level_access("SELECT * FROM frontrun_warmup WHERE id = 1")
    except Exception:
        # Optional dependency missing or parser warmup failed. The actual SQL
        # interception path remains best-effort and will fall back naturally.
        pass
    # Force the lazy Dialect import so that sqlglot's _import_lock (which
    # guards __getattr__ for dialect loading) is exercised while threading
    # primitives are still real (not cooperative).
    try:
        from sqlglot.dialects import Dialect  # noqa: F401
    except Exception:
        pass


def _summarize_sql_for_trace(operation: Any, parameters: Any, paramstyle: str) -> str | None:
    """Return a short SQL summary suitable for trace output."""
    if not isinstance(operation, str):
        return None
    sql = operation
    try:
        if parameters is not None:
            sql = resolve_parameters(operation, parameters, paramstyle)
    except Exception:
        sql = operation
    sql = " ".join(sql.split())
    if len(sql) > 160:
        sql = f"{sql[:157]}..."
    return f"SQL: {sql}"


def _current_user_call_chain() -> list[str] | None:
    """Return a best-effort call chain rooted at the first traced user frame."""
    frame = sys._getframe(1)
    while frame is not None and not _should_trace_file(frame.f_code.co_filename):
        frame = frame.f_back
    if frame is None:
        return None
    return build_call_chain(frame, filter_fn=_should_trace_file)


def _set_active_sql_io_context(operation: Any, parameters: Any, paramstyle: str) -> None:
    """Remember the current SQL statement for C-level socket I/O trace rendering."""
    tid = threading.get_native_id()
    summary = _summarize_sql_for_trace(operation, parameters, paramstyle)
    chain = _current_user_call_chain()
    with _suppress_lock:
        _ACTIVE_SQL_IO_CONTEXTS[tid] = (summary, chain)


def get_active_sql_io_context(tid: int) -> tuple[str | None, list[str] | None]:
    """Return the most recent SQL trace context for a native thread id."""
    with _suppress_lock:
        return _ACTIVE_SQL_IO_CONTEXTS.get(tid, (None, None))


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
        return tid in _suppress_tids or tid in _permanently_suppressed_tids


# Persistent suppression: SQL socket endpoints whose LD_PRELOAD events
# should be suppressed because the SQL layer reports at a higher granularity
# (table/row level).  Keyed by resource_id (e.g. "socket:127.0.0.1:5432",
# "socket:unix:/var/run/postgresql/.s.PGSQL.5432").
#
# The temporary _suppress_endpoint_io() context manager has a timing
# problem: LD_PRELOAD events travel through an async pipe, so by the time
# they're read the context has exited.  Permanent endpoint suppression
# persists across the entire DPOR execution.
_suppressed_sql_endpoints: set[str] = set()


def _socket_resource_id_from_fd(fd: int) -> str | None:
    """Derive the LD_PRELOAD-style resource_id from a socket file descriptor.

    Returns e.g. ``"socket:127.0.0.1:5432"`` for TCP or
    ``"socket:unix:/var/run/postgresql/.s.PGSQL.5432"`` for Unix domain sockets.
    Returns ``None`` if the fd is not a connected socket.
    """
    import socket as _socket

    # Duplicate the fd so we don't accidentally close the connection's socket
    # when the temporary socket object is garbage-collected.
    dup_fd = os.dup(fd)
    try:
        sock = _socket.socket(fileno=dup_fd)
        try:
            peer = sock.getpeername()
        except (OSError, ValueError):
            return None
        finally:
            sock.detach()  # detach so sock.__del__ doesn't close dup_fd
    finally:
        os.close(dup_fd)
    if isinstance(peer, str):
        # Unix domain socket — peer is a path string
        return f"socket:unix:{peer}" if peer else None
    if isinstance(peer, tuple) and len(peer) >= 2:
        return f"socket:{peer[0]}:{peer[1]}"
    return None


def _resource_id_from_connection(conn: Any) -> str | None:
    """Extract the LD_PRELOAD-compatible socket resource_id from a DB connection."""
    fileno_fn = getattr(conn, "fileno", None)
    if fileno_fn is None:
        return None
    try:
        fd = fileno_fn()
    except Exception:
        return None
    if not isinstance(fd, int) or fd < 0:
        return None
    return _socket_resource_id_from_fd(fd)


def suppress_sql_endpoint(conn: Any) -> None:
    """Register a SQL connection's socket endpoint for LD_PRELOAD suppression."""
    resource_id = _resource_id_from_connection(conn)
    if resource_id is not None:
        with _suppress_lock:
            _suppressed_sql_endpoints.add(resource_id)


def suppress_tid_permanently(tid: int | None = None) -> None:
    """Mark a thread as permanently suppressed for LD_PRELOAD events.

    .. deprecated::
        Prefer :func:`suppress_sql_endpoint` which suppresses by socket
        endpoint rather than by thread, so non-SQL file I/O remains visible.
        Kept for the connect-time path where the connection is not yet
        established and we must suppress by thread temporarily.
    """
    if tid is None:
        tid = threading.get_native_id()
    with _suppress_lock:
        _permanently_suppressed_tids.add(tid)


_permanently_suppressed_tids: set[int] = set()


def is_sql_endpoint_suppressed(resource_id: str) -> bool:
    """Check if a resource_id matches a known SQL socket endpoint."""
    with _suppress_lock:
        return resource_id in _suppressed_sql_endpoints


def clear_permanent_suppressions() -> None:
    """Clear all permanent suppressions (between DPOR executions)."""
    with _suppress_lock:
        _permanently_suppressed_tids.clear()
        _suppressed_sql_endpoints.clear()


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

    # Track resources that need row-lock arbitration.
    # SELECT FOR UPDATE (force_immediate) always needs arbitration.
    # Any write inside a transaction (INSERT, UPDATE, DELETE) also needs
    # arbitration because PG row locks (e.g. from UNIQUE constraints or
    # row-level locks) can cause the cooperative scheduler to deadlock
    # when one thread blocks in the kernel waiting for another's lock
    # (defect #6).
    if in_tx and (force_immediate or kind == "write"):
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


# Global to track primary column set per (db_scope, table) for cross-column
# conflict detection.  Keyed by (db_scope, table) rather than just table to
# avoid cross-database contamination when the same table name exists in
# multiple databases with different schemas/access patterns.
_table_primary_colset: dict[tuple[str | None, str], tuple[str, ...]] = {}


def _get_primary_colset(table: str, colset: tuple[str, ...], *, db_scope: str | None = None) -> tuple[str, ...]:
    """Return the primary column set for a table, initializing it if necessary."""
    return _table_primary_colset.setdefault((db_scope, table), colset)


def _stable_db_scope(identity: str) -> str:
    """Return a short deterministic token for a database identity string."""
    return hashlib.sha1(identity.encode("utf-8"), usedforsecurity=False).hexdigest()[:12]


def _register_connection_db_scope(connection: Any, identity: str) -> str:
    """Associate a stable database scope with a connection object."""
    scope = _stable_db_scope(identity)
    _CONNECTION_DB_SCOPES[id(connection)] = scope
    try:
        setattr(connection, _DB_SCOPE_ATTR, scope)
    except AttributeError:
        pass
    return scope


def _normalize_sqlite_db_identity(*args: Any, **kwargs: Any) -> str | None:
    """Best-effort canonical identity for a sqlite3 connection target."""
    database = kwargs.get("database")
    if database is None and args:
        database = args[0]
    if database is None:
        return None

    database_path = os.fspath(database)
    if isinstance(database_path, bytes):
        database_str = database_path.decode("utf-8", errors="surrogateescape")
    else:
        database_str = database_path
    use_uri = bool(kwargs.get("uri"))
    if database_str == ":memory:" and not use_uri:
        return None
    if use_uri or database_str.startswith("file:"):
        return f"sqlite-uri:{database_str}"
    return f"sqlite-path:{os.path.abspath(database_str)}"


def _normalize_mapping_db_identity(driver: str, mapping: dict[str, Any]) -> str | None:
    """Build a canonical DB identity from connect kwargs or driver info dicts."""
    items = [(key, value) for key, value in sorted(mapping.items()) if value not in (None, "")]
    if not items:
        return None
    return f"{driver}:{repr(items)}"


def _infer_db_identity_from_connection(connection: Any) -> str | None:
    """Infer a stable database identity from a live connection object."""
    info = getattr(connection, "info", None)
    dsn_params = getattr(info, "dsn_parameters", None)
    if isinstance(dsn_params, dict):
        relevant = {key: dsn_params.get(key) for key in ("host", "port", "dbname") if dsn_params.get(key)}
        identity = _normalize_mapping_db_identity("postgres", relevant)
        if identity is not None:
            return identity

    dsn = getattr(connection, "dsn", None)
    if isinstance(dsn, str) and dsn:
        return f"dsn:{dsn}"

    relevant = {
        "host": getattr(connection, "host", None),
        "port": getattr(connection, "port", None),
        "database": getattr(connection, "database", None),
        "db": getattr(connection, "db", None),
        "dbname": getattr(getattr(connection, "info", None), "dbname", None),
    }
    identity = _normalize_mapping_db_identity("dbapi", relevant)
    if identity is not None:
        return identity

    path = getattr(connection, "filename", None)
    if isinstance(path, str) and path:
        return f"sqlite-path:{os.path.abspath(path)}"

    return None


def _get_connection_db_scope(db_obj: Any) -> str | None:
    """Resolve the stable database scope for a cursor/connection-like object."""
    if db_obj is None:
        return None
    if type(db_obj).__module__.startswith("unittest.mock"):
        return None

    seen: set[int] = set()
    pending = [db_obj]
    while pending:
        candidate = pending.pop(0)
        if type(candidate).__module__.startswith("unittest.mock"):
            continue
        candidate_id = id(candidate)
        if candidate_id in seen:
            continue
        seen.add(candidate_id)

        scope = getattr(candidate, _DB_SCOPE_ATTR, None)
        if isinstance(scope, str):
            return scope

        mapped_scope = _CONNECTION_DB_SCOPES.get(candidate_id)
        if mapped_scope is not None:
            return mapped_scope

        for attr in ("connection", "_conn", "_connection"):
            nested = getattr(candidate, attr, None)
            if nested is not None:
                pending.append(nested)

    connection = getattr(db_obj, "connection", None)
    if connection is None:
        connection = getattr(db_obj, "_conn", None)
    if connection is None:
        connection = db_obj

    identity = _infer_db_identity_from_connection(connection)
    if identity is None:
        return None
    return _register_connection_db_scope(connection, identity)


def _sql_resource_id(
    table: str,
    predicates: list[Any],
    temporal: str | None = None,
    *,
    db_scope: str | None = None,
) -> str:
    """Build a resource ID from table name and optional predicates."""
    resource = f"sql:{table}"
    if temporal:
        resource = f"{resource}:history:{temporal}"
    if db_scope is not None:
        resource = f"{resource}:db={db_scope}"
    if not predicates:
        return resource
    pred_key = tuple(sorted((p.column, p.value) for p in predicates))
    return f"{resource}:{pred_key}"


def _sql_sequence_resource_id(table: str, *, db_scope: str | None = None) -> str:
    """Build the shared sequence resource ID for INSERT ordering on a table."""
    return f"{_sql_resource_id(table, [], db_scope=db_scope)}:seq"


def _report_sql_access(
    operation: Any,
    parameters: Any = None,
    *,
    db_obj: Any = None,
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
            dpor_ctx = _get_dpor_context()
            db_scope = _get_connection_db_scope(db_obj)

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

            lock_update = access.lock_intent in (LockIntent.UPDATE, LockIntent.UPDATE_SKIP_LOCKED)

            # Track which tables already reported their bridge resource in this op.
            reported_bridges: set[str] = set()

            def report_or_buffer(table: str, kind: str, rows: list[list[Any]]) -> None:
                temporal = access.temporal_clauses.get(table) if access.temporal_clauses else None
                has_row_level_predicates = bool(rows and rows[0])

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
                if dpor_ctx is not None and table not in reported_bridges and has_row_level_predicates:
                    colset = tuple(sorted(p.column for p in rows[0]))
                    primary = _get_primary_colset(table, colset, db_scope=db_scope)

                    if colset == primary:
                        # Primary row-level reads use a shared READ bridge so
                        # they still conflict conservatively with non-primary
                        # accesses, while primary row-level writes stay fully
                        # row-granular.
                        if kind == "read":
                            _report_or_buffer(
                                reporter,
                                _sql_resource_id(table, [], db_scope=db_scope),
                                "read",
                                force_immediate=lock_update,
                            )
                            reported_bridges.add(table)
                    else:
                        # Non-primary colset conflicts conservatively at table scope.
                        _report_or_buffer(
                            reporter,
                            _sql_resource_id(table, [], db_scope=db_scope),
                            "write",
                            force_immediate=lock_update,
                        )
                        reported_bridges.add(table)

                for row_preds in rows:
                    # Check if any predicate value matches a captured INSERT ID
                    alias = None
                    for pred in row_preds:
                        if isinstance(pred, EqualityPredicate):
                            alias = resolve_alias(table, pred.value, db_scope=db_scope)
                            if alias is not None:
                                break
                    res_id = (
                        alias if alias is not None else _sql_resource_id(table, row_preds, temporal, db_scope=db_scope)
                    )
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

            # Phantom read detection (sequence-number tracking):
            # SELECT depends on which rows exist in a table.  If a concurrent
            # INSERT adds a row (or DELETE removes one), the SELECT's result
            # changes.  Row-level conflict tracking misses this because the
            # new/removed row has a different resource ID than the SELECT's
            # table-level or row-level resource.
            #
            # Fix: use the table's :seq resource as a "membership" marker.
            # - Pure-read tables (SELECT) report READ on :seq.
            # - INSERT tables report WRITE on :seq (moved here from
            #   _capture_insert_id so the write is flushed at the INSERT's
            #   scheduling point, not left as orphaned pending_io).
            # - DELETE tables report WRITE on :seq.
            # - UPDATE tables report READ on :seq (defect #6 fix): UPDATE
            #   results depend on which rows exist (like SELECT), so
            #   concurrent INSERTs that add rows matching the UPDATE's WHERE
            #   clause are phantom reads. We use READ (not WRITE) to avoid
            #   false write-write conflicts between UPDATEs on different rows.
            #
            # This creates READ-WRITE conflicts between SELECT/UPDATE and
            # INSERT/DELETE, detecting phantom read races.
            pure_read_tables = access.read_tables - access.write_tables
            for table in pure_read_tables:
                _report_or_buffer(
                    reporter,
                    _sql_sequence_resource_id(table, db_scope=db_scope),
                    "read",
                )
            # INSERT targets: in write_tables but NOT in read_tables (INSERT
            # doesn't read the target table, unlike UPDATE/DELETE).
            insert_tables = access.write_tables - access.read_tables
            for table in insert_tables:
                _report_or_buffer(
                    reporter,
                    _sql_sequence_resource_id(table, db_scope=db_scope),
                    "write",
                )
            delete_tables = access.delete_tables or set()
            for table in delete_tables:
                _report_or_buffer(
                    reporter,
                    _sql_sequence_resource_id(table, db_scope=db_scope),
                    "write",
                )
            # UPDATE targets: in both write_tables and read_tables (UPDATE
            # reads the WHERE clause and writes matched rows), excluding
            # DELETE tables. Report READ on :seq so DPOR creates conflict
            # arcs with concurrent INSERTs (which WRITE :seq). This lets
            # DPOR explore interleavings where both UPDATEs run before either
            # INSERT — the pattern that causes phantom races (defect #6).
            update_tables = (access.write_tables & access.read_tables) - delete_tables
            for table in update_tables:
                _report_or_buffer(
                    reporter,
                    _sql_sequence_resource_id(table, db_scope=db_scope),
                    "read",
                )

    return reported


def _capture_insert_id(cursor: Any, table: str) -> None:
    """Capture lastrowid after INSERT and report indexical alias.

    The shared sequence resource (sql:<table>:seq) WRITE is now reported
    in ``_report_sql_access`` instead of here.  This ensures the :seq write
    is flushed at the INSERT's scheduling point (via ``report_and_wait``),
    rather than being left as orphaned ``pending_io`` when the INSERT is
    the last operation before the thread exits.
    """
    reporter = get_io_reporter()
    if reporter is None:
        return

    lastrowid = getattr(cursor, "lastrowid", None)
    db_scope = _get_connection_db_scope(cursor)

    alias = record_insert(table, lastrowid, db_scope=db_scope)

    # Report the logical alias as a write (indexical tracking for determinism)
    _report_or_buffer(reporter, alias, "write")


def _execute_with_retry(original_method: Any, cursor: Any, operation: Any, parameters: Any = None) -> Any:
    """Execute a DB-API method, retrying transient SQLite lock errors."""
    for i in range(50):
        try:
            if parameters is not None:
                return original_method(cursor, operation, parameters)
            return original_method(cursor, operation)
        except sqlite3.OperationalError as e:
            if "locked" not in str(e).lower():
                raise
            time.sleep(0.01 * (i + 1))

    if parameters is not None:
        return original_method(cursor, operation, parameters)
    return original_method(cursor, operation)


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
    from frontrun._cooperative import suppress_sync_reporting, unsuppress_sync_reporting

    insert_match = _RE_INSERT_TABLE.match(operation) if isinstance(operation, str) else None
    update_match = _RE_UPDATE_TABLE.match(operation) if isinstance(operation, str) else None

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

    # Permanently suppress LD_PRELOAD *socket* events for this thread.
    # SQL-level reporting (table/row granularity) supersedes socket-level
    # I/O.  The listener only applies tid suppression to socket events,
    # so non-SQL file I/O from this thread passes through.
    # The patched connect() also registers the connection's socket endpoint
    # for endpoint-based suppression (which handles remote connections).
    suppress_tid_permanently()

    reported = _report_sql_access(
        operation,
        parameters,
        db_obj=self,
        is_executemany=is_executemany,
        paramstyle=paramstyle,
    )

    # Block if another DPOR thread holds a conflicting row lock
    _acquire_pending_row_locks()

    # Force a scheduling point so the scheduler can interleave between
    # SQL operations.  Without this, all code inside frontrun/ is skipped
    # by the tracer, so pending_io is never flushed between back-to-back
    # SQL calls.  This is needed both during DPOR exploration (to report
    # accesses) and during replay (to consume schedule entries that DPOR
    # generated for SQL statements).
    _dpor_ctx = _get_dpor_context()
    if _dpor_ctx is not None and (reported or isinstance(operation, str)):
        _dpor_ctx[0].report_and_wait(None, _dpor_ctx[1])

    _set_active_sql_io_context(operation, parameters, paramstyle)
    # Suppress cooperative lock sync events during the actual DB call.
    # Internal psycopg2/driver locks are implementation details.
    suppress_sync_reporting()
    try:
        if reported:
            with _suppress_endpoint_io():
                result = _execute_with_retry(original_method, self, operation, parameters)
        else:
            result = _execute_with_retry(original_method, self, operation, parameters)
    except Exception:
        # Release row locks on execution failure to prevent framework-induced
        # deadlocks.  Without this, if a SQL statement raises (e.g.,
        # OperationalError from SQLite lock contention), any row locks
        # acquired by _acquire_pending_row_locks remain held until thread
        # exit, blocking other DPOR threads indefinitely.
        _release_dpor_row_locks()
        raise
    finally:
        unsuppress_sync_reporting()

    # Defect #6 fix: release row locks for 0-row UPDATEs.
    # In PostgreSQL, an UPDATE that matches 0 rows acquires no row locks
    # (there are no rows to lock).  But frontrun's row-lock arbitration
    # acquired a scheduler-level lock based on the WHERE-clause resource ID
    # regardless of whether any rows matched.  This over-serialization
    # prevents DPOR from exploring interleavings where both 0-row UPDATEs
    # execute before either INSERT (the UPDATE-INSERT phantom race pattern).
    if update_match is not None and reported:
        rowcount = getattr(self, "rowcount", -1)
        if rowcount == 0:
            _release_dpor_row_locks()

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

# Global lock_timeout (milliseconds) to inject on new PostgreSQL connections.
# Set by explore_dpor(lock_timeout=...) and cleared after exploration.
_lock_timeout_ms: int | None = None


def set_lock_timeout(ms: int | None) -> None:
    """Set the global lock_timeout that will be injected on new PG connections."""
    global _lock_timeout_ms  # noqa: PLW0603
    _lock_timeout_ms = ms


def get_lock_timeout() -> int | None:
    """Return the current global lock_timeout (milliseconds), or None."""
    return _lock_timeout_ms


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
        conn = orig_connect(*args, **kwargs)
        identity = _normalize_sqlite_db_identity(*args, **kwargs)
        if identity is None:
            identity = f"sqlite-memory:{id(conn)}"
        _register_connection_db_scope(conn, identity)
        return conn

    sqlite3.connect = wrap_method_metadata(patched_connect, orig_connect, name="connect")  # type: ignore[assignment]
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
        _is_executemany = method_name == "executemany"

        def _make_patched(orig: Any, mname: str = method_name, ps: str = paramstyle) -> Any:
            def _patched(self: Any, operation: Any, parameters: Any = None, *args: Any, **kwargs: Any) -> Any:
                return _intercept_execute(
                    orig, self, operation, parameters, is_executemany=_is_executemany, paramstyle=ps
                )

            return wrap_method_metadata(_patched, orig, name=mname)

        patch_method(
            cls,
            method_name,
            originals=_ORIGINAL_METHODS,
            patches=_PATCHES,
            make_wrapper=_make_patched,
        )


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

    _warm_sql_parsers()

    # sqlite3 requires factory-injection approach
    _patch_sqlite3()

    # Pure-Python drivers can be patched directly
    for target in PYTHON_CURSOR_TARGETS:
        try:
            mod = importlib.import_module(target.module_path)
            cls = getattr(mod, target.class_name)
            driver_mod = importlib.import_module(target.paramstyle_module)
            paramstyle = getattr(driver_mod, "paramstyle", "format")
            _patch_class_methods(cls, paramstyle)
        except (ImportError, AttributeError):
            pass  # driver not installed — skip silently

    def _make_patched_connect(orig: Any, default_cursor_cls: type, paramstyle: str, driver: str) -> Any:
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
            from frontrun._cooperative import suppress_sync_reporting as _ssr
            from frontrun._cooperative import unsuppress_sync_reporting as _usr

            # Suppress LD_PRELOAD events BEFORE the actual connect call.
            # The background pipe reader may process events from connect()
            # before we return; suppressing the tid first ensures those
            # events are dropped in the listener() callback.  After the
            # connection is established we register the *endpoint* for
            # permanent suppression and remove the thread-level suppression.
            suppress_tid_permanently()
            _ssr()
            try:
                conn = orig(*args, **kwargs)
            finally:
                _usr()
            # Now that the connection is established, register its socket
            # endpoint for permanent suppression.  The thread-level tid
            # suppression remains as a belt-and-suspenders fallback for
            # any socket events that raced through the pipe before the
            # endpoint was registered.  The listener only uses tid
            # suppression for *socket* events, so file I/O passes through.
            suppress_sql_endpoint(conn)
            identity = _infer_db_identity_from_connection(conn)
            if identity is None and args and isinstance(args[0], str):
                identity = f"{driver}-dsn:{args[0]}"
            if identity is None:
                relevant = {
                    "host": kwargs.get("host"),
                    "port": kwargs.get("port"),
                    "dbname": kwargs.get("dbname") or kwargs.get("database") or kwargs.get("db"),
                }
                identity = _normalize_mapping_db_identity(driver, relevant)
            if identity is not None:
                _register_connection_db_scope(conn, identity)
            # Inject SET lock_timeout if configured (defect #6 workaround).
            # Use the *original* cursor class to avoid triggering DPOR
            # scheduling points during connection setup.
            if _lock_timeout_ms is not None and driver in ("psycopg2", "psycopg"):
                _ssr()
                try:
                    _was_autocommit = conn.autocommit
                    conn.autocommit = True
                    _lt_cur = conn.cursor(cursor_factory=default_cursor_cls)
                    try:
                        _lt_cur.execute(f"SET lock_timeout = '{int(_lock_timeout_ms)}ms'")
                    finally:
                        _lt_cur.close()
                    conn.autocommit = _was_autocommit
                finally:
                    _usr()
            return conn

        return patched_connect

    # psycopg2: patch via cursor_factory injection into connect()
    for target in CONNECT_FACTORY_TARGETS:
        try:
            driver_mod = importlib.import_module(target.module_name)
            cursor_mod = importlib.import_module(target.cursor_module_name)
            orig_cursor_cls = getattr(cursor_mod, target.cursor_attr_name)
            orig_connect = driver_mod.connect
            setattr(
                driver_mod,
                "connect",
                _make_patched_connect(
                    orig_connect, orig_cursor_cls, paramstyle=target.paramstyle, driver=target.driver
                ),
            )
            _PATCHES.append((driver_mod, "connect", orig_connect))
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
    for attr in ("_in_transaction", "_is_autobegin", "_tx_buffer", "_tx_savepoints", "_pending_row_locks"):
        if hasattr(_io_tls, attr):
            delattr(_io_tls, attr)


def clear_sql_metadata() -> None:
    """Reset all global SQL resource tracking metadata.

    Call this between DPOR exploration sessions to ensure test isolation.
    """
    _table_primary_colset.clear()
    _CONNECTION_DB_SCOPES.clear()


def unpatch_sql() -> None:
    """Restore original DBAPI cursor methods and connect functions."""
    global _sql_patched  # noqa: PLW0603
    if not _sql_patched:
        return
    restore_patches(_PATCHES)
    _PATCHES.clear()
    _ORIGINAL_METHODS.clear()
    _sql_patched = False
