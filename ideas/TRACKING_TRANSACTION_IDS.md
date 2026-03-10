# Tracking Database Connection and Transaction Identity

## Problem

The LD_PRELOAD library currently identifies database connections by socket
path (e.g., `socket:unix:/var/run/postgresql/.s.PGSQL.5432`).  This is too
coarse — every connection to the same PostgreSQL server produces the same
resource_id, whether the threads have independent connections or are sharing
one.

We removed a spurious "shared socket" warning that keyed on resource_id, but
the underlying question remains: how should DPOR distinguish independent
connections from shared ones, and how should it track which operations actually
conflict at the database level?

## Python drivers already expose what we need

Before considering wire-protocol parsing in the LD_PRELOAD library, it's
worth noting that Python database drivers already expose connection and
transaction metadata at the Python level.  The SQL cursor patching layer
(`_sql_cursor.py`) intercepts `cursor.execute()` and has access to
`cursor.connection` — so it can read these attributes directly.

### psycopg2 (PostgreSQL)

```python
conn = psycopg2.connect(dsn)
conn.info.backend_pid        # 34827 — unique per server-side process
conn.info.transaction_status  # 0=idle, 1=active, 2=in-trans, 3=error
conn.info.socket              # 5 — the fd number
conn.info.dbname              # 'mydb'
```

`backend_pid` uniquely identifies the server-side connection.  Two threads
sharing a pooled connection see the same PID; independent connections get
different PIDs.  `transaction_status` gives live transaction boundaries
(maps to the wire protocol's `ReadyForQuery` status byte).

### SQLite

```python
conn = sqlite3.connect('mydb.sqlite')
conn.in_transaction  # bool — whether a transaction is active
```

No equivalent of `backend_pid` — SQLite is in-process, so there's no
server-side connection to identify.  The meaningful identifier is the
database file path.  Two connections to the same file share the database
via file locks, not a wire protocol.  Transaction tracking is available
via `conn.in_transaction`.

### SQLAlchemy / Django ORM

Both provide access to the underlying driver connection:

```python
# SQLAlchemy
raw_conn = engine.raw_connection()  # → psycopg2 connection
raw_conn.info.backend_pid

# Django
from django.db import connection
connection.connection.info.backend_pid  # underlying psycopg2
```

### Summary

| Driver    | Connection ID         | Transaction state           |
|-----------|-----------------------|-----------------------------|
| psycopg2  | `info.backend_pid`    | `info.transaction_status`   |
| SQLite    | file path             | `in_transaction`            |
| psycopg3  | `pgconn.backend_pid`  | `pgconn.transaction_status` |
| asyncpg   | `con.get_server_pid()`| implicit (protocol-level)   |

The common Python database drivers all expose enough to identify connections
and track transaction state.  Wire-protocol parsing in the LD_PRELOAD library
would only be needed for code that bypasses Python's cursor layer entirely
(direct libpq FFI), which is a niche case.

## Where to use this

### 1. Connection identity in `_sql_cursor.py`

The SQL cursor patching layer already intercepts `execute()`.  It could
read `backend_pid` (or equivalent) from the connection and include it in
the resource_id reported to DPOR:

```python
# Instead of just:  resource_id = f"sql:{table}"
# Include:          resource_id = f"sql:{table}:conn={backend_pid}"
```

This would let DPOR distinguish operations from independent connections vs.
shared connections.  Currently this distinction doesn't matter much because
the SQL-level conflict detection already identifies conflicts by table/row,
but it could be useful for:

- Detecting when two threads are incorrectly sharing a single connection
  (diagnostic, replaces the removed warning)
- More precise conflict tracking when connection pools are involved

### 2. Transaction boundaries in `_sql_cursor.py`

The cursor patching layer already tracks `_in_transaction` state.  Using the
driver's own transaction status (rather than inferring it from `BEGIN`/
`COMMIT` SQL parsing) would be more reliable — it handles edge cases like
implicit transactions, savepoints, and driver-specific autocommit behavior.

This is already partially done — `_io_detection.py` tracks `_in_transaction`
and `_is_autobegin`.  Checking `conn.info.transaction_status` directly would
be a more robust source of truth.

## Wire-protocol approach (LD_PRELOAD level)

For completeness, here's what the PostgreSQL wire protocol offers, in case
we need C-level tracking for non-Python drivers:

### BackendKeyData (`K`, server → client)

Sent once during connection startup:

    'K' | int32 len(12) | int32 backend_pid | int32 secret_key

Could be parsed in the `recv()` hook during the first few recv calls after
`connect()`.  ~10 lines of Rust.

### ReadyForQuery (`Z`, server → client)

Sent after every command completion:

    'Z' | int32 len(5) | byte status   (I=idle, T=in-trans, E=error)

Gives transaction boundaries.  Requires a small message framer per fd since
recv buffers may contain multiple/partial messages.  Medium complexity.

### CommandComplete (`C`, server → client)

Command tag with row counts (`INSERT 0 1`, `UPDATE 3`).  Marginal value
over send-side SQL extraction.

### Existing send-side parsing

`sql_extract.rs` already parses Simple Query (`Q`) and Parse (`P`) messages.
The backend messages above would extend this to the recv side.

## Recommendation

**Use the Python driver APIs first.**  The `_sql_cursor.py` layer already
has access to `cursor.connection` and can read `backend_pid` and
`transaction_status` directly.  This covers psycopg2, SQLite, and any ORM
built on top of them — which is the vast majority of real-world usage.

Reserve wire-protocol parsing in the LD_PRELOAD library for the rare case
of C extensions that bypass Python's database cursor entirely.

## Relationship to the shared-socket warning

The removed warning tried to detect shared connections by keying on socket
path.  With `backend_pid` available from the driver, the same detection
becomes trivial and correct: if two threads' cursors report the same
`backend_pid`, they are genuinely sharing a database connection.  Whether
that warrants a warning or is just a shared resource for DPOR to explore
is a separate design question.
