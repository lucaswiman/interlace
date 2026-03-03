# Problem

All SQL to the same `(host, port)` collapses to a single DPOR `ObjectId`. Two threads doing `INSERT INTO logs` and `SELECT * FROM users` appear to conflict because both produce `send()`/`recv()` to `localhost:5432`. DPOR explores O(n!) interleavings that can never actually race. Similarly, two concurrent `SELECT`s on the same table look like write-write conflicts (because they're both "writes to the socket").

**Fix:** Parse SQL at the DBAPI layer, derive per-table `ObjectId`s with correct `AccessKind` (Read/Write), and suppress the coarser endpoint-level reports.
