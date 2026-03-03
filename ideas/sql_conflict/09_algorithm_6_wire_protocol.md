# Algorithm 6: Wire Protocol SQL Extraction (LD_PRELOAD Enhancement)

For C-extension drivers (psycopg2 uses libpq, which calls `send()` directly), the DBAPI monkey-patch may not fire. The LD_PRELOAD library already intercepts `send()` buffers. We can parse the PostgreSQL wire protocol to extract SQL:

## 6a. PostgreSQL Simple Query Protocol

```
Byte1('Q')        — message type
Int32             — message length (including self)
String            — the SQL query text (null-terminated)
```

## 6b. PostgreSQL Extended Query Protocol (Prepared Statements)

```
Parse:    Byte1('P') Int32-len String-name String-query Int16-nparams ...
Bind:     Byte1('B') Int32-len String-portal String-stmt Int16-nformats ...
Execute:  Byte1('E') Int32-len String-portal Int32-maxrows
```

## 6c. Extraction in Rust (crates/io/)

```rust
// crates/io/src/sql_extract.rs

/// Extract SQL query text from a PostgreSQL wire protocol buffer.
/// Returns None if the buffer doesn't contain a recognizable query message.
pub fn extract_pg_query(buf: &[u8]) -> Option<&str> {
    if buf.is_empty() {
        return None;
    }
    match buf[0] {
        b'Q' => {
            // Simple query: 'Q' + i32 len + null-terminated SQL
            if buf.len() < 5 {
                return None;
            }
            let len = i32::from_be_bytes([buf[1], buf[2], buf[3], buf[4]]) as usize;
            if buf.len() < 1 + len {
                return None;
            }
            let sql_bytes = &buf[5..1 + len - 1]; // exclude null terminator
            std::str::from_utf8(sql_bytes).ok()
        }
        b'P' => {
            // Parse message: 'P' + i32 len + name(str0) + query(str0) + i16 nparams
            if buf.len() < 5 {
                return None;
            }
            let len = i32::from_be_bytes([buf[1], buf[2], buf[3], buf[4]]) as usize;
            if buf.len() < 1 + len {
                return None;
            }
            let payload = &buf[5..1 + len];
            // Skip statement name (null-terminated)
            let name_end = payload.iter().position(|&b| b == 0)?;
            let query_start = name_end + 1;
            let remaining = &payload[query_start..];
            let query_end = remaining.iter().position(|&b| b == 0)?;
            std::str::from_utf8(&remaining[..query_end]).ok()
        }
        _ => None,
    }
}
```

Then in the LD_PRELOAD `send()` hook, after intercepting the buffer, attempt SQL extraction. If successful, write SQL-enriched events to the pipe instead of raw socket events. The Python-side `_PreloadBridge.listener` can then parse the SQL and report at table level.

**This is Phase 3 work.** The DBAPI monkey-patching (Phase 1) covers most cases. Wire protocol parsing is only needed for C-extension drivers that bypass the Python DBAPI layer entirely (rare in practice — even psycopg2's `cursor.execute()` goes through the Python method).
