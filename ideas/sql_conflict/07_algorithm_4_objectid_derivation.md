# Algorithm 4: ObjectId Derivation

Uses the existing `_make_object_key` function from `dpor.py:569`:

```python
def _make_object_key(obj_id: int, name: Any) -> int:
    """Create a non-negative u64 object key for the Rust engine."""
    return hash((obj_id, name)) & 0xFFFFFFFFFFFFFFFF
```

For SQL, the resource_id is `"sql:{table}"`. The io_reporter in `_setup_dpor_tls` already computes:

```python
object_key = _make_object_key(hash(resource_id), resource_id)
```

So `"sql:accounts"` and `"sql:users"` get different `ObjectId`s. Two threads on different tables → different `ObjectId`s → no conflict → DPOR skips the interleaving.

Two threads on the same table with one reading and one writing → same `ObjectId` `hash("sql:accounts")`, but one reports `kind="read"` and the other `kind="write"` → the Rust engine's `ObjectState::dependent_accesses` correctly identifies this as a RW conflict → DPOR explores the interleaving.

Two threads both doing `SELECT` on the same table → same `ObjectId`, both `kind="read"` → `dependent_accesses(Read, thread_id)` returns only writes by other threads → no writes → no conflict → DPOR skips.

**This is the key correctness property: the Adya Read/Write classification maps directly to `AccessKind::Read`/`AccessKind::Write`, and `ObjectState::dependent_accesses` implements exactly the right conflict rules.**
