# Reducing DPOR Trace Overcounting

## Problem Statement

The exact Mazurkiewicz trace count tests (`test_exact_mazurkiewicz_trace_count.py`)
reveal that the DPOR implementation explores more traces than the theoretical
minimum for several test families. After removing initial thread rotation (which
was redundant — see "Resolved" section), the remaining overcounting is:

| Test | Expected | Actual | Factor |
|------|----------|--------|--------|
| 2 threads, 1 shared var (loop `for v in s.vars`) | 2 | 3 | 1.5× |
| 2 threads, 2 shared vars (loop) | 4 | 9 | 2.25× |
| 2 threads, 3 shared vars (loop) | 8 | 27 | 3.375× |
| 2 threads, lock, N=2 | 2 | 3 | 1.5× |
| 2 threads, lock over 1 var | 2 | 3 | 1.5× |
| N=3 threads, lock | 6 | ? | >2× |

This document analyzes the root causes and proposes fixes.

## Resolved: Initial Thread Rotation (removed in #149)

The DPOR engine previously used **initial thread rotation**: after exhausting the
exploration tree starting with thread 0, it re-explored starting with thread 1,
then thread 2, etc. This was a workaround for a bug where DPOR only explored traces
starting with thread 0. The fix went too far — it added N−1 extra exploration rounds,
roughly multiplying the total executions by N.

This was removed entirely in commit 3fcb62c (#149). Standard DPOR race detection
already handles all cases: when initial operations conflict, DPOR detects the race
at position 0 and inserts the alternative thread into the wakeup tree. When initial
operations are independent, swapping them produces a Mazurkiewicz-equivalent trace.

## Root Cause 1: Multi-Opcode Decomposition of "Single" Operations

### Attribute Set: `obj.attr = value`

The Python statement `s.slot.value = tid` compiles to:

```
LOAD_SMALL_INT  tid       # push value
LOAD_FAST       s         # push object
LOAD_ATTR       slot      # READ(State, "slot") + WKREAD(_Slot, "__cmethods__")
STORE_ATTR      value     # WRITE(_Slot, "value")
```

What the abstract model treats as one atomic write operation, the DPOR system sees
as **two scheduling points** (LOAD_ATTR and STORE_ATTR), each of which is a potential
context switch location. The DPOR engine correctly determines that the LOAD_ATTR reads
don't conflict with any writes from other threads (they're on different object keys),
but the two scheduling points create positions in the execution tree where wakeup
insertions can direct alternative thread orderings.

### Iterated Attribute Set: `for v in s.vars: v.value = tid`

This compiles to even more scheduling points per variable:

```
LOAD_FAST       s         # no report
LOAD_ATTR       vars      # READ(State, "vars") + WKREAD(list, "__cmethods__")
GET_ITER                  # READ(list, "__cmethods__")
FOR_ITER                  # READ(list, "0") + READ(list, "__cmethods__")
STORE_FAST      v         # no report
LOAD_SMALL_INT  tid       # no report
LOAD_FAST       v         # no report
STORE_ATTR      value     # WRITE(_Slot, "value")
FOR_ITER (exhausted)      # READ(list, "1") + READ(list, "__cmethods__")
```

That's **5 scheduling points** per variable per thread (LOAD_ATTR, GET_ITER, FOR_ITER,
STORE_ATTR, FOR_ITER-exhausted), versus the abstract model's 1.

### Lock Acquisition: `with s.lock:`

The `with` statement compiles to:

```
LOAD_FAST       s         # no report
LOAD_ATTR       lock      # READ(State, "lock") + WKREAD(Lock, "__cmethods__")
COPY 1                    # no report
LOAD_SPECIAL    __exit__  # READ(Lock, "__exit__")
SWAP, SWAP                # no report
LOAD_SPECIAL    __enter__ # READ(Lock, "__enter__")
CALL            __enter__ # cooperative lock_acquire → WRITE(lock_io_obj) via io_vv
POP_TOP                   # no report
```

That's **4 scheduling points** before the lock is even acquired (LOAD_ATTR, LOAD_SPECIAL
×2, CALL). Inside the lock body, `s.shared.value = tid` adds 2 more (LOAD_ATTR +
STORE_ATTR). Lock release via `CALL __exit__` adds 1 more. Total: **~7 scheduling
points** per thread for a single lock-protected write.

## Root Cause 2: Lock Release/Acquire Races via `io_vv`

Lock operations use `io_vv` (which excludes lock-based happens-before edges) to detect
races. This is intentional: it ensures TOCTOU races across lock boundaries are caught.
However, it creates **two independent race dimensions** for each lock:

1. Lock acquire WRITE(lock_io_obj) — races between all threads
2. Lock release WRITE(lock_rel_io_obj) — races between all threads

Both use different virtual object IDs (XOR with `LOCK_OBJECT_XOR` and
`LOCK_RELEASE_XOR` respectively). Each creates a write-write race between threads.
Combined with the multiple pre-lock scheduling points, this produces:

- N=2: 3 traces (expected 2) — 1.5× overcounting
- N=3: much larger (combinatorial explosion)

## Proposed Fixes

### Fix 1: Coalesce Non-Conflicting Opcodes Into Atomic Groups (Medium, High Impact)

**Idea**: When consecutive opcodes from the same thread access different objects and
none of those accesses conflict with operations from other threads, treat them as a
single atomic scheduling point.

**Concrete approach**: Identify "read-only preamble" patterns:
- `LOAD_ATTR` → `STORE_ATTR`: The LOAD_ATTR reads object X, STORE_ATTR writes object
  Y. If X ≠ Y (always true for `s.slot.value = v`), the LOAD_ATTR commutes with
  everything the STORE_ATTR conflicts with. So the pair can be a single scheduling point.
- `LOAD_ATTR` → `LOAD_SPECIAL` → `LOAD_SPECIAL` → `CALL` (lock acquire pattern):
  The three pre-acquire reads don't conflict with anything. They could be a single
  scheduling point with the CALL.

**Implementation**: After executing a non-conflicting opcode (pure read that creates no
races), skip the scheduler yield and continue executing the same thread. Only yield when
an opcode actually creates or could create a conflict.

Specifically, modify `_process_opcode` to return a flag indicating whether the access
was potentially conflicting. If not, the scheduler can skip the yield and continue the
current thread without creating a new scheduling point.

**Impact**: Would dramatically reduce scheduling points. For `s.slot.value = tid`:
from 2 scheduling points to 1. For `with s.lock: s.shared.value = tid`: from ~7 to ~3
(ACQ, WRITE, REL).

**Risk**: This is sound only if the skipped opcodes truly commute with all possible
operations from other threads at that point. A read can only be skipped if no other
thread has written to the same object. Since we're checking at execution time (not
statically), this is a dynamic optimization that preserves correctness.

### Fix 2: Collapse Lock Acquire/Release Into Single Race Point (Medium, Medium Impact)

**Idea**: Instead of creating two independent race objects for lock acquire and lock
release, create only ONE virtual race object per lock. The lock acquire and release from
the same thread both write to the same virtual object. This way, the release doesn't
create an additional independent race dimension.

**Current behavior**:
- `lock_acquire` → WRITE(`lock_id ^ LOCK_OBJECT_XOR`)
- `lock_release` → WRITE(`lock_id ^ LOCK_RELEASE_XOR`)

These are different objects, so each creates an independent set of races.

**Proposed behavior**:
- `lock_acquire` → WRITE(`lock_id ^ LOCK_OBJECT_XOR`)
- `lock_release` → no virtual write (or write to the same object as acquire)

Since the acquire already creates the necessary races for exploring different lock
orderings, the release write is redundant. The happens-before edge from release to the
next acquire is still established via `dpor_vv` / `lock_release_vv`.

**Impact**: Removes one independent race dimension per lock. Halves the lock-related
overcounting.

**Risk**: Removing the release race point might miss some TOCTOU patterns where the
timing of release matters independently of acquire. Need to analyze whether any real
bugs require exploring different release orderings that can't be reached through
different acquire orderings.

### Fix 3: Eliminate Weak Read on LOAD_ATTR (Easy, Low-Medium Impact)

**Idea**: When `LOAD_ATTR` loads an attribute value that is a mutable object, it
currently reports a `weak_read` on `(returned_obj, "__cmethods__")`. For the common
pattern `s.slot.value = tid`, this creates a spurious `weak_read` on the `_Slot` object
that is about to be written to.

While weak reads don't conflict with weak writes, they DO conflict with strong writes
(AccessKind::Write). So if another thread writes to `(_Slot, "__cmethods__")` (which
doesn't happen in these tests, but could happen with container methods), the weak read
would create a race.

More importantly, the weak read adds an access to the DPOR state that participates in
sleep set propagation and trace caching, potentially preventing sleep set optimizations
from pruning equivalent executions.

**Proposed change**: Only emit the weak_read when the loaded value is a container type
(list, dict, set) that could have C-level method mutations. Skip it for simple objects
with only Python-level attribute access.

**Impact**: Removes ~1 spurious access per LOAD_ATTR. Small but compounds with other
fixes.

### Fix 4: Batch Iterator Operations (Medium, Medium Impact)

**Idea**: The `GET_ITER` → `FOR_ITER` → `FOR_ITER(exhausted)` sequence for iterating a
list creates 3+ scheduling points for what is conceptually "iterate over the list."
Since these are all reads (no writes), they could be collapsed into a single scheduling
point.

**Implementation**: When `GET_ITER` detects iteration over a mutable container, report
the container-level read once, then skip `FOR_ITER` yields until the iteration is
complete or a write occurs inside the loop body.

**Impact**: Reduces scheduling points per loop from 2+N (GET_ITER + N×FOR_ITER) to 1
(for the read) + body scheduling points. For `for v in s.vars: v.value = tid`, this
would reduce from ~5 scheduling points per variable to ~2.

## Priority Ordering

1. **Fix 1** (coalesce non-conflicting opcodes): highest impact, addresses all test
   families. Medium complexity but conceptually clean.
2. **Fix 2** (collapse lock races): medium impact for lock tests specifically.
3. **Fix 4** (batch iterators): medium impact for loop-heavy code.
4. **Fix 3** (eliminate weak reads): lowest priority, small incremental improvement.

## Implementation Notes

Fix 1 deserves more detail since it's the most impactful. The key insight is that an
opcode that performs only reads on objects that no other thread has written to is
*guaranteed* to commute with all concurrent operations. At that point in the execution,
the DPOR engine has already checked for races (via `process_access`) and found none.
Therefore, no wakeup tree insertion occurred, and the scheduler can safely continue the
current thread without creating a new scheduling point.

The implementation would add a return value to `process_access` (and similar functions)
indicating whether any race was detected. The Python scheduler would check this flag
and, if no race was found and the access was a read, skip the yield.

Note: even if the read doesn't create a race in THIS execution, DPOR might need the
scheduling point in future executions when different thread interleavings are explored.
The correct approach is to check whether the read could POSSIBLY conflict with any
concurrent operation — which is exactly what `process_access` already does.
