# Reducing DPOR Trace Overcounting

## Problem Statement

The exact Mazurkiewicz trace count tests (`test_exact_mazurkiewicz_trace_count.py`)
reveal that the DPOR implementation explores more traces than the theoretical
minimum for several test families. After removing initial thread rotation (which
was redundant — see "Resolved" section), the remaining overcounting is:

| Test | Expected | Baseline | After Fixes 3+4+5+6 | Factor |
|------|----------|----------|---------------------|--------|
| 2 threads, 1 shared var (loop `for v in s.vars`) | 2 | 3 | **2** ✅ | 1.0× |
| 2 threads, 2 shared vars (loop) | 4 | 9 | 6 | 1.5× |
| 2 threads, 3 shared vars (loop) | 8 | 27 | 18 | 2.25× |
| 2 threads, lock, N=2 | 2 | 3 | 3 | 1.5× |
| 2 threads, lock over 1 var | 2 | 3 | 3 | 1.5× |
| N=2 threads, lock | 2 | 3 | 3 | 1.5× |
| N=3 threads, lock | 6 | 17 | 17 | 2.83× |

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

## Root Cause 3: Trace Cache Merge Escalation (FIXED — Fix 6)

When computing the trace cache (`prev_thread_all_accesses` in `step()`), the engine
merges a thread's accesses across all scheduling points. **Previously**, any mismatch
in AccessKind was naively upgraded to Write:

```rust
// BEFORE (buggy):
if *existing != *kind {
    *existing = AccessKind::Write;
}
```

This caused Read + WeakRead (which commonly occurs when LOAD_ATTR emits WeakRead on
`__cmethods__` and GET_ITER/FOR_ITER emit Read on the same object) to be stored as
Write. The sleeping thread then appeared to *write* to the container object, causing
it to be falsely woken from the sleep set whenever another thread merely *read* the
same container. Each false wakeup opened a wakeup tree insertion site → extra execution.

**Fix 6** added `AccessKind::merge()` which correctly handles Read + WeakRead → Read
(Read's conflict set `{Write, WeakWrite}` is a superset of WeakRead's `{Write}`):

```rust
// AFTER (correct):
*existing = existing.merge(*kind);
```

**Important**: This fix applies ONLY to the trace cache merge in `step()`. The
per-branch `record_access()` merge must remain conservative (upgrade to Write),
because the per-branch merge affects `active_accesses` which participates in
`propagate_sleep()` independence checks during the same execution. Making that
less conservative caused a regression in `test_independent_file_writes[2]` — the
exact mechanism needs further investigation.

**Impact**: Fixed the simplest overcounting case (2 threads, 1 shared var via loop)
from 3 to the theoretical minimum of 2. Reduced 2-var from 9 to 6 and 3-var from 27
to 18.

## Implemented Fixes

### Fix 3: Restrict LOAD_ATTR Weak Read to Container Types (Implemented)

Restricted the LOAD_ATTR `weak_read` on `__cmethods__` to only fire when the loaded
value is `isinstance(val, (list, dict, set))`. Previously fired for all non-immutable
types, creating spurious weak reads on user-defined objects like `_Slot`.

**Impact on trace counts**: Zero — weak_read-weak_read pairs are non-conflicting.
This is a correct optimization that reduces per-execution overhead.

### Fix 4+5: First-Access Semantics for Iterator Reads (Implemented)

Changed GET_ITER and FOR_ITER from `_report_read` to `_report_first_read`, pinning
repeated reads of `__cmethods__` and per-element keys to the first scheduling point.

**Impact on trace counts**: Zero — read-read pairs don't generate wakeup insertions.
Correct optimization that reduces unnecessary work in the DPOR engine.

### Fix 6: Trace Cache Merge Fix (Implemented)

See Root Cause 3 above. The only fix that actually reduced trace counts.

## Proposed Fixes (Not Yet Implemented)

### Fix 2: Collapse Lock Acquire/Release Into Single Race Point

**Status**: Previously attempted and abandoned — removing the lock release race broke
`test_prometheus_summary_observe_pattern` and `test_transfer_between_two_locked_accounts`
(multi-lock tests in `test_defect11_multi_lock_races.py`). The lock release race is
needed to create scheduling opportunities between consecutive critical sections using
DIFFERENT locks.

**Possible sound approach**: Only collapse acquire/release for the SAME lock within the
same thread's contiguous critical section. When a thread does `lock_A.acquire();
lock_A.release(); lock_B.acquire()`, the release of lock_A is needed to create a
scheduling point between the two critical sections. But when a thread does a simple
`with lock: body`, the release of the same lock doesn't add a new race dimension
beyond what the acquire already covers.

**Implementation idea**: Track whether the lock being released is the same lock most
recently acquired by this thread (no intervening acquire of a different lock). If so,
skip the release race. If a different lock was acquired in between, emit the release
race as before.

**Risk**: Medium. The heuristic might miss edge cases.

### Fix 7: Improve AccessKind Merge for WeakWrite + WeakRead

**Idea**: Extend `AccessKind::merge()` to handle additional combinations:

```rust
// WeakWrite conflicts with {Read, Write}
// WeakRead conflicts with {Write}
// WeakWrite ⊇ WeakRead in conflict sets → merge to WeakWrite
(WeakWrite, WeakRead) | (WeakRead, WeakWrite) => WeakWrite,
```

Currently these merge to Write (which conflicts with everything). Merging to
WeakWrite instead prevents false wakeups when another thread does only WeakWrite
or WeakRead on the same object.

**Impact**: Low — only applies when both WeakWrite and WeakRead occur on the
same object across different scheduling points of the same thread.

### Fix 8: Per-Branch Merge Investigation

**Idea**: Apply `AccessKind::merge()` to the per-branch `record_access()` as well,
not just the trace cache. This would fix `active_accesses` at the scheduling-point
level, preventing the Read + WeakRead → Write escalation within a single scheduling
point from propagating through sleep set checks.

**Status**: Attempted and caused a regression in `test_independent_file_writes[2]`
(independent writes got 2 instead of 1). The root cause of the regression needs
investigation — it may be that per-branch merge interacts with I/O access tracking
(`process_io_access` uses the same `record_access` path).

**Potential approach**: Only apply the improved merge for non-I/O accesses. Or
investigate whether the regression is actually caused by a pre-existing bug that
the conservative Write merge was masking.

### Fix 9: Reduce Scheduling Points for Non-Conflicting Opcodes

**Idea**: Skip the scheduling point (don't call `report_and_wait`) for opcodes
whose accesses are provably non-conflicting with any other thread's possible
accesses. For example, LOAD_ATTR on a read-only shared object where no thread
ever writes to it could be executed without a scheduling point.

**Implementation**: This requires static analysis or dynamic tracking of which
objects are written by any thread. The DPOR engine could maintain a
"potentially-written objects" set, and the Python trace handler could check
whether an opcode's target object is in this set before calling report_and_wait.

**Soundness**: Safe if the "potentially-written" set is conservative (includes all
objects that could be written). The challenge is building this set without running
the full execution first.

**Alternative**: A simpler version: skip scheduling points for opcodes on objects
that are known immutable (e.g., reading attributes of a module or class that is
never modified during the test). This is already partially done by the
`_IMMUTABLE_TYPES` check.

## Priority Ordering

1. **Fix 2 (sound variant)**: Collapse lock release race for same-lock critical
   sections. Addresses all lock-related overcounting (3→2 for 2 threads).
2. **Fix 8**: Investigate per-branch merge regression. If fixable, addresses
   remaining overcounting in shared-vars tests.
3. **Fix 7**: Extend merge function for WeakWrite + WeakRead. Small incremental.
4. **Fix 9**: Reduce scheduling points. Larger effort, potentially high impact.

## Analysis: Why Fixes 3+4+5 Had Zero Impact

Fixes 3, 4, and 5 reduce the number and kind of access reports, but they had
zero effect on trace counts because:

1. **Read-read pairs don't generate wakeup insertions.** The only actual conflict
   in the shared-vars tests is the Write-Write race on `_Slot.value`. The iterator
   reads on `list.__cmethods__` are all Read (or WeakRead), which never conflict
   with each other. No matter how many reads are reported, they never trigger
   `insert_wakeup()`.

2. **The overcounting came from the trace cache, not from access reports.** Fix 6
   (trace cache merge) was the actual fix because it corrected a bug where the
   SLEEP SET was incorrectly waking threads. The trace cache merged Read + WeakRead
   into Write, making the engine think a sleeping thread *writes* to the container.
   This caused false wakeups → false wakeup insertions → extra executions.

3. **Fixes 3+4+5 are still worthwhile** as optimizations: they reduce the volume
   of accesses the DPOR engine processes per execution, which speeds up each
   execution even if the number of executions doesn't change.

## Implementation Notes

The remaining overcounting falls into two categories:

1. **Lock-related**: 3 instead of 2 for all lock tests. Root cause is the
   separate lock release race dimension (Root Cause 2). Fix 2 (sound variant)
   would address this.

2. **Multi-variable iterator**: 6 instead of 4 for 2 vars, 18 instead of 8
   for 3 vars. Root cause is likely residual trace cache imprecision or
   per-branch merge escalation (Fix 8). The pattern 9/3=3, 6/2=3 → the
   N=1 case is fixed but N>1 still has an overcounting factor that grows.
   For N=2, 6/4=1.5×; for N=3, 18/8=2.25×. This matches the baseline
   pattern 9/4=2.25 → the fix reduced the factor but didn't eliminate it.
