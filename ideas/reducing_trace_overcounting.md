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

### Fix 1: Skip Scheduling Points on Non-Conflicting Reads — UNSOUND in General

**Original idea**: After executing a non-conflicting opcode (pure read that creates no
races with prior accesses), skip the scheduler yield and continue executing the same
thread. Only yield when an opcode actually creates or could create a conflict.

**Why this is unsound**: A read that has no conflict with *prior* accesses in the current
execution may still race with a *later* write from another thread. DPOR detects such
races when the later write's `process_access` finds the earlier read — but only if the
read was recorded as a scheduling point. If the read was skipped, the write has no
dependent prior access to race against, and the interleaving where the write precedes the
read is never explored.

Concrete example:
```python
def thread1():
    shared.x = 1
    _ = shared.y         # READ(y) — no prior write to y

def thread2():
    if shared.x == 0:
        shared.y = 99    # WRITE(y) — conditional on interleaving
```
In a T1-first execution, `shared.y` is never written. But DPOR discovers the race on `x`
and explores T2-first, where T2 *does* write `y`. In that execution, T1's READ(y) follows
T2's WRITE(y), so it has a prior write and yields normally — the race on `y` is detected.
So far so good. But DPOR then needs to explore the interleaving where T1 READ(y) happens
*before* T2 WRITE(y). It inserts a wakeup at T1 READ(y)'s position. In the re-exploration
of that prefix, T2's WRITE(y) hasn't happened yet, so READ(y) again sees no prior write.
Under Fix 1, it would be skipped — making the wakeup impossible to honor.

**Sound variant — never-written objects**: Skipping scheduling points on reads of objects
that have never been written in *any* explored execution is technically sound: if no
WRITE(Y) exists in any trace, READ(Y) can never race with anything, so DPOR would never
insert a wakeup at that position anyway. The scheduling point is dead weight.

However, this optimization **does not reduce overcounting** — precisely because DPOR
never inserts wakeups at those points, they don't generate additional traces. The overhead
is limited to the per-access cost of `process_access` + yield (which immediately returns
the same thread), negligible compared to the cost of full execution replays.

**Conclusion**: Fix 1 as originally proposed is unsound. The sound variant (never-written
objects) is correct but doesn't address overcounting. The real overcounting comes from
excess scheduling points on objects that *do* have conflicts — addressed by Fixes 2–4.

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

1. **Fix 2** (collapse lock races): medium impact for lock tests specifically.
2. **Fix 4** (batch iterators): medium impact for loop-heavy code.
3. **Fix 3** (eliminate weak reads): lowest priority, small incremental improvement.
4. ~~**Fix 1**~~: unsound in general; sound variant doesn't reduce overcounting. See above.

## Implementation Notes

The remaining fixes (2–4) all reduce scheduling points on objects that genuinely
participate in conflicts. This is where overcounting comes from: each scheduling point on
a conflicting object is a potential wakeup insertion site, and each wakeup means another
full execution. Non-conflicting reads (Fix 1's target) never receive wakeups regardless,
so they don't multiply the execution count — only the per-execution cost.
