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

**Caveat — pure-read loops**: When the loop body has no writes (no scheduling points
inside the body), batching removes ALL mid-iteration scheduling points. If another
thread writes to the iterated container between iterations, the interleaving where
some elements are read before the write and others after is not explored. However,
this interleaving is a distinct Mazurkiewicz class only for the elements whose read
ordering relative to the write differs — and the `__cmethods__` race at GET_ITER
still distinguishes "all reads before write" from "write before all reads." In
practice, the test cases always have writes in the loop body (STORE_ATTR), which
preserves mid-iteration scheduling points.

### Fix 5: Collate Repeated Reads of the Same Key (Medium, High Impact)

**Idea**: When consecutive scheduling points by the same thread include reads of the
exact same object_id (same Python object + same key), record all of them at the
**first** read's path position rather than creating separate wakeup insertion sites
for each. Different keys remain at their own scheduling points.

**Motivation — cascade elimination**: When an object key is involved in a conflict,
DPOR inserts wakeups at each scheduling point where it's read. With N reads of the
same key, `process_access` cascades backward — the write races with the last read,
triggering re-exploration, where it races with the next-to-last, etc. This cascade
produces up to N+1 executions. Collation reduces this to 2 (write before all reads,
or after all reads).

**Target pattern**: The `__cmethods__` key on container objects is read at LOAD_ATTR
(as WKREAD), GET_ITER, every FOR_ITER, and FOR_ITER(exhausted). For 1 variable, that's
4 reads of `(list, "__cmethods__")` across 4 scheduling points. Collation merges them
to 1. The per-element reads `(list, "0")`, `(list, "1")` are different keys and stay
at their own positions.

**Soundness**: Collating reads of the same key is **unsound in general** — it misses
the Mazurkiewicz class where a write interposes between two reads:

| Class | Ordering | Thread A observes |
|-------|----------|-------------------|
| 1 | WRITE, READ₁, READ₂ | new, new |
| 2 | READ₁, WRITE, READ₂ | old, new (missed!) |
| 3 | READ₁, READ₂, WRITE | old, old |

Collation keeps classes 1 and 3 but eliminates class 2.

**Can we scope collation to be sound?** The TOCTOU scenario requires user code that
reads the same (object, key) at two scheduling points — e.g., `x = s.obj; y = s.obj`.
However, the intervening STORE_FAST creates a scheduling point gap. If "adjacent" is
defined strictly as "no intervening scheduling point (even a non-reporting one)," then
user-code double-reads are never collated. The repeated `__cmethods__` reads across
GET_ITER → FOR_ITER ARE truly adjacent (no intervening opcode), so they would be
collated.

But this adjacency definition is fragile — it depends on CPython bytecode details and
could break across Python versions. A more robust approach:

**Scope to instrumentation-generated reads only**: Only collate `__cmethods__` reads
that are generated by the iterator instrumentation (GET_ITER, FOR_ITER handlers), not
arbitrary same-key reads. This is a whitelist approach: the trace handler knows which
reads are instrumentation artifacts vs user-visible operations. Since `__cmethods__`
is a synthetic key that never appears in user code, any read of it is by definition
an instrumentation artifact.

This makes collation **sound for all user-observable behavior**: the missed TOCTOU
class (2 above) can only occur on `__cmethods__`, which is invisible to user code.
The program cannot branch on whether a `__cmethods__` read sees old or new, because
the program never reads `__cmethods__` directly.

**Alternative — just use Fix 4**: Fix 4 (batch iterators) already eliminates the
FOR_ITER scheduling points, which are the main source of repeated `__cmethods__`
reads. The remaining reads (LOAD_ATTR + GET_ITER) are only 2 scheduling points,
making the cascade depth 3 instead of N+1. This avoids the soundness question
entirely, at the cost of a slightly larger (but constant) cascade.

**Implementation**: In the Python trace handler, track the last-reported (object, key)
per thread. When the next opcode reports a read on the same (object, key), use
`process_first_access` (which already exists — `engine.rs:194`) to pin the read to the
first position rather than creating a new scheduling point. Alternatively, simply skip
the `report_and_wait` for the duplicate read entirely, since the first read already
provides the wakeup insertion site.

**Impact**: For the iterator pattern, reduces the cascade from O(N) scheduling points
on `__cmethods__` to O(1). Combined with Fix 4 (batch FOR_ITER yields), the
`for v in s.vars: v.value = tid` pattern drops from ~5 scheduling points per variable
to ~2 (one for the collated container read, one for STORE_ATTR).

## Priority Ordering

1. **Fix 2** (collapse lock races): medium impact for lock tests specifically.
2. **Fix 5** (collate same-key reads): high impact on cascade depth; unsound in
   general (misses same-key TOCTOU). Consider combining with Fix 4 instead.
3. **Fix 4** (batch iterators): medium impact for loop-heavy code.
4. **Fix 3** (eliminate weak reads): lowest priority, small incremental improvement.

## Implementation Notes

Fixes 2–5 all reduce scheduling points or wakeup insertion sites on objects that
genuinely participate in conflicts. This is where overcounting comes from: each
scheduling point on a conflicting object is a potential wakeup insertion site, and each
wakeup means another full execution.
