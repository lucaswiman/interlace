# Reducing DPOR Trace Overcounting

## Problem Statement

The exact Mazurkiewicz trace count tests (`test_exact_mazurkiewicz_trace_count.py`)
reveal that the DPOR implementation explores more traces than the theoretical
minimum for several test families. After removing initial thread rotation (which
was redundant — see "Resolved" section), the remaining overcounting is:

| Test | Expected | Baseline | After Fixes 3+4+5+6 | After Fix 8 | Factor |
|------|----------|----------|---------------------|-------------|--------|
| 2 threads, 1 shared var (loop `for v in s.vars`) | 2 | 3 | **2** ✅ | **2** ✅ | 1.0× |
| 2 threads, 2 shared vars (loop) | 4 | 9 | 6 | 6 | 1.5× |
| 2 threads, 3 shared vars (loop) | 8 | 27 | 18 | 18 | 2.25× |
| N=2 threads, lock | 2 | 3 | 3 | **2** ✅ | 1.0× |
| 2 threads, lock over 1 var | 2 | 3 | 3 | **2** ✅ | 1.0× |
| 2 threads, lock over 2 vars | 2 | 3 | 3 | **2** ✅ | 1.0× |
| 2 threads, lock over 3 vars | 2 | 3 | 3 | **2** ✅ | 1.0× |
| N=3 threads, lock | 6 | 17 | 17 | **6** ✅ | 1.0× |

This document analyzes the root causes and proposes fixes.

## Scope: "Optimal" Counts Depend on the Event Model

One important clarification: **exact Mazurkiewicz trace counts are defined relative
to an event alphabet**. The theoretical counts in
`test_exact_mazurkiewicz_trace_count.py` treat each logical operation as a single
event:

- writing one `_Slot.value`
- iterating one shared list element
- entering/leaving one critical section

The current DPOR implementation does **not** operate on that alphabet. It explores
at the Python opcode level, with some higher-level synthetic events layered on top
(e.g. lock acquire/release via `report_sync()`).

That means there are two distinct sources of "extra" traces:

1. **Real over-exploration caused by DPOR imprecision**: false wakeups, overly
   conservative merges, coarse future summaries.
2. **Model mismatch**: the engine is exploring a finer-grained event system than
   the abstract proofs in the tests assume.

Fixing (1) can reduce clearly spurious executions. Eliminating (2) requires either
coarsening the event model or making the tests' notion of "exact" match the opcode
alphabet.

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

## Root Cause 1: Event-Model Mismatch From Multi-Opcode Decomposition

### Attribute Set: `obj.attr = value`

The Python statement `s.slot.value = tid` compiles to:

```
LOAD_SMALL_INT  tid       # push value
LOAD_FAST       s         # push object
LOAD_ATTR       slot      # READ(State, "slot") + WKREAD(_Slot, "__cmethods__")
STORE_ATTR      value     # WRITE(_Slot, "value")
```

What the abstract model treats as one atomic write operation, the DPOR system sees
as **two scheduling points** (LOAD_ATTR and STORE_ATTR). This does **not** by itself
prove a DPOR bug: if sleep sets and wakeups were perfectly precise, many such extra
positions would still collapse to the same Mazurkiewicz class under the opcode-level
alphabet.

However, the finer-grained decomposition makes any imprecision more expensive:
additional positions create more places where a false wakeup or conservative merge
can open an unnecessary branch.

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
STORE_ATTR, FOR_ITER-exhausted), versus the abstract model's 1. Again, this is best
understood as an **amplifier** of overcounting rather than a proven root cause on its
own.

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

For built-in locks this is partly redundant with the synthetic synchronization events:
the real lock ordering is already modeled by `report_sync(lock_acquire/lock_release)`.
So some of these Python-level reports are modeling noise rather than essential
semantic events.

## Root Cause 2: Lock Modeling Via `io_vv` Is Intentionally Conservative

Lock operations use `io_vv` (which excludes lock-based happens-before edges) to detect
races. This is intentional: it ensures TOCTOU races across lock boundaries are caught.
However, it creates **two independent synthetic race dimensions** for each lock:

1. Lock acquire WRITE(lock_io_obj) — races between all threads
2. Lock release WRITE(lock_rel_io_obj) — races between all threads

Both use different virtual object IDs (XOR with `LOCK_OBJECT_XOR` and
`LOCK_RELEASE_XOR` respectively). Each creates a write-write race between threads.
This is a pragmatic over-approximation: good for bug-finding, but not a principled
representation of lock semantics if the goal is exact trace counts.

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

## Proven Exact Under The Current Python Event Alphabet

### Case: 2 Threads, 1 Shared Var Via `for v in s.vars`

After Fix 6, this case reaches the exact minimum of **2 traces even under the
current opcode-level event model**.

Each thread executes the following shared events:

1. `LOAD_ATTR vars` on the `State` object: `READ(State, "vars")`
2. weak read of the list object from `LOAD_ATTR vars`: `WEAK_READ(list, "__cmethods__")`
3. `GET_ITER`: `READ(list, "__cmethods__")` (first-access semantics)
4. `FOR_ITER` for element 0: `READ(list, "0")` and `READ(list, "__cmethods__")`
5. `STORE_ATTR value`: `WRITE(_Slot, "value")`
6. exhausted `FOR_ITER`: `READ(list, "1")` and `READ(list, "__cmethods__")`

Let thread A's events be `a1, a2, a3, a4, aw, a5` and thread B's events be
`b1, b2, b3, b4, bw, b5`, where `aw` and `bw` are the `STORE_ATTR value`
events on the shared `_Slot`.

**Dependency relation under the current alphabet**:

- `aw` and `bw` are dependent: both are `WRITE(_Slot, "value")`.
- Every other cross-thread pair is independent:
  - `READ(State, "vars")` vs `READ(State, "vars")` is read-read.
  - `WEAK_READ(list, "__cmethods__")` vs `READ/WEAK_READ(list, "__cmethods__")`
    is non-conflicting.
  - `READ(list, "0")` / `READ(list, "1")` across threads is read-read.
  - None of the list/state reads share an object key with `_Slot.value`.

So the only cross-thread dependent pair is `aw` vs `bw`.

**Upper bound <= 2**:

Two executions with the same order of `aw` and `bw` differ only in the placement of
cross-thread read events. Since all those cross-thread read events are independent,
they can be swapped past each other by Mazurkiewicz commutations until the executions
match. Therefore a trace is determined entirely by whether `aw < bw` or `bw < aw`.

**Lower bound >= 2**:

Both orders are realizable by scheduling either thread's `STORE_ATTR` first. Since
`aw` and `bw` are dependent, those two orders belong to different traces.

Therefore the exact count under the current Python event alphabet is **2**.

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

### Fix 2: Suppress Redundant Python-Level Lock Metadata Reports

**Idea**: For lock types that already emit `report_sync(lock_acquire/lock_release)`,
avoid also treating Python-level metadata lookup as independent shared-memory events.
In particular:

- `LOAD_ATTR lock` on the owning object
- `LOAD_SPECIAL __enter__`
- `LOAD_SPECIAL __exit__`

are mostly artifact of how `with lock:` is compiled, not meaningful semantic events
for the lock-count tests.

**Why this is sounder than collapsing release races**: It removes redundant
front-end noise while keeping the actual synthetic synchronization events intact.
We still preserve the acquire/release backtrack points that are currently needed for
multi-lock race detection.

**Likely impact**: Reduces lock-test overcounting and shrinks the number of branch
positions around `with lock:` without weakening the lock model itself.

### Fix 3: Position-Sensitive Future Access Cache

**Idea**: Replace `prev_thread_all_accesses` with a more precise cache keyed by
position as well as thread, e.g.:

```text
future_accesses[(path_pos, thread_id)] = union of that thread's accesses
from its next step onward in the previous execution
```

`propagate_sleep()` would then use the summary appropriate to the current replay
prefix, rather than the union of **all** accesses the thread performed anywhere in
the previous execution.

**Why it helps**: the current cache is sound but very coarse. For `for v in s.vars`,
once a thread has ever touched all slots in a prior execution, the cache makes it
look like the sleeping thread may immediately touch all of them again. That creates
false wakeups at later positions.

**Soundness**: This remains conservative as long as each cached summary is a superset
of what the thread can still do from that replay position onward.

**Likely impact**: Best candidate for reducing the remaining `2 vars -> 6` and
`3 vars -> 18` overcounting in the shared-vars tests.

### Fix 4: Provenance-Tagged Access Summaries

**Idea**: Extend `active_accesses` / trace-cache entries to track not just
`object_id -> AccessKind`, but also the origin of the access:

- Python-memory access
- synthetic lock access
- real I/O access

**Why this matters**: today the path layer loses that distinction, so any policy like
"merge more precisely for non-I/O but stay conservative for I/O" cannot actually be
implemented cleanly.

**Likely impact**: Enables safer follow-on fixes, especially around the regression
seen when relaxing the per-branch merge.

### Fix 5: Improve AccessKind Merge for WeakWrite + WeakRead

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

### Fix 6: Per-Branch Merge Investigation

**Idea**: Apply `AccessKind::merge()` to the per-branch `record_access()` as well,
not just the trace cache. This would fix `active_accesses` at the scheduling-point
level, preventing the Read + WeakRead → Write escalation within a single scheduling
point from propagating through sleep set checks.

**Status**: Attempted and caused a regression in `test_independent_file_writes[2]`
(independent writes got 2 instead of 1). The root cause of the regression needs
investigation — it may be that per-branch merge interacts with I/O access tracking
(`process_io_access` uses the same `record_access` path).

**Potential approach**:

1. First add provenance tags (Fix 4), because the current path data structure does
   not distinguish I/O from non-I/O.
2. Then selectively relax only Python-memory merges while keeping I/O-originated
   accesses conservative if needed.
3. Also investigate whether the failing file-I/O case exposed a real pre-existing
   bug that the conservative Write merge happened to mask.

### Fix 7: Reduce Scheduling Points for Redundant Shared Opcodes

**Idea**: The scheduler already coarsens many thread-local opcodes. The remaining
opportunity is not a blanket "skip more opcodes" pass, but narrower suppression of
shared opcodes that are redundant with stronger semantic events. Examples:

- lock metadata lookups already covered by `report_sync()`
- repeated container-iteration reads already pinned to first access
- known immutable-object lookups already covered by `_IMMUTABLE_TYPES`

**Caution**: A broad "potentially-written objects" optimization is easy to make
unsound. The promising version is targeted and semantics-aware, not a global
analysis pass.

### Fix 8: Lock-Aware DPOR via Deferred Release Backtracking (IMPLEMENTED)

**Previous idea**: Remove lock acquire/release from the generic `io_vv` machinery and model
them directly as synchronization events in the DPOR engine.

`io_vv` is a good fit for interactions we do not control directly (file/socket/DB
activity observed from the side). Locks are different:

- lock state is fully under the scheduler's control
- blocked/runnable transitions are already explicit
- the relevant happens-before edges are deterministic
- the "resource" is not an opaque external object; it is a synchronization primitive

So treating `lock_acquire` / `lock_release` as fake I/O writes to virtual objects is
not conceptually aligned with the rest of the design.

**Goal**: preserve the useful multi-lock bug-finding behavior while replacing the
current lock-object approximation with a model that matches lock semantics more
directly and gives cleaner trace counts.

#### Concrete implementation plan

**Phase 1: Separate lock races from I/O races in the engine API**

Add a dedicated lock-sync path instead of routing lock events through
`process_io_access_at()`:

- keep `process_sync()` for HB updates and runnable/blocking state
- add a dedicated helper, e.g. `process_lock_event(...)`
- stop XOR-ing `lock_id` into fake object IDs for acquire/release
- stop using `io_vv` for lock events entirely

This makes the code reflect the actual distinction:

- `process_io_access*()` is for external effects whose ordering we observe
- `process_lock_event()` is for synchronization primitives we control

**Phase 2: Represent lock events explicitly in the path**

Extend the path/branch representation with lock-event metadata for each scheduling
position, e.g.:

```text
LockEvent {
    thread_id,
    lock_id,
    kind: Acquire | Release,
}
```

Record these in parallel with `active_accesses`, but do **not** merge them into the
 ordinary object-access maps. They have different semantics and should participate in
independence checks separately.

**Phase 3: Define lock-specific dependency rules**

Use lock-aware dependence instead of generic object-access dependence.

For the same lock `L`:

- `Acquire(L)` vs `Acquire(L)` are dependent
- `Release(L)` vs `Acquire(L)` are dependent
- `Release(L)` vs `Release(L)` are usually not a useful independent race dimension
  for exact counting; in normal mutex semantics they can be treated as dependent only
  insofar as they order future acquires, not as standalone conflicting "writes"

For different locks `L1 != L2`:

- `Acquire(L1)` vs `Acquire(L2)` are independent
- `Release(L1)` vs `Acquire(L2)` are independent at the primitive level

That last rule is the critical difference from the current `io_vv` model. The
multi-lock bugs do **not** fundamentally require pretending that releasing `L1` races
with acquiring `L2`. What they require is that DPOR can backtrack to the boundary
between adjacent critical sections in the same thread.

**Phase 4: Preserve inter-critical-section backtracking explicitly**

The current virtual-object trick uses release/acquire "races" to create a scheduling
point between:

```python
with lock_a:
    ...
with lock_b:
    ...
```

Instead of faking that via `io_vv`, add an explicit backtracking rule at lock-section
boundaries:

- when thread T executes `Release(L)` and remains runnable
- and another thread U is runnable immediately after that release
- record a backtrack opportunity at the release position to allow U to run before
  T's next action

This is closer to what we actually want:

- not "release writes an external resource"
- but "the end of a critical section is a semantically important scheduling boundary"

In other words, use the path/wakeup machinery to preserve post-release alternatives
directly, instead of encoding them as fake data races.

**Phase 5: Incorporate lock events into sleep-set propagation**

Update the independence check used by `propagate_sleep()` so that it consults both:

- normal access summaries for shared-memory / I/O objects
- lock-event summaries for synchronization behavior

This likely needs a second summary structure, analogous to `active_accesses`, but for
future lock events:

```text
future_lock_events[(path_pos, thread_id)] = summary of remaining lock actions
```

At minimum, the summary needs:

- which locks may be acquired next
- whether the thread may perform a release that ends a critical section while
  remaining runnable

This is the lock analogue of Fix 3's position-sensitive future-access cache.

**Phase 6: Remove lock-specific overcounting from the frontend**

Once lock events are modeled explicitly in the engine:

- suppress redundant Python-level lock metadata reports (`LOAD_ATTR lock`,
  `LOAD_SPECIAL __enter__`, `LOAD_SPECIAL __exit__`) for lock types covered by the
  lock-aware path
- keep the Python-level reports only for objects that do not participate in the
  explicit lock-sync path

That should eliminate a large amount of front-end lock noise while leaving the actual
lock semantics intact.

#### Why this should still catch the multi-lock bugs

Consider the motivating pattern:

```python
with lock_a:
    write_a()
with lock_b:
    write_b()
```

and a competing thread:

```python
with lock_a:
    read_a()
with lock_b:
    read_b()
```

The bug exists because another thread can run **after** `Release(lock_a)` and
**before** the next `Acquire(lock_b)`.

The lock-aware plan preserves this by:

- treating `Release(lock_a)` as a first-class scheduling boundary
- adding an explicit backtrack opportunity at that boundary when another thread is
  runnable
- continuing to use normal shared-memory races inside the critical sections for
  `write_a/read_a` and `write_b/read_b`

So the multi-lock behavior does not depend on fake `io_vv` conflicts between
different lock objects; it depends on preserving the release boundary as a place
where another runnable thread may take over.

#### Minimum migration strategy

To reduce risk, implement this in two steps:

1. **Dual-path stage**
   Keep the current `io_vv` lock modeling behind a flag, add the lock-aware path
   behind another flag, and run both implementations against the same tests.
2. **Cutover stage**
   Once the lock-aware path passes the lock-count tests and the multi-lock defect
   tests, remove the `io_vv` lock route.

#### Test plan

The minimum test matrix for this change should include:

- `test_n_threads_single_lock`
- `test_two_threads_locked_n_vars`
- `test_prometheus_summary_observe_pattern`
- `test_transfer_between_two_locked_accounts`
- existing deadlock / lock-wait tests

Add one new focused unit test at the Rust-path layer for the key behavior:

- when a thread releases `lock_a` and immediately continues to unrelated work,
  another runnable thread must still be insertable at that release boundary even
  though there is no synthetic object race

#### Expected payoff

- More principled lock semantics
- Cleaner exact-count behavior for lock-only tests
- Less dependence on `io_vv` for phenomena that are not actually I/O
- A better foundation for future lock-specific optimizations than the current
  acquire/release-as-virtual-write approximation

**Original risk assessment**: High. This is still a larger algorithmic change than the
fixes above, but unlike the current approximation it has a clear correctness story.

#### What was actually implemented

The 6-phase plan above was the original design. The actual implementation was much
simpler — only a subset of Phase 1 and Phase 4 was needed:

1. **Lock acquire**: unchanged (still a Write to `LOCK_OBJECT_XOR`-masked virtual
   object via `io_vv`), plus records a `DeferredLockAcquire { thread_id, path_id }`.
2. **Lock release**: only does the HB update (`lock_release_vv`), NO `io_vv` access.
   Records a `DeferredLockRelease { thread_id, path_id }`. The `LOCK_RELEASE_XOR`
   constant was removed.
3. **At execution end** (`next_execution()`), `process_deferred_lock_releases()` checks:
   for each release by thread T at position P, did T later acquire any lock at
   position P' > P? If so, insert backtrack opportunities at position P for all other
   threads via `path.insert_wakeup()`.

**Key insight**: Acquire-acquire races (via the existing `LOCK_OBJECT_XOR` mechanism)
already handle lock ordering correctly. The only missing piece was inter-critical-section
backtracking — which the deferred release mechanism provides precisely. Phases 2
(explicit lock events in path), 3 (lock-specific dependency rules), 5 (lock events in
sleep-set propagation), and 6 (suppress frontend noise) were all unnecessary.

**Impact**:
- `test_n_threads_single_lock[2]`: 3 → **2** ✅ exact
- `test_n_threads_single_lock[3]`: 17 → **6** ✅ exact
- `test_two_threads_locked_n_vars[1,2,3]`: 3 → **2** ✅ exact
- Multi-lock defect tests: PASS (Prometheus summary, bank transfer)
- Full test suite: zero regressions

## Priority Ordering (Updated)

1. **Fix 3**: Position-sensitive future access cache. Best shot at reducing the
   remaining shared-vars overcounting without changing the event model.
2. **Fix 2**: Suppress redundant Python-level lock metadata reports. May still help
   with file-I/O lock overcounting (`test_n_threads_locked_file_writes`).
3. **Fix 4 + Fix 6**: Add provenance tags, then revisit per-branch merge
   relaxation with enough information to do it safely.
4. **Fix 5**: Extend merge function for WeakWrite + WeakRead. Small incremental.
5. ~~**Fix 8**~~: ✅ Done. Lock trace counts are now exact.

## Analysis: Why Fixes 3+4+5 Had Zero Impact

Fixes 3, 4, and 5 reduce the number and kind of access reports, but they had
zero effect on trace counts because:

1. **Read-read pairs don't generate wakeup insertions.** The only actual conflict
   in the shared-vars tests is the Write-Write race on `_Slot.value`. The iterator
   reads on `list.__cmethods__` are all Read (or WeakRead), which never conflict
   with each other. No matter how many reads are reported, they never trigger
   `insert_wakeup()`.

2. **The demonstrated overcounting came from the trace cache, not from access reports.** Fix 6
   (trace cache merge) was the actual fix because it corrected a bug where the
   SLEEP SET was incorrectly waking threads. The trace cache merged Read + WeakRead
   into Write, making the engine think a sleeping thread *writes* to the container.
   This caused false wakeups → false wakeup insertions → extra executions.

3. **Fixes 3+4+5 are still worthwhile** as optimizations: they reduce the volume
   of accesses the DPOR engine processes per execution, which speeds up each
   execution even if the number of executions doesn't change.

## Implementation Notes

The remaining overcounting falls into two categories:

1. ~~**Lock-related**~~: ✅ Fixed by Fix 8 (deferred release backtracking). All
   lock-only tests now produce exact Mazurkiewicz trace counts.

2. **File-I/O lock overcounting**: `test_n_threads_locked_file_writes` still shows
   3 instead of 2 (N=2) and 17 instead of 6 (N=3). These use file I/O which still
   goes through the dual virtual-object model. Fix 2 (suppress redundant lock
   metadata) may help here.

3. **Multi-variable iterator**: 6 instead of 4 for 2 vars, 18 instead of 8
   for 3 vars. Root cause is likely residual future-summary imprecision or
   per-branch merge escalation (Fix 3 / Fix 6). The pattern 9/3=3, 6/2=3 → the
   N=1 case is fixed but N>1 still has an overcounting factor that grows.
   For N=2, 6/4=1.5×; for N=3, 18/8=2.25×. This matches the baseline
   pattern 9/4=2.25 → the fix reduced the factor but didn't eliminate it.

4. **Event-model mismatch**: even after fixing clear DPOR imprecision, some gap may
   remain because the engine explores opcode-level events while the tests reason
   about higher-level logical operations. That is a limitation of the current
   instrumentation model, not a small fix item in this plan.
