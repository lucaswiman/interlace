# Future Work

## Cooperative Wrappers for Threading Primitives

Currently, only `threading.Lock` has a cooperative wrapper (`_CooperativeLock` in
`bytecode.py`). Under opcode-level scheduling, any blocking call that occurs in
C code is invisible to the scheduler — the blocked thread holds a scheduler turn
but can't make progress, causing a deadlock. The cooperative lock solves this for
`Lock` by spinning with non-blocking attempts and yielding scheduler turns
between each attempt.

The same pattern needs to be extended to all commonly-used threading primitives.

### The General Pattern

Every cooperative wrapper follows the same structure:

1. Try the non-blocking variant of the operation.
2. If it succeeds, return immediately.
3. If it fails, call `scheduler.wait_for_turn(thread_id)` to yield.
4. Go to step 1.

Each primitive has a non-blocking equivalent:

| Primitive             | Blocking call | Non-blocking equivalent                   |
|-----------------------|---------------|-------------------------------------------|
| `threading.Lock`      | `.acquire()`  | `.acquire(blocking=False)` *(done)*       |
| `threading.RLock`     | `.acquire()`  | `.acquire(blocking=False)`                |
| `threading.Semaphore` | `.acquire()`  | `.acquire(blocking=False)`                |
| `threading.Event`     | `.wait()`     | `.wait(timeout=0)`                        |
| `threading.Condition` | `.wait()`     | `.wait(timeout=0)`                        |
| `queue.Queue`         | `.get()`      | `.get(block=False)` + catch `queue.Empty` |
| `queue.Queue`         | `.put()`      | `.put(block=False)` + catch `queue.Full`  |
| `threading.Barrier`   | `.wait()`     | No non-blocking variant (see below)       |

### Implementation Plan

#### Phase 1: RLock

**Priority: High** — `RLock` is the second most common lock primitive.

`threading.RLock` is a reentrant lock: the same thread can acquire it multiple
times without deadlocking. The cooperative wrapper must:

- Track the owning thread (use `threading.get_ident()`).
- Track the recursion count.
- On `acquire()`: if the current thread already owns the lock, increment the
  count and return immediately. Otherwise, spin-yield as with `_CooperativeLock`.
- On `release()`: decrement the count; only release the underlying lock when the
  count reaches zero.

```python
class _CooperativeRLock:
    def __init__(self):
        self._lock = _real_lock()
        self._owner = None
        self._count = 0

    def acquire(self, blocking=True, timeout=-1):
        me = threading.get_ident()
        if self._owner == me:
            self._count += 1
            return True
        # ... spin-yield loop same as _CooperativeLock ...
        self._owner = me
        self._count = 1
        return True

    def release(self):
        if self._owner != threading.get_ident():
            raise RuntimeError("cannot release un-acquired lock")
        self._count -= 1
        if self._count == 0:
            self._owner = None
            self._lock.release()
```

Monkey-patching: save `_real_rlock = threading.RLock` at module load, patch
`threading.RLock = _CooperativeRLock` in `_patch_locks()`.

#### Phase 2: Semaphore and BoundedSemaphore

**Priority: Medium** — Used in connection pools, rate limiters, and resource
management.

`threading.Semaphore` is a counter-based primitive. `acquire()` blocks when the
counter is zero. The cooperative wrapper:

- Wraps a real `Semaphore` instance.
- `acquire()`: try `self._sem.acquire(blocking=False)`. If it fails, spin-yield.
- `release()`: delegate directly to `self._sem.release()`.

`BoundedSemaphore` is the same but raises on over-release — the wrapper can
delegate to a real `BoundedSemaphore` for that check.

```python
class _CooperativeSemaphore:
    def __init__(self, value=1):
        self._sem = _real_semaphore(value)

    def acquire(self, blocking=True, timeout=-1):
        if self._sem.acquire(blocking=False):
            return True
        if not blocking:
            return False
        # ... spin-yield loop ...
```

Monkey-patching: `threading.Semaphore = _CooperativeSemaphore` and
`threading.BoundedSemaphore = _CooperativeBoundedSemaphore`.

#### Phase 3: Event

**Priority: Medium** — Common for signaling between threads (e.g., "init
complete", "shutdown requested").

`threading.Event` uses an internal `Condition` and flag. The cooperative wrapper:

- `wait(timeout=None)`: spin-yield checking `self._event.wait(timeout=0)` until
  the event is set or timeout elapses.
- `set()`, `clear()`, `is_set()`: delegate directly.

```python
class _CooperativeEvent:
    def __init__(self):
        self._event = _real_event()

    def wait(self, timeout=None):
        if self._event.is_set():
            return True
        # ... spin-yield loop, respecting timeout ...

    def set(self):
        self._event.set()
```

Monkey-patching: `threading.Event = _CooperativeEvent`.

#### Phase 4: queue.Queue

**Priority: High** — `queue.Queue` is the primary inter-thread communication
primitive in idiomatic Python. Many concurrency bugs involve queue interactions
(e.g., TOCTOU between `empty()` and `get()`, or producer-consumer ordering).

The cooperative wrapper needs to handle both `get()` and `put()`:

- `get(block=True, timeout=None)`: try `self._queue.get(block=False)`. If
  `queue.Empty`, spin-yield.
- `put(item, block=True, timeout=None)`: try `self._queue.put(item, block=False)`.
  If `queue.Full`, spin-yield.
- All other methods (`qsize()`, `empty()`, `full()`, `task_done()`, `join()`):
  delegate directly.

```python
class _CooperativeQueue:
    def __init__(self, maxsize=0):
        self._queue = _real_queue(maxsize)

    def get(self, block=True, timeout=None):
        try:
            return self._queue.get(block=False)
        except queue.Empty:
            if not block:
                raise
            # ... spin-yield loop ...

    def put(self, item, block=True, timeout=None):
        try:
            self._queue.put(item, block=False)
        except queue.Full:
            if not block:
                raise
            # ... spin-yield loop ...
```

Monkey-patching: `queue.Queue = _CooperativeQueue`. Also provide wrappers for
`queue.LifoQueue` and `queue.PriorityQueue` (they subclass `Queue`, so the
cooperative version should mirror that hierarchy).

Note: `queue.Queue` internally uses `threading.Condition` and `threading.Lock`.
If we patch Lock/Condition first, Queue *may* work without its own wrapper. But
an explicit wrapper is still preferred because it lets us avoid spinning through
Queue's internal Condition machinery opcode-by-opcode — the wrapper can spin at
the semantic level (`get`/`put`) which is much cheaper.

#### Phase 5: Condition

**Priority: Low** — Most user code doesn't use `Condition` directly (it's used
internally by `Queue`, `Event`, etc.). But code that does use it needs a wrapper.

This is the trickiest primitive because **the scheduler itself uses
`threading.Condition`** (see `OpcodeScheduler.__init__`). Patching Condition
globally would cause infinite recursion: the cooperative Condition's spin loop
would call `scheduler.wait_for_turn()`, which acquires the scheduler's own
Condition, which would try to cooperate, and so on.

Mitigation strategies:

1. **Use `_real_condition`**: The scheduler already uses `_real_lock()` for its
   internal lock. Extend this pattern: save `_real_condition = threading.Condition`
   at module load, and have the scheduler use `_real_condition`. The cooperative
   wrapper only affects user code.

2. **Guard with `_active_scheduler` TLS**: In the cooperative Condition, check
   whether we're in a managed thread. If `_active_scheduler.scheduler is None`,
   fall back to real blocking (same pattern as `_CooperativeLock` line 160-162).

The cooperative wrapper needs to handle:
- `wait(timeout=None)`: release the associated lock, spin-yield until notified or
  timeout, re-acquire the lock.
- `wait_for(predicate, timeout=None)`: spin-yield until predicate is true.
- `notify(n=1)` and `notify_all()`: delegate directly (they don't block).

```python
class _CooperativeCondition:
    def __init__(self, lock=None):
        self._lock = lock or _CooperativeLock()
        self._waiters = []

    def wait(self, timeout=None):
        # Release lock, add self to waiters, spin-yield, re-acquire
        ...
```

#### Phase 6: Barrier

**Priority: Low** — Rarely used in application code.

`threading.Barrier` has no non-blocking variant for `wait()`. Two approaches:

1. **Spin on internal state**: Access `barrier._count` or similar internals to
   check if all parties have arrived, and spin-yield until they have. Fragile
   (depends on CPython internals).

2. **Reimplement using cooperative primitives**: Build a `_CooperativeBarrier`
   from `_CooperativeLock` and `_CooperativeCondition`. This is cleaner but
   requires Phase 5 to be complete first.

### Patching Infrastructure

The current `_patch_locks()` / `_unpatch_locks()` methods in `BytecodeInterlace`
need to be extended to handle all primitives. Consider:

- A registry of `(module, attr_name, real_factory, cooperative_factory)` tuples.
- A single `_patch_all()` / `_unpatch_all()` that iterates the registry.
- Save all real factories at module load time, before any patching can occur.

```python
_real_lock = threading.Lock
_real_rlock = threading.RLock
_real_semaphore = threading.Semaphore
_real_bounded_semaphore = threading.BoundedSemaphore
_real_event = threading.Event
_real_condition = threading.Condition
_real_queue = queue.Queue

_PATCHES = [
    (threading, 'Lock', _CooperativeLock),
    (threading, 'RLock', _CooperativeRLock),
    (threading, 'Semaphore', _CooperativeSemaphore),
    (threading, 'BoundedSemaphore', _CooperativeBoundedSemaphore),
    (threading, 'Event', _CooperativeEvent),
    (threading, 'Condition', _CooperativeCondition),
    (queue, 'Queue', _CooperativeQueue),
    (queue, 'LifoQueue', _CooperativeLifoQueue),
    (queue, 'PriorityQueue', _CooperativePriorityQueue),
]
```

### Risks and Open Questions

1. **Scheduler recursion**: The scheduler's own `threading.Condition` must never
   be patched. Using saved `_real_*` factories for all scheduler internals is
   sufficient, but this invariant must be maintained as the codebase evolves.

2. **Third-party code**: Libraries that create locks at import time (before
   patching) will hold real locks. This is generally fine — the cooperative
   wrappers only affect locks created *during* the controlled run. But it means
   we can't test lock interactions inside third-party code without more invasive
   patching.

3. **Performance**: Each cooperative primitive adds a spin loop. For opcode-level
   scheduling this is acceptable (the scheduler is already calling
   `wait_for_turn()` at every opcode), but the overhead compounds with more
   wrapped primitives.

4. **`timeout` semantics**: Each cooperative wrapper must correctly handle the
   `timeout` parameter. The spin-yield loop needs wall-clock tracking to respect
   caller-specified timeouts.

---

## Deferred: Explicit Cooperative Primitives (Loom-Style)

An alternative to monkey-patching is providing explicit cooperative replacements
that users opt into, inspired by the Rust
[loom](https://github.com/tokio-rs/loom) library.

In loom, instead of `std::sync::Mutex`, you use `loom::sync::Mutex`. Under test,
loom's Mutex cooperates with loom's scheduler. In production, it compiles down to
the real Mutex. The user changes imports, not runtime behavior.

The equivalent for interlace:

```python
# Instead of:
import threading
lock = threading.Lock()

# User writes:
from interlace.sync import Lock
lock = Lock()
```

Under `controlled_interleaving`, `interlace.sync.Lock` behaves like
`_CooperativeLock`. Outside of it, it delegates to `threading.Lock` with zero
overhead.

### Advantages over monkey-patching

- **No global state mutation**: patching `threading.Lock` affects the entire
  process, including the scheduler, unrelated libraries, and other test threads.
  Explicit imports are scoped to the code that uses them.
- **No recursion risk**: the scheduler uses real `threading.Condition` by
  construction — there's nothing to accidentally patch.
- **Clearer intent**: reading the code makes it obvious which primitives are
  under scheduler control.
- **Composable**: multiple test harnesses can coexist without conflicting patches.

### Disadvantages

- **Requires import changes**: can't test unmodified code. This is the main
  drawback — the whole point of the bytecode approach is testing code as-is.
- **Dual-import maintenance**: `interlace.sync` must mirror the `threading` and
  `queue` APIs exactly, and stay in sync as CPython evolves.

### Why deferred

The monkey-patching approach (Phase 1-6 above) is the right first step because
it preserves the library's core value proposition: testing unmodified concurrent
code. The loom-style approach can be layered on later as an opt-in alternative
for users who want tighter control and are willing to change imports. It also
becomes more attractive once the cooperative wrappers exist — `interlace.sync`
would just be thin dispatch wrappers over the same cooperative implementations.

---

## Deferred: Dynamic Partial Order Reduction (DPOR)

The current exploration strategy in `explore_interleavings()` generates random
schedules — essentially a random walk through the interleaving space. This works
well for finding bugs quickly (a random schedule has a reasonable probability of
hitting most concurrency bugs), but it provides no coverage guarantees and may
redundantly explore equivalent interleavings.

**Dynamic Partial Order Reduction (DPOR)** is a family of algorithms that
systematically explore only the *distinct* interleavings — those that differ in
the ordering of conflicting (dependent) operations. Two interleavings that differ
only in the ordering of independent operations (e.g., two threads writing to
different variables) are equivalent and only one needs to be tested.

### How DPOR works

1. Execute the program under one interleaving, recording which operations each
   thread performs and which operations *conflict* (access the same shared state
   with at least one write).
2. Identify **backtrack points**: positions in the execution where a different
   thread ordering would produce a genuinely different result (because the
   reordered operations conflict).
3. Explore those alternative orderings, repeating the process.
4. The algorithm terminates when all non-equivalent interleavings have been
   explored.

### Variants

- **Classic DPOR** (Flanagan & Godefroid, 2005): The original algorithm. Uses
  persistent sets to identify necessary backtrack points.
- **Optimal DPOR** (Abdulla et al., 2014): Guarantees that each distinct
  Mazurkiewicz trace is explored exactly once. Eliminates redundant explorations
  that classic DPOR can still produce.
- **Source-DPOR** and **context-sensitive DPOR**: Further refinements that reduce
  the explored state space.

### What interlace would need

1. **Conflict detection**: Track which memory locations (Python object
   attributes, dictionary keys, list indices) each thread reads and writes at
   each opcode. The opcode trace infrastructure already intercepts every
   instruction — it would need to additionally record the accessed
   addresses.

2. **Happens-before tracking**: Maintain a partial order of events based on
   synchronization (lock acquire/release pairs establish ordering). This lets the
   algorithm determine which operations are truly concurrent vs. already ordered.

3. **Backtrack set computation**: After each full execution, compute which
   alternative thread orderings at conflict points would produce distinct
   behaviors, and add those to the exploration queue.

4. **Execution replay**: The ability to replay an execution up to a backtrack
   point and then diverge. The current schedule-based infrastructure supports
   this naturally — a schedule prefix defines the execution up to a point, and
   the suffix can be varied.

### Why deferred

DPOR is a significant undertaking:

- Conflict detection requires instrumenting memory access at the opcode level,
  which is substantially more complex than the current trace function (which only
  calls `wait_for_turn()`).
- The algorithm implementation itself is non-trivial, especially the optimal
  variant.
- The current random exploration already finds bugs effectively for practical
  program sizes. DPOR's value increases with program complexity and interleaving
  space size — for the small, focused tests that interlace targets, random
  exploration with a few hundred attempts is usually sufficient.

DPOR becomes worthwhile when users want *exhaustive* verification (proving the
absence of bugs, not just finding them) or when the interleaving space is large
enough that random exploration misses rare bugs. It is a natural evolution of the
library once the core cooperative primitives are solid.

### References

- Flanagan, C. and Godefroid, P. "Dynamic Partial-Order Reduction for Model
  Checking Software." POPL 2005.
- Abdulla, P. et al. "Optimal Dynamic Partial Order Reduction." POPL 2014.
- The [loom](https://github.com/tokio-rs/loom) Rust library implements a variant
  of DPOR for exhaustive async interleaving exploration.
