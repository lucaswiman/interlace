# Missing Features / Ideas

## 1. All-threads-waiting instant deadlock detection for sync schedulers

`InterleavedLoop` (async) already has instant all-tasks-waiting deadlock detection:
when `_waiting_count >= alive`, it fires immediately instead of waiting for the
5-second timeout.

`OpcodeScheduler` and `DporScheduler` lack this.  Adding it would turn most
5-second timeout waits into instant detections:

```python
# In OpcodeScheduler.wait_for_turn:
self._waiting_count += 1
try:
    alive = self.num_threads - len(self._threads_done)
    if self._waiting_count >= alive:
        self._error = TimeoutError("All threads waiting, none can proceed")
        self._condition.notify_all()
        return False
    ...
finally:
    self._waiting_count -= 1
```

## 2. Async DPOR ✅ DONE

Implemented in `frontrun/async_dpor.py` as `explore_async_dpor()`.  Uses the
same Rust DPOR engine with `_AutoPauseCoroutine` to insert scheduling points
at every `await`.  Supports SQL/Redis detection, asyncio.Lock deadlock
detection via WaitForGraph, and row-lock tracking.

## 3. Schedule shrinking / minimization — WON'T DO

This is a labeled special case of Hypothesis shrinking.  The bytecode
`schedule_strategy` already integrates with Hypothesis, which has built-in
shrinking.  The interaction with threading makes shrinking unreliable (the
docstring recommends `phases=[Phase.generate]` to skip shrinking), and
DPOR-based exploration already produces minimal counterexamples by
construction (each execution explores a distinct Mazurkiewicz trace).

## 4. Progress reporting / callbacks

DPOR can explore thousands of interleavings over minutes.  There's currently no
callback or progress reporting mechanism.  A simple callback would help:

```python
result = explore_dpor(
    ...,
    on_progress=lambda explored, total_estimate: print(f"{explored} explored"),
)
```

## 5. `threading.Barrier` in cooperative primitives

`_cooperative.py` covers Lock, RLock, Semaphore, BoundedSemaphore, Event,
Condition, and all Queue variants.  `threading.Barrier` is absent.  User code
that uses `Barrier` will deadlock under the scheduler because the real `Barrier`
blocks in C.

## 6. Dynamic thread creation in DPOR

`PyDporEngine` takes a fixed `num_threads` at creation.  If user code spawns
threads dynamically (a common pattern), those threads aren't tracked by the
DPOR engine.  Supporting dynamic thread creation would require the engine to
grow its vector clocks on the fly.

## 7. File include/exclude patterns for tracing

`_tracing.py` uses an automatic heuristic (skip stdlib, site-packages, frontrun
itself).  Users can't control which files to trace.  This matters when:

- Testing code inside third-party libraries the user owns
- The automatic detection misclassifies files (e.g., editable installs)
- Users want to exclude specific modules from tracing for performance

An include/exclude pattern list (glob-based) would help.
