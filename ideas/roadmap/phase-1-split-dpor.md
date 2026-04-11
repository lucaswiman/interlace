# Phase 1: Split `frontrun/dpor.py`

## Why this phase comes first

`frontrun/dpor.py` currently mixes several distinct responsibilities:

- DPOR scheduler logic
- fixed-schedule replay logic
- thread runner and opcode/monitoring setup
- LD_PRELOAD bridge handling
- DPOR thread-local setup and teardown
- report generation and counterexample reproduction
- top-level `explore_dpor()` orchestration

That makes the file hard to navigate and raises the cost of every follow-on refactor. This phase creates explicit internal seams first, without trying to reduce total logic yet.

## Success criteria

- `frontrun/dpor.py` becomes a thin public entrypoint plus compatibility layer.
- Internal code is split into focused modules with stable ownership.
- Public imports and behavior remain unchanged.
- The test suite passes without requiring user-facing migration.

## Target module layout

Suggested new files under `frontrun/_dpor/`:

- `__init__.py`
- `scheduler.py`
  - `DporScheduler`
  - `_ReplayDporScheduler`
  - `_IOAnchoredReplayScheduler`
- `runner.py`
  - `DporBytecodeRunner`
  - opcode/monitoring setup specific to threaded DPOR runs
- `tls.py`
  - DPOR thread-local state helpers
  - `_setup_dpor_tls`
  - `_teardown_dpor_tls`
  - lock-depth and pending-I/O lifecycle helpers
- `preload_bridge.py`
  - `_PreloadBridge`
- `replay.py`
  - `_run_dpor_schedule`
  - `_reproduce_dpor_counterexample`
  - replay-only helper types
- `explore.py`
  - `explore_dpor`
  - top-level result construction
  - trace/report assembly
- `types.py` or `engine.py`
  - light wrappers around Rust engine imports and shared aliases if needed

The public file `frontrun/dpor.py` should re-export the stable API from these modules.

## Extraction sequence

### Step 1: create the private package

- Add `frontrun/_dpor/__init__.py`.
- Move no logic yet.
- Re-export nothing publicly from the package until later steps compile.

### Step 2: isolate the preload bridge

- Move `_PreloadBridge` out of `frontrun/dpor.py`.
- Keep constructor and method names unchanged.
- Keep all imports in `frontrun/dpor.py` pointing at the new home.

This is low-risk because the bridge has a narrow interface and no public API.

### Step 3: isolate scheduler classes

- Move `DporScheduler`, `_ReplayEngine`, `_ReplayExecution`, `_ReplayDporScheduler`, and `_IOAnchoredReplayScheduler` into `scheduler.py`.
- Keep helper functions that are only used by the scheduler next to it.
- Do not simplify logic during the move.

This is the most important boundary because replay and runner logic both depend on it.

### Step 4: isolate the bytecode runner

- Move `DporBytecodeRunner` into `runner.py`.
- Initially leave DPOR TLS setup as instance methods if needed.
- After the move, extract TLS setup/teardown into a dedicated helper module only if that reduces import tangling.

### Step 5: isolate replay orchestration

- Move `_run_dpor_schedule` and `_reproduce_dpor_counterexample` into `replay.py`.
- Keep the same function signatures during this phase.
- Import them back into `frontrun/dpor.py` for compatibility.

### Step 6: isolate top-level exploration orchestration

- Move `explore_dpor()` into `explore.py`.
- Move closely-related helper functions used only by `explore_dpor()` with it.
- Leave public import compatibility in `frontrun/dpor.py`.

### Step 7: shrink the public module

Final expected responsibilities of `frontrun/dpor.py`:

- import-time Rust extension checks
- public re-exports
- any explicit compatibility aliases that external users may import directly

## Invariants to preserve

- `explore_dpor()` signature and semantics remain unchanged.
- Existing reproduction behavior remains unchanged.
- Trace formatting and race reporting output remain byte-for-byte stable where practical.
- Import paths used in tests continue to work:
  - `frontrun.dpor`
  - any tests importing private names from `frontrun.dpor`

For private names, preserving imports during the transition is more important than preserving the old file layout internally.

## Likely friction points

- Circular imports between runner, scheduler, and SQL/Redis reporting helpers.
- DPOR TLS setup currently closes over many objects from `dpor.py`; extracting it may need a small context object rather than many free variables.
- `DporBytecodeRunner` currently owns both runtime behavior and integration plumbing. Avoid trying to “improve” that ownership during the move.
- Replay code may implicitly depend on symbols that only exist because everything currently lives in one module.

## Tactics to keep the refactor safe

- Prefer move-only commits before cleanup commits.
- Keep symbol names unchanged until after the file split is merged.
- If a helper is shared across multiple new modules, first extract it unchanged, then rename later.
- Add a package-private import façade if necessary to prevent cycles.

## Validation

Minimum validation for each PR in this phase:

- `make test-3.14 PYTEST_ARGS="-q tests/test_dpor.py tests/test_dpor_replay.py tests/test_dpor_io.py tests/test_dpor_preload.py tests/test_dpor_sql_scheduling.py"`
- `make test-3.14 PYTEST_ARGS="-q tests/test_integration_redis_dpor.py tests/test_integration_async_dpor_sqlalchemy.py tests/test_integration_defect15_dpor_conflict_explosion.py"`
- `make check`

If the split is done as multiple PRs, the smaller PRs can use narrower targeted test subsets, but the final phase-completion PR should run the full `make test-3.14` and `make check`.

## Expected outcome

This phase may not reduce LOC very much on its own. Its value is that every later reduction becomes much safer:

- async DPOR and sync DPOR lifecycle code become easier to compare
- replay logic stops being entangled with exploration logic
- instrumentation setup becomes a visible subsystem instead of hidden glue

That is the prerequisite for the later size reductions.
