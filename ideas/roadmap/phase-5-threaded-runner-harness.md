# Phase 5: Unify threaded runner lifecycle

## Problem

There is repeated “threaded execution harness” code across:

- `frontrun/bytecode.py`
- `frontrun/dpor.py`
- parts of `frontrun/trace_markers.py`

The repeated concerns include:

- patch/unpatch lifecycle
- thread startup and joining
- timeout handling
- monitoring versus `sys.settrace` selection
- error capture and propagation

## Goal

Extract a shared threaded harness that can host different scheduling policies and tracing callbacks.

## Proposed shape

Suggested module:

- `frontrun/_threaded_runner.py`

Possible contents:

- thread launch/join orchestration
- shared timeout handling
- shared patch lifecycle scaffolding
- monitoring backend selection
- common exception collection

Specialized modules would still own:

- DPOR-specific TLS and engine reporting
- bytecode random-schedule behavior
- marker-specific trace behavior

## Sequence

1. Extract neutral thread lifecycle helpers first.
2. Move bytecode and DPOR runner patch/unpatch scaffolding behind shared helpers.
3. Evaluate whether marker execution should also use the harness or remain separate.
4. Avoid forcing async code into this abstraction.

## Special caution

Do not merge the schedulers themselves prematurely. The duplication worth removing is lifecycle plumbing, not the scheduling models.

## Validation

- `make test-3.14 PYTEST_ARGS="-q tests/test_bytecode.py tests/test_dpor.py tests/test_trace_markers.py tests/test_sleep_patching.py tests/test_threading_primitives.py"`
- `make check`

## Expected payoff

- Moderate LOC reduction
- More uniform execution semantics across threaded exploration modes
- Easier maintenance of patching and timeout behavior
