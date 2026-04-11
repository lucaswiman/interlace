# Phase 2: Share async auto-pause machinery

## Problem

`frontrun/async_shuffler.py` and `frontrun/async_dpor.py` both implement the same async scheduling wrapper pattern:

- scheduler and task contextvars
- `await_point()`
- `_AutoPauseIterator`
- `_AutoPauseCoroutine`
- `_in_scheduler_pause` handling

The two versions are nearly the same, but drift independently.

## Goal

Extract one shared async auto-pause module and make both async exploration modes use it.

## Proposed shape

Suggested module:

- `frontrun/_async_autopause.py`

Suggested contents:

- shared contextvars
- `await_point()`
- `_AutoPauseIterator`
- `_AutoPauseCoroutine`
- a minimal scheduler protocol or typed callback contract

## Sequence

1. Move the generic auto-pause implementation into the shared module.
2. Keep `async_shuffler.py` and `async_dpor.py` re-exporting or forwarding old names if tests depend on them.
3. Move any truly DPOR-specific behavior behind small callbacks instead of forking the whole wrapper.
4. Remove duplicate local implementations after behavior matches.

## Boundaries

Keep these concerns out of the shared module:

- DPOR engine scheduling decisions
- async SQL/Redis reporting
- `asyncio.Lock` deadlock handling
- result formatting

The shared module should only own “make every natural await a scheduler boundary”.

## Validation

- `make test-3.14 PYTEST_ARGS="-q tests/test_async_shuffler.py tests/test_async_dpor.py tests/test_bugfix_cache_deadlock_async.py tests/test_async_exact_mazurkiewicz_trace_count.py"`
- `make check`

## Expected payoff

- Moderate LOC reduction
- Lower drift risk between async execution modes
- Easier future work on async scheduler semantics
