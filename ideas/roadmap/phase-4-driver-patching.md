# Phase 4: Replace hand-written patching with registries

## Problem

SQL and Redis detection rely on a lot of repetitive patch registration code:

- import module
- find class or connect function
- stash original
- build patched wrapper
- assign patched method
- remember how to unpatch

That pattern repeats across sync and async clients, even though the mechanics are mostly the same.

## Goal

Move patch registration to declarative driver tables plus a small shared patching toolkit.

## Scope

Primary targets:

- `frontrun/_sql_cursor.py`
- `frontrun/_sql_cursor_async.py`
- `frontrun/_redis_client.py`
- `frontrun/_redis_client_async.py`

## Proposed shape

Suggested helpers:

- `frontrun/_patching.py`
  - generic patch record storage
  - patch/unpatch helpers
  - method-wrapper helpers
- `frontrun/_sql_patch_registry.py`
- `frontrun/_redis_patch_registry.py`

Suggested registry data:

- module path
- target class/function path
- sync vs async method kind
- parameter style or reporting mode
- optional connect-time factory injection behavior

## Sequence

1. Extract generic patch bookkeeping helpers first.
2. Convert Redis patching to registry-driven form.
3. Convert async Redis patching.
4. Convert SQL patching last, because it has the most special cases.

## Special caution

SQL patching is not just “method wrapping”. It also contains:

- connection identity tracking
- endpoint suppression
- lock-timeout injection
- cursor-factory substitution

Do not force all SQL cases into one overly-generic abstraction. Use registries to remove obvious repetition, not to hide important behavioral distinctions.

## Validation

- `make test-3.14 PYTEST_ARGS="-q tests/test_sql_cursor.py tests/test_sql_cursor_async.py tests/test_integration_redis.py tests/test_integration_redis_dpor.py tests/test_sql_conflict_integration.py"`
- `make check`

## Expected payoff

- Moderate LOC reduction
- Easier support for new drivers
- Fewer patch/unpatch bugs caused by copy-paste divergence
