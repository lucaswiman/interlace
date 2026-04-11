# Phase 3: Consolidate framework adapters

## Problem

The Django and SQLAlchemy integration helpers have separate sync and async implementations that mostly differ in:

- how they acquire a connection
- whether the user callback is called or awaited
- whether the underlying engine exposes sync or async APIs

The surrounding setup, wrapping, and option passthrough are largely the same.

## Goal

Keep the public sync and async entrypoints, but factor their implementation through shared wrapper builders.

## Scope

Primary targets:

- `frontrun/contrib/django/_sync.py`
- `frontrun/contrib/django/_async.py`
- `frontrun/contrib/sqlalchemy/_sync.py`
- `frontrun/contrib/sqlalchemy/_async.py`

## Proposed shape

Suggested private helpers:

- `frontrun/contrib/django/_common.py`
- `frontrun/contrib/sqlalchemy/_common.py`

Shared helper responsibilities:

- normalize `trace_packages` defaults
- build wrapped setup functions
- build per-thread/per-task connection-scoped wrappers
- centralize lock-timeout injection policy

## Sequence

1. Extract read-only shared constants and option normalization.
2. Extract wrapper-builder helpers without changing public functions.
3. Convert sync and async entrypoints to thin frontends over those helpers.
4. Only then consider small naming/API cleanups.

## Special caution

The SQLAlchemy sync adapter has extra sync-report suppression during connection lifecycle. Do not erase those differences by over-abstracting too early. The shared helper should allow sync-specific setup/teardown hooks.

## Validation

- `make test-3.14 PYTEST_ARGS="-q tests/test_integration_django.py tests/test_integration_sqlalchemy_dpor.py tests/test_integration_async_dpor_sqlalchemy.py tests/test_contrib_bugs.py"`
- `make check`

## Expected payoff

- Small-to-moderate LOC reduction
- Cleaner framework integration surface
- Less risk of feature skew between sync and async adapters
