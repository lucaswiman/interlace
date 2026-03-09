# Easier Connection Management in DPOR Integration Tests

## The Problem

Writing `explore_dpor` tests against real databases requires careful per-thread connection management that is tedious and error-prone. Every test author has to remember the same boilerplate:

**Django:**
```python
class _State:
    def __init__(self):
        from django.db import connections
        connections.close_all()  # must close before threads fork
        # ... setup ...

def _thread_fn(state):
    from django.db import connections
    connections.close_all()  # each thread needs its own connection
    # ... actual test logic ...
```

**SQLAlchemy / Flask-SQLAlchemy:**
```python
class _State:
    def __init__(self):
        db.engine.dispose()  # drop pooled connections
        with app.app_context():
            # ... setup ...

def _thread_fn(state):
    with app.app_context():
        db.engine.dispose()  # each thread needs fresh connection
        # ... actual test logic ...
```

If you forget `connections.close_all()` or `db.engine.dispose()`, threads silently share a connection and you get:
- Intermittent `InterfaceError: connection already closed`
- Queries from one thread arriving on another thread's connection
- Confusing `InternalError: current transaction is aborted` errors
- DPOR seeing no conflicts because all SQL goes through one socket

This is the #1 friction point when writing SQL DPOR tests. The actual invariant + thread logic is usually 10-20 lines; the connection management ceremony is another 10-15 lines of defensive boilerplate that has nothing to do with the race condition being tested.

## Possible Solutions

### Option A: Auto-dispose in `explore_dpor` when `detect_io=True`

When `detect_io=True`, frontrun already monkey-patches `psycopg2.cursor.execute`. It could also:

1. **Before calling `setup()`:** Close all Django connections and dispose SQLAlchemy engines that exist in the process.
2. **Before each thread function:** Do the same, ensuring each thread gets a fresh connection from the driver.

This would be a best-effort heuristic — detect known frameworks and close their connections:

```python
def _close_all_db_connections():
    """Best-effort cleanup of DB connections from known frameworks."""
    # Django
    try:
        from django.db import connections
        connections.close_all()
    except ImportError:
        pass
    # SQLAlchemy (global engines are harder to find, but Flask-SQLAlchemy
    # stores them on the app extension)
    try:
        from flask import current_app
        current_app.extensions["sqlalchemy"].engine.dispose()
    except Exception:
        pass
```

**Pros:** Zero boilerplate for test authors. Just works.
**Cons:** Magic. Might close connections the user wants to keep open. Hard to enumerate all ORMs/drivers.

### Option B: `connection_factory` parameter on `explore_dpor`

Add an explicit parameter that frontrun calls to set up per-thread connections:

```python
result = explore_dpor(
    setup=_State,
    threads=[thread_a, thread_b],
    invariant=_invariant,
    detect_io=True,
    connection_setup=lambda: django.db.connections.close_all(),
)
```

Frontrun calls `connection_setup()` before `setup()` and before each thread function. The user provides framework-specific logic once.

**Pros:** Explicit, no magic, works with any framework.
**Cons:** Still requires the user to know *what* to call. Slightly less boilerplate but not zero.

### Option C: Framework-specific helpers

Provide helper decorators/context managers for common frameworks:

```python
from frontrun.contrib.django import django_dpor
from frontrun.contrib.flask import flask_dpor

# Django
result = django_dpor(
    setup=_State,
    threads=[thread_a, thread_b],
    invariant=_invariant,
)

# Flask
result = flask_dpor(
    app=app,
    setup=_State,
    threads=[thread_a, thread_b],
    invariant=_invariant,
)
```

These wrappers would:
1. Close all connections before setup
2. Wrap each thread function in the appropriate context (Django: `connections.close_all()`, Flask: `app.app_context()` + `db.engine.dispose()`)
3. Set `detect_io=True` by default

```python
# frontrun/contrib/django.py
def django_dpor(*, setup, threads, invariant, **kwargs):
    from django.db import connections

    original_setup = setup
    def _wrapped_setup():
        connections.close_all()
        return original_setup()

    def _wrap_thread(fn):
        def _wrapped(state):
            connections.close_all()
            return fn(state)
        return _wrapped

    return explore_dpor(
        setup=_wrapped_setup,
        threads=[_wrap_thread(t) for t in threads],
        invariant=invariant,
        detect_io=True,
        **kwargs,
    )
```

**Pros:** Minimal boilerplate. Framework-aware. Easy to document with examples.
**Cons:** Maintenance burden for each supported framework. Users of unsupported frameworks get no help.

### Option D: Detect and warn

Instead of automatically fixing the problem, detect when multiple threads are sending SQL over the same socket file descriptor and emit a clear warning:

```
WARNING: Threads 0 and 1 are sharing database socket fd=7.
This usually means connections weren't closed before thread creation.
Add `connections.close_all()` (Django) or `db.engine.dispose()` (SQLAlchemy)
at the start of each thread function and in your setup callable.
```

**Pros:** Non-invasive. Educates the user. No risk of breaking anything.
**Cons:** Doesn't actually fix the problem. User still has to write the boilerplate.

## Recommendation

**Option C (framework helpers) + Option D (detect and warn).**

The helpers eliminate boilerplate for the common cases (Django, Flask-SQLAlchemy, plain SQLAlchemy). The warning catches everyone else and also catches cases where the helpers aren't used.

The shared-socket-fd detection (Option D) is cheap — frontrun already intercepts `send`/`recv` at the C level and knows which fd each thread is using. If two threads use the same fd, that's almost certainly a bug in test setup, not a real race condition.

## Other Pain Points Observed

- **Django can only be `settings.configure()`'d once per process.** Running multiple Django-based case studies in the same pytest session is fragile. This isn't frontrun's problem to solve, but it's worth documenting in a "writing SQL DPOR tests" guide.

- **Flask app context must be pushed in each thread.** This is a Flask requirement, not frontrun's, but the `flask_dpor` helper could handle it automatically.

- **`db.engine.dispose()` inside `app.app_context()` ordering matters.** If you call `dispose()` outside the app context, Flask-SQLAlchemy raises `RuntimeError: Working outside of application context`. The helper should get this right so users don't have to.
