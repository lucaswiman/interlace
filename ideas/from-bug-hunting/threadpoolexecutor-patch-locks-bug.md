# Bug: `patch_locks()` breaks `concurrent.futures.ThreadPoolExecutor` import

## Summary

Calling `frontrun._cooperative.patch_locks()` before `concurrent.futures.ThreadPoolExecutor` is first accessed causes an `ImportError`. This breaks any library that lazily imports `ThreadPoolExecutor` after lock patching — notably Django (via `asgiref.sync`), which makes it impossible to run Django-based integration tests with the default pytest plugin configuration.

## Root Cause

Python 3.10+'s `concurrent.futures.__init__.py` uses `__getattr__`-based lazy loading:

```python
# /usr/lib/python3.10/concurrent/futures/__init__.py
def __getattr__(name):
    global ProcessPoolExecutor, ThreadPoolExecutor
    if name == 'ThreadPoolExecutor':
        from .thread import ThreadPoolExecutor as te
        ThreadPoolExecutor = te
        return te
    raise AttributeError(...)
```

When `ThreadPoolExecutor` is first accessed, it triggers `from .thread import ThreadPoolExecutor`, which imports `concurrent.futures.thread`. That module uses `threading.Lock` at module level to construct internal locks.

`patch_locks()` replaces `threading.Lock` with `CooperativeLock`. The `concurrent.futures.thread` module fails to import because `CooperativeLock` doesn't behave identically to the real `threading.Lock` during module initialization (specifically, the C-level `_thread.LockType` check fails).

## Reproduction

```python
from frontrun._cooperative import patch_locks
patch_locks()
from concurrent.futures import ThreadPoolExecutor  # ImportError!
```

Works fine if `ThreadPoolExecutor` is imported **before** patching:

```python
from concurrent.futures import ThreadPoolExecutor  # OK
from frontrun._cooperative import patch_locks
patch_locks()
# ThreadPoolExecutor already cached in concurrent.futures module globals
```

## Impact

- **Django integration tests fail** when running under `frontrun pytest` (which calls `patch_locks()` in `pytest_configure`, before test modules are collected)
- Any library using `asgiref` (Django's async support) triggers the lazy import path
- The frontrun pytest plugin's `--no-frontrun-patch-locks` flag works around the issue, but `explore_dpor()` still calls `patch_locks()` internally — this works because by the time `explore_dpor` runs, the test module has already imported Django (and thus `concurrent.futures.ThreadPoolExecutor`)

## Affected Libraries

- Django (via `asgiref.sync`)
- Any ASGI framework using `asgiref`
- Anything that lazily imports `concurrent.futures.ThreadPoolExecutor` after pytest starts

## Workarounds

1. **Use `--no-frontrun-patch-locks`** when running Django-based tests:
   ```
   frontrun pytest --no-frontrun-patch-locks tests/test_django_*.py
   ```

2. **Pre-import `concurrent.futures.thread`** in `conftest.py`:
   ```python
   import concurrent.futures.thread  # noqa: F401  — must precede patch_locks()
   ```

## Suggested Fix

In `patch_locks()` (or `pytest_configure`), pre-import `concurrent.futures.thread` before replacing `threading.Lock`:

```python
def patch_locks() -> None:
    # Ensure concurrent.futures.thread is imported before we replace
    # threading.Lock, since it uses Lock at module level.
    import concurrent.futures.thread  # noqa: F401

    threading.Lock = CooperativeLock
    # ... rest of patching
```

This ensures the real `threading.Lock` is used during `concurrent.futures.thread` module initialization, and the module-level locks are already constructed before patching takes effect.
