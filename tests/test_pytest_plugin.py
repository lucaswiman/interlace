"""Tests for the frontrun pytest plugin and lock-patching infrastructure."""

import _thread
import threading

from frontrun._cooperative import (
    CooperativeLock,
    CooperativeRLock,
    patch_locks,
    unpatch_locks,
)
from frontrun.pytest_plugin import (
    _is_frontrun_internal,
    _try_replace,
)

# ---------------------------------------------------------------------------
# patch_locks / unpatch_locks reference counting
# ---------------------------------------------------------------------------


class TestPatchLocksRefCounting:
    """Test that patch_locks/unpatch_locks reference counting is correct."""

    def test_single_patch_unpatch(self):
        """Basic patch then unpatch restores originals."""
        assert threading.Lock is not CooperativeLock
        patch_locks()
        try:
            assert threading.Lock is CooperativeLock
        finally:
            unpatch_locks()
        assert threading.Lock is not CooperativeLock

    def test_nested_patch_unpatch(self):
        """Multiple patch calls require matching unpatch calls."""
        assert threading.Lock is not CooperativeLock
        patch_locks()
        patch_locks()
        try:
            assert threading.Lock is CooperativeLock
            unpatch_locks()
            # Still patched — one more unpatch needed
            assert threading.Lock is CooperativeLock
        finally:
            unpatch_locks()
        assert threading.Lock is not CooperativeLock

    def test_over_unpatch_is_harmless(self):
        """Calling unpatch_locks when not patched is a no-op."""
        assert threading.Lock is not CooperativeLock
        unpatch_locks()
        assert threading.Lock is not CooperativeLock

    def test_over_unpatch_does_not_corrupt_count(self):
        """Over-unpatch doesn't make the count negative, breaking future patches."""
        unpatch_locks()  # no-op
        unpatch_locks()  # no-op
        patch_locks()
        try:
            assert threading.Lock is CooperativeLock
        finally:
            unpatch_locks()
        assert threading.Lock is not CooperativeLock


# ---------------------------------------------------------------------------
# _try_replace
# ---------------------------------------------------------------------------


class TestTryReplace:
    """Test the best-effort replacement helper."""

    def test_replace_in_dict(self):
        old = object()
        new = object()
        d = {"key": old, "other": 42}
        _try_replace(d, old, new)
        assert d["key"] is new
        assert d["other"] == 42

    def test_replace_in_list(self):
        old = object()
        new = object()
        lst = [1, old, 3]
        _try_replace(lst, old, new)
        assert lst[1] is new

    def test_replace_in_object_dict(self):
        old = object()
        new = object()

        class Holder:
            pass

        h = Holder()
        h.lock = old  # type: ignore[attr-defined]
        _try_replace(h, old, new)
        assert h.lock is new  # type: ignore[attr-defined]

    def test_no_crash_on_tuple(self):
        """Tuples are immutable — _try_replace should be a no-op."""
        old = object()
        new = object()
        t = (old,)
        _try_replace(t, old, new)
        assert t[0] is old  # unchanged


# ---------------------------------------------------------------------------
# _is_frontrun_internal
# ---------------------------------------------------------------------------


class TestIsFrontrunInternal:
    def test_frontrun_module_dict(self):
        d = {"__name__": "frontrun._cooperative", "__file__": "whatever.py"}
        assert _is_frontrun_internal(d) is True

    def test_real_threading_dict(self):
        d = {"__name__": "something", "__file__": "/path/to/_real_threading.py"}
        assert _is_frontrun_internal(d) is True

    def test_normal_dict(self):
        d = {"__name__": "mymodule", "__file__": "/path/to/mymodule.py"}
        assert _is_frontrun_internal(d) is False

    def test_non_dict(self):
        assert _is_frontrun_internal([1, 2, 3]) is False


# ---------------------------------------------------------------------------
# CooperativeRLock wrapper construction (aggressive gc path)
#
# We test wrapper construction directly rather than calling
# _replace_preexisting_locks() — that function walks ALL gc objects
# including stdlib-internal locks (logging, threading) whose replacement
# causes infinite recursion at process shutdown.
# ---------------------------------------------------------------------------


def _make_lock_wrapper(real_lock_obj: object) -> CooperativeLock:
    """Build a CooperativeLock wrapper the same way the plugin does."""
    wrapper = CooperativeLock.__new__(CooperativeLock)
    wrapper._lock = real_lock_obj  # type: ignore[attr-defined]
    wrapper._object_id = id(wrapper)
    wrapper._owner_thread_id = None
    return wrapper


def _make_rlock_wrapper(real_rlock_obj: object) -> CooperativeRLock:
    """Build a CooperativeRLock wrapper the same way the plugin does."""
    wrapper = CooperativeRLock.__new__(CooperativeRLock)
    wrapper._lock = real_rlock_obj  # type: ignore[attr-defined]
    wrapper._owner = None
    wrapper._count = 0
    wrapper._object_id = id(wrapper)
    wrapper._owner_thread_id = None
    return wrapper


class TestAggressiveWrapperConstruction:
    """Test that wrappers built via __new__ (as in aggressive mode) are usable."""

    def test_rlock_wrapper_has_all_fields(self):
        """Regression: CooperativeRLock wrapper must have _owner and _count."""
        real = _thread.RLock()
        wrapper = _make_rlock_wrapper(real)
        assert wrapper._owner is None
        assert wrapper._count == 0
        assert wrapper._owner_thread_id is None
        assert wrapper._lock is real

    def test_rlock_wrapper_acquire_release(self):
        """A wrapper-constructed RLock can be acquired and released."""
        real = _thread.RLock()
        wrapper = _make_rlock_wrapper(real)
        # This would AttributeError before the fix (missing _owner/_count)
        wrapper.acquire()
        wrapper.release()

    def test_rlock_wrapper_context_manager(self):
        """A wrapper-constructed RLock works as a context manager."""
        real = _thread.RLock()
        wrapper = _make_rlock_wrapper(real)
        with wrapper:
            pass

    def test_rlock_wrapper_reentrant(self):
        """A wrapper-constructed RLock supports reentrant acquisition."""
        real = _thread.RLock()
        wrapper = _make_rlock_wrapper(real)
        wrapper.acquire()
        wrapper.acquire()  # reentrant
        wrapper.release()
        wrapper.release()

    def test_lock_wrapper_acquire_release(self):
        """A wrapper-constructed Lock can be acquired and released."""
        real = _thread.allocate_lock()
        wrapper = _make_lock_wrapper(real)
        wrapper.acquire()
        wrapper.release()

    def test_lock_wrapper_context_manager(self):
        """A wrapper-constructed Lock works as a context manager."""
        real = _thread.allocate_lock()
        wrapper = _make_lock_wrapper(real)
        with wrapper:
            pass

    def test_lock_wrapper_nonblocking(self):
        """Non-blocking acquire on a wrapper-constructed Lock."""
        real = _thread.allocate_lock()
        wrapper = _make_lock_wrapper(real)
        assert wrapper.acquire(blocking=False) is True
        assert wrapper.acquire(blocking=False) is False
        wrapper.release()


class TestIsFrontrunInternalLive:
    """Live test that _is_frontrun_internal identifies real frontrun module dicts."""

    def test_skips_frontrun_internals(self):
        import frontrun._cooperative as mod

        assert _is_frontrun_internal(mod.__dict__) is True
