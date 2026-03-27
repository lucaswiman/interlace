"""
Pytest plugin for frontrun - deterministic concurrency testing.

This plugin provides fixtures and hooks for using frontrun with pytest,
enabling monkey-patching of threading.Lock when tests need deterministic
thread scheduling.

Usage:
    - Place this conftest.py in your tests directory
    - Use frontrun fixtures directly, or mark tests with @pytest.mark.frontrun
    - Fixtures will automatically patch threading.Lock for those tests
"""

import os
import sys
import threading

import pytest

# Add parent directory to path so we can import frontrun
_frontrun_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _frontrun_path not in sys.path:
    sys.path.insert(0, _frontrun_path)

from frontrun._cooperative import CooperativeLock


def pytest_configure(config):
    """Hook that runs before test collection.

    Registers pytest markers for frontrun tests.
    """
    config.addinivalue_line("markers", "frontrun: mark test as using frontrun concurrency testing")
    config.addinivalue_line(
        "markers",
        "intentionally_leaves_dangling_threads: mark test as intentionally leaving threads alive (e.g., deadlock tests that cannot be cleaned up)",
    )
    config.addinivalue_line(
        "markers",
        "integration: mark test as requiring external services (Redis or Postgres)",
    )


def _global_patch_active() -> bool:
    """Return True if the pytest plugin already patched locks globally."""
    from frontrun._cooperative import _patched

    return _patched



@pytest.fixture(autouse=True)
def _patch_locks_for_marked_tests(request):
    """Auto-used fixture that patches threading.Lock for tests marked with @pytest.mark.frontrun.

    This allows tests that directly call frontrun functions (without using fixtures)
    to still have cooperative lock behavior.

    If the ``--frontrun-patch-locks`` plugin already patched globally, this
    is a no-op.
    """
    if _global_patch_active():
        yield
        return
    if request.node.get_closest_marker("frontrun"):
        original_lock = threading.Lock
        threading.Lock = CooperativeLock
        try:
            yield
        finally:
            threading.Lock = original_lock
    else:
        yield


@pytest.fixture(autouse=True)
def _check_thread_cleanup(request):
    """Auto-used fixture that verifies all non-daemon threads are cleaned up after each test.

    This catches tests that start threads but don't properly join them, which would
    otherwise cause pytest to hang at exit. By checking explicitly, we get clear
    error messages instead of silent hangs.

    Also prints diagnostic info about daemon threads that are still alive (these won't
    block exit but may indicate tests that create intentionally deadlocked threads).
    """
    # Record threads before test
    initial_threads = set(threading.enumerate())

    yield

    # Check for lingering threads after test
    final_threads = set(threading.enumerate())
    new_threads = final_threads - initial_threads

    # Separate into daemon and non-daemon threads (excluding main thread)
    main_thread = threading.main_thread()
    alive_threads = [t for t in new_threads if t != main_thread and t.is_alive()]

    daemon_threads = [t for t in alive_threads if t.daemon]
    non_daemon_threads = [t for t in alive_threads if not t.daemon]

    # Print diagnostic info for all alive threads
    if alive_threads:
        print(f"\n[THREAD CLEANUP] Test: {request.node.nodeid}")
        for t in alive_threads:
            status = "daemon" if t.daemon else "NON-DAEMON"
            print(f"  - {t.name} (ident={t.ident}, {status}, alive={t.is_alive()})")

    # Non-daemon threads block pytest exit — always fail on these
    if non_daemon_threads:
        thread_info = ", ".join(f"{t.name} (ident={t.ident})" for t in non_daemon_threads)
        pytest.fail(
            f"Test {request.node.nodeid} left {len(non_daemon_threads)} non-daemon thread(s) running: {thread_info}. "
            f"All threads must be joined before test completion."
        )

    # Daemon threads don't block exit but still indicate cleanup issues
    if daemon_threads and not request.node.get_closest_marker("intentionally_leaves_dangling_threads"):
        thread_info = ", ".join(f"{t.name} (ident={t.ident})" for t in daemon_threads)
        pytest.fail(
            f"Test {request.node.nodeid} left {len(daemon_threads)} daemon thread(s) running: {thread_info}. "
            f"All threads must be joined before test completion. "
            f"If this is intentional (e.g., testing deadlocks that cannot be cleaned up), "
            f"mark the test with @pytest.mark.intentionally_leaves_dangling_threads"
        )


