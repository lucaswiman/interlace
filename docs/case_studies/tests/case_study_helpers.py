"""Shared utilities for case study test runs."""

import signal
from contextlib import contextmanager


@contextmanager
def timeout_minutes(minutes=10):
    """Raise TimeoutError if the block takes longer than `minutes` minutes.

    Uses SIGALRM so it only works on Unix and must be called from the
    main thread.  Restores the previous SIGALRM handler on exit.
    """

    def _handler(signum, frame):
        raise TimeoutError(f"Test timed out after {minutes} minute(s)")

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(int(minutes * 60))
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def print_exploration_result(result):
    """Print single exploration result."""
    print(f"\nExplored {result.num_explored} interleavings")
    print(f"Property holds: {result.property_holds}")
    if result.counterexample:
        print(f"Counterexample schedule length: {len(result.counterexample)}")


def print_seed_sweep_results(found_seeds, total_explored):
    """Print seed sweep results."""
    print(f"\nTotal interleavings explored: {total_explored}")
    print(f"Seeds that found the bug: {len(found_seeds)} / 20")
    for seed, n in found_seeds:
        print(f"  seed={seed}: found after {n} interleavings")
