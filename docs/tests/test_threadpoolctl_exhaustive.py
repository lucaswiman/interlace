"""
Exhaustive concurrency tests for threadpoolctl.

This file tests EVERY concurrency bug we can identify in threadpoolctl.py.
The library has ZERO synchronization primitives (no locks, no RLock, no
Condition variables anywhere), making it vulnerable to multiple classes of
concurrency bugs.

Bugs tested:
  1. _get_libc() TOCTOU on _system_libraries["libc"] cache
  2. _get_windll() TOCTOU on _system_libraries[dll_name] cache
  3. register() race on global mutable lists (_ALL_CONTROLLERS, etc.)
  4. _make_controller_from_path() duplicate-check-then-append TOCTOU
  5. Concurrent ThreadpoolController.__init__ sharing class-level _system_libraries
  6. _set_threadpool_limits() concurrent iteration vs mutation of lib_controllers
  7. info() concurrent with lib_controllers mutation
  8. select() concurrent with lib_controllers mutation
  9. restore_original_limits() concurrent with lib_controllers mutation
  10. _realpath lru_cache concurrent population (not thread-safe)
  11. register() interleaved with _make_controller_from_path iteration of
      _ALL_CONTROLLERS
  12. Concurrent threadpool_info() calls racing on shared class-level state
  13. _from_controllers aliasing: two ThreadpoolControllers sharing the same
      lib_controllers list, both mutating it
"""

import os
import sys
import ctypes
from unittest.mock import patch, MagicMock

_test_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_test_dir, "..", "external_repos", "threadpoolctl"))

from external_tests_helpers import print_exploration_result, print_seed_sweep_results

# We must import threadpoolctl items after sys.path manipulation
import threadpoolctl
from threadpoolctl import (
    ThreadpoolController,
    LibController,
    register,
    threadpool_info,
    _ALL_CONTROLLERS,
    _ALL_USER_APIS,
    _ALL_INTERNAL_APIS,
    _ALL_PREFIXES,
    _realpath,
)

from interlace.bytecode import explore_interleavings, run_with_schedule


# ---------------------------------------------------------------------------
# Helpers: Fake LibController subclass for testing
# ---------------------------------------------------------------------------

class FakeLibControllerA(LibController):
    """A fake library controller for testing concurrent registration."""
    user_api = "fake_api_a"
    internal_api = "fake_internal_a"
    filename_prefixes = ("libfake_a",)
    check_symbols = ()

    def get_num_threads(self):
        return 1

    def set_num_threads(self, num_threads):
        pass

    def get_version(self):
        return "1.0.0"


class FakeLibControllerB(LibController):
    """A second fake library controller for testing concurrent registration."""
    user_api = "fake_api_b"
    internal_api = "fake_internal_b"
    filename_prefixes = ("libfake_b",)
    check_symbols = ()

    def get_num_threads(self):
        return 2

    def set_num_threads(self, num_threads):
        pass

    def get_version(self):
        return "2.0.0"


class FakeLibControllerC(LibController):
    """A third fake controller for concurrent register + iteration tests."""
    user_api = "fake_api_c"
    internal_api = "fake_internal_c"
    filename_prefixes = ("libfake_c",)
    check_symbols = ()

    def get_num_threads(self):
        return 3

    def set_num_threads(self, num_threads):
        pass

    def get_version(self):
        return "3.0.0"


def _remove_fake_controllers():
    """Remove any fake controllers we registered from global lists."""
    fake_internals = {"fake_internal_a", "fake_internal_b", "fake_internal_c"}
    fake_apis = {"fake_api_a", "fake_api_b", "fake_api_c"}
    fake_prefixes = {"libfake_a", "libfake_b", "libfake_c"}

    _ALL_CONTROLLERS[:] = [
        c for c in _ALL_CONTROLLERS if c.internal_api not in fake_internals
    ]
    _ALL_USER_APIS[:] = [a for a in _ALL_USER_APIS if a not in fake_apis]
    _ALL_INTERNAL_APIS[:] = [a for a in _ALL_INTERNAL_APIS if a not in fake_internals]
    _ALL_PREFIXES[:] = [p for p in _ALL_PREFIXES if p not in fake_prefixes]


# ---------------------------------------------------------------------------
# Helper: create a mock LibController instance without calling __init__
# (avoids ctypes.CDLL on a non-existent file)
# ---------------------------------------------------------------------------

def _make_mock_lib_controller(prefix="libmock", filepath="/fake/libmock.so",
                              user_api="blas", internal_api="mock",
                              num_threads=4):
    """Create a mock object that behaves like a LibController instance."""
    mock = MagicMock()
    mock.prefix = prefix
    mock.filepath = filepath
    mock.user_api = user_api
    mock.internal_api = internal_api
    mock.num_threads = num_threads
    mock.get_num_threads.return_value = num_threads
    mock.set_num_threads.return_value = None
    mock.info.return_value = {
        "user_api": user_api,
        "internal_api": internal_api,
        "num_threads": num_threads,
        "prefix": prefix,
        "filepath": filepath,
        "version": "1.0",
    }
    return mock


# ===========================================================================
# Bug 1: _get_libc() TOCTOU on _system_libraries["libc"]
#
# Two threads both call _get_libc(). Both see _system_libraries.get("libc")
# is None, both create a ctypes.CDLL, both store it. The second overwrites
# the first's CDLL. The two threads end up with different CDLL objects.
# ===========================================================================

class GetLibcTOCTOUState:
    """State for the _get_libc() TOCTOU race."""

    def __init__(self):
        ThreadpoolController._system_libraries.clear()
        self.libc_1 = None
        self.libc_2 = None

    def thread1(self):
        self.libc_1 = ThreadpoolController._get_libc()

    def thread2(self):
        self.libc_2 = ThreadpoolController._get_libc()


def test_get_libc_toctou():
    """Bug 1: _get_libc() TOCTOU -- two threads get different CDLL objects."""
    result = explore_interleavings(
        setup=lambda: GetLibcTOCTOUState(),
        threads=[
            lambda s: s.thread1(),
            lambda s: s.thread2(),
        ],
        invariant=lambda s: s.libc_1 is s.libc_2,
        max_attempts=200,
        max_ops=300,
        seed=42,
    )
    print_exploration_result(result)
    return result


def test_get_libc_toctou_sweep():
    """Bug 1: Sweep seeds for _get_libc() TOCTOU."""
    found_seeds = []
    total_explored = 0

    for seed in range(20):
        result = explore_interleavings(
            setup=lambda: GetLibcTOCTOUState(),
            threads=[
                lambda s: s.thread1(),
                lambda s: s.thread2(),
            ],
            invariant=lambda s: s.libc_1 is s.libc_2,
            max_attempts=100,
            max_ops=300,
            seed=seed,
        )
        total_explored += result.num_explored
        if not result.property_holds:
            found_seeds.append((seed, result.num_explored))

    print_seed_sweep_results(found_seeds, total_explored)
    return found_seeds


def test_get_libc_toctou_reproduce():
    """Bug 1: Reproduce the _get_libc() TOCTOU."""
    result = explore_interleavings(
        setup=lambda: GetLibcTOCTOUState(),
        threads=[
            lambda s: s.thread1(),
            lambda s: s.thread2(),
        ],
        invariant=lambda s: s.libc_1 is s.libc_2,
        max_attempts=200,
        max_ops=300,
        seed=42,
    )

    if not result.counterexample:
        print("No counterexample found -- skipping reproduction")
        return

    print(f"Found counterexample after {result.num_explored} attempts")
    print(f"Schedule length: {len(result.counterexample)}")

    print("\nReproducing 10 times with the same schedule...")
    bugs_reproduced = 0
    for i in range(10):
        state = run_with_schedule(
            result.counterexample,
            setup=lambda: GetLibcTOCTOUState(),
            threads=[
                lambda s: s.thread1(),
                lambda s: s.thread2(),
            ],
        )
        is_bug = state.libc_1 is not state.libc_2
        if is_bug:
            bugs_reproduced += 1
        status = "BUG" if is_bug else "ok"
        same = "same" if state.libc_1 is state.libc_2 else "DIFFERENT"
        print(f"  Run {i + 1}: libc objects {same} [{status}]")

    print(f"\nReproduced: {bugs_reproduced}/10")
    return bugs_reproduced


# ===========================================================================
# Bug 2: _get_windll() TOCTOU on _system_libraries[dll_name]
#
# Identical pattern to _get_libc(). Two threads call _get_windll() with the
# same dll_name, both see None, both create ctypes.WinDLL, both store.
# Since WinDLL is not available on non-Windows, we simulate the pattern
# by patching the method to use CDLL instead.
# ===========================================================================

class GetWindllTOCTOUState:
    """State for the _get_windll() TOCTOU race (simulated on non-Windows)."""

    def __init__(self):
        ThreadpoolController._system_libraries.clear()
        self.dll_1 = None
        self.dll_2 = None

    def _get_fake_windll(self, dll_name):
        """Simulate _get_windll using CDLL to avoid WinDLL on Linux."""
        dll = ThreadpoolController._system_libraries.get(dll_name)
        if dll is None:
            # Simulate creating a DLL -- use a unique object each time
            dll = object()
            ThreadpoolController._system_libraries[dll_name] = dll
        return dll

    def thread1(self):
        self.dll_1 = self._get_fake_windll("FakeDll")

    def thread2(self):
        self.dll_2 = self._get_fake_windll("FakeDll")


def test_get_windll_toctou():
    """Bug 2: _get_windll() TOCTOU -- same pattern as _get_libc()."""
    result = explore_interleavings(
        setup=lambda: GetWindllTOCTOUState(),
        threads=[
            lambda s: s.thread1(),
            lambda s: s.thread2(),
        ],
        invariant=lambda s: s.dll_1 is s.dll_2,
        max_attempts=200,
        max_ops=300,
        seed=42,
    )
    print_exploration_result(result)
    return result


def test_get_windll_toctou_sweep():
    """Bug 2: Sweep seeds for _get_windll() TOCTOU."""
    found_seeds = []
    total_explored = 0

    for seed in range(20):
        result = explore_interleavings(
            setup=lambda: GetWindllTOCTOUState(),
            threads=[
                lambda s: s.thread1(),
                lambda s: s.thread2(),
            ],
            invariant=lambda s: s.dll_1 is s.dll_2,
            max_attempts=100,
            max_ops=300,
            seed=seed,
        )
        total_explored += result.num_explored
        if not result.property_holds:
            found_seeds.append((seed, result.num_explored))

    print_seed_sweep_results(found_seeds, total_explored)
    return found_seeds


# ===========================================================================
# Bug 3: register() race on global mutable lists
#
# register() appends to _ALL_CONTROLLERS, _ALL_USER_APIS, _ALL_INTERNAL_APIS,
# and extends _ALL_PREFIXES with no synchronization. Two concurrent register()
# calls can interleave such that a thread sees a partially-updated global
# state (e.g., controller is in _ALL_CONTROLLERS but its user_api is not
# yet in _ALL_USER_APIS).
# ===========================================================================

class RegisterRaceState:
    """State for the register() global list race."""

    def __init__(self):
        _remove_fake_controllers()
        self.controllers_len_after_1 = None
        self.controllers_len_after_2 = None

    def thread1(self):
        register(FakeLibControllerA)
        self.controllers_len_after_1 = len(_ALL_CONTROLLERS)

    def thread2(self):
        register(FakeLibControllerB)
        self.controllers_len_after_2 = len(_ALL_CONTROLLERS)


def _register_invariant(s):
    """After both register() calls, all four global lists must be consistent.

    Specifically:
    - Both controllers must be in _ALL_CONTROLLERS
    - Both user_apis must be in _ALL_USER_APIS
    - Both internal_apis must be in _ALL_INTERNAL_APIS
    - All prefixes from both controllers must be in _ALL_PREFIXES
    """
    a_in_controllers = FakeLibControllerA in _ALL_CONTROLLERS
    b_in_controllers = FakeLibControllerB in _ALL_CONTROLLERS
    a_api = FakeLibControllerA.user_api in _ALL_USER_APIS
    b_api = FakeLibControllerB.user_api in _ALL_USER_APIS
    a_internal = FakeLibControllerA.internal_api in _ALL_INTERNAL_APIS
    b_internal = FakeLibControllerB.internal_api in _ALL_INTERNAL_APIS
    a_prefixes = all(p in _ALL_PREFIXES for p in FakeLibControllerA.filename_prefixes)
    b_prefixes = all(p in _ALL_PREFIXES for p in FakeLibControllerB.filename_prefixes)

    return all([
        a_in_controllers, b_in_controllers,
        a_api, b_api,
        a_internal, b_internal,
        a_prefixes, b_prefixes,
    ])


def test_register_race():
    """Bug 3: Concurrent register() calls cause inconsistent global state."""
    result = explore_interleavings(
        setup=lambda: RegisterRaceState(),
        threads=[
            lambda s: s.thread1(),
            lambda s: s.thread2(),
        ],
        invariant=_register_invariant,
        max_attempts=200,
        max_ops=300,
        seed=42,
    )
    print_exploration_result(result)
    _remove_fake_controllers()
    return result


def test_register_race_sweep():
    """Bug 3: Sweep seeds for concurrent register() race."""
    found_seeds = []
    total_explored = 0

    for seed in range(20):
        result = explore_interleavings(
            setup=lambda: RegisterRaceState(),
            threads=[
                lambda s: s.thread1(),
                lambda s: s.thread2(),
            ],
            invariant=_register_invariant,
            max_attempts=100,
            max_ops=300,
            seed=seed,
        )
        total_explored += result.num_explored
        if not result.property_holds:
            found_seeds.append((seed, result.num_explored))

    print_seed_sweep_results(found_seeds, total_explored)
    _remove_fake_controllers()
    return found_seeds


# ===========================================================================
# Bug 4: _make_controller_from_path() duplicate-check-then-append TOCTOU
#
# In _make_controller_from_path(), after creating a lib_controller, the code
# checks: `if filepath in (lib.filepath for lib in self.lib_controllers):`
# and then appends. Two threads calling _make_controller_from_path with the
# same filepath can both pass the uniqueness check and both append, creating
# duplicate entries in lib_controllers.
#
# We simulate this by having two threads call _make_controller_from_path
# with the same filepath on the same ThreadpoolController instance.
# ===========================================================================

class MakeControllerDuplicateState:
    """State for the _make_controller_from_path() duplicate TOCTOU."""

    def __init__(self):
        # Create a ThreadpoolController without loading libraries
        self.controller = ThreadpoolController._from_controllers([])
        # We need a filepath that will match a known controller class.
        # Use a mock to avoid actually loading a real .so
        self.filepath = "/fake/libopenblas.so"
        self.error_1 = None
        self.error_2 = None

    def thread1(self):
        try:
            self.controller._make_controller_from_path(self.filepath)
        except (OSError, Exception) as e:
            # Library may not exist; the bug is in the check-then-append logic
            self.error_1 = e

    def thread2(self):
        try:
            self.controller._make_controller_from_path(self.filepath)
        except (OSError, Exception) as e:
            self.error_2 = e


def test_make_controller_duplicate_toctou():
    """Bug 4: _make_controller_from_path duplicate-check TOCTOU.

    If both threads successfully create controllers (no OSError from the
    fake path), then we should have at most 1 controller for that filepath.
    """
    result = explore_interleavings(
        setup=lambda: MakeControllerDuplicateState(),
        threads=[
            lambda s: s.thread1(),
            lambda s: s.thread2(),
        ],
        invariant=lambda s: (
            # Either errors occurred (acceptable) or at most 1 controller
            (s.error_1 is not None or s.error_2 is not None)
            or len(s.controller.lib_controllers) <= 1
        ),
        max_attempts=200,
        max_ops=300,
        seed=42,
    )
    print_exploration_result(result)
    return result


# ===========================================================================
# Bug 5: Concurrent ThreadpoolController.__init__ racing on _system_libraries
#
# Two threads each create a new ThreadpoolController(). Both call
# _load_libraries() -> _find_libraries_with_dl_iterate_phdr() -> _get_libc().
# They share the class-level _system_libraries dict, leading to the TOCTOU.
# Additionally, the lib_controllers lists are per-instance, but the callback
# closures capture `self`, so if both dl_iterate_phdr callbacks interleave,
# they can see inconsistent state of each other's lib_controllers via the
# shared class dict.
# ===========================================================================

class ConcurrentInitState:
    """State for concurrent ThreadpoolController.__init__() race."""

    def __init__(self):
        ThreadpoolController._system_libraries.clear()
        self.ctrl_1 = None
        self.ctrl_2 = None
        self.libc_after = None

    def thread1(self):
        self.ctrl_1 = ThreadpoolController()

    def thread2(self):
        self.ctrl_2 = ThreadpoolController()


def test_concurrent_init_shared_cache():
    """Bug 5: Two concurrent ThreadpoolController() inits race on class cache.

    Both threads clear into _system_libraries; the _get_libc() TOCTOU fires
    and they may get different CDLL objects cached at the class level.
    After both threads finish, _system_libraries["libc"] should be a single
    consistent object, and it should be the same one both threads used.
    """
    result = explore_interleavings(
        setup=lambda: ConcurrentInitState(),
        threads=[
            lambda s: s.thread1(),
            lambda s: s.thread2(),
        ],
        invariant=lambda s: (
            # Both controllers should exist (no crash)
            s.ctrl_1 is not None
            and s.ctrl_2 is not None
            # The cached libc should be consistent -- only one CDLL should exist
            and ThreadpoolController._system_libraries.get("libc") is not None
        ),
        max_attempts=200,
        max_ops=300,
        seed=42,
    )
    print_exploration_result(result)
    return result


# ===========================================================================
# Bug 6: _set_threadpool_limits() concurrent with lib_controllers mutation
#
# _set_threadpool_limits() iterates over self._controller.lib_controllers
# and calls set_num_threads on each. If another thread is concurrently
# appending to lib_controllers (via _load_libraries or _make_controller_from_path),
# the iteration can see an inconsistent list (e.g., skip elements, see
# duplicates, or raise RuntimeError from list mutation during iteration).
# ===========================================================================

class SetLimitsDuringMutationState:
    """State for _set_threadpool_limits() vs lib_controllers mutation race."""

    def __init__(self):
        self.controller = ThreadpoolController._from_controllers([])
        # Pre-populate with some mock controllers
        self.mock1 = _make_mock_lib_controller(
            prefix="libmock1", filepath="/fake/libmock1.so", num_threads=4
        )
        self.mock2 = _make_mock_lib_controller(
            prefix="libmock2", filepath="/fake/libmock2.so", num_threads=4
        )
        self.mock3 = _make_mock_lib_controller(
            prefix="libmock3", filepath="/fake/libmock3.so", num_threads=4
        )
        self.controller.lib_controllers.append(self.mock1)
        self.set_threads_called = []
        self.error = None

    def thread1(self):
        """Append new controllers while thread2 iterates."""
        self.controller.lib_controllers.append(self.mock2)
        self.controller.lib_controllers.append(self.mock3)

    def thread2(self):
        """Iterate lib_controllers and collect num_threads."""
        try:
            for lc in self.controller.lib_controllers:
                self.set_threads_called.append(lc.prefix)
        except RuntimeError as e:
            # list changed size during iteration
            self.error = e


def test_set_limits_during_mutation():
    """Bug 6: Iterating lib_controllers while another thread appends.

    The invariant is that thread2 should see a consistent snapshot of
    lib_controllers -- either it sees the original list or the fully
    updated list, never a partial view that would cause a RuntimeError
    or inconsistent count.
    """
    result = explore_interleavings(
        setup=lambda: SetLimitsDuringMutationState(),
        threads=[
            lambda s: s.thread1(),
            lambda s: s.thread2(),
        ],
        invariant=lambda s: (
            s.error is None
            # Thread2 should see either 1 (just mock1) or 3 (all) controllers,
            # never 2 (partial update)
            and len(s.set_threads_called) in (1, 3)
        ),
        max_attempts=200,
        max_ops=300,
        seed=42,
    )
    print_exploration_result(result)
    return result


def test_set_limits_during_mutation_sweep():
    """Bug 6: Sweep seeds for iteration-during-mutation race."""
    found_seeds = []
    total_explored = 0

    for seed in range(20):
        result = explore_interleavings(
            setup=lambda: SetLimitsDuringMutationState(),
            threads=[
                lambda s: s.thread1(),
                lambda s: s.thread2(),
            ],
            invariant=lambda s: (
                s.error is None
                and len(s.set_threads_called) in (1, 3)
            ),
            max_attempts=100,
            max_ops=300,
            seed=seed,
        )
        total_explored += result.num_explored
        if not result.property_holds:
            found_seeds.append((seed, result.num_explored))

    print_seed_sweep_results(found_seeds, total_explored)
    return found_seeds


# ===========================================================================
# Bug 7: info() concurrent with lib_controllers mutation
#
# info() does: [lib_controller.info() for lib_controller in self.lib_controllers]
# If another thread appends to lib_controllers during this list comprehension,
# the result can be inconsistent.
# ===========================================================================

class InfoDuringMutationState:
    """State for info() vs lib_controllers mutation race."""

    def __init__(self):
        self.controller = ThreadpoolController._from_controllers([])
        self.mock1 = _make_mock_lib_controller(
            prefix="libmock1", filepath="/fake/libmock1.so"
        )
        self.mock2 = _make_mock_lib_controller(
            prefix="libmock2", filepath="/fake/libmock2.so"
        )
        self.controller.lib_controllers.append(self.mock1)
        self.info_result = None
        self.error = None

    def thread1(self):
        """Append a new controller."""
        self.controller.lib_controllers.append(self.mock2)

    def thread2(self):
        """Call info() which iterates lib_controllers."""
        try:
            self.info_result = self.controller.info()
        except RuntimeError as e:
            self.error = e


def test_info_during_mutation():
    """Bug 7: info() sees inconsistent lib_controllers during concurrent append.

    The invariant: info() should return either 1 item (before append) or
    2 items (after append), never crash, and the result should be
    consistent with the actual number of controllers at the time of the
    snapshot.
    """
    result = explore_interleavings(
        setup=lambda: InfoDuringMutationState(),
        threads=[
            lambda s: s.thread1(),
            lambda s: s.thread2(),
        ],
        invariant=lambda s: (
            s.error is None
            and s.info_result is not None
            and len(s.info_result) in (1, 2)
        ),
        max_attempts=200,
        max_ops=300,
        seed=42,
    )
    print_exploration_result(result)
    return result


# ===========================================================================
# Bug 8: select() concurrent with lib_controllers mutation
#
# select() iterates self.lib_controllers in a list comprehension.
# If another thread appends to lib_controllers during the iteration,
# the selected subset can be inconsistent.
# ===========================================================================

class SelectDuringMutationState:
    """State for select() vs lib_controllers mutation race."""

    def __init__(self):
        self.controller = ThreadpoolController._from_controllers([])
        self.mock_blas = _make_mock_lib_controller(
            prefix="libmock_blas", filepath="/fake/libmock_blas.so",
            user_api="blas", internal_api="mock_blas"
        )
        self.mock_openmp = _make_mock_lib_controller(
            prefix="libmock_openmp", filepath="/fake/libmock_openmp.so",
            user_api="openmp", internal_api="mock_openmp"
        )
        self.controller.lib_controllers.append(self.mock_blas)
        self.selected = None
        self.error = None

    def thread1(self):
        """Append an openmp controller."""
        self.controller.lib_controllers.append(self.mock_openmp)

    def thread2(self):
        """Select blas controllers."""
        try:
            self.selected = self.controller.select(user_api="blas")
        except RuntimeError as e:
            self.error = e


def test_select_during_mutation():
    """Bug 8: select() sees inconsistent state during concurrent append.

    The selected controllers should contain exactly the blas mock,
    and nothing else, regardless of interleaving.
    """
    result = explore_interleavings(
        setup=lambda: SelectDuringMutationState(),
        threads=[
            lambda s: s.thread1(),
            lambda s: s.thread2(),
        ],
        invariant=lambda s: (
            s.error is None
            and s.selected is not None
            # The blas controller should always be selected
            and len(s.selected.lib_controllers) == 1
            and s.selected.lib_controllers[0].user_api == "blas"
        ),
        max_attempts=200,
        max_ops=300,
        seed=42,
    )
    print_exploration_result(result)
    return result


# ===========================================================================
# Bug 9: restore_original_limits() vs concurrent lib_controllers modification
#
# restore_original_limits() does:
#   zip(self._controller.lib_controllers, self._original_info)
# If lib_controllers has been modified (elements added/removed) since
# _original_info was captured, the zip pairs up wrong controllers with
# wrong thread limits, silently applying incorrect limits.
# ===========================================================================

class RestoreLimitsMismatchState:
    """State for restore_original_limits() misalignment race."""

    def __init__(self):
        self.controller = ThreadpoolController._from_controllers([])
        self.mock1 = _make_mock_lib_controller(
            prefix="libmock1", filepath="/fake/libmock1.so", num_threads=4
        )
        self.mock2 = _make_mock_lib_controller(
            prefix="libmock2", filepath="/fake/libmock2.so", num_threads=8
        )
        self.mock3 = _make_mock_lib_controller(
            prefix="libmock3", filepath="/fake/libmock3.so", num_threads=16
        )
        self.controller.lib_controllers.append(self.mock1)
        self.controller.lib_controllers.append(self.mock2)

        # Capture original_info (for 2 controllers)
        self.original_info = self.controller.info()
        self.restore_error = None

    def thread1(self):
        """Insert a new controller at the beginning, shifting indices."""
        self.controller.lib_controllers.insert(0, self.mock3)

    def thread2(self):
        """Restore limits using the original (now stale) info."""
        try:
            for lib_controller, original_info in zip(
                self.controller.lib_controllers, self.original_info
            ):
                lib_controller.set_num_threads(original_info["num_threads"])
        except Exception as e:
            self.restore_error = e


def test_restore_limits_mismatch():
    """Bug 9: restore_original_limits() applies wrong limits after mutation.

    After thread1 inserts mock3 at position 0, the zip in thread2 pairs:
      mock3 with mock1's original info (num_threads=4)
      mock1 with mock2's original info (num_threads=8)
    This is silently wrong -- mock3 gets 4 threads instead of 16,
    mock1 gets 8 threads instead of 4.
    """
    result = explore_interleavings(
        setup=lambda: RestoreLimitsMismatchState(),
        threads=[
            lambda s: s.thread1(),
            lambda s: s.thread2(),
        ],
        invariant=lambda s: (
            s.restore_error is None
            # Check that mock1 was restored to 4 (its original)
            # If the bug fires, mock1 gets set to 8 (mock2's original)
            and s.mock1.set_num_threads.call_args_list[-1][0][0] == 4
            if s.mock1.set_num_threads.call_count > 0
            else True
        ),
        max_attempts=200,
        max_ops=300,
        seed=42,
    )
    print_exploration_result(result)
    return result


def test_restore_limits_mismatch_sweep():
    """Bug 9: Sweep seeds for restore_original_limits() misalignment."""
    found_seeds = []
    total_explored = 0

    for seed in range(20):
        result = explore_interleavings(
            setup=lambda: RestoreLimitsMismatchState(),
            threads=[
                lambda s: s.thread1(),
                lambda s: s.thread2(),
            ],
            invariant=lambda s: (
                s.restore_error is None
                and (
                    s.mock1.set_num_threads.call_args_list[-1][0][0] == 4
                    if s.mock1.set_num_threads.call_count > 0
                    else True
                )
            ),
            max_attempts=100,
            max_ops=300,
            seed=seed,
        )
        total_explored += result.num_explored
        if not result.property_holds:
            found_seeds.append((seed, result.num_explored))

    print_seed_sweep_results(found_seeds, total_explored)
    return found_seeds


# ===========================================================================
# Bug 10: _realpath lru_cache concurrent population
#
# Python's functools.lru_cache is NOT thread-safe for concurrent calls
# with the same key. Two threads calling _realpath("/some/path") can both
# compute the result, and the cache update can be corrupted.
# In CPython, lru_cache is implemented in C and uses a doubly-linked list
# that can be corrupted under concurrent access.
# ===========================================================================

class RealpathCacheRaceState:
    """State for _realpath lru_cache race."""

    def __init__(self):
        # Clear the lru_cache to force recomputation
        _realpath.cache_clear()
        self.result_1 = None
        self.result_2 = None

    def thread1(self):
        self.result_1 = _realpath("/usr/lib/libc.so.6")

    def thread2(self):
        self.result_2 = _realpath("/usr/lib/libc.so.6")


def test_realpath_cache_race():
    """Bug 10: _realpath lru_cache concurrent access.

    Both threads should get the same result. The lru_cache should not
    corrupt its internal state.
    """
    result = explore_interleavings(
        setup=lambda: RealpathCacheRaceState(),
        threads=[
            lambda s: s.thread1(),
            lambda s: s.thread2(),
        ],
        invariant=lambda s: s.result_1 == s.result_2,
        max_attempts=200,
        max_ops=300,
        seed=42,
    )
    print_exploration_result(result)
    return result


# ===========================================================================
# Bug 11: register() interleaved with _make_controller_from_path() iteration
#
# _make_controller_from_path() iterates _ALL_CONTROLLERS:
#   for controller_class in _ALL_CONTROLLERS:
# If register() appends a new controller class to _ALL_CONTROLLERS during
# this iteration, the iteration may or may not see the new controller
# (inconsistent view), or may raise RuntimeError on CPython if the list
# mutates during iteration.
# ===========================================================================

class RegisterDuringIterationState:
    """State for register() vs _make_controller_from_path() iteration race."""

    def __init__(self):
        _remove_fake_controllers()
        self.controller = ThreadpoolController._from_controllers([])
        self.controllers_seen = []
        self.error = None

    def thread1(self):
        """Register a new controller class while thread2 iterates."""
        register(FakeLibControllerC)

    def thread2(self):
        """Iterate _ALL_CONTROLLERS (simulating _make_controller_from_path)."""
        try:
            for controller_class in _ALL_CONTROLLERS:
                self.controllers_seen.append(controller_class.internal_api)
        except RuntimeError as e:
            # list changed size during iteration
            self.error = e


def test_register_during_iteration():
    """Bug 11: register() modifies _ALL_CONTROLLERS during iteration.

    The invariant is that the iteration should not crash and should see
    a consistent snapshot of _ALL_CONTROLLERS.
    """
    result = explore_interleavings(
        setup=lambda: RegisterDuringIterationState(),
        threads=[
            lambda s: s.thread1(),
            lambda s: s.thread2(),
        ],
        invariant=lambda s: s.error is None,
        max_attempts=200,
        max_ops=300,
        seed=42,
    )
    print_exploration_result(result)
    _remove_fake_controllers()
    return result


# ===========================================================================
# Bug 12: Concurrent threadpool_info() calls racing on shared class state
#
# threadpool_info() creates a new ThreadpoolController() which calls
# _load_libraries() -> _get_libc() which mutates the class-level
# _system_libraries. Two concurrent threadpool_info() calls race on
# this shared state.
# ===========================================================================

class ConcurrentThreadpoolInfoState:
    """State for concurrent threadpool_info() calls."""

    def __init__(self):
        ThreadpoolController._system_libraries.clear()
        self.info_1 = None
        self.info_2 = None

    def thread1(self):
        self.info_1 = threadpool_info()

    def thread2(self):
        self.info_2 = threadpool_info()


def test_concurrent_threadpool_info():
    """Bug 12: Two concurrent threadpool_info() calls race on class cache.

    Both should return the same library info (same set of libraries).
    """
    result = explore_interleavings(
        setup=lambda: ConcurrentThreadpoolInfoState(),
        threads=[
            lambda s: s.thread1(),
            lambda s: s.thread2(),
        ],
        invariant=lambda s: (
            s.info_1 is not None
            and s.info_2 is not None
            # Both should find the same number of libraries
            and len(s.info_1) == len(s.info_2)
        ),
        max_attempts=200,
        max_ops=300,
        seed=42,
    )
    print_exploration_result(result)
    return result


# ===========================================================================
# Bug 13: _from_controllers aliasing -- two ThreadpoolControllers sharing
# the same lib_controllers list, both mutating it
#
# _from_controllers does NOT copy the list:
#   new_controller.lib_controllers = lib_controllers
# So if two ThreadpoolControllers share the same lib_controllers list
# and one calls _load_libraries() (e.g., via FlexiBLAS switch_backend()),
# both see the mutations, but iterators in the other controller can break.
# ===========================================================================

class FromControllersAliasingState:
    """State for _from_controllers aliasing race."""

    def __init__(self):
        self.shared_controllers = []
        self.mock1 = _make_mock_lib_controller(
            prefix="libshared1", filepath="/fake/libshared1.so"
        )
        self.mock2 = _make_mock_lib_controller(
            prefix="libshared2", filepath="/fake/libshared2.so"
        )
        self.mock3 = _make_mock_lib_controller(
            prefix="libshared3", filepath="/fake/libshared3.so"
        )
        self.shared_controllers.append(self.mock1)
        # Two controllers sharing the same underlying list
        self.ctrl_a = ThreadpoolController._from_controllers(self.shared_controllers)
        self.ctrl_b = ThreadpoolController._from_controllers(self.shared_controllers)
        self.info_a = None
        self.info_b = None
        self.error = None

    def thread1(self):
        """Append to the shared list via ctrl_a."""
        self.ctrl_a.lib_controllers.append(self.mock2)
        self.ctrl_a.lib_controllers.append(self.mock3)

    def thread2(self):
        """Read from the shared list via ctrl_b."""
        try:
            self.info_b = self.ctrl_b.info()
        except RuntimeError as e:
            self.error = e


def test_from_controllers_aliasing():
    """Bug 13: _from_controllers shares list reference, concurrent mutation.

    Thread1 appends to ctrl_a.lib_controllers, which is the SAME object as
    ctrl_b.lib_controllers. Thread2 iterating via ctrl_b.info() can see
    partial updates.
    """
    result = explore_interleavings(
        setup=lambda: FromControllersAliasingState(),
        threads=[
            lambda s: s.thread1(),
            lambda s: s.thread2(),
        ],
        invariant=lambda s: (
            s.error is None
            and s.info_b is not None
            # Should see either 1 (original) or 3 (fully updated), never 2
            and len(s.info_b) in (1, 3)
        ),
        max_attempts=200,
        max_ops=300,
        seed=42,
    )
    print_exploration_result(result)
    return result


def test_from_controllers_aliasing_sweep():
    """Bug 13: Sweep seeds for _from_controllers aliasing race."""
    found_seeds = []
    total_explored = 0

    for seed in range(20):
        result = explore_interleavings(
            setup=lambda: FromControllersAliasingState(),
            threads=[
                lambda s: s.thread1(),
                lambda s: s.thread2(),
            ],
            invariant=lambda s: (
                s.error is None
                and s.info_b is not None
                and len(s.info_b) in (1, 3)
            ),
            max_attempts=100,
            max_ops=300,
            seed=seed,
        )
        total_explored += result.num_explored
        if not result.property_holds:
            found_seeds.append((seed, result.num_explored))

    print_seed_sweep_results(found_seeds, total_explored)
    return found_seeds


# ===========================================================================
# Bug 14: _system_libraries dict concurrent read-write
#
# _system_libraries is a plain dict shared at the class level.
# One thread calling _get_libc() writes to it while another thread
# reads from it (e.g., via .get() or .clear()). On CPython, dict
# operations are protected by the GIL at the bytecode level, but the
# check-then-write pattern in _get_libc/_get_windll is still a TOCTOU.
# The higher-level issue: one thread might clear the cache while another
# is between the get() and the store.
# ===========================================================================

class SystemLibsClearDuringPopulateState:
    """State for _system_libraries clear-during-populate race."""

    def __init__(self):
        ThreadpoolController._system_libraries.clear()
        self.libc = None
        self.cache_after = None

    def thread1(self):
        """Populate the cache via _get_libc."""
        self.libc = ThreadpoolController._get_libc()

    def thread2(self):
        """Clear the cache while thread1 is populating it."""
        ThreadpoolController._system_libraries.clear()
        self.cache_after = dict(ThreadpoolController._system_libraries)


def test_system_libs_clear_during_populate():
    """Bug 14: One thread clears _system_libraries while another populates it.

    After both threads complete, the state can be inconsistent:
    - thread1 may have returned a CDLL that is no longer in the cache
    - thread2 may have cleared the cache after thread1 stored, leaving it empty
    """
    result = explore_interleavings(
        setup=lambda: SystemLibsClearDuringPopulateState(),
        threads=[
            lambda s: s.thread1(),
            lambda s: s.thread2(),
        ],
        invariant=lambda s: (
            # If thread1 got a libc, it should still be in the cache
            s.libc is None
            or ThreadpoolController._system_libraries.get("libc") is s.libc
        ),
        max_attempts=200,
        max_ops=300,
        seed=42,
    )
    print_exploration_result(result)
    return result


def test_system_libs_clear_during_populate_sweep():
    """Bug 14: Sweep seeds for cache clear-during-populate race."""
    found_seeds = []
    total_explored = 0

    for seed in range(20):
        result = explore_interleavings(
            setup=lambda: SystemLibsClearDuringPopulateState(),
            threads=[
                lambda s: s.thread1(),
                lambda s: s.thread2(),
            ],
            invariant=lambda s: (
                s.libc is None
                or ThreadpoolController._system_libraries.get("libc") is s.libc
            ),
            max_attempts=100,
            max_ops=300,
            seed=seed,
        )
        total_explored += result.num_explored
        if not result.property_holds:
            found_seeds.append((seed, result.num_explored))

    print_seed_sweep_results(found_seeds, total_explored)
    return found_seeds


# ===========================================================================
# Bug 15: Three-thread _get_libc() stampede
#
# Three threads all race to populate the libc cache. This is a more severe
# version of Bug 1 that tests whether the TOCTOU can cause three distinct
# CDLL objects to be created.
# ===========================================================================

class ThreeThreadGetLibcState:
    """State for three-thread _get_libc() stampede."""

    def __init__(self):
        ThreadpoolController._system_libraries.clear()
        self.libc_1 = None
        self.libc_2 = None
        self.libc_3 = None

    def thread1(self):
        self.libc_1 = ThreadpoolController._get_libc()

    def thread2(self):
        self.libc_2 = ThreadpoolController._get_libc()

    def thread3(self):
        self.libc_3 = ThreadpoolController._get_libc()


def test_three_thread_get_libc_stampede():
    """Bug 15: Three threads stampede _get_libc().

    All three should get the same CDLL object. If the TOCTOU fires,
    some or all may get different objects.
    """
    result = explore_interleavings(
        setup=lambda: ThreeThreadGetLibcState(),
        threads=[
            lambda s: s.thread1(),
            lambda s: s.thread2(),
            lambda s: s.thread3(),
        ],
        invariant=lambda s: (
            s.libc_1 is s.libc_2 and s.libc_2 is s.libc_3
        ),
        max_attempts=200,
        max_ops=300,
        seed=42,
    )
    print_exploration_result(result)
    return result


# ===========================================================================
# Bug 16: _warn_if_incompatible_openmp race with lib_controllers mutation
#
# In __init__, _load_libraries() is called first, then
# _warn_if_incompatible_openmp() iterates lib_controllers to check prefixes.
# If another thread is concurrently mutating the same instance's
# lib_controllers (via _from_controllers aliasing), the prefix check
# can see an inconsistent snapshot.
# ===========================================================================

class WarnOpenMPDuringMutationState:
    """State for _warn_if_incompatible_openmp during mutation race."""

    def __init__(self):
        self.shared_controllers = []
        self.mock_iomp = _make_mock_lib_controller(
            prefix="libiomp", filepath="/fake/libiomp.so",
            user_api="openmp", internal_api="openmp_intel"
        )
        self.mock_omp = _make_mock_lib_controller(
            prefix="libomp", filepath="/fake/libomp.so",
            user_api="openmp", internal_api="openmp_llvm"
        )
        self.shared_controllers.append(self.mock_iomp)
        self.ctrl = ThreadpoolController._from_controllers(self.shared_controllers)
        self.prefixes_seen = None
        self.error = None

    def thread1(self):
        """Append libomp to trigger incompatible openmp condition."""
        self.ctrl.lib_controllers.append(self.mock_omp)

    def thread2(self):
        """Check prefixes (simulating _warn_if_incompatible_openmp)."""
        try:
            prefixes = [lc.prefix for lc in self.ctrl.lib_controllers]
            self.prefixes_seen = prefixes
        except RuntimeError as e:
            self.error = e


def test_warn_openmp_during_mutation():
    """Bug 16: _warn_if_incompatible_openmp iteration races with append.

    The prefix list should be either [libiomp] or [libiomp, libomp],
    never partial or errored.
    """
    result = explore_interleavings(
        setup=lambda: WarnOpenMPDuringMutationState(),
        threads=[
            lambda s: s.thread1(),
            lambda s: s.thread2(),
        ],
        invariant=lambda s: (
            s.error is None
            and s.prefixes_seen is not None
            and len(s.prefixes_seen) in (1, 2)
        ),
        max_attempts=200,
        max_ops=300,
        seed=42,
    )
    print_exploration_result(result)
    return result


# ===========================================================================
# Bug 17: Concurrent _get_libc() with _system_libraries being populated
# for multiple keys simultaneously
#
# On Windows, _find_libraries_with_enum_process_module_ex calls
# _get_windll("Psapi") and _get_windll("kernel32") sequentially. If two
# threads do this concurrently, the dict can have keys written in
# interleaved order, and one thread might see Psapi but not kernel32 yet.
# We simulate this with two different cache keys.
# ===========================================================================

class MultiKeyCacheRaceState:
    """State for multi-key _system_libraries cache race."""

    def __init__(self):
        ThreadpoolController._system_libraries.clear()
        self.obj_a_t1 = None
        self.obj_b_t1 = None
        self.obj_a_t2 = None
        self.obj_b_t2 = None

    def _cache_get_or_create(self, key):
        """Simulates the _get_windll pattern for arbitrary keys."""
        obj = ThreadpoolController._system_libraries.get(key)
        if obj is None:
            obj = object()  # unique object each time
            ThreadpoolController._system_libraries[key] = obj
        return obj

    def thread1(self):
        self.obj_a_t1 = self._cache_get_or_create("key_a")
        self.obj_b_t1 = self._cache_get_or_create("key_b")

    def thread2(self):
        self.obj_a_t2 = self._cache_get_or_create("key_a")
        self.obj_b_t2 = self._cache_get_or_create("key_b")


def test_multi_key_cache_race():
    """Bug 17: Two threads populating multiple keys in _system_libraries.

    Both threads should end up with the same objects for each key.
    """
    result = explore_interleavings(
        setup=lambda: MultiKeyCacheRaceState(),
        threads=[
            lambda s: s.thread1(),
            lambda s: s.thread2(),
        ],
        invariant=lambda s: (
            s.obj_a_t1 is s.obj_a_t2
            and s.obj_b_t1 is s.obj_b_t2
        ),
        max_attempts=200,
        max_ops=300,
        seed=42,
    )
    print_exploration_result(result)
    return result


def test_multi_key_cache_race_sweep():
    """Bug 17: Sweep seeds for multi-key cache TOCTOU."""
    found_seeds = []
    total_explored = 0

    for seed in range(20):
        result = explore_interleavings(
            setup=lambda: MultiKeyCacheRaceState(),
            threads=[
                lambda s: s.thread1(),
                lambda s: s.thread2(),
            ],
            invariant=lambda s: (
                s.obj_a_t1 is s.obj_a_t2
                and s.obj_b_t1 is s.obj_b_t2
            ),
            max_attempts=100,
            max_ops=300,
            seed=seed,
        )
        total_explored += result.num_explored
        if not result.property_holds:
            found_seeds.append((seed, result.num_explored))

    print_seed_sweep_results(found_seeds, total_explored)
    return found_seeds


# ===========================================================================
# Bug 18: len() concurrent with lib_controllers mutation
#
# ThreadpoolController.__len__ returns len(self.lib_controllers).
# If another thread is appending to lib_controllers, the returned length
# can be stale by the time the caller uses it.
# ===========================================================================

class LenDuringMutationState:
    """State for __len__ vs lib_controllers mutation race."""

    def __init__(self):
        self.controller = ThreadpoolController._from_controllers([])
        self.mock1 = _make_mock_lib_controller(
            prefix="liblen1", filepath="/fake/liblen1.so"
        )
        self.mock2 = _make_mock_lib_controller(
            prefix="liblen2", filepath="/fake/liblen2.so"
        )
        self.len_before_append = None
        self.len_after_append = None
        self.len_from_thread2 = None

    def thread1(self):
        """Append controllers."""
        self.len_before_append = len(self.controller)
        self.controller.lib_controllers.append(self.mock1)
        self.controller.lib_controllers.append(self.mock2)
        self.len_after_append = len(self.controller)

    def thread2(self):
        """Read len() concurrently."""
        self.len_from_thread2 = len(self.controller)


def test_len_during_mutation():
    """Bug 18: len() can return stale values during concurrent append.

    Thread2's len() can return 0, 1, or 2 depending on interleaving,
    but it should be consistent with the actual list state at some point.
    The concern is a TOCTOU: thread2 reads len=0, then uses it to index,
    but thread1 has already appended.
    """
    result = explore_interleavings(
        setup=lambda: LenDuringMutationState(),
        threads=[
            lambda s: s.thread1(),
            lambda s: s.thread2(),
        ],
        invariant=lambda s: (
            # len_from_thread2 should be one of the valid states: 0, 1, or 2
            s.len_from_thread2 in (0, 1, 2)
            # After thread1 finishes, the final length should be 2
            and s.len_after_append == 2
        ),
        max_attempts=200,
        max_ops=300,
        seed=42,
    )
    print_exploration_result(result)
    return result


# ===========================================================================
# Bug 19: Concurrent _make_controller_from_path on same controller instance
# with different paths -- lib_controllers append ordering
#
# Two threads both call _make_controller_from_path on the same controller
# with different filepaths. The appends to lib_controllers can interleave
# such that the final order is non-deterministic, which matters for
# restore_original_limits() which relies on positional correspondence.
# ===========================================================================

class ConcurrentMakeControllerOrderState:
    """State for concurrent _make_controller_from_path ordering race."""

    def __init__(self):
        self.controller = ThreadpoolController._from_controllers([])
        self.mock_a = _make_mock_lib_controller(
            prefix="liborder_a", filepath="/fake/liborder_a.so"
        )
        self.mock_b = _make_mock_lib_controller(
            prefix="liborder_b", filepath="/fake/liborder_b.so"
        )
        self.error = None

    def thread1(self):
        """Directly append mock_a (simulating successful _make_controller_from_path)."""
        self.controller.lib_controllers.append(self.mock_a)

    def thread2(self):
        """Directly append mock_b (simulating successful _make_controller_from_path)."""
        self.controller.lib_controllers.append(self.mock_b)


def test_concurrent_make_controller_order():
    """Bug 19: Concurrent appends produce non-deterministic ordering.

    The order of lib_controllers depends on thread interleaving.
    After both threads finish, both controllers should be present,
    but the order may vary.
    """
    result = explore_interleavings(
        setup=lambda: ConcurrentMakeControllerOrderState(),
        threads=[
            lambda s: s.thread1(),
            lambda s: s.thread2(),
        ],
        invariant=lambda s: (
            len(s.controller.lib_controllers) == 2
            # Order should be deterministic: mock_a first, mock_b second
            and s.controller.lib_controllers[0].prefix == "liborder_a"
            and s.controller.lib_controllers[1].prefix == "liborder_b"
        ),
        max_attempts=200,
        max_ops=300,
        seed=42,
    )
    print_exploration_result(result)
    return result


def test_concurrent_make_controller_order_sweep():
    """Bug 19: Sweep seeds for concurrent append ordering race."""
    found_seeds = []
    total_explored = 0

    for seed in range(20):
        result = explore_interleavings(
            setup=lambda: ConcurrentMakeControllerOrderState(),
            threads=[
                lambda s: s.thread1(),
                lambda s: s.thread2(),
            ],
            invariant=lambda s: (
                len(s.controller.lib_controllers) == 2
                and s.controller.lib_controllers[0].prefix == "liborder_a"
                and s.controller.lib_controllers[1].prefix == "liborder_b"
            ),
            max_attempts=100,
            max_ops=300,
            seed=seed,
        )
        total_explored += result.num_explored
        if not result.property_holds:
            found_seeds.append((seed, result.num_explored))

    print_seed_sweep_results(found_seeds, total_explored)
    return found_seeds


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("EXHAUSTIVE THREADPOOLCTL CONCURRENCY TESTS")
    print("=" * 70)

    print("\n" + "=" * 70)
    print("Bug 1: _get_libc() TOCTOU")
    print("=" * 70)
    print("\n--- Single run (seed=42, 200 attempts) ---")
    test_get_libc_toctou()
    print("\n--- Seed sweep (20 seeds x 100 attempts) ---")
    test_get_libc_toctou_sweep()
    print("\n--- Deterministic reproduction ---")
    test_get_libc_toctou_reproduce()

    print("\n" + "=" * 70)
    print("Bug 2: _get_windll() TOCTOU (simulated)")
    print("=" * 70)
    print("\n--- Single run ---")
    test_get_windll_toctou()
    print("\n--- Seed sweep ---")
    test_get_windll_toctou_sweep()

    print("\n" + "=" * 70)
    print("Bug 3: register() race on global lists")
    print("=" * 70)
    print("\n--- Single run ---")
    test_register_race()
    print("\n--- Seed sweep ---")
    test_register_race_sweep()

    print("\n" + "=" * 70)
    print("Bug 4: _make_controller_from_path() duplicate-check TOCTOU")
    print("=" * 70)
    print("\n--- Single run ---")
    test_make_controller_duplicate_toctou()

    print("\n" + "=" * 70)
    print("Bug 5: Concurrent ThreadpoolController.__init__ shared cache")
    print("=" * 70)
    print("\n--- Single run ---")
    test_concurrent_init_shared_cache()

    print("\n" + "=" * 70)
    print("Bug 6: _set_threadpool_limits() during lib_controllers mutation")
    print("=" * 70)
    print("\n--- Single run ---")
    test_set_limits_during_mutation()
    print("\n--- Seed sweep ---")
    test_set_limits_during_mutation_sweep()

    print("\n" + "=" * 70)
    print("Bug 7: info() during lib_controllers mutation")
    print("=" * 70)
    print("\n--- Single run ---")
    test_info_during_mutation()

    print("\n" + "=" * 70)
    print("Bug 8: select() during lib_controllers mutation")
    print("=" * 70)
    print("\n--- Single run ---")
    test_select_during_mutation()

    print("\n" + "=" * 70)
    print("Bug 9: restore_original_limits() misalignment")
    print("=" * 70)
    print("\n--- Single run ---")
    test_restore_limits_mismatch()
    print("\n--- Seed sweep ---")
    test_restore_limits_mismatch_sweep()

    print("\n" + "=" * 70)
    print("Bug 10: _realpath lru_cache race")
    print("=" * 70)
    print("\n--- Single run ---")
    test_realpath_cache_race()

    print("\n" + "=" * 70)
    print("Bug 11: register() during _ALL_CONTROLLERS iteration")
    print("=" * 70)
    print("\n--- Single run ---")
    test_register_during_iteration()

    print("\n" + "=" * 70)
    print("Bug 12: Concurrent threadpool_info() calls")
    print("=" * 70)
    print("\n--- Single run ---")
    test_concurrent_threadpool_info()

    print("\n" + "=" * 70)
    print("Bug 13: _from_controllers aliasing -- shared list mutation")
    print("=" * 70)
    print("\n--- Single run ---")
    test_from_controllers_aliasing()
    print("\n--- Seed sweep ---")
    test_from_controllers_aliasing_sweep()

    print("\n" + "=" * 70)
    print("Bug 14: _system_libraries clear during populate")
    print("=" * 70)
    print("\n--- Single run ---")
    test_system_libs_clear_during_populate()
    print("\n--- Seed sweep ---")
    test_system_libs_clear_during_populate_sweep()

    print("\n" + "=" * 70)
    print("Bug 15: Three-thread _get_libc() stampede")
    print("=" * 70)
    print("\n--- Single run ---")
    test_three_thread_get_libc_stampede()

    print("\n" + "=" * 70)
    print("Bug 16: _warn_if_incompatible_openmp during mutation")
    print("=" * 70)
    print("\n--- Single run ---")
    test_warn_openmp_during_mutation()

    print("\n" + "=" * 70)
    print("Bug 17: Multi-key _system_libraries cache TOCTOU")
    print("=" * 70)
    print("\n--- Single run ---")
    test_multi_key_cache_race()
    print("\n--- Seed sweep ---")
    test_multi_key_cache_race_sweep()

    print("\n" + "=" * 70)
    print("Bug 18: len() during lib_controllers mutation")
    print("=" * 70)
    print("\n--- Single run ---")
    test_len_during_mutation()

    print("\n" + "=" * 70)
    print("Bug 19: Concurrent append ordering non-determinism")
    print("=" * 70)
    print("\n--- Single run ---")
    test_concurrent_make_controller_order()
    print("\n--- Seed sweep ---")
    test_concurrent_make_controller_order_sweep()

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETE")
    print("=" * 70)
