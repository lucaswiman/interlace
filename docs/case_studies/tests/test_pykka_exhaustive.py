"""
Exhaustive concurrency tests for pykka using interlace bytecode exploration.

Tests multiple distinct concurrency bugs in pykka's actor system:

1. ask() TOCTOU ghost ask -- same pattern as tell() but for ask()
2. Concurrent tell() + tell() message ordering/loss
3. ask() + stop() race -- future that may never resolve or ghost response
4. ActorRegistry concurrent register() race
5. ActorRegistry concurrent unregister() race
6. ActorRegistry.stop_all() while actor is still starting
7. Double stop() race on the same actor
8. tell() ghost message with concurrent stop (baseline, from existing test)

Repository: https://github.com/jodal/pykka
"""

import os
import sys

_test_dir = os.path.dirname(os.path.abspath(__file__))
_repo_src = os.path.join(_test_dir, "..", "external_repos", "pykka", "src")
# Insert local repo path FIRST so interlace can trace it (site-packages are excluded).
sys.path.insert(0, os.path.abspath(_repo_src))

import pykka  # noqa: E402
from case_study_helpers import (  # noqa: E402
    print_exploration_result,
    print_seed_sweep_results,
    timeout_minutes,
)
from pykka import ActorDeadError, Timeout  # noqa: E402

from interlace.bytecode import explore_interleavings, run_with_schedule  # noqa: E402


# ---------------------------------------------------------------------------
# Helper: daemon actor base
# ---------------------------------------------------------------------------
class DaemonActor(pykka.ThreadingActor):
    """ThreadingActor that runs on a daemon thread so it never blocks exit."""
    use_daemon_thread = True


# ===========================================================================
# Test 1: ask() TOCTOU -- ghost ask
#
# ActorRef.ask() has the same TOCTOU as tell():
#
#     def ask(self, message, ...):
#         future = self.actor_class._create_future()
#         try:
#             if not self.is_alive():          # CHECK
#                 raise ActorDeadError(...)
#         except ActorDeadError:
#             future.set_exception()
#         else:
#             self.actor_inbox.put(...)         # ACT  <-- no lock
#         ...
#
# If stop() fires between the is_alive() check and the inbox.put(), the
# message is enqueued into a dead actor's inbox.  The actor loop has exited,
# so the future's reply_to is never fulfilled.  If block=False, the caller
# gets a future that hangs forever.  With block=True and a timeout, it
# raises Timeout instead of ActorDeadError -- a semantic surprise.
#
# Invariant: if ask(block=False) returns a future without setting an
# exception immediately, the actor must have actually processed the message.
# ===========================================================================

class AskTOCTOUState:
    """State for ask() TOCTOU ghost-ask test."""

    def __init__(self):
        self.received = []
        received = self.received

        class MyActor(DaemonActor):
            def on_receive(self, message):
                received.append(message)
                return "ok"

        self.actor_ref = MyActor.start()
        self.ask_successes = 0  # ask returned a future (no immediate exception)
        self.ask_errors = 0     # ask raised ActorDeadError immediately
        self.ask_timeout = 0    # future.get() timed out (ghost ask)

    def thread1(self):
        """Send an ask() to the actor (non-blocking) then try to get result."""
        try:
            future = self.actor_ref.ask("ping", block=False)
            self.ask_successes += 1
            # Try to get the result with a short timeout
            try:
                future.get(timeout=2.0)
            except ActorDeadError:
                # Actor died after enqueue but before processing -- that is
                # handled in teardown, so the future gets an exception.  Fine.
                pass
            except Timeout:
                # Future never resolved -- ghost ask!
                self.ask_timeout += 1
        except ActorDeadError:
            self.ask_errors += 1

    def thread2(self):
        """Stop the actor."""
        try:
            self.actor_ref.stop(block=True, timeout=2.0)
        except ActorDeadError:
            pass
        except Timeout:
            pass


def _no_ghost_asks(s: AskTOCTOUState) -> bool:
    """If ask() succeeded (no immediate error), the actor must have processed
    or properly rejected the message.  A timeout means the future was orphaned."""
    return s.ask_timeout == 0


def test_pykka_ask_toctou():
    """Find the ask() ghost-ask TOCTOU in real pykka ActorRef."""
    with timeout_minutes(10):
        result = explore_interleavings(
            setup=lambda: AskTOCTOUState(),
            threads=[lambda s: s.thread1(), lambda s: s.thread2()],
            invariant=_no_ghost_asks,
            max_attempts=500,
            max_ops=500,
            seed=42,
        )
    print_exploration_result(result)
    return result


# ===========================================================================
# Test 2: tell() ghost message (same as existing test, included for
# completeness in the exhaustive file)
#
# tell() checks is_alive() then puts -- classic TOCTOU.
# ===========================================================================

class TellGhostState:
    """State for tell() TOCTOU ghost-message test."""

    def __init__(self):
        self.received = []
        received = self.received

        class MyActor(DaemonActor):
            def on_receive(self, message):
                received.append(message)

        self.actor_ref = MyActor.start()
        self.tell_successes = 0
        self.tell_errors = 0

    def thread1(self):
        try:
            self.actor_ref.tell("ping")
            self.tell_successes += 1
        except ActorDeadError:
            self.tell_errors += 1

    def thread2(self):
        try:
            self.actor_ref.stop(block=True, timeout=2.0)
        except ActorDeadError:
            pass
        except Timeout:
            pass


def _no_ghost_tells(s: TellGhostState) -> bool:
    return s.tell_successes == len(s.received)


def test_pykka_tell_ghost():
    """Find the tell() ghost-message TOCTOU."""
    with timeout_minutes(10):
        result = explore_interleavings(
            setup=lambda: TellGhostState(),
            threads=[lambda s: s.thread1(), lambda s: s.thread2()],
            invariant=_no_ghost_tells,
            max_attempts=500,
            max_ops=500,
            seed=42,
        )
    print_exploration_result(result)
    return result


# ===========================================================================
# Test 3: Concurrent tell() + tell() -- message loss or duplication
#
# Two threads both call tell() concurrently.  The actor should receive
# exactly the messages that were successfully sent (no tell raised
# ActorDeadError).  Due to the TOCTOU in tell(), one thread might enqueue
# a ghost message after the actor stopped, but the other thread's tell
# might also slip through.
#
# We test: number of messages received == number of successful tells.
# ===========================================================================

class ConcurrentTellState:
    """State for concurrent tell() + tell() test."""

    def __init__(self):
        self.received = []
        received = self.received

        class MyActor(DaemonActor):
            def on_receive(self, message):
                received.append(message)

        self.actor_ref = MyActor.start()
        self.tell1_ok = 0
        self.tell1_err = 0
        self.tell2_ok = 0
        self.tell2_err = 0

    def thread1(self):
        """First concurrent tell."""
        try:
            self.actor_ref.tell("msg_from_t1")
            self.tell1_ok += 1
        except ActorDeadError:
            self.tell1_err += 1

    def thread2(self):
        """Second concurrent tell then stop."""
        try:
            self.actor_ref.tell("msg_from_t2")
            self.tell2_ok += 1
        except ActorDeadError:
            self.tell2_err += 1
        # Now stop the actor
        try:
            self.actor_ref.stop(block=True, timeout=2.0)
        except (ActorDeadError, Timeout):
            pass


def _concurrent_tells_consistent(s: ConcurrentTellState) -> bool:
    """Every successful tell must be received by the actor."""
    total_successes = s.tell1_ok + s.tell2_ok
    return total_successes == len(s.received)


def test_pykka_concurrent_tells():
    """Two concurrent tells + stop: message count must match successes."""
    with timeout_minutes(10):
        result = explore_interleavings(
            setup=lambda: ConcurrentTellState(),
            threads=[lambda s: s.thread1(), lambda s: s.thread2()],
            invariant=_concurrent_tells_consistent,
            max_attempts=500,
            max_ops=500,
            seed=42,
        )
    print_exploration_result(result)
    return result


# ===========================================================================
# Test 4: ask() + stop() race -- orphaned future
#
# Thread 1 calls ask(block=False), getting a future.
# Thread 2 calls stop(block=True).
#
# If the ask's envelope lands in the inbox *after* the actor loop has
# exited but *before* teardown drains the inbox, the future might be
# properly rejected (ActorDeadError).  But if the envelope lands *after*
# teardown has already finished draining, the future is orphaned -- no one
# will ever set a value or exception on it.
#
# Additionally, there is a subtlety: ask() creates the future, checks
# is_alive(), then puts.  If the actor dies between check and put, the
# message lands in the inbox.  The teardown loop (in _actor_loop_teardown)
# calls actor_inbox.empty() then actor_inbox.get().  But if the put happens
# *after* empty() returns True, the teardown loop exits and the message is
# stranded.
#
# We check: the future must resolve (with either a value or an exception)
# within a reasonable timeout.
# ===========================================================================

class AskStopRaceState:
    """State for ask + stop race test."""

    def __init__(self):
        self.received = []
        received = self.received

        class MyActor(DaemonActor):
            def on_receive(self, message):
                received.append(message)
                return "reply"

        self.actor_ref = MyActor.start()
        self.future = None
        self.future_resolved = False
        self.future_orphaned = False
        self.ask_raised = False

    def thread1(self):
        """Send ask(block=False) and store the future."""
        try:
            self.future = self.actor_ref.ask("hello", block=False)
        except ActorDeadError:
            self.ask_raised = True

    def thread2(self):
        """Stop the actor."""
        try:
            self.actor_ref.stop(block=True, timeout=2.0)
        except (ActorDeadError, Timeout):
            pass

        # After stop completes, try to resolve the future
        if self.future is not None:
            try:
                self.future.get(timeout=1.0)
                self.future_resolved = True
            except ActorDeadError:
                self.future_resolved = True  # rejected properly
            except Timeout:
                self.future_orphaned = True  # orphaned!
            except Exception:
                self.future_resolved = True  # some other exception, but resolved


def _no_orphaned_futures(s: AskStopRaceState) -> bool:
    """The future must resolve (value or exception), never orphan."""
    if s.ask_raised:
        return True  # ask itself raised, no future to worry about
    if s.future is None:
        return True  # thread1 hasn't run yet (shouldn't happen after both threads)
    return not s.future_orphaned


def test_pykka_ask_stop_orphaned_future():
    """ask() + stop() race: future must not be orphaned."""
    with timeout_minutes(10):
        result = explore_interleavings(
            setup=lambda: AskStopRaceState(),
            threads=[lambda s: s.thread1(), lambda s: s.thread2()],
            invariant=_no_orphaned_futures,
            max_attempts=500,
            max_ops=500,
            seed=42,
        )
    print_exploration_result(result)
    return result


# ===========================================================================
# Test 5: Double stop() race
#
# Two threads both call stop() on the same actor concurrently.
# ActorRef.stop() calls self.ask(_ActorStop(), block=False) which itself
# calls ask() which has the TOCTOU.  The _ActorStop message triggers
# Actor._stop() which calls:
#   - ActorRegistry.unregister(self.actor_ref)
#   - self.actor_stopped.set()
#
# With two concurrent stops, the second _ActorStop message may be enqueued
# after the first one has already stopped the actor.  The second stop's
# future will then be orphaned (never resolved) because the actor loop
# has exited and teardown may have already drained the inbox.
#
# Invariant: both stop calls must return without hanging.
# ===========================================================================

class DoubleStopState:
    """State for double stop() race test."""

    def __init__(self):
        class MyActor(DaemonActor):
            def on_receive(self, message):
                return "ok"

        self.actor_ref = MyActor.start()
        self.stop1_result = None
        self.stop2_result = None
        self.stop1_timeout = False
        self.stop2_timeout = False

    def thread1(self):
        """First concurrent stop."""
        try:
            self.stop1_result = self.actor_ref.stop(block=True, timeout=2.0)
        except ActorDeadError:
            self.stop1_result = "dead"
        except Timeout:
            self.stop1_timeout = True

    def thread2(self):
        """Second concurrent stop."""
        try:
            self.stop2_result = self.actor_ref.stop(block=True, timeout=2.0)
        except ActorDeadError:
            self.stop2_result = "dead"
        except Timeout:
            self.stop2_timeout = True


def _double_stop_no_hang(s: DoubleStopState) -> bool:
    """Neither stop() call should time out (hang)."""
    return not s.stop1_timeout and not s.stop2_timeout


def test_pykka_double_stop():
    """Two concurrent stop() calls on the same actor must not hang."""
    with timeout_minutes(10):
        result = explore_interleavings(
            setup=lambda: DoubleStopState(),
            threads=[lambda s: s.thread1(), lambda s: s.thread2()],
            invariant=_double_stop_no_hang,
            max_attempts=500,
            max_ops=500,
            seed=42,
        )
    print_exploration_result(result)
    return result


# ===========================================================================
# Test 6: ActorRegistry concurrent register() -- duplicate entries
#
# ActorRegistry.register() acquires _actor_refs_lock then appends.
# If two actors start concurrently, register() is individually atomic
# (thanks to the lock), but the question is whether the registry ends up
# in a consistent state and both actors are findable.
#
# More interesting: what if register() and stop_all() race?
# stop_all() calls get_all() (snapshot under lock) then iterates,
# calling ref.stop() on each.  An actor registered *after* the snapshot
# is taken will not be stopped by stop_all().
#
# Invariant: after stop_all(block=True), get_all() should return empty.
# ===========================================================================

class RegistryStopAllRaceState:
    """State for registry stop_all() vs late registration race."""

    def __init__(self):
        # Start with a clean registry -- stop and unregister everything
        # from previous tests.
        try:
            pykka.ActorRegistry.stop_all(block=True, timeout=2.0)
        except Exception:
            pass

        self.late_ref = None
        self.stop_all_done = False
        self.registry_empty_after_stop_all = None

    def thread1(self):
        """Start a new actor (which registers it in the registry)."""
        class LateActor(DaemonActor):
            def on_receive(self, message):
                return "late"

        self.late_ref = LateActor.start()

    def thread2(self):
        """Call stop_all(), then check the registry is empty."""
        try:
            pykka.ActorRegistry.stop_all(block=True, timeout=2.0)
        except Exception:
            pass
        self.stop_all_done = True
        remaining = pykka.ActorRegistry.get_all()
        self.registry_empty_after_stop_all = len(remaining) == 0

    def cleanup(self):
        """Ensure everything is stopped for next test."""
        try:
            pykka.ActorRegistry.stop_all(block=True, timeout=2.0)
        except Exception:
            pass


def _stop_all_clears_registry(s: RegistryStopAllRaceState) -> bool:
    """After stop_all completes, the registry should be empty.
    If an actor was registered after stop_all's snapshot, it survives."""
    s.cleanup()
    if s.registry_empty_after_stop_all is None:
        return True  # stop_all didn't complete yet
    return s.registry_empty_after_stop_all


def test_pykka_registry_stop_all_race():
    """stop_all() vs concurrent actor start: registry must be empty after."""
    with timeout_minutes(10):
        result = explore_interleavings(
            setup=lambda: RegistryStopAllRaceState(),
            threads=[lambda s: s.thread1(), lambda s: s.thread2()],
            invariant=_stop_all_clears_registry,
            max_attempts=500,
            max_ops=500,
            seed=42,
        )
    print_exploration_result(result)
    return result


# ===========================================================================
# Test 7: ActorProxy creation + stop() race
#
# ActorProxy.__init__() checks is_alive() and then introspects the actor.
# If the actor dies between the check and the introspection, we may get
# a proxy to a dead actor.  Subsequent calls through the proxy will
# enqueue messages to a dead inbox.
#
# Invariant: if proxy creation succeeds, a call through the proxy must
# either succeed or raise ActorDeadError -- never time out.
# ===========================================================================

class ProxyStopRaceState:
    """State for proxy creation + stop race test."""

    def __init__(self):
        class MyActor(DaemonActor):
            def on_receive(self, message):
                return "proxied"

            def greet(self):
                return "hello"

        self.actor_ref = MyActor.start()
        self.proxy = None
        self.proxy_created = False
        self.proxy_call_timeout = False
        self.proxy_creation_error = False

    def thread1(self):
        """Create a proxy and try to call a method through it."""
        try:
            self.proxy = self.actor_ref.proxy()
            self.proxy_created = True
        except ActorDeadError:
            self.proxy_creation_error = True
            return

        # The proxy was created (is_alive was True at that instant).
        # Now try to use it.
        try:
            # proxy.greet() returns a future
            result = self.proxy.greet().get(timeout=2.0)
        except ActorDeadError:
            pass  # actor died, proper error
        except Timeout:
            self.proxy_call_timeout = True  # orphaned call!
        except AttributeError:
            pass  # proxy introspection issue

    def thread2(self):
        """Stop the actor."""
        try:
            self.actor_ref.stop(block=True, timeout=2.0)
        except (ActorDeadError, Timeout):
            pass


def _proxy_call_no_timeout(s: ProxyStopRaceState) -> bool:
    """Proxy call must not time out (orphaned future)."""
    return not s.proxy_call_timeout


def test_pykka_proxy_stop_race():
    """Proxy creation + stop race: proxy calls must not orphan."""
    with timeout_minutes(10):
        result = explore_interleavings(
            setup=lambda: ProxyStopRaceState(),
            threads=[lambda s: s.thread1(), lambda s: s.thread2()],
            invariant=_proxy_call_no_timeout,
            max_attempts=500,
            max_ops=500,
            seed=42,
        )
    print_exploration_result(result)
    return result


# ===========================================================================
# Test 8: tell() + ask() interleaved with stop -- mixed ghost messages
#
# Thread 1 does tell("t") then ask("a", block=False).
# Thread 2 does stop().
#
# Possible bugs:
#   - tell succeeds (ghost) but ask properly raises ActorDeadError
#   - ask succeeds (ghost) but tell properly raises ActorDeadError
#   - Both ghost (tell + ask succeed but actor never processes either)
#
# Invariant: for tell, successes == received tells.
#            for ask, if the future was created, it must resolve.
# ===========================================================================

class TellAskStopState:
    """State for interleaved tell + ask + stop."""

    def __init__(self):
        self.tell_received = []
        self.ask_received = []
        tell_received = self.tell_received
        ask_received = self.ask_received

        class MyActor(DaemonActor):
            def on_receive(self, message):
                if message == "tell_msg":
                    tell_received.append(message)
                elif message == "ask_msg":
                    ask_received.append(message)
                    return "ask_reply"

        self.actor_ref = MyActor.start()
        self.tell_ok = 0
        self.ask_future = None
        self.ask_ok = False
        self.ask_timeout = False

    def thread1(self):
        """tell then ask."""
        try:
            self.actor_ref.tell("tell_msg")
            self.tell_ok += 1
        except ActorDeadError:
            pass

        try:
            self.ask_future = self.actor_ref.ask("ask_msg", block=False)
            self.ask_ok = True
        except ActorDeadError:
            pass

    def thread2(self):
        """Stop the actor, then check futures."""
        try:
            self.actor_ref.stop(block=True, timeout=2.0)
        except (ActorDeadError, Timeout):
            pass

        # After stop, try to resolve the ask future
        if self.ask_future is not None:
            try:
                self.ask_future.get(timeout=1.0)
            except ActorDeadError:
                pass  # properly rejected
            except Timeout:
                self.ask_timeout = True


def _tell_ask_stop_consistent(s: TellAskStopState) -> bool:
    """tell successes must match received count AND ask future must resolve."""
    tell_consistent = s.tell_ok == len(s.tell_received)
    no_orphan = not s.ask_timeout
    return tell_consistent and no_orphan


def test_pykka_tell_ask_stop():
    """Interleaved tell + ask + stop: no ghost messages or orphaned futures."""
    with timeout_minutes(10):
        result = explore_interleavings(
            setup=lambda: TellAskStopState(),
            threads=[lambda s: s.thread1(), lambda s: s.thread2()],
            invariant=_tell_ask_stop_consistent,
            max_attempts=500,
            max_ops=500,
            seed=42,
        )
    print_exploration_result(result)
    return result


# ===========================================================================
# Test 9: _handle_failure race -- unregister + actor_stopped.set()
#
# Actor._handle_failure() does:
#   ActorRegistry.unregister(self.actor_ref)
#   self.actor_stopped.set()
#
# Actor._stop() does:
#   ActorRegistry.unregister(self.actor_ref)
#   self.actor_stopped.set()
#   self.on_stop()
#
# If on_receive raises and _handle_failure runs, but concurrently a stop()
# message is also being processed, both _handle_failure and _stop can race
# on unregister + actor_stopped.set().  The unregister itself is safe (lock),
# but the actor_stopped.set() happening twice is just redundant.  However,
# the ordering can cause tell() to see is_alive()=False in a window where
# the actor is actually still processing.
#
# We trigger this by having on_receive raise on a specific message while
# concurrently stopping.
# ===========================================================================

class FailureStopRaceState:
    """State for _handle_failure + stop race."""

    def __init__(self):
        self.processed = []
        processed = self.processed

        class MyActor(DaemonActor):
            def on_receive(self, message):
                if message == "crash":
                    raise ValueError("intentional crash")
                processed.append(message)

        self.actor_ref = MyActor.start()
        self.tell_ok = 0
        self.tell_err = 0

    def thread1(self):
        """Send a crashing message."""
        try:
            self.actor_ref.tell("crash")
            self.tell_ok += 1
        except ActorDeadError:
            self.tell_err += 1

    def thread2(self):
        """Stop the actor."""
        try:
            self.actor_ref.stop(block=True, timeout=2.0)
        except (ActorDeadError, Timeout):
            pass


def _failure_stop_no_hang(s: FailureStopRaceState) -> bool:
    """After both threads complete, the actor should be stopped.
    This test mainly checks that no deadlock/hang occurs and that
    the actor_stopped event is set."""
    return s.actor_ref.actor_stopped.is_set()


def test_pykka_failure_stop_race():
    """_handle_failure + stop race: actor must end up stopped, no hang."""
    with timeout_minutes(10):
        result = explore_interleavings(
            setup=lambda: FailureStopRaceState(),
            threads=[lambda s: s.thread1(), lambda s: s.thread2()],
            invariant=_failure_stop_no_hang,
            max_attempts=500,
            max_ops=500,
            seed=42,
        )
    print_exploration_result(result)
    return result


# ===========================================================================
# Test 10: Seed sweep for tell() ghost message
# ===========================================================================

def test_pykka_tell_ghost_sweep():
    """Sweep 20 seeds to measure tell() ghost detection reliability."""
    found_seeds = []
    total_explored = 0
    for seed in range(20):
        with timeout_minutes(10):
            result = explore_interleavings(
                setup=lambda: TellGhostState(),
                threads=[lambda s: s.thread1(), lambda s: s.thread2()],
                invariant=_no_ghost_tells,
                max_attempts=200,
                max_ops=500,
                seed=seed,
            )
        total_explored += result.num_explored
        if not result.property_holds:
            found_seeds.append((seed, result.num_explored))
    print_seed_sweep_results(found_seeds, total_explored)
    return found_seeds


# ===========================================================================
# Test 11: Seed sweep for ask() ghost ask
# ===========================================================================

def test_pykka_ask_toctou_sweep():
    """Sweep 20 seeds to measure ask() ghost detection reliability."""
    found_seeds = []
    total_explored = 0
    for seed in range(20):
        with timeout_minutes(10):
            result = explore_interleavings(
                setup=lambda: AskTOCTOUState(),
                threads=[lambda s: s.thread1(), lambda s: s.thread2()],
                invariant=_no_ghost_asks,
                max_attempts=200,
                max_ops=500,
                seed=seed,
            )
        total_explored += result.num_explored
        if not result.property_holds:
            found_seeds.append((seed, result.num_explored))
    print_seed_sweep_results(found_seeds, total_explored)
    return found_seeds


# ===========================================================================
# Test 12: Deterministic reproduction for tell() ghost
# ===========================================================================

def test_pykka_tell_ghost_reproduce():
    """Find a tell() ghost counterexample then reproduce it 10 times."""
    with timeout_minutes(10):
        result = explore_interleavings(
            setup=lambda: TellGhostState(),
            threads=[lambda s: s.thread1(), lambda s: s.thread2()],
            invariant=_no_ghost_tells,
            max_attempts=500,
            max_ops=500,
            seed=42,
        )

    if not result.counterexample:
        print("No counterexample found -- skipping reproduction")
        return 0

    print(f"Found counterexample after {result.num_explored} attempts")
    print(f"Schedule length: {len(result.counterexample)}")

    print("\nReproducing 10 times with the same schedule...")
    bugs_reproduced = 0
    for i in range(10):
        state = run_with_schedule(
            result.counterexample,
            setup=lambda: TellGhostState(),
            threads=[lambda s: s.thread1(), lambda s: s.thread2()],
        )
        is_bug = not _no_ghost_tells(state)
        bugs_reproduced += is_bug
        print(
            f"  Run {i + 1}: tell_successes={state.tell_successes},"
            f" received={len(state.received)} [{'BUG' if is_bug else 'ok'}]"
        )

    print(f"\nReproduced: {bugs_reproduced}/10")
    return bugs_reproduced


# ===========================================================================
# Test 13: Deterministic reproduction for ask() ghost
# ===========================================================================

def test_pykka_ask_toctou_reproduce():
    """Find an ask() ghost counterexample then reproduce it 10 times."""
    with timeout_minutes(10):
        result = explore_interleavings(
            setup=lambda: AskTOCTOUState(),
            threads=[lambda s: s.thread1(), lambda s: s.thread2()],
            invariant=_no_ghost_asks,
            max_attempts=500,
            max_ops=500,
            seed=42,
        )

    if not result.counterexample:
        print("No counterexample found -- skipping reproduction")
        return 0

    print(f"Found counterexample after {result.num_explored} attempts")
    print(f"Schedule length: {len(result.counterexample)}")

    print("\nReproducing 10 times with the same schedule...")
    bugs_reproduced = 0
    for i in range(10):
        state = run_with_schedule(
            result.counterexample,
            setup=lambda: AskTOCTOUState(),
            threads=[lambda s: s.thread1(), lambda s: s.thread2()],
        )
        is_bug = not _no_ghost_asks(state)
        bugs_reproduced += is_bug
        print(
            f"  Run {i + 1}: ask_timeout={state.ask_timeout},"
            f" ask_successes={state.ask_successes} [{'BUG' if is_bug else 'ok'}]"
        )

    print(f"\nReproduced: {bugs_reproduced}/10")
    return bugs_reproduced


# ===========================================================================
# Main: run all tests
# ===========================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("EXHAUSTIVE PYKKA CONCURRENCY TESTS")
    print("=" * 70)

    print("\n=== Test 1: tell() ghost message (TOCTOU) ===")
    test_pykka_tell_ghost()

    print("\n=== Test 2: ask() ghost ask (TOCTOU) ===")
    test_pykka_ask_toctou()

    print("\n=== Test 3: Concurrent tell() + tell() + stop ===")
    test_pykka_concurrent_tells()

    print("\n=== Test 4: ask() + stop() orphaned future ===")
    test_pykka_ask_stop_orphaned_future()

    print("\n=== Test 5: Double stop() race ===")
    test_pykka_double_stop()

    print("\n=== Test 6: Registry stop_all() vs concurrent start ===")
    test_pykka_registry_stop_all_race()

    print("\n=== Test 7: Proxy creation + stop race ===")
    test_pykka_proxy_stop_race()

    print("\n=== Test 8: Interleaved tell + ask + stop ===")
    test_pykka_tell_ask_stop()

    print("\n=== Test 9: _handle_failure + stop race ===")
    test_pykka_failure_stop_race()

    print("\n=== Test 10: tell() ghost sweep (20 seeds) ===")
    test_pykka_tell_ghost_sweep()

    print("\n=== Test 11: ask() ghost sweep (20 seeds) ===")
    test_pykka_ask_toctou_sweep()

    print("\n=== Test 12: tell() ghost deterministic reproduction ===")
    test_pykka_tell_ghost_reproduce()

    print("\n=== Test 13: ask() ghost deterministic reproduction ===")
    test_pykka_ask_toctou_reproduce()

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETE")
    print("=" * 70)
