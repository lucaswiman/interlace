"""
Exhaustive concurrency tests for PyDispatcher.

Tests every concurrency bug arising from PyDispatcher's THREE module-level
global dictionaries (connections, senders, sendersBack) with ZERO
synchronization.

Each test targets a distinct race condition pattern:

 1. connect()+connect() TOCTOU on signals dict (covered by existing test,
    included here for completeness with a different invariant angle)
 2. disconnect()+disconnect() race: two threads disconnecting same receiver
 3. connect()+disconnect() race on same sender/signal
 4. send()+disconnect() race: send iterates receivers while disconnect mutates
 5. send()+connect() race: send misses a newly connected receiver
 6. Two sends racing: shared mutable getAllReceivers iteration
 7. connect() racing on different signals for the same sender
 8. sendersBack dict corruption during concurrent connect()
 9. _cleanupConnections race between disconnect() calls
10. connect()+send() race on the Any/Anonymous wildcard paths
11. disconnect()+send() via sendExact (separate code path)
12. liveReceivers iteration during concurrent connect/disconnect
"""

import os
import sys

_test_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_test_dir, "..", "external_repos", "pydispatcher"))

from external_tests_helpers import print_exploration_result, print_seed_sweep_results
from pydispatch import dispatcher
from pydispatch.dispatcher import Any, Anonymous

from interlace.bytecode import explore_interleavings, run_with_schedule


# ---------------------------------------------------------------------------
# Module-level receiver functions (stable id() -- never use lambdas)
# ---------------------------------------------------------------------------
def recv1(**kw):
    pass


def recv2(**kw):
    pass


def recv3(**kw):
    pass


def recv4(**kw):
    pass


def recv5(**kw):
    pass


def recv6(**kw):
    pass


def recv7(**kw):
    pass


def recv8(**kw):
    pass


# Receivers that record calls for send-related tests
_call_log = []


def recv_log_a(**kw):
    _call_log.append("a")


def recv_log_b(**kw):
    _call_log.append("b")


def recv_log_c(**kw):
    _call_log.append("c")


def recv_log_d(**kw):
    _call_log.append("d")


# ---------------------------------------------------------------------------
# Helper: reset PyDispatcher global state
# ---------------------------------------------------------------------------
def _reset_dispatcher():
    """Clear all three global dictionaries."""
    dispatcher.connections.clear()
    dispatcher.senders.clear()
    dispatcher.sendersBack.clear()


# ===================================================================
# TEST 1: connect()+connect() TOCTOU -- two receivers, same signal
#
# Bug: connect() does `if senderkey in connections` then creates a new
# signals dict.  Two threads can both see the key absent, both create
# separate dicts, and the second overwrites the first.
# ===================================================================
class ConnectConnectState:
    def __init__(self):
        _reset_dispatcher()
        self.sender = object()
        self.signal = "sig1"

    def thread1(self):
        dispatcher.connect(recv1, signal=self.signal, sender=self.sender, weak=False)

    def thread2(self):
        dispatcher.connect(recv2, signal=self.signal, sender=self.sender, weak=False)


def test_connect_connect_race():
    """Two threads connect different receivers to the same (sender, signal)."""
    result = explore_interleavings(
        setup=lambda: ConnectConnectState(),
        threads=[
            lambda s: s.thread1(),
            lambda s: s.thread2(),
        ],
        invariant=lambda s: (
            len(dispatcher.connections.get(id(s.sender), {}).get(s.signal, [])) == 2
        ),
        max_attempts=500,
        max_ops=300,
        seed=42,
    )
    print_exploration_result(result)
    return result


# ===================================================================
# TEST 2: disconnect()+disconnect() race -- same receiver
#
# Bug: disconnect() does connections[senderkey] then signals[signal]
# to get the receivers list, then calls _removeOldBackRefs which does
# receivers.index(receiver) + del receivers[index].  Two threads
# racing to disconnect the same receiver can both find it, then one
# deletes it and the second raises ValueError/DispatcherKeyError, or
# worse, deletes the wrong entry.
# ===================================================================
class DisconnectDisconnectState:
    def __init__(self):
        _reset_dispatcher()
        self.sender = object()
        self.signal = "sig_dd"
        # Pre-connect a receiver that both threads will try to disconnect
        dispatcher.connect(recv1, signal=self.signal, sender=self.sender, weak=False)
        self.errors = []

    def thread1(self):
        try:
            dispatcher.disconnect(recv1, signal=self.signal, sender=self.sender, weak=False)
        except Exception as e:
            self.errors.append(("t1", type(e).__name__, str(e)))

    def thread2(self):
        try:
            dispatcher.disconnect(recv1, signal=self.signal, sender=self.sender, weak=False)
        except Exception as e:
            self.errors.append(("t2", type(e).__name__, str(e)))


def test_disconnect_disconnect_race():
    """Two threads disconnect the same receiver simultaneously.

    Invariant: exactly one disconnect should succeed and one should raise
    DispatcherKeyError. If both succeed silently, or if an unexpected error
    type is raised, or if the receiver is still present, we have a bug.
    """
    result = explore_interleavings(
        setup=lambda: DisconnectDisconnectState(),
        threads=[
            lambda s: s.thread1(),
            lambda s: s.thread2(),
        ],
        invariant=lambda s: (
            # The receiver should be fully gone
            len(dispatcher.connections.get(id(s.sender), {}).get(s.signal, [])) == 0
            # Exactly one thread should have errored (DispatcherKeyError)
            and len(s.errors) == 1
        ),
        max_attempts=500,
        max_ops=300,
        seed=42,
    )
    print_exploration_result(result)
    return result


# ===================================================================
# TEST 3: connect()+disconnect() race -- same sender/signal
#
# Bug: Thread 1 connects recv1 while Thread 2 disconnects recv2 (already
# connected).  The disconnect path calls _cleanupConnections which may
# delete the entire signals dict or senderkey entry while connect is
# in the middle of adding to it.  This can cause KeyError in connect
# or silently lose the new connection.
# ===================================================================
class ConnectDisconnectState:
    def __init__(self):
        _reset_dispatcher()
        self.sender = object()
        self.signal = "sig_cd"
        # Pre-connect recv2 so thread2 can disconnect it
        dispatcher.connect(recv2, signal=self.signal, sender=self.sender, weak=False)
        self.errors = []

    def thread1(self):
        """Connect a new receiver."""
        try:
            dispatcher.connect(recv1, signal=self.signal, sender=self.sender, weak=False)
        except Exception as e:
            self.errors.append(("t1_connect", type(e).__name__, str(e)))

    def thread2(self):
        """Disconnect the existing receiver."""
        try:
            dispatcher.disconnect(recv2, signal=self.signal, sender=self.sender, weak=False)
        except Exception as e:
            self.errors.append(("t2_disconnect", type(e).__name__, str(e)))


def test_connect_disconnect_race():
    """One thread connects while another disconnects on the same sender/signal.

    After both complete: recv2 should be gone, recv1 should be present.
    The race in _cleanupConnections can delete the signals dict while
    connect() is still writing to it.
    """
    result = explore_interleavings(
        setup=lambda: ConnectDisconnectState(),
        threads=[
            lambda s: s.thread1(),
            lambda s: s.thread2(),
        ],
        invariant=lambda s: (
            # recv1 should be present, recv2 should be gone
            len(s.errors) == 0
            and recv1 in dispatcher.connections.get(id(s.sender), {}).get(s.signal, [])
            and recv2 not in dispatcher.connections.get(id(s.sender), {}).get(s.signal, [])
        ),
        max_attempts=500,
        max_ops=300,
        seed=42,
    )
    print_exploration_result(result)
    return result


# ===================================================================
# TEST 4: send()+disconnect() race
#
# Bug: send() calls getAllReceivers() which iterates over
# connections[senderkey][signal].  If disconnect() modifies that list
# concurrently (via _removeOldBackRefs which does del receivers[index]),
# the iteration can skip receivers, see stale data, or raise RuntimeError
# (dictionary changed size during iteration) / IndexError.
# ===================================================================
class SendDisconnectState:
    def __init__(self):
        _reset_dispatcher()
        self.sender = object()
        self.signal = "sig_sd"
        # Pre-connect both receivers
        dispatcher.connect(recv1, signal=self.signal, sender=self.sender, weak=False)
        dispatcher.connect(recv2, signal=self.signal, sender=self.sender, weak=False)
        self.send_results = None
        self.errors = []

    def thread1(self):
        """Send signal -- should reach connected receivers."""
        try:
            self.send_results = dispatcher.send(
                signal=self.signal, sender=self.sender
            )
        except Exception as e:
            self.errors.append(("t1_send", type(e).__name__, str(e)))

    def thread2(self):
        """Disconnect recv2 while send may be iterating."""
        try:
            dispatcher.disconnect(recv2, signal=self.signal, sender=self.sender, weak=False)
        except Exception as e:
            self.errors.append(("t2_disconnect", type(e).__name__, str(e)))


def test_send_disconnect_race():
    """send() iterates receivers while disconnect() removes one.

    Invariant: No unhandled exceptions should occur (KeyError, RuntimeError,
    ValueError from list mutation during iteration). If errors is non-empty
    that's a bug manifestation.
    """
    result = explore_interleavings(
        setup=lambda: SendDisconnectState(),
        threads=[
            lambda s: s.thread1(),
            lambda s: s.thread2(),
        ],
        invariant=lambda s: len(s.errors) == 0,
        max_attempts=500,
        max_ops=300,
        seed=42,
    )
    print_exploration_result(result)
    return result


# ===================================================================
# TEST 5: send()+connect() race -- send misses new receiver
#
# Bug: Thread 1 calls send() which snapshots receivers via
# getAllReceivers().  Thread 2 connects recv3 concurrently.  Due to the
# non-atomic read of connections[senderkey][signal], send may or may not
# see recv3.  More critically, if connect creates a new signals dict
# while getAllReceivers is iterating, the dict can change size.
# ===================================================================
class SendConnectState:
    def __init__(self):
        _reset_dispatcher()
        self.sender = object()
        self.signal = "sig_sc"
        global _call_log
        _call_log = []
        # Pre-connect recv_log_a
        dispatcher.connect(recv_log_a, signal=self.signal, sender=self.sender, weak=False)
        self.send_results = None
        self.errors = []

    def thread1(self):
        """Send signal."""
        try:
            self.send_results = dispatcher.send(
                signal=self.signal, sender=self.sender
            )
        except Exception as e:
            self.errors.append(("t1_send", type(e).__name__, str(e)))

    def thread2(self):
        """Connect a new receiver during send."""
        try:
            dispatcher.connect(recv_log_b, signal=self.signal, sender=self.sender, weak=False)
        except Exception as e:
            self.errors.append(("t2_connect", type(e).__name__, str(e)))


def test_send_connect_race():
    """send() races with connect() adding a new receiver.

    Invariant: No exceptions should occur. After completion, both
    receivers should be registered (even if the send didn't see the
    new one). The bug manifests as exceptions or lost registrations.
    """
    result = explore_interleavings(
        setup=lambda: SendConnectState(),
        threads=[
            lambda s: s.thread1(),
            lambda s: s.thread2(),
        ],
        invariant=lambda s: (
            len(s.errors) == 0
            and len(dispatcher.connections.get(id(s.sender), {}).get(s.signal, [])) == 2
        ),
        max_attempts=500,
        max_ops=300,
        seed=42,
    )
    print_exploration_result(result)
    return result


# ===================================================================
# TEST 6: Two concurrent sends -- shared mutable structures in
#          getAllReceivers
#
# Bug: getAllReceivers() uses a local dict `receivers = {}` but iterates
# over the *global* connections dict entries.  Two sends racing access
# the same global lists; if one send's iteration interleaves with
# modifications from any source, we get undefined behavior.  More
# subtly, send() returns responses list -- two sends sharing mutable
# global state can corrupt each other's iteration.
# ===================================================================
class SendSendState:
    def __init__(self):
        _reset_dispatcher()
        self.sender = object()
        self.signal = "sig_ss"
        global _call_log
        _call_log = []
        dispatcher.connect(recv_log_a, signal=self.signal, sender=self.sender, weak=False)
        dispatcher.connect(recv_log_b, signal=self.signal, sender=self.sender, weak=False)
        self.results1 = None
        self.results2 = None
        self.errors = []

    def thread1(self):
        try:
            self.results1 = dispatcher.send(
                signal=self.signal, sender=self.sender
            )
        except Exception as e:
            self.errors.append(("t1", type(e).__name__, str(e)))

    def thread2(self):
        try:
            self.results2 = dispatcher.send(
                signal=self.signal, sender=self.sender
            )
        except Exception as e:
            self.errors.append(("t2", type(e).__name__, str(e)))


def test_send_send_race():
    """Two threads send the same signal simultaneously.

    Invariant: No exceptions, and each send should see both receivers
    (2 responses each).
    """
    result = explore_interleavings(
        setup=lambda: SendSendState(),
        threads=[
            lambda s: s.thread1(),
            lambda s: s.thread2(),
        ],
        invariant=lambda s: (
            len(s.errors) == 0
            and s.results1 is not None
            and s.results2 is not None
            and len(s.results1) == 2
            and len(s.results2) == 2
        ),
        max_attempts=500,
        max_ops=300,
        seed=42,
    )
    print_exploration_result(result)
    return result


# ===================================================================
# TEST 7: connect() racing on DIFFERENT signals, same sender
#
# Bug: connect() does `if senderkey in connections` and then either
# reads the existing signals dict or creates a new one.  Two threads
# connecting to different signals on the same sender can race on this
# check-then-act: both see senderkey absent, both create separate
# empty dicts, and one overwrites the other -- losing a signal.
# ===================================================================
class ConnectDifferentSignalsState:
    def __init__(self):
        _reset_dispatcher()
        self.sender = object()
        self.signal_a = "sig_a"
        self.signal_b = "sig_b"

    def thread1(self):
        dispatcher.connect(recv1, signal=self.signal_a, sender=self.sender, weak=False)

    def thread2(self):
        dispatcher.connect(recv2, signal=self.signal_b, sender=self.sender, weak=False)


def test_connect_different_signals_race():
    """Two threads connect to different signals on the same sender.

    Invariant: Both signals should exist in the connections dict for this
    sender, each with their respective receiver.
    """
    result = explore_interleavings(
        setup=lambda: ConnectDifferentSignalsState(),
        threads=[
            lambda s: s.thread1(),
            lambda s: s.thread2(),
        ],
        invariant=lambda s: (
            s.signal_a in dispatcher.connections.get(id(s.sender), {})
            and s.signal_b in dispatcher.connections.get(id(s.sender), {})
            and recv1 in dispatcher.connections[id(s.sender)][s.signal_a]
            and recv2 in dispatcher.connections[id(s.sender)][s.signal_b]
        ),
        max_attempts=500,
        max_ops=300,
        seed=42,
    )
    print_exploration_result(result)
    return result


# ===================================================================
# TEST 8: sendersBack corruption during concurrent connect()
#
# Bug: connect() does:
#   current = sendersBack.get(receiverID)
#   if current is None:
#       sendersBack[receiverID] = current = []
#   if senderkey not in current:
#       current.append(senderkey)
#
# Two threads connecting different receivers to different senders can
# race on sendersBack.  More critically, if the same receiver is
# connected to two different senders concurrently, both threads read
# `current is None`, both create separate lists, and the second
# overwrites the first -- losing the back-reference to the first sender.
# ===================================================================
class SendersBackCorruptionState:
    def __init__(self):
        _reset_dispatcher()
        self.sender1 = object()
        self.sender2 = object()
        self.signal = "sig_sb"

    def thread1(self):
        dispatcher.connect(recv1, signal=self.signal, sender=self.sender1, weak=False)

    def thread2(self):
        dispatcher.connect(recv1, signal=self.signal, sender=self.sender2, weak=False)


def test_sendersback_corruption():
    """Same receiver connected to two different senders concurrently.

    Invariant: sendersBack[id(recv1)] should list both sender keys,
    so that cleanup works correctly for both senders.
    """
    result = explore_interleavings(
        setup=lambda: SendersBackCorruptionState(),
        threads=[
            lambda s: s.thread1(),
            lambda s: s.thread2(),
        ],
        invariant=lambda s: (
            id(recv1) in dispatcher.sendersBack
            and id(s.sender1) in dispatcher.sendersBack[id(recv1)]
            and id(s.sender2) in dispatcher.sendersBack[id(recv1)]
        ),
        max_attempts=500,
        max_ops=300,
        seed=42,
    )
    print_exploration_result(result)
    return result


# ===================================================================
# TEST 9: _cleanupConnections race between two disconnect() calls
#
# Bug: _cleanupConnections checks `if not receivers` then deletes
# signals[signal], then checks `if not signals` and calls
# _removeSender(senderkey) which deletes connections[senderkey].
# Two threads disconnecting the last two receivers of the same sender
# can both see `not signals` as True after deleting their respective
# signal entries, leading to double-delete of connections[senderkey]
# or corruption of the senders/sendersBack dicts.
# ===================================================================
class CleanupRaceState:
    def __init__(self):
        _reset_dispatcher()
        self.sender = object()
        self.signal1 = "sig_c1"
        self.signal2 = "sig_c2"
        # Connect one receiver per signal, so disconnecting each leaves
        # that signal's list empty, triggering _cleanupConnections.
        dispatcher.connect(recv1, signal=self.signal1, sender=self.sender, weak=False)
        dispatcher.connect(recv2, signal=self.signal2, sender=self.sender, weak=False)
        self.errors = []

    def thread1(self):
        try:
            dispatcher.disconnect(recv1, signal=self.signal1, sender=self.sender, weak=False)
        except Exception as e:
            self.errors.append(("t1", type(e).__name__, str(e)))

    def thread2(self):
        try:
            dispatcher.disconnect(recv2, signal=self.signal2, sender=self.sender, weak=False)
        except Exception as e:
            self.errors.append(("t2", type(e).__name__, str(e)))


def test_cleanup_connections_race():
    """Two threads disconnect last receivers of different signals on same sender.

    Invariant: After both disconnects, the sender should be fully cleaned
    up with no errors.  connections should not contain the senderkey.
    No unexpected exceptions.
    """
    result = explore_interleavings(
        setup=lambda: CleanupRaceState(),
        threads=[
            lambda s: s.thread1(),
            lambda s: s.thread2(),
        ],
        invariant=lambda s: (
            len(s.errors) == 0
            and id(s.sender) not in dispatcher.connections
        ),
        max_attempts=500,
        max_ops=300,
        seed=42,
    )
    print_exploration_result(result)
    return result


# ===================================================================
# TEST 10: connect()+send() race using Any wildcard
#
# Bug: getAllReceivers() queries 4 different (sender, signal) combos
# including (Any, signal), (sender, Any), and (Any, Any).  A concurrent
# connect() to Any can race with the iteration over these wildcard
# entries. The connections dict is mutated mid-iteration.
# ===================================================================
class AnyWildcardRaceState:
    def __init__(self):
        _reset_dispatcher()
        self.sender = object()
        self.signal = "sig_any"
        global _call_log
        _call_log = []
        # Connect recv_log_a to catch-all (Any sender, Any signal)
        dispatcher.connect(recv_log_a, signal=Any, sender=Any, weak=False)
        self.send_results = None
        self.errors = []

    def thread1(self):
        """Send specific signal from specific sender."""
        try:
            self.send_results = dispatcher.send(
                signal=self.signal, sender=self.sender
            )
        except Exception as e:
            self.errors.append(("t1_send", type(e).__name__, str(e)))

    def thread2(self):
        """Connect another wildcard receiver concurrently."""
        try:
            dispatcher.connect(recv_log_b, signal=Any, sender=Any, weak=False)
        except Exception as e:
            self.errors.append(("t2_connect", type(e).__name__, str(e)))


def test_any_wildcard_race():
    """send() queries Any entries while connect() modifies them.

    Invariant: No exceptions. After completion, both wildcard receivers
    should be registered.
    """
    result = explore_interleavings(
        setup=lambda: AnyWildcardRaceState(),
        threads=[
            lambda s: s.thread1(),
            lambda s: s.thread2(),
        ],
        invariant=lambda s: (
            len(s.errors) == 0
            and len(dispatcher.connections.get(id(Any), {}).get(Any, [])) == 2
        ),
        max_attempts=500,
        max_ops=300,
        seed=42,
    )
    print_exploration_result(result)
    return result


# ===================================================================
# TEST 11: disconnect()+sendExact() race
#
# Bug: sendExact() uses getReceivers() which directly returns the
# live list reference: `return connections[id(sender)][signal]`.
# liveReceivers() then iterates this list.  If disconnect() mutates
# the list via _removeOldBackRefs (del receivers[index]) during
# iteration, we get skipped receivers or RuntimeError.
# ===================================================================
class SendExactDisconnectState:
    def __init__(self):
        _reset_dispatcher()
        self.sender = object()
        self.signal = "sig_se"
        dispatcher.connect(recv1, signal=self.signal, sender=self.sender, weak=False)
        dispatcher.connect(recv2, signal=self.signal, sender=self.sender, weak=False)
        dispatcher.connect(recv3, signal=self.signal, sender=self.sender, weak=False)
        self.send_results = None
        self.errors = []

    def thread1(self):
        """sendExact iterates receivers."""
        try:
            self.send_results = dispatcher.sendExact(
                signal=self.signal, sender=self.sender
            )
        except Exception as e:
            self.errors.append(("t1_sendExact", type(e).__name__, str(e)))

    def thread2(self):
        """Disconnect recv2 during iteration."""
        try:
            dispatcher.disconnect(recv2, signal=self.signal, sender=self.sender, weak=False)
        except Exception as e:
            self.errors.append(("t2_disconnect", type(e).__name__, str(e)))


def test_sendexact_disconnect_race():
    """sendExact() iterates the raw receiver list while disconnect mutates it.

    Invariant: No exceptions should occur.
    """
    result = explore_interleavings(
        setup=lambda: SendExactDisconnectState(),
        threads=[
            lambda s: s.thread1(),
            lambda s: s.thread2(),
        ],
        invariant=lambda s: len(s.errors) == 0,
        max_attempts=500,
        max_ops=300,
        seed=42,
    )
    print_exploration_result(result)
    return result


# ===================================================================
# TEST 12: Triple connect race -- three threads, same sender+signal
#
# Bug: The TOCTOU in connect() is amplified with three threads.
# All three can see senderkey as absent, create three separate dicts,
# and only the last one survives -- losing two receivers.
# ===================================================================
class TripleConnectState:
    def __init__(self):
        _reset_dispatcher()
        self.sender = object()
        self.signal = "sig_tc"

    def thread1(self):
        dispatcher.connect(recv1, signal=self.signal, sender=self.sender, weak=False)

    def thread2(self):
        dispatcher.connect(recv2, signal=self.signal, sender=self.sender, weak=False)

    def thread3(self):
        dispatcher.connect(recv3, signal=self.signal, sender=self.sender, weak=False)


def test_triple_connect_race():
    """Three threads connect to the same (sender, signal) simultaneously.

    Invariant: All three receivers should be registered.
    """
    result = explore_interleavings(
        setup=lambda: TripleConnectState(),
        threads=[
            lambda s: s.thread1(),
            lambda s: s.thread2(),
            lambda s: s.thread3(),
        ],
        invariant=lambda s: (
            len(dispatcher.connections.get(id(s.sender), {}).get(s.signal, [])) == 3
        ),
        max_attempts=500,
        max_ops=300,
        seed=42,
    )
    print_exploration_result(result)
    return result


# ===================================================================
# TEST 13: connect + disconnect + send -- three-way race
#
# Bug: The most dangerous real-world scenario.  One thread is sending,
# another is connecting, a third is disconnecting.  All three are
# touching connections/senders/sendersBack concurrently.
# ===================================================================
class ThreeWayRaceState:
    def __init__(self):
        _reset_dispatcher()
        self.sender = object()
        self.signal = "sig_3w"
        global _call_log
        _call_log = []
        # Pre-connect recv_log_a so there's something to send to and disconnect
        dispatcher.connect(recv_log_a, signal=self.signal, sender=self.sender, weak=False)
        self.send_results = None
        self.errors = []

    def thread_send(self):
        try:
            self.send_results = dispatcher.send(
                signal=self.signal, sender=self.sender
            )
        except Exception as e:
            self.errors.append(("send", type(e).__name__, str(e)))

    def thread_connect(self):
        try:
            dispatcher.connect(recv_log_b, signal=self.signal, sender=self.sender, weak=False)
        except Exception as e:
            self.errors.append(("connect", type(e).__name__, str(e)))

    def thread_disconnect(self):
        try:
            dispatcher.disconnect(recv_log_a, signal=self.signal, sender=self.sender, weak=False)
        except Exception as e:
            self.errors.append(("disconnect", type(e).__name__, str(e)))


def test_three_way_race():
    """send + connect + disconnect all racing on the same sender/signal.

    Invariant: No unhandled exceptions. After completion, recv_log_b
    should be registered and recv_log_a should not.
    """
    result = explore_interleavings(
        setup=lambda: ThreeWayRaceState(),
        threads=[
            lambda s: s.thread_send(),
            lambda s: s.thread_connect(),
            lambda s: s.thread_disconnect(),
        ],
        invariant=lambda s: (
            len(s.errors) == 0
            and recv_log_b in dispatcher.connections.get(id(s.sender), {}).get(s.signal, [])
            and recv_log_a not in dispatcher.connections.get(id(s.sender), {}).get(s.signal, [])
        ),
        max_attempts=500,
        max_ops=300,
        seed=42,
    )
    print_exploration_result(result)
    return result


# ===================================================================
# TEST 14: Multiple senders with shared signal name -- cross-sender
#          interference during connect
#
# Bug: connect() uses id(sender) as the key.  If two different senders
# are used but happen to be in the same code path, the sendersBack
# tracking can get confused if both threads are modifying it
# simultaneously.
# ===================================================================
class CrossSenderConnectState:
    def __init__(self):
        _reset_dispatcher()
        self.sender1 = object()
        self.sender2 = object()
        self.signal = "shared_sig"

    def thread1(self):
        dispatcher.connect(recv1, signal=self.signal, sender=self.sender1, weak=False)

    def thread2(self):
        dispatcher.connect(recv2, signal=self.signal, sender=self.sender2, weak=False)


def test_cross_sender_connect():
    """Connect to the same signal from different senders concurrently.

    Invariant: Each sender should have its own receiver registered
    independently. No cross-contamination.
    """
    result = explore_interleavings(
        setup=lambda: CrossSenderConnectState(),
        threads=[
            lambda s: s.thread1(),
            lambda s: s.thread2(),
        ],
        invariant=lambda s: (
            recv1 in dispatcher.connections.get(id(s.sender1), {}).get(s.signal, [])
            and recv2 in dispatcher.connections.get(id(s.sender2), {}).get(s.signal, [])
        ),
        max_attempts=500,
        max_ops=300,
        seed=42,
    )
    print_exploration_result(result)
    return result


# ===================================================================
# TEST 15: Disconnect + connect same receiver (re-registration race)
#
# Bug: Thread 1 disconnects recv1, Thread 2 re-connects recv1.  The
# disconnect path calls _removeOldBackRefs and _cleanupConnections.
# If connect() runs between these steps, it may connect to a signals
# dict that is about to be deleted, or the sendersBack entry may be
# corrupted.
# ===================================================================
class ReRegistrationRaceState:
    def __init__(self):
        _reset_dispatcher()
        self.sender = object()
        self.signal = "sig_rereg"
        dispatcher.connect(recv1, signal=self.signal, sender=self.sender, weak=False)
        self.errors = []

    def thread1(self):
        """Disconnect recv1."""
        try:
            dispatcher.disconnect(recv1, signal=self.signal, sender=self.sender, weak=False)
        except Exception as e:
            self.errors.append(("t1_disconnect", type(e).__name__, str(e)))

    def thread2(self):
        """Re-connect recv1 to same sender/signal."""
        try:
            dispatcher.connect(recv1, signal=self.signal, sender=self.sender, weak=False)
        except Exception as e:
            self.errors.append(("t2_connect", type(e).__name__, str(e)))


def test_re_registration_race():
    """Disconnect and re-connect the same receiver concurrently.

    Invariant: After completion, recv1 should be registered exactly once
    (the connect should win if disconnect ran first, or vice versa the
    state should be consistent). No errors.
    """
    result = explore_interleavings(
        setup=lambda: ReRegistrationRaceState(),
        threads=[
            lambda s: s.thread1(),
            lambda s: s.thread2(),
        ],
        invariant=lambda s: (
            len(s.errors) == 0
            # recv1 should be present exactly once after both operations
            and dispatcher.connections.get(id(s.sender), {}).get(s.signal, []).count(recv1) == 1
        ),
        max_attempts=500,
        max_ops=300,
        seed=42,
    )
    print_exploration_result(result)
    return result


# ===================================================================
# TEST 16: connect() + connect() for signals dict TOCTOU on the
#          *signal* level (not senderkey level)
#
# Bug: connect() also has a TOCTOU on `if signal in signals`:
#   if signal in signals:
#       receivers = signals[signal]
#   else:
#       receivers = signals[signal] = []
# Two threads connecting to the same (sender, signal) can both see
# the signal absent, both create empty lists, and the second
# overwrites the first's list (losing the first receiver).
# ===================================================================
class SignalLevelTOCTOUState:
    def __init__(self):
        _reset_dispatcher()
        self.sender = object()
        self.signal = "sig_toctou2"
        # Pre-populate the senderkey so the first TOCTOU doesn't fire;
        # this isolates the second TOCTOU at the signal level.
        senderkey = id(self.sender)
        dispatcher.connections[senderkey] = {}

    def thread1(self):
        dispatcher.connect(recv1, signal=self.signal, sender=self.sender, weak=False)

    def thread2(self):
        dispatcher.connect(recv2, signal=self.signal, sender=self.sender, weak=False)


def test_signal_level_toctou():
    """Isolate the signal-level TOCTOU in connect().

    The senderkey already exists so only the `if signal in signals`
    branch can race.

    Invariant: Both receivers should be in the list for this signal.
    """
    result = explore_interleavings(
        setup=lambda: SignalLevelTOCTOUState(),
        threads=[
            lambda s: s.thread1(),
            lambda s: s.thread2(),
        ],
        invariant=lambda s: (
            len(dispatcher.connections.get(id(s.sender), {}).get(s.signal, [])) == 2
        ),
        max_attempts=500,
        max_ops=300,
        seed=42,
    )
    print_exploration_result(result)
    return result


# ===================================================================
# Main entry point: run all tests
# ===================================================================
if __name__ == "__main__":
    tests = [
        ("TEST 1:  connect()+connect() same signal", test_connect_connect_race),
        ("TEST 2:  disconnect()+disconnect() same receiver", test_disconnect_disconnect_race),
        ("TEST 3:  connect()+disconnect() same sender/signal", test_connect_disconnect_race),
        ("TEST 4:  send()+disconnect() race", test_send_disconnect_race),
        ("TEST 5:  send()+connect() race", test_send_connect_race),
        ("TEST 6:  send()+send() race", test_send_send_race),
        ("TEST 7:  connect() different signals same sender", test_connect_different_signals_race),
        ("TEST 8:  sendersBack corruption", test_sendersback_corruption),
        ("TEST 9:  _cleanupConnections race", test_cleanup_connections_race),
        ("TEST 10: Any wildcard + send race", test_any_wildcard_race),
        ("TEST 11: sendExact()+disconnect() race", test_sendexact_disconnect_race),
        ("TEST 12: triple connect race (3 threads)", test_triple_connect_race),
        ("TEST 13: three-way send+connect+disconnect", test_three_way_race),
        ("TEST 14: cross-sender connect", test_cross_sender_connect),
        ("TEST 15: disconnect+re-connect same receiver", test_re_registration_race),
        ("TEST 16: signal-level TOCTOU (isolated)", test_signal_level_toctou),
    ]

    bugs_found = 0
    bugs_not_found = 0
    errors = 0

    for name, test_fn in tests:
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")
        try:
            result = test_fn()
            if result and not result.property_holds:
                print(f"  >>> BUG FOUND after {result.num_explored} interleavings")
                bugs_found += 1
            else:
                explored = result.num_explored if result else "?"
                print(f"  --- No bug found in {explored} interleavings")
                bugs_not_found += 1
        except Exception as e:
            print(f"  !!! Test error: {type(e).__name__}: {e}")
            errors += 1

    print(f"\n{'='*60}")
    print(f"  SUMMARY: {bugs_found} bugs found, {bugs_not_found} held, {errors} errors")
    print(f"{'='*60}")
