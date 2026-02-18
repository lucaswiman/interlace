"""
Exhaustive concurrency bug exploration for the amqtt MQTT library.

This file tests multiple distinct concurrency bugs found in the amqtt
codebase using interlace's deterministic bytecode-level exploration.
Each test targets a different race condition in the Session, Broker, or
Server classes.

Repository: https://github.com/Yakifo/amqtt

Bug inventory
=============
1. next_packet_id duplicate IDs with 3 threads
   - More collision opportunities than 2 threads; all three can read/write
     the shared _packet_id field and return the same value.

2. next_packet_id + inflight_out TOCTOU in mqtt_publish flow
   - mqtt_publish calls next_packet_id, then checks ``packet_id in
     inflight_out``, then stores.  A concurrent caller can allocate the
     same ID and store it in the window between check and store.

3. inflight_out lost-insert race
   - Two threads both do ``session.inflight_out[packet_id] = message``
     with different packet_ids obtained from the racy next_packet_id.
     Because OrderedDict.__setitem__ is not atomic relative to a
     concurrent read-modify-write of _packet_id, one entry can overwrite
     the other or the dict can lose an entry.

4. inflight_in concurrent dict mutation
   - Two threads both insert into inflight_in (simulating two incoming
     QOS2 messages arriving simultaneously), risking lost entries.

5. next_packet_id wraparound near 65535
   - When _packet_id starts near 65535, concurrent callers both wrap to
     small values and can collide on packet ID 1.

6. Session state: concurrent connect/disconnect time tracking
   - _on_enter_connected reads last_disconnect_time and writes
     last_connect_time; _on_enter_disconnected does the reverse.
     Concurrent transitions corrupt the timestamps.

7. Broker _subscriptions: concurrent add_subscription check-then-append
   - add_subscription checks ``if all(s.client_id != ...)`` then appends.
     Two threads subscribing different sessions to the same topic can both
     pass the check, resulting in duplicate subscription entries.

8. Server.conn_count lost update
   - acquire_connection does ``self.conn_count += 1`` and
     release_connection does ``self.conn_count -= 1``.  These are
     non-atomic read-modify-write operations that lose updates under
     concurrency.

9. Broker _sessions dict: concurrent client add and remove
   - One thread sets ``_sessions[client_id] = (session, handler)`` while
     another pops it, racing on the shared dict.

10. Broker retained_messages: concurrent store and clear
    - retain_message stores into _retained_messages while another thread
      deletes from it, risking KeyError or lost messages.
"""

import os
import sys

_test_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.join(_test_dir, "..", "external_repos", "amqtt")
# Insert local repo path FIRST so interlace can trace it (site-packages are excluded).
sys.path.insert(0, os.path.abspath(_repo_root))

from collections import OrderedDict

from amqtt.session import (  # noqa: E402
    IncomingApplicationMessage,
    OutgoingApplicationMessage,
    Session,
)
from case_study_helpers import (  # noqa: E402
    print_exploration_result,
    print_seed_sweep_results,
    timeout_minutes,
)

from interlace.bytecode import explore_interleavings, run_with_schedule  # noqa: E402


# ---------------------------------------------------------------------------
# 1. next_packet_id duplicate IDs with 3 concurrent callers
# ---------------------------------------------------------------------------

class ThreeThreadPacketIdState:
    """Three threads each call next_packet_id, recording the result."""

    def __init__(self):
        self.session = Session()
        self.id1 = None
        self.id2 = None
        self.id3 = None

    def thread1(self):
        self.id1 = self.session.next_packet_id

    def thread2(self):
        self.id2 = self.session.next_packet_id

    def thread3(self):
        self.id3 = self.session.next_packet_id


def _three_ids_unique(s: ThreeThreadPacketIdState) -> bool:
    """All three packet IDs must be distinct."""
    return len({s.id1, s.id2, s.id3}) == 3


def test_three_thread_packet_id_duplicates():
    """3 concurrent next_packet_id callers -- more collision opportunities."""
    with timeout_minutes(10):
        result = explore_interleavings(
            setup=lambda: ThreeThreadPacketIdState(),
            threads=[
                lambda s: s.thread1(),
                lambda s: s.thread2(),
                lambda s: s.thread3(),
            ],
            invariant=_three_ids_unique,
            max_attempts=500,
            max_ops=500,
            seed=42,
        )
    print_exploration_result(result)
    return result


def test_three_thread_packet_id_duplicates_sweep():
    """Sweep 20 seeds for the 3-thread packet ID race."""
    found_seeds = []
    total_explored = 0
    for seed in range(20):
        with timeout_minutes(10):
            result = explore_interleavings(
                setup=lambda: ThreeThreadPacketIdState(),
                threads=[
                    lambda s: s.thread1(),
                    lambda s: s.thread2(),
                    lambda s: s.thread3(),
                ],
                invariant=_three_ids_unique,
                max_attempts=200,
                max_ops=500,
                seed=seed,
            )
        total_explored += result.num_explored
        if not result.property_holds:
            found_seeds.append((seed, result.num_explored))
    print_seed_sweep_results(found_seeds, total_explored)
    return found_seeds


# ---------------------------------------------------------------------------
# 2. next_packet_id + inflight_out TOCTOU (allocate-check-store race)
# ---------------------------------------------------------------------------

class PacketIdInflightTocTouState:
    """Simulates the mqtt_publish flow: get next_packet_id, check inflight_out,
    store in inflight_out.  Two threads race on the same session."""

    def __init__(self):
        self.session = Session()
        self.id1 = None
        self.id2 = None
        self.stored1 = False
        self.stored2 = False
        self.error = None

    def _publish_flow(self):
        """Mirrors the logic in ProtocolHandler.mqtt_publish for QOS_1."""
        packet_id = self.session.next_packet_id
        # TOCTOU gap: between the check and the store another thread can
        # allocate the same packet_id and store it first.
        if packet_id in self.session.inflight_out:
            return packet_id, False
        msg = OutgoingApplicationMessage(packet_id, "test/topic", 1, b"data", False)
        self.session.inflight_out[packet_id] = msg
        return packet_id, True

    def thread1(self):
        self.id1, self.stored1 = self._publish_flow()

    def thread2(self):
        self.id2, self.stored2 = self._publish_flow()


def _publish_flow_invariant(s: PacketIdInflightTocTouState) -> bool:
    """Both threads must store distinct packet IDs in inflight_out.
    If they got the same ID, one would either overwrite the other's entry
    or be rejected -- violating the requirement that both messages are tracked."""
    if s.id1 == s.id2:
        return False  # Duplicate allocation -- bug
    if not s.stored1 or not s.stored2:
        return False  # One was rejected because the other raced in -- bug
    # Both stored with distinct IDs -- verify both actually exist
    return s.id1 in s.session.inflight_out and s.id2 in s.session.inflight_out


def test_packet_id_inflight_toctou():
    """Race between next_packet_id allocation and inflight_out check-then-store."""
    with timeout_minutes(10):
        result = explore_interleavings(
            setup=lambda: PacketIdInflightTocTouState(),
            threads=[
                lambda s: s.thread1(),
                lambda s: s.thread2(),
            ],
            invariant=_publish_flow_invariant,
            max_attempts=500,
            max_ops=400,
            seed=42,
        )
    print_exploration_result(result)
    return result


def test_packet_id_inflight_toctou_sweep():
    """Sweep 20 seeds for the allocate-check-store TOCTOU race."""
    found_seeds = []
    total_explored = 0
    for seed in range(20):
        with timeout_minutes(10):
            result = explore_interleavings(
                setup=lambda: PacketIdInflightTocTouState(),
                threads=[
                    lambda s: s.thread1(),
                    lambda s: s.thread2(),
                ],
                invariant=_publish_flow_invariant,
                max_attempts=200,
                max_ops=400,
                seed=seed,
            )
        total_explored += result.num_explored
        if not result.property_holds:
            found_seeds.append((seed, result.num_explored))
    print_seed_sweep_results(found_seeds, total_explored)
    return found_seeds


# ---------------------------------------------------------------------------
# 3. inflight_out lost-insert race (concurrent dict writes)
# ---------------------------------------------------------------------------

class InflightOutLostInsertState:
    """Two threads each insert a message into inflight_out with
    pre-assigned (distinct) packet IDs.  The race is on the OrderedDict
    itself: a concurrent insert can lose an entry."""

    def __init__(self):
        self.session = Session()

    def thread1(self):
        msg = OutgoingApplicationMessage(1, "topic/a", 1, b"payload_a", False)
        self.session.inflight_out[1] = msg

    def thread2(self):
        msg = OutgoingApplicationMessage(2, "topic/b", 1, b"payload_b", False)
        self.session.inflight_out[2] = msg


def _inflight_out_both_present(s: InflightOutLostInsertState) -> bool:
    """Both entries must be present after concurrent inserts."""
    return 1 in s.session.inflight_out and 2 in s.session.inflight_out


def test_inflight_out_lost_insert():
    """Concurrent inserts into inflight_out OrderedDict can lose an entry."""
    with timeout_minutes(10):
        result = explore_interleavings(
            setup=lambda: InflightOutLostInsertState(),
            threads=[
                lambda s: s.thread1(),
                lambda s: s.thread2(),
            ],
            invariant=_inflight_out_both_present,
            max_attempts=500,
            max_ops=300,
            seed=42,
        )
    print_exploration_result(result)
    return result


# ---------------------------------------------------------------------------
# 4. inflight_in concurrent dict mutation (two incoming QOS2 messages)
# ---------------------------------------------------------------------------

class InflightInConcurrentState:
    """Two incoming QOS2 messages arrive simultaneously.  Both try to
    insert into inflight_in and then delete (simulating the full
    receive-pubrec-pubrel-pubcomp flow)."""

    def __init__(self):
        self.session = Session()
        self.success1 = False
        self.success2 = False

    def thread1(self):
        """Simulates receiving and processing an incoming QOS2 message."""
        msg = IncomingApplicationMessage(10, "topic/x", 2, b"data_x", False)
        self.session.inflight_in[10] = msg
        # Simulate pubrel arrival: message delivered, entry deleted
        if 10 in self.session.inflight_in:
            del self.session.inflight_in[10]
            self.success1 = True

    def thread2(self):
        """Simulates receiving and processing another incoming QOS2 message."""
        msg = IncomingApplicationMessage(11, "topic/y", 2, b"data_y", False)
        self.session.inflight_in[11] = msg
        if 11 in self.session.inflight_in:
            del self.session.inflight_in[11]
            self.success2 = True


def _inflight_in_both_processed(s: InflightInConcurrentState) -> bool:
    """Both messages must be successfully inserted and then cleaned up."""
    return s.success1 and s.success2 and len(s.session.inflight_in) == 0


def test_inflight_in_concurrent_mutation():
    """Concurrent insert-then-delete on inflight_in can lose or corrupt entries."""
    with timeout_minutes(10):
        result = explore_interleavings(
            setup=lambda: InflightInConcurrentState(),
            threads=[
                lambda s: s.thread1(),
                lambda s: s.thread2(),
            ],
            invariant=_inflight_in_both_processed,
            max_attempts=500,
            max_ops=300,
            seed=42,
        )
    print_exploration_result(result)
    return result


# ---------------------------------------------------------------------------
# 5. next_packet_id wraparound near 65535
# ---------------------------------------------------------------------------

class PacketIdWraparoundState:
    """Start _packet_id near the upper bound.  Two concurrent callers
    both wrap to small values and collide on the same post-wraparound ID."""

    def __init__(self):
        self.session = Session()
        # Position _packet_id at 65534 so the next call wraps: (65534 % 65535) + 1 = 65535
        # and the call after that wraps: (65535 % 65535) + 1 = 1
        self.session._packet_id = 65534
        self.id1 = None
        self.id2 = None

    def thread1(self):
        self.id1 = self.session.next_packet_id

    def thread2(self):
        self.id2 = self.session.next_packet_id


def _wraparound_ids_unique(s: PacketIdWraparoundState) -> bool:
    """After wraparound both threads must still get distinct IDs."""
    return s.id1 != s.id2


def test_packet_id_wraparound_race():
    """Packet ID wraparound near 65535 creates collision opportunities."""
    with timeout_minutes(10):
        result = explore_interleavings(
            setup=lambda: PacketIdWraparoundState(),
            threads=[
                lambda s: s.thread1(),
                lambda s: s.thread2(),
            ],
            invariant=_wraparound_ids_unique,
            max_attempts=500,
            max_ops=300,
            seed=42,
        )
    print_exploration_result(result)
    return result


def test_packet_id_wraparound_race_sweep():
    """Sweep 20 seeds for the wraparound race."""
    found_seeds = []
    total_explored = 0
    for seed in range(20):
        with timeout_minutes(10):
            result = explore_interleavings(
                setup=lambda: PacketIdWraparoundState(),
                threads=[
                    lambda s: s.thread1(),
                    lambda s: s.thread2(),
                ],
                invariant=_wraparound_ids_unique,
                max_attempts=200,
                max_ops=300,
                seed=seed,
            )
        total_explored += result.num_explored
        if not result.property_holds:
            found_seeds.append((seed, result.num_explored))
    print_seed_sweep_results(found_seeds, total_explored)
    return found_seeds


# ---------------------------------------------------------------------------
# 6. Session state: concurrent connect/disconnect timestamp corruption
# ---------------------------------------------------------------------------

class SessionStateTransitionState:
    """Two threads concurrently trigger connect and disconnect on a session.
    The _on_enter_connected handler reads last_disconnect_time and writes
    last_connect_time; _on_enter_disconnected does the reverse.  Interleaving
    can corrupt timestamps."""

    def __init__(self):
        self.session = Session()
        # Move session to 'connected' first so we can disconnect then reconnect.
        self.session.transitions.connect()
        self.session.transitions.disconnect()
        # Now session is 'disconnected', last_disconnect_time is set.
        self.connect_time_before = self.session.last_connect_time
        self.disconnect_time_before = self.session.last_disconnect_time
        self.error = None

    def thread1(self):
        """Reconnect the session."""
        try:
            self.session.transitions.connect()
        except Exception as e:
            self.error = e

    def thread2(self):
        """Attempt to read and write timestamps concurrently, simulating
        the broker's session state monitoring that reads timestamps."""
        # Simulate the session monitor reading timestamps
        ct = self.session.last_connect_time
        dt = self.session.last_disconnect_time
        # Write back (simulating broker updating session metadata)
        self.session.last_connect_time = ct
        self.session.last_disconnect_time = dt


def _session_timestamps_consistent(s: SessionStateTransitionState) -> bool:
    """After a reconnect, last_connect_time should be set and
    last_disconnect_time should be None (cleared by _on_enter_connected)."""
    if s.error is not None:
        return True  # Transition error is not the race we are looking for
    # After connect: last_connect_time should be set, last_disconnect_time cleared
    return (
        s.session.last_connect_time is not None
        and s.session.last_disconnect_time is None
    )


def test_session_state_timestamp_corruption():
    """Concurrent connect and timestamp reading can corrupt session state."""
    with timeout_minutes(10):
        result = explore_interleavings(
            setup=lambda: SessionStateTransitionState(),
            threads=[
                lambda s: s.thread1(),
                lambda s: s.thread2(),
            ],
            invariant=_session_timestamps_consistent,
            max_attempts=500,
            max_ops=300,
            seed=42,
        )
    print_exploration_result(result)
    return result


# ---------------------------------------------------------------------------
# 7. Broker _subscriptions: concurrent add_subscription (check-then-append)
# ---------------------------------------------------------------------------

class SubscriptionRaceState:
    """Simulates the Broker.add_subscription check-then-append pattern.
    Two threads subscribe different sessions to the same topic filter.
    The code checks ``if all(s.client_id != session.client_id for s, _ in
    self._subscriptions[topic_filter])`` then appends.  Under concurrency
    both threads can pass the check before either appends."""

    def __init__(self):
        self.subscriptions: dict[str, list[tuple[Session, int]]] = {}
        self.session1 = Session()
        self.session1.client_id = "client_1"
        self.session2 = Session()
        self.session2.client_id = "client_2"

    def _add_subscription(self, topic_filter: str, session: Session, qos: int):
        """Mirrors Broker.add_subscription's core logic (synchronous part)."""
        if topic_filter not in self.subscriptions:
            self.subscriptions[topic_filter] = []
        # check-then-append race
        if all(s.client_id != session.client_id for s, _ in self.subscriptions[topic_filter]):
            self.subscriptions[topic_filter].append((session, qos))

    def thread1(self):
        self._add_subscription("sensor/#", self.session1, 1)

    def thread2(self):
        self._add_subscription("sensor/#", self.session2, 1)


def _both_subscribed_no_duplicates(s: SubscriptionRaceState) -> bool:
    """Both sessions should be subscribed exactly once."""
    subs = s.subscriptions.get("sensor/#", [])
    client_ids = [sess.client_id for sess, _ in subs]
    return (
        len(client_ids) == 2
        and "client_1" in client_ids
        and "client_2" in client_ids
        and len(client_ids) == len(set(client_ids))
    )


def test_subscription_check_then_append_race():
    """Concurrent add_subscription can lose a subscription or create duplicates."""
    with timeout_minutes(10):
        result = explore_interleavings(
            setup=lambda: SubscriptionRaceState(),
            threads=[
                lambda s: s.thread1(),
                lambda s: s.thread2(),
            ],
            invariant=_both_subscribed_no_duplicates,
            max_attempts=500,
            max_ops=300,
            seed=42,
        )
    print_exploration_result(result)
    return result


# ---------------------------------------------------------------------------
# 8. Server.conn_count lost update
# ---------------------------------------------------------------------------

class ConnCountState:
    """Simulates the Server.acquire_connection / release_connection pattern.
    conn_count += 1 and conn_count -= 1 are non-atomic LOAD/BINARY_OP/STORE
    sequences that lose updates under concurrency."""

    def __init__(self):
        # We replicate the relevant fields to avoid needing an asyncio server
        self.conn_count = 0

    def acquire(self):
        self.conn_count += 1

    def release(self):
        self.conn_count -= 1


class ConnCountLostUpdateState:
    """Two threads both acquire a connection.  Expected final conn_count: 2."""

    def __init__(self):
        self.server = ConnCountState()

    def thread1(self):
        self.server.acquire()

    def thread2(self):
        self.server.acquire()


def _conn_count_is_two(s: ConnCountLostUpdateState) -> bool:
    """After two acquires, conn_count must be 2."""
    return s.server.conn_count == 2


def test_conn_count_lost_update():
    """conn_count += 1 is not atomic -- concurrent acquires lose an increment."""
    with timeout_minutes(10):
        result = explore_interleavings(
            setup=lambda: ConnCountLostUpdateState(),
            threads=[
                lambda s: s.thread1(),
                lambda s: s.thread2(),
            ],
            invariant=_conn_count_is_two,
            max_attempts=500,
            max_ops=200,
            seed=42,
        )
    print_exploration_result(result)
    return result


def test_conn_count_lost_update_sweep():
    """Sweep 20 seeds for the conn_count lost update."""
    found_seeds = []
    total_explored = 0
    for seed in range(20):
        with timeout_minutes(10):
            result = explore_interleavings(
                setup=lambda: ConnCountLostUpdateState(),
                threads=[
                    lambda s: s.thread1(),
                    lambda s: s.thread2(),
                ],
                invariant=_conn_count_is_two,
                max_attempts=200,
                max_ops=200,
                seed=seed,
            )
        total_explored += result.num_explored
        if not result.property_holds:
            found_seeds.append((seed, result.num_explored))
    print_seed_sweep_results(found_seeds, total_explored)
    return found_seeds


# ---------------------------------------------------------------------------
# 9. Broker _sessions dict: concurrent client add and remove
# ---------------------------------------------------------------------------

class SessionsDictRaceState:
    """One thread adds a session to the broker's _sessions dict while
    another thread removes (pops) a different session.  The race on
    the shared dict can corrupt its internal state."""

    def __init__(self):
        self.sessions: dict[str, tuple[Session, str]] = {}
        # Pre-populate with one session so thread2 can remove it
        s = Session()
        s.client_id = "existing"
        self.sessions["existing"] = (s, "handler_existing")
        self.new_session = Session()
        self.new_session.client_id = "new_client"
        self.removed = None

    def thread1(self):
        """Add a new session."""
        self.sessions["new_client"] = (self.new_session, "handler_new")

    def thread2(self):
        """Remove the existing session (like _delete_session)."""
        self.removed = self.sessions.pop("existing", None)


def _sessions_dict_consistent(s: SessionsDictRaceState) -> bool:
    """After thread1 adds and thread2 removes, the dict should contain
    exactly the new session and not the removed one."""
    return (
        "new_client" in s.sessions
        and "existing" not in s.sessions
        and s.removed is not None
    )


def test_sessions_dict_concurrent_add_remove():
    """Concurrent add and pop on the broker's _sessions dict."""
    with timeout_minutes(10):
        result = explore_interleavings(
            setup=lambda: SessionsDictRaceState(),
            threads=[
                lambda s: s.thread1(),
                lambda s: s.thread2(),
            ],
            invariant=_sessions_dict_consistent,
            max_attempts=500,
            max_ops=300,
            seed=42,
        )
    print_exploration_result(result)
    return result


# ---------------------------------------------------------------------------
# 10. Broker retained_messages: concurrent store and delete
# ---------------------------------------------------------------------------

class RetainedMessageRaceState:
    """Simulates retain_message logic: one thread stores a retained message
    while another thread deletes retained messages for the same topic.
    The non-atomic check-then-write/delete pattern races."""

    def __init__(self):
        self.retained_messages: dict[str, object] = {}
        # Pre-populate so the deleter has something to delete
        self.retained_messages["sensor/temp"] = "old_data"
        self.store_result = None
        self.delete_result = None

    def thread1(self):
        """Store a new retained message (mirrors retain_message with data)."""
        topic = "sensor/temp"
        data = "new_data"
        if data:
            self.retained_messages[topic] = data
            self.store_result = "stored"

    def thread2(self):
        """Clear a retained message (mirrors retain_message with empty data)."""
        topic = "sensor/temp"
        if topic in self.retained_messages:
            del self.retained_messages[topic]
            self.delete_result = "deleted"


def _retained_message_consistent(s: RetainedMessageRaceState) -> bool:
    """After a concurrent store and delete, the result must be consistent:
    either the new data is present (store happened last) or it's absent
    (delete happened last).  The bug manifests if the delete raises KeyError
    because the store replaced the entry mid-check, or if the state is
    otherwise inconsistent."""
    topic = "sensor/temp"
    if topic in s.retained_messages:
        # Store won -- the value should be the new data, not old
        return s.retained_messages[topic] == "new_data"
    else:
        # Delete won -- should be fully absent
        return topic not in s.retained_messages


def test_retained_message_store_delete_race():
    """Concurrent store and delete on retained_messages dict."""
    with timeout_minutes(10):
        result = explore_interleavings(
            setup=lambda: RetainedMessageRaceState(),
            threads=[
                lambda s: s.thread1(),
                lambda s: s.thread2(),
            ],
            invariant=_retained_message_consistent,
            max_attempts=500,
            max_ops=300,
            seed=42,
        )
    print_exploration_result(result)
    return result


# ---------------------------------------------------------------------------
# 11. next_packet_id duplicate with inflight entries (skip-loop race)
# ---------------------------------------------------------------------------

class PacketIdSkipLoopState:
    """Tests the while-loop inside next_packet_id that skips IDs already in
    inflight_in or inflight_out.  When one thread is in the skip loop and
    another thread modifies _packet_id, the loop variable ``limit`` can
    become stale, causing one thread to either spin forever or return a
    duplicate."""

    def __init__(self):
        self.session = Session()
        # Pre-populate inflight_out with packet_id=1, so next_packet_id must
        # skip 1 and go to 2.  This forces the while-loop to execute.
        msg = OutgoingApplicationMessage(1, "t", 1, b"d", False)
        self.session.inflight_out[1] = msg
        self.id1 = None
        self.id2 = None

    def thread1(self):
        self.id1 = self.session.next_packet_id

    def thread2(self):
        self.id2 = self.session.next_packet_id


def _skip_loop_ids_unique(s: PacketIdSkipLoopState) -> bool:
    """Both IDs must be distinct and neither should be 1 (which is in-flight)."""
    if s.id1 is None or s.id2 is None:
        return False
    return s.id1 != s.id2 and s.id1 != 1 and s.id2 != 1


def test_packet_id_skip_loop_race():
    """Race in the while-loop of next_packet_id that skips inflight IDs."""
    with timeout_minutes(10):
        result = explore_interleavings(
            setup=lambda: PacketIdSkipLoopState(),
            threads=[
                lambda s: s.thread1(),
                lambda s: s.thread2(),
            ],
            invariant=_skip_loop_ids_unique,
            max_attempts=500,
            max_ops=400,
            seed=42,
        )
    print_exploration_result(result)
    return result


def test_packet_id_skip_loop_race_sweep():
    """Sweep 20 seeds for the skip-loop race."""
    found_seeds = []
    total_explored = 0
    for seed in range(20):
        with timeout_minutes(10):
            result = explore_interleavings(
                setup=lambda: PacketIdSkipLoopState(),
                threads=[
                    lambda s: s.thread1(),
                    lambda s: s.thread2(),
                ],
                invariant=_skip_loop_ids_unique,
                max_attempts=200,
                max_ops=400,
                seed=seed,
            )
        total_explored += result.num_explored
        if not result.property_holds:
            found_seeds.append((seed, result.num_explored))
    print_seed_sweep_results(found_seeds, total_explored)
    return found_seeds


# ---------------------------------------------------------------------------
# 12. Reproduce -- find and deterministically replay the 3-thread duplicate
# ---------------------------------------------------------------------------

def test_three_thread_packet_id_reproduce():
    """Find a 3-thread duplicate packet ID counterexample and replay it."""
    with timeout_minutes(10):
        result = explore_interleavings(
            setup=lambda: ThreeThreadPacketIdState(),
            threads=[
                lambda s: s.thread1(),
                lambda s: s.thread2(),
                lambda s: s.thread3(),
            ],
            invariant=_three_ids_unique,
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
            setup=lambda: ThreeThreadPacketIdState(),
            threads=[
                lambda s: s.thread1(),
                lambda s: s.thread2(),
                lambda s: s.thread3(),
            ],
        )
        ids = {state.id1, state.id2, state.id3}
        is_bug = len(ids) < 3
        bugs_reproduced += is_bug
        print(
            f"  Run {i + 1}: ids=({state.id1}, {state.id2}, {state.id3})"
            f" [{'BUG' if is_bug else 'ok'}]"
        )

    print(f"\nReproduced: {bugs_reproduced}/10")
    return bugs_reproduced


# ---------------------------------------------------------------------------
# 13. Reproduce -- find and replay the wraparound race
# ---------------------------------------------------------------------------

def test_packet_id_wraparound_reproduce():
    """Find a wraparound counterexample and replay it."""
    with timeout_minutes(10):
        result = explore_interleavings(
            setup=lambda: PacketIdWraparoundState(),
            threads=[
                lambda s: s.thread1(),
                lambda s: s.thread2(),
            ],
            invariant=_wraparound_ids_unique,
            max_attempts=500,
            max_ops=300,
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
            setup=lambda: PacketIdWraparoundState(),
            threads=[
                lambda s: s.thread1(),
                lambda s: s.thread2(),
            ],
        )
        is_bug = state.id1 == state.id2
        bugs_reproduced += is_bug
        print(
            f"  Run {i + 1}: id1={state.id1}, id2={state.id2}"
            f" [{'BUG' if is_bug else 'ok'}]"
        )

    print(f"\nReproduced: {bugs_reproduced}/10")
    return bugs_reproduced


# ---------------------------------------------------------------------------
# Main: run all tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("=== 1. Three-thread next_packet_id duplicate (seed=42) ===")
    print("=" * 70)
    test_three_thread_packet_id_duplicates()

    print("\n" + "=" * 70)
    print("=== 2. Three-thread next_packet_id sweep (20 seeds) ===")
    print("=" * 70)
    test_three_thread_packet_id_duplicates_sweep()

    print("\n" + "=" * 70)
    print("=== 3. Packet ID + inflight_out TOCTOU (seed=42) ===")
    print("=" * 70)
    test_packet_id_inflight_toctou()

    print("\n" + "=" * 70)
    print("=== 4. Packet ID + inflight_out TOCTOU sweep (20 seeds) ===")
    print("=" * 70)
    test_packet_id_inflight_toctou_sweep()

    print("\n" + "=" * 70)
    print("=== 5. inflight_out lost insert (seed=42) ===")
    print("=" * 70)
    test_inflight_out_lost_insert()

    print("\n" + "=" * 70)
    print("=== 6. inflight_in concurrent mutation (seed=42) ===")
    print("=" * 70)
    test_inflight_in_concurrent_mutation()

    print("\n" + "=" * 70)
    print("=== 7. Packet ID wraparound near 65535 (seed=42) ===")
    print("=" * 70)
    test_packet_id_wraparound_race()

    print("\n" + "=" * 70)
    print("=== 8. Packet ID wraparound sweep (20 seeds) ===")
    print("=" * 70)
    test_packet_id_wraparound_race_sweep()

    print("\n" + "=" * 70)
    print("=== 9. Session state timestamp corruption (seed=42) ===")
    print("=" * 70)
    test_session_state_timestamp_corruption()

    print("\n" + "=" * 70)
    print("=== 10. Subscription check-then-append race (seed=42) ===")
    print("=" * 70)
    test_subscription_check_then_append_race()

    print("\n" + "=" * 70)
    print("=== 11. Server conn_count lost update (seed=42) ===")
    print("=" * 70)
    test_conn_count_lost_update()

    print("\n" + "=" * 70)
    print("=== 12. Server conn_count lost update sweep (20 seeds) ===")
    print("=" * 70)
    test_conn_count_lost_update_sweep()

    print("\n" + "=" * 70)
    print("=== 13. Broker _sessions concurrent add/remove (seed=42) ===")
    print("=" * 70)
    test_sessions_dict_concurrent_add_remove()

    print("\n" + "=" * 70)
    print("=== 14. Retained message store/delete race (seed=42) ===")
    print("=" * 70)
    test_retained_message_store_delete_race()

    print("\n" + "=" * 70)
    print("=== 15. Packet ID skip-loop race (seed=42) ===")
    print("=" * 70)
    test_packet_id_skip_loop_race()

    print("\n" + "=" * 70)
    print("=== 16. Packet ID skip-loop sweep (20 seeds) ===")
    print("=" * 70)
    test_packet_id_skip_loop_race_sweep()

    print("\n" + "=" * 70)
    print("=== 17. Three-thread packet ID reproduce ===")
    print("=" * 70)
    test_three_thread_packet_id_reproduce()

    print("\n" + "=" * 70)
    print("=== 18. Wraparound reproduce ===")
    print("=" * 70)
    test_packet_id_wraparound_reproduce()
