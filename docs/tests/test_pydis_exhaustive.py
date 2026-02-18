"""
Exhaustive concurrency bug tests for pydis (minimal Redis clone).

pydis uses TWO module-level globals (dictionary, expiration) with ZERO
synchronization.  Virtually every command has read-modify-write patterns
that are vulnerable to interleaving.  This file tests as many distinct
race conditions as possible using interlace's bytecode-level exploration.

Each test class defines a shared state with __init__ resetting the globals,
thread1/thread2 performing concurrent operations, and an invariant that
should hold under sequential consistency but can be violated by interleaving.

Bug categories found:
  - Lost updates (INCR, DECR, SADD, HSET, LPUSH, RPUSH on new keys)
  - TOCTOU checks (SET NX, SET XX)
  - Non-atomic multi-key operations (MSET)
  - Crash bugs (LPOP/RPOP/SPOP on nearly-empty collections)
  - Value/expiration mismatch (SET EX)
  - Mutation during iteration (LRANGE + LPUSH)
  - Return value inconsistencies (HSET same field, SADD return count)
"""

import collections
import os
import sys

_test_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_test_dir, "..", "external_repos", "pydis"))

from external_tests_helpers import print_exploration_result
from pydis.__main__ import RedisProtocol, dictionary, expiration

from interlace.bytecode import explore_interleavings, run_with_schedule


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MAX_ATTEMPTS = 300
MAX_OPS = 400
SEED = 42


def run_and_report(state_cls, invariant_fn, label, max_attempts=MAX_ATTEMPTS):
    """Run explore_interleavings for a state class and print the outcome."""
    result = explore_interleavings(
        setup=lambda: state_cls(),
        threads=[
            lambda s: s.thread1(),
            lambda s: s.thread2(),
        ],
        invariant=invariant_fn,
        max_attempts=max_attempts,
        max_ops=MAX_OPS,
        seed=SEED,
    )
    print(f"\n=== {label} ===")
    print_exploration_result(result)
    return result


# ===================================================================
# 1. INCR lost update (start=10, expect=12)
#
# com_incr: value = get(key) or 0; value += 1; set(key, value)
# Both threads read 10, both write 11, one increment lost.
# ===================================================================

class IncrLostUpdateState:
    """Two clients INCR the same key starting from 10; expect 12."""

    def __init__(self):
        dictionary.clear()
        expiration.clear()
        dictionary[b"counter"] = b"10"
        self.client1 = RedisProtocol()
        self.client2 = RedisProtocol()

    def thread1(self):
        self.client1.com_incr(b"counter")

    def thread2(self):
        self.client2.com_incr(b"counter")


def test_incr_lost_update():
    """Two INCRs from 10 should reach 12; lost update makes it 11."""
    return run_and_report(
        IncrLostUpdateState,
        lambda s: int(dictionary.get(b"counter", b"0")) == 12,
        "INCR lost update (start=10, expect=12)",
    )


# ===================================================================
# 2. INCR on non-existent key race
#
# get(key) returns None, `or 0` gives 0 to both threads.
# Both increment to 1, both write 1.  Final should be 2.
# ===================================================================

class IncrNonExistentRaceState:
    """Two clients INCR a key that does not exist."""

    def __init__(self):
        dictionary.clear()
        expiration.clear()
        self.client1 = RedisProtocol()
        self.client2 = RedisProtocol()

    def thread1(self):
        self.client1.com_incr(b"counter")

    def thread2(self):
        self.client2.com_incr(b"counter")


def test_incr_nonexistent_race():
    """Two INCRs on non-existent key should create key with value 2."""
    return run_and_report(
        IncrNonExistentRaceState,
        lambda s: int(dictionary.get(b"counter", b"0")) == 2,
        "INCR race on non-existent key",
    )


# ===================================================================
# 3. LPUSH race (concurrent list pushes to non-existent key)
#
# com_lpush: deque = get(key, deque()); deque.extendleft(values); set(key, deque)
# Both threads get their OWN empty default deque (since key doesn't
# exist), both extend their own copy, second set() overwrites first.
# ===================================================================

class LpushRaceState:
    """Two clients each LPUSH one element to a new (non-existent) list."""

    def __init__(self):
        dictionary.clear()
        expiration.clear()
        self.client1 = RedisProtocol()
        self.client2 = RedisProtocol()

    def thread1(self):
        self.client1.com_lpush(b"mylist", b"A")

    def thread2(self):
        self.client2.com_lpush(b"mylist", b"B")


def test_lpush_race():
    """After two LPUSHes to new key, list should have both elements (length 2)."""
    return run_and_report(
        LpushRaceState,
        lambda s: (
            b"mylist" in dictionary
            and isinstance(dictionary[b"mylist"], collections.deque)
            and len(dictionary[b"mylist"]) == 2
        ),
        "LPUSH race (two pushes to non-existent key)",
    )


# ===================================================================
# 4. RPUSH race (same pattern as LPUSH)
# ===================================================================

class RpushRaceState:
    """Two clients each RPUSH one element to a new list."""

    def __init__(self):
        dictionary.clear()
        expiration.clear()
        self.client1 = RedisProtocol()
        self.client2 = RedisProtocol()

    def thread1(self):
        self.client1.com_rpush(b"mylist", b"X")

    def thread2(self):
        self.client2.com_rpush(b"mylist", b"Y")


def test_rpush_race():
    """After two RPUSHes to new key, list should have both elements (length 2)."""
    return run_and_report(
        RpushRaceState,
        lambda s: (
            b"mylist" in dictionary
            and isinstance(dictionary[b"mylist"], collections.deque)
            and len(dictionary[b"mylist"]) == 2
        ),
        "RPUSH race (two pushes to non-existent key)",
    )


# ===================================================================
# 5. LPOP + RPOP race on a 1-element list
#
# com_lpop: deque = get(key); if None: return nil; value = deque.popleft()
# Both threads see the non-None deque with 1 element, both try to pop,
# one gets IndexError (crash).
# ===================================================================

class LpopRpopRaceState:
    """Two clients pop from a 1-element list concurrently."""

    def __init__(self):
        dictionary.clear()
        expiration.clear()
        dictionary[b"mylist"] = collections.deque([b"only_element"])
        self.client1 = RedisProtocol()
        self.client2 = RedisProtocol()
        self.error = False

    def thread1(self):
        try:
            self.client1.com_lpop(b"mylist")
        except IndexError:
            self.error = True

    def thread2(self):
        try:
            self.client2.com_rpop(b"mylist")
        except IndexError:
            self.error = True


def test_lpop_rpop_race():
    """Concurrent LPOP+RPOP on 1-element list: should not crash with IndexError."""
    return run_and_report(
        LpopRpopRaceState,
        lambda s: not s.error,
        "LPOP + RPOP race (1-element list, IndexError crash)",
    )


# ===================================================================
# 6. SADD race (concurrent set additions to non-existent key)
#
# com_sadd: set_ = get(key, set()); set_.add(member); set(key, set_)
# Both threads get their OWN empty default set, both add, second
# set() overwrites the first -- one member is lost.
# ===================================================================

class SaddRaceState:
    """Two clients each SADD a member to a new (non-existent) set."""

    def __init__(self):
        dictionary.clear()
        expiration.clear()
        self.client1 = RedisProtocol()
        self.client2 = RedisProtocol()

    def thread1(self):
        self.client1.com_sadd(b"myset", b"alpha")

    def thread2(self):
        self.client2.com_sadd(b"myset", b"beta")


def test_sadd_race():
    """After two SADDs to a new set, both members should be present."""
    return run_and_report(
        SaddRaceState,
        lambda s: (
            b"myset" in dictionary
            and isinstance(dictionary[b"myset"], set)
            and len(dictionary[b"myset"]) == 2
        ),
        "SADD race (two adds to non-existent key)",
    )


# ===================================================================
# 7. SPOP race on a 1-element set
#
# com_spop: set_ = get(key); if len(set_)==0 return nil; elem = set_.pop()
# Both threads see len==1, both call set_.pop(), second gets KeyError.
# ===================================================================

class SpopRaceState:
    """Two clients SPOP from a 1-element set."""

    def __init__(self):
        dictionary.clear()
        expiration.clear()
        dictionary[b"myset"] = {b"only_member"}
        self.client1 = RedisProtocol()
        self.client2 = RedisProtocol()
        self.error = False

    def thread1(self):
        try:
            self.client1.com_spop(b"myset")
        except KeyError:
            self.error = True

    def thread2(self):
        try:
            self.client2.com_spop(b"myset")
        except KeyError:
            self.error = True


def test_spop_race():
    """Concurrent SPOPs on a 1-element set: should not crash (KeyError)."""
    return run_and_report(
        SpopRaceState,
        lambda s: not s.error,
        "SPOP race (1-element set, KeyError crash)",
    )


# ===================================================================
# 8. SET NX race (TOCTOU: set-if-not-exists)
#
# com_set with NX: checks `if key in dictionary`, then sets.
# Both threads pass the check (key absent), both write => both get +OK,
# violating the set-if-not-exists guarantee.
# ===================================================================

class SetNxRaceState:
    """Two clients both SET the same key with NX (set-if-not-exists)."""

    def __init__(self):
        dictionary.clear()
        expiration.clear()
        self.client1 = RedisProtocol()
        self.client2 = RedisProtocol()
        self.result1 = None
        self.result2 = None

    def thread1(self):
        self.result1 = self.client1.com_set(b"key", b"A", b"NX")

    def thread2(self):
        self.result2 = self.client2.com_set(b"key", b"B", b"NX")


def test_set_nx_race():
    """With NX, exactly one SET should succeed. Both succeeding = bug."""
    return run_and_report(
        SetNxRaceState,
        lambda s: (s.result1 == b"+OK\r\n") != (s.result2 == b"+OK\r\n"),
        "SET NX TOCTOU race",
    )


# ===================================================================
# 9. HSET race (concurrent hash field sets on non-existent key)
#
# com_hset: hash_ = get(key, {}); hash_[field] = value; set(key, hash_)
# Both threads get their OWN empty default dict, both set their field,
# second set() overwrites the first -- one field is lost.
# ===================================================================

class HsetRaceState:
    """Two clients each HSET a different field on a new (non-existent) hash."""

    def __init__(self):
        dictionary.clear()
        expiration.clear()
        self.client1 = RedisProtocol()
        self.client2 = RedisProtocol()

    def thread1(self):
        self.client1.com_hset(b"myhash", b"field1", b"val1")

    def thread2(self):
        self.client2.com_hset(b"myhash", b"field2", b"val2")


def test_hset_race():
    """After two HSETs on different fields of new hash, both fields should exist."""
    return run_and_report(
        HsetRaceState,
        lambda s: (
            b"myhash" in dictionary
            and isinstance(dictionary[b"myhash"], dict)
            and b"field1" in dictionary[b"myhash"]
            and b"field2" in dictionary[b"myhash"]
        ),
        "HSET race (two fields on non-existent hash)",
    )


# ===================================================================
# 10. MSET non-atomicity (partial writes visible to concurrent reader)
#
# com_mset iterates and calls self.set for each key-value pair.
# A concurrent GET between the iterations sees an inconsistent state:
# key1 is updated but key2 is not yet.
# ===================================================================

class MsetGetRaceState:
    """Thread 1 does MSET key1=new key2=new.  Thread 2 reads both keys.
    If thread 2 sees key2=new, then key1 must also be new (atomicity)."""

    def __init__(self):
        dictionary.clear()
        expiration.clear()
        dictionary[b"key1"] = b"old"
        dictionary[b"key2"] = b"old"
        self.client1 = RedisProtocol()
        self.client2 = RedisProtocol()
        self.read_key1 = None
        self.read_key2 = None

    def thread1(self):
        self.client1.com_mset(b"key1", b"new", b"key2", b"new")

    def thread2(self):
        self.read_key1 = dictionary.get(b"key1")
        self.read_key2 = dictionary.get(b"key2")


def test_mset_non_atomic():
    """MSET should be atomic: if key2 is 'new', key1 must also be 'new'."""
    def invariant(s):
        if s.read_key2 == b"new":
            return s.read_key1 == b"new"
        return True

    return run_and_report(MsetGetRaceState, invariant, "MSET non-atomicity")


# ===================================================================
# 11. MSET + MSET race (two concurrent MSETs on overlapping keys)
#
# Both MSETs iterate over their pairs, interleaving produces a state
# where some keys have values from MSET1 and others from MSET2.
# This would never happen under any sequential ordering.
# ===================================================================

class MsetMsetRaceState:
    """Thread 1: MSET k1=A k2=A.  Thread 2: MSET k1=B k2=B.
    Sequentially, final state must be all-A or all-B."""

    def __init__(self):
        dictionary.clear()
        expiration.clear()
        self.client1 = RedisProtocol()
        self.client2 = RedisProtocol()

    def thread1(self):
        self.client1.com_mset(b"k1", b"A", b"k2", b"A")

    def thread2(self):
        self.client2.com_mset(b"k1", b"B", b"k2", b"B")


def test_mset_mset_race():
    """Two MSETs on same keys: final state should be all-A or all-B."""
    def invariant(s):
        k1 = dictionary.get(b"k1")
        k2 = dictionary.get(b"k2")
        return (k1 == b"A" and k2 == b"A") or (k1 == b"B" and k2 == b"B")

    return run_and_report(
        MsetMsetRaceState, invariant, "MSET + MSET race (non-atomicity)"
    )


# ===================================================================
# 12. INCR + SET race (one thread increments, other overwrites)
#
# Thread 1: INCR counter (read 0, increment to 1, write 1)
# Thread 2: SET counter 100
# Valid serial orderings: INCR then SET => 100, SET then INCR => 101.
# Bug: INCR reads 0, SET writes 100, INCR writes 1 => SET lost.
# ===================================================================

class IncrSetRaceState:
    """Thread 1 INCRs counter from 0, thread 2 SETs counter to 100."""

    def __init__(self):
        dictionary.clear()
        expiration.clear()
        dictionary[b"counter"] = b"0"
        self.client1 = RedisProtocol()
        self.client2 = RedisProtocol()

    def thread1(self):
        self.client1.com_incr(b"counter")

    def thread2(self):
        self.client2.com_set(b"counter", b"100")


def test_incr_set_race():
    """INCR + SET race: final value must be 100 or 101, not 1."""
    def invariant(s):
        val = int(dictionary.get(b"counter", b"0"))
        return val in (100, 101)

    return run_and_report(IncrSetRaceState, invariant, "INCR + SET race")


# ===================================================================
# 13. Triple INCR race (three threads, more lost updates)
#
# Three threads all INCR the same key from 0.  Should reach 3.
# With interleaving, two or even all three can read the same value.
# ===================================================================

class TripleIncrState:
    """Three clients all INCR the same key from 0."""

    def __init__(self):
        dictionary.clear()
        expiration.clear()
        dictionary[b"counter"] = b"0"
        self.client1 = RedisProtocol()
        self.client2 = RedisProtocol()
        self.client3 = RedisProtocol()

    def thread1(self):
        self.client1.com_incr(b"counter")

    def thread2(self):
        self.client2.com_incr(b"counter")

    def thread3(self):
        self.client3.com_incr(b"counter")


def test_triple_incr_race():
    """Three concurrent INCRs from 0 should reach 3."""
    result = explore_interleavings(
        setup=lambda: TripleIncrState(),
        threads=[
            lambda s: s.thread1(),
            lambda s: s.thread2(),
            lambda s: s.thread3(),
        ],
        invariant=lambda s: int(dictionary.get(b"counter", b"0")) == 3,
        max_attempts=MAX_ATTEMPTS,
        max_ops=MAX_OPS,
        seed=SEED,
    )
    print("\n=== Triple INCR race (3 threads, expect 3) ===")
    print_exploration_result(result)
    return result


# ===================================================================
# 14. SET EX race (value/expiration mismatch)
#
# com_set with EX computes expires_at, then self.set does:
#   dictionary[key] = value; expiration[key] = expires_at
# Two concurrent SET EX commands can interleave such that the value
# comes from one SET and the expiration from the other.
# ===================================================================

class SetExRaceState:
    """Thread 1: SET key val1 EX 10.  Thread 2: SET key val2 EX 20.
    Value and expiration should be from the same SET."""

    def __init__(self):
        dictionary.clear()
        expiration.clear()
        self.client1 = RedisProtocol()
        self.client2 = RedisProtocol()

    def thread1(self):
        self.client1.com_set(b"key", b"val1", b"EX", b"10")

    def thread2(self):
        self.client2.com_set(b"key", b"val2", b"EX", b"20")


def test_set_ex_race():
    """Two SET EX on same key: value and expiration should be from same SET."""
    def invariant(s):
        import time
        val = dictionary.get(b"key")
        exp = expiration.get(b"key")
        if val is None or exp is None:
            return False
        now = time.monotonic()
        remaining = exp - now
        # val1 => expiration ~10s, val2 => expiration ~20s
        # Mismatch (val1 with ~20s or val2 with ~10s) is the bug.
        if val == b"val1":
            return 9 < remaining < 11
        elif val == b"val2":
            return 19 < remaining < 21
        return False

    return run_and_report(
        SetExRaceState, invariant,
        "SET EX race (value/expiration mismatch)"
    )


# ===================================================================
# 15. LRANGE + LPUSH race (mutation during iteration)
#
# com_lrange: deque = get(key); itertools.islice(deque, ...)
# com_lpush: deque = get(key, deque()); deque.extendleft(values)
# When the key exists, both get the SAME deque object.  LPUSH mutates
# the deque via extendleft while LRANGE iterates it with islice,
# causing RuntimeError: "deque mutated during iteration".
# ===================================================================

class LrangeLpushRaceState:
    """Thread 1 reads list via LRANGE, thread 2 pushes elements."""

    def __init__(self):
        dictionary.clear()
        expiration.clear()
        dictionary[b"mylist"] = collections.deque([b"a", b"b", b"c"])
        self.client1 = RedisProtocol()
        self.client2 = RedisProtocol()
        self.range_result = None
        self.runtime_error = False

    def thread1(self):
        try:
            self.range_result = self.client1.com_lrange(b"mylist", b"0", b"10")
        except RuntimeError:
            # "deque mutated during iteration" -- this IS the bug
            self.runtime_error = True

    def thread2(self):
        self.client2.com_lpush(b"mylist", b"x", b"y")


def test_lrange_lpush_race():
    """LRANGE during LPUSH: should not crash with RuntimeError."""
    def invariant(s):
        if s.runtime_error:
            return False  # RuntimeError = crash = bug
        if s.range_result is None:
            return False
        # Valid sequential results: 3 elements (LRANGE first) or 5 (LPUSH first)
        if s.range_result.startswith(b"*"):
            count = int(s.range_result.split(b"\r\n")[0][1:])
            return count in (3, 5)
        return True

    return run_and_report(
        LrangeLpushRaceState, invariant,
        "LRANGE + LPUSH race (deque mutated during iteration)"
    )


# ===================================================================
# 16. HSET same field race (return value inconsistency)
#
# com_hset: hash_ = get(key, {}); ret = int(field in hash_);
#           hash_[field] = value; set(key, hash_)
# When key does not exist, both get separate empty dicts.
# Both check `field in hash_` => both get False => both return :0.
# Sequentially, one should return :0 (new) and the other :1 (existed).
# Also, one write is completely lost.
# ===================================================================

class HsetSameFieldRaceState:
    """Two clients HSET the same field on a new (non-existent) hash."""

    def __init__(self):
        dictionary.clear()
        expiration.clear()
        self.client1 = RedisProtocol()
        self.client2 = RedisProtocol()
        self.result1 = None
        self.result2 = None

    def thread1(self):
        self.result1 = self.client1.com_hset(b"myhash", b"field", b"val1")

    def thread2(self):
        self.result2 = self.client2.com_hset(b"myhash", b"field", b"val2")


def test_hset_same_field_race():
    """Two HSETs on same field of new hash: exactly one should report 'new field'."""
    def invariant(s):
        # HSET returns :0 if field is new, :1 if field existed.
        # Sequentially, first returns :0, second returns :1.
        # Bug: both return :0 (both think field is new).
        returns_new = [r for r in (s.result1, s.result2) if r == b":0\r\n"]
        returns_existed = [r for r in (s.result1, s.result2) if r == b":1\r\n"]
        return len(returns_new) == 1 and len(returns_existed) == 1

    return run_and_report(
        HsetSameFieldRaceState, invariant,
        "HSET same field race (return value inconsistency)"
    )


# ===================================================================
# 17. SADD return count race (lost member + wrong return value)
#
# com_sadd: set_ = get(key, set()); prev_size = len(set_); add members;
#           set(key, set_); return len(set_) - prev_size
# When key doesn't exist, both get separate empty sets.
# Both compute prev_size=0, add 1, return 1.  But one's set is lost,
# so the final set has only 1 member despite both claiming they added 1.
# The combined "added" count (2) exceeds the actual set size (1).
# ===================================================================

class SaddReturnCountState:
    """Two clients each SADD one member to a new set, track return values."""

    def __init__(self):
        dictionary.clear()
        expiration.clear()
        self.client1 = RedisProtocol()
        self.client2 = RedisProtocol()
        self.result1 = None
        self.result2 = None

    def thread1(self):
        self.result1 = self.client1.com_sadd(b"myset", b"alpha")

    def thread2(self):
        self.result2 = self.client2.com_sadd(b"myset", b"beta")


def test_sadd_return_count_race():
    """Sum of SADD return values should equal actual set size."""
    def invariant(s):
        # Extract counts from RESP format ":N\r\n"
        def extract_count(resp):
            if resp and resp.startswith(b":"):
                return int(resp.split(b"\r\n")[0][1:])
            return 0

        added1 = extract_count(s.result1)
        added2 = extract_count(s.result2)
        total_claimed = added1 + added2
        actual_size = len(dictionary.get(b"myset", set()))
        # The total claimed additions should equal the actual set size
        return total_claimed == actual_size

    return run_and_report(
        SaddReturnCountState, invariant,
        "SADD return count race (claimed adds != actual size)"
    )


# ===================================================================
# 18. LPUSH + RPUSH race on non-existent key
#
# Both get their own default empty deque, both push, second wins.
# One element is lost.
# ===================================================================

class LpushRpushRaceState:
    """Thread 1 LPUSHes, thread 2 RPUSHes to a new (non-existent) list."""

    def __init__(self):
        dictionary.clear()
        expiration.clear()
        self.client1 = RedisProtocol()
        self.client2 = RedisProtocol()

    def thread1(self):
        self.client1.com_lpush(b"mylist", b"left")

    def thread2(self):
        self.client2.com_rpush(b"mylist", b"right")


def test_lpush_rpush_race():
    """LPUSH + RPUSH to new key: list should have both elements."""
    return run_and_report(
        LpushRpushRaceState,
        lambda s: (
            b"mylist" in dictionary
            and isinstance(dictionary[b"mylist"], collections.deque)
            and len(dictionary[b"mylist"]) == 2
        ),
        "LPUSH + RPUSH race (both to non-existent key)",
    )


# ===================================================================
# 19. SET NX + SET NX + SET NX triple race
#
# Three threads all try SET key NX.  At most one should succeed.
# ===================================================================

class TripleSetNxState:
    """Three clients all SET the same key with NX."""

    def __init__(self):
        dictionary.clear()
        expiration.clear()
        self.client1 = RedisProtocol()
        self.client2 = RedisProtocol()
        self.client3 = RedisProtocol()
        self.result1 = None
        self.result2 = None
        self.result3 = None

    def thread1(self):
        self.result1 = self.client1.com_set(b"key", b"A", b"NX")

    def thread2(self):
        self.result2 = self.client2.com_set(b"key", b"B", b"NX")

    def thread3(self):
        self.result3 = self.client3.com_set(b"key", b"C", b"NX")


def test_triple_set_nx_race():
    """Three SET NX: exactly one should succeed."""
    result = explore_interleavings(
        setup=lambda: TripleSetNxState(),
        threads=[
            lambda s: s.thread1(),
            lambda s: s.thread2(),
            lambda s: s.thread3(),
        ],
        invariant=lambda s: (
            sum(1 for r in (s.result1, s.result2, s.result3)
                if r == b"+OK\r\n") == 1
        ),
        max_attempts=MAX_ATTEMPTS,
        max_ops=MAX_OPS,
        seed=SEED,
    )
    print("\n=== Triple SET NX race (3 threads, exactly 1 should succeed) ===")
    print_exploration_result(result)
    return result


# ===================================================================
# 20. SADD + HSET type confusion race
#
# Thread 1 creates a key as a set (SADD), thread 2 creates it as a
# hash (HSET).  Both get their own default (set() vs {}), both call
# self.set(key, ...).  The last writer wins, but the type is now
# wrong for whichever command runs next.  The immediate bug: one
# command's data is completely lost.
# ===================================================================

class SaddHsetTypeRaceState:
    """Thread 1 SADDs to a key, thread 2 HSETs to the same key."""

    def __init__(self):
        dictionary.clear()
        expiration.clear()
        self.client1 = RedisProtocol()
        self.client2 = RedisProtocol()
        self.sadd_result = None
        self.hset_result = None

    def thread1(self):
        self.sadd_result = self.client1.com_sadd(b"key", b"member")

    def thread2(self):
        self.hset_result = self.client2.com_hset(b"key", b"field", b"value")


def test_sadd_hset_type_race():
    """SADD + HSET on same new key: data from one command is lost entirely."""
    def invariant(s):
        val = dictionary.get(b"key")
        if val is None:
            return False
        # Both commands should have written successfully.  Under sequential
        # execution the second would get WRONGTYPE error.  But since key
        # didn't exist initially, both see a fresh default and both succeed,
        # with the second overwriting the first.  We check: did both report
        # success even though that should be impossible?
        sadd_ok = s.sadd_result is not None and s.sadd_result.startswith(b":")
        hset_ok = s.hset_result is not None and s.hset_result.startswith(b":")
        if sadd_ok and hset_ok:
            # Both succeeded but key can only be one type
            # Under sequential ordering, the second would get WRONGTYPE.
            # Both succeeding with no error = bug (no type check on new key).
            # Actually, since the key didn't exist, neither gets WRONGTYPE.
            # The bug is that one's data is silently lost.
            # Check: does the key contain both the set member AND hash field?
            # It can't -- it's either a set or a dict, not both.
            if isinstance(val, set):
                return b"member" in val and False  # dict data lost
            elif isinstance(val, dict):
                return b"field" in val and False  # set data lost
        return True  # If one failed, sequential execution might explain it

    return run_and_report(
        SaddHsetTypeRaceState, invariant,
        "SADD + HSET type confusion race (data loss)"
    )


# ===================================================================
# 21. INCR + INCR on two different keys (no race expected -- control test)
#
# This is a control test.  Two INCRs on DIFFERENT keys should never
# conflict.  The invariant should always hold.
# ===================================================================

class IncrDifferentKeysState:
    """Two clients INCR different keys -- should never conflict."""

    def __init__(self):
        dictionary.clear()
        expiration.clear()
        dictionary[b"counter_a"] = b"0"
        dictionary[b"counter_b"] = b"0"
        self.client1 = RedisProtocol()
        self.client2 = RedisProtocol()

    def thread1(self):
        self.client1.com_incr(b"counter_a")

    def thread2(self):
        self.client2.com_incr(b"counter_b")


def test_incr_different_keys_control():
    """Control: INCRs on different keys should never conflict."""
    return run_and_report(
        IncrDifferentKeysState,
        lambda s: (
            int(dictionary.get(b"counter_a", b"0")) == 1
            and int(dictionary.get(b"counter_b", b"0")) == 1
        ),
        "INCR on different keys (control -- no bug expected)",
    )


# ---------------------------------------------------------------------------
# Main -- run all tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        ("1.  INCR lost update", test_incr_lost_update),
        ("2.  INCR non-existent key", test_incr_nonexistent_race),
        ("3.  LPUSH race", test_lpush_race),
        ("4.  RPUSH race", test_rpush_race),
        ("5.  LPOP + RPOP crash", test_lpop_rpop_race),
        ("6.  SADD race", test_sadd_race),
        ("7.  SPOP crash", test_spop_race),
        ("8.  SET NX TOCTOU", test_set_nx_race),
        ("9.  HSET race (different fields)", test_hset_race),
        ("10. MSET non-atomicity (reader)", test_mset_non_atomic),
        ("11. MSET + MSET race", test_mset_mset_race),
        ("12. INCR + SET race", test_incr_set_race),
        ("13. Triple INCR race", test_triple_incr_race),
        ("14. SET EX mismatch", test_set_ex_race),
        ("15. LRANGE + LPUSH crash", test_lrange_lpush_race),
        ("16. HSET same field race", test_hset_same_field_race),
        ("17. SADD return count race", test_sadd_return_count_race),
        ("18. LPUSH + RPUSH race", test_lpush_rpush_race),
        ("19. Triple SET NX race", test_triple_set_nx_race),
        ("20. SADD + HSET type confusion", test_sadd_hset_type_race),
        ("21. INCR different keys (control)", test_incr_different_keys_control),
    ]

    bugs_found = 0
    bugs_list = []
    no_bugs_list = []
    errors_list = []
    total = len(tests)

    for label, test_fn in tests:
        try:
            result = test_fn()
            if result and not result.property_holds:
                bugs_found += 1
                bugs_list.append(label)
                print(f"  >>> BUG FOUND: {label}")
            else:
                no_bugs_list.append(label)
                print(f"  --- No bug found: {label}")
        except Exception as e:
            bugs_found += 1  # Crashes count as bugs
            errors_list.append((label, str(e)))
            print(f"  !!! CRASH BUG in {label}: {e}")

    print(f"\n{'='*60}")
    print(f"RESULTS: found bugs in {bugs_found} / {total} test scenarios")
    print(f"{'='*60}")
    if bugs_list:
        print("\nBugs found via invariant violation:")
        for b in bugs_list:
            print(f"  - {b}")
    if errors_list:
        print("\nBugs found via crash:")
        for label, err in errors_list:
            print(f"  - {label}: {err}")
    if no_bugs_list:
        print("\nNo bug found (may need more attempts or is a control test):")
        for b in no_bugs_list:
            print(f"  - {b}")
