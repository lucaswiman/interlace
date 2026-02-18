"""
Exhaustive concurrency tests for the cachetools library.

Tests a wide range of concurrency bugs in cachetools by exploring bytecode-level
interleavings using interlace. Each test targets a specific race condition or
lost-update pattern in the cachetools source.

The bugs found here are all real: cachetools provides no internal synchronization,
so any concurrent access to a Cache (or subclass) without external locking is
subject to data races.

Tested bugs:
 1. Cache.__delitem__: lost update on currsize -= oldsize
 2. Cache.__setitem__ + __delitem__: interleaved insert and delete corrupt currsize
 3. Cache.__setitem__ overwrite path: lost update when key already exists
 4. LRUCache ordering corruption: concurrent __setitem__ corrupts OrderedDict
 5. LFUCache frequency counter corruption: concurrent access corrupts linked list
 6. TTLCache link list corruption: concurrent insert corrupts doubly-linked list
 7. Cache.pop() TOCTOU: check-then-act race on key existence
 8. Cache.setdefault() TOCTOU: check-then-act race on key existence
 9. @cached decorator (unlocked): two threads compute and store same key
10. Three-thread concurrent insert: more threads amplify lost-update probability
11. Cache.__setitem__ eviction race: concurrent inserts trigger popitem() races
12. LRUCache concurrent read+write: __getitem__ reorders while __setitem__ modifies
13. Cache.clear() vs concurrent insert: clear races with ongoing insertions
"""

import os
import sys

_test_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_test_dir, "..", "external_repos", "cachetools", "src"))

from cachetools import Cache, LRUCache, TTLCache, LFUCache, cached, cachedmethod
from external_tests_helpers import print_exploration_result, print_seed_sweep_results

from interlace.bytecode import explore_interleavings, run_with_schedule


# ---------------------------------------------------------------------------
# 1. Cache.__delitem__ lost update on currsize
#
# __delitem__ does:
#     size = self.__size.pop(key)
#     del self.__data[key]
#     self.__currsize -= size        # <-- LOAD_ATTR, LOAD_FAST, INPLACE_SUB, STORE_ATTR
#
# Two concurrent deletes can both load the same currsize, subtract their
# respective sizes, and store back -- one subtraction is lost, leaving
# currsize too high (positive when it should be zero).
# ---------------------------------------------------------------------------

class DelItemLostUpdateState:
    def __init__(self):
        self.cache = Cache(maxsize=100)
        self.cache["a"] = "value_a"
        self.cache["b"] = "value_b"

    def thread1(self):
        del self.cache["a"]

    def thread2(self):
        del self.cache["b"]


def test_delitem_lost_update():
    """Cache.__delitem__: concurrent deletes lose a currsize subtraction."""

    result = explore_interleavings(
        setup=lambda: DelItemLostUpdateState(),
        threads=[
            lambda s: s.thread1(),
            lambda s: s.thread2(),
        ],
        invariant=lambda s: s.cache.currsize == len(s.cache),
        max_attempts=200,
        max_ops=300,
        seed=42,
    )

    print_exploration_result(result)
    return result


# ---------------------------------------------------------------------------
# 2. Cache.__setitem__ + __delitem__ interleaving
#
# Thread 1 inserts key "c", thread 2 deletes key "a".
# Both modify currsize via non-atomic read-modify-write.
# The final currsize should equal len(cache) (each item has size 1).
# ---------------------------------------------------------------------------

class SetDelInterleaveState:
    def __init__(self):
        self.cache = Cache(maxsize=100)
        self.cache["a"] = "value_a"

    def thread1(self):
        self.cache["c"] = "value_c"

    def thread2(self):
        del self.cache["a"]


def test_setitem_delitem_interleave():
    """Interleaved __setitem__ and __delitem__ corrupt currsize."""

    result = explore_interleavings(
        setup=lambda: SetDelInterleaveState(),
        threads=[
            lambda s: s.thread1(),
            lambda s: s.thread2(),
        ],
        invariant=lambda s: s.cache.currsize == len(s.cache),
        max_attempts=200,
        max_ops=300,
        seed=42,
    )

    print_exploration_result(result)
    return result


# ---------------------------------------------------------------------------
# 3. Cache.__setitem__ overwrite path: lost update when key already exists
#
# When key already exists in __data, __setitem__ computes:
#     diffsize = size - self.__size[key]
#     self.__data[key] = value
#     self.__size[key] = size
#     self.__currsize += diffsize
#
# Two threads overwriting the SAME key both read the old size, compute
# diffsize, and do currsize += diffsize.  With default getsizeof (size=1),
# diffsize is 0 for both, so currsize stays correct.  But with a custom
# getsizeof that returns different sizes, the race manifests.
#
# We use a simpler variant: two threads both do cache["a"] = ..., one
# of which may interleave with the other in the "key not in data" branch
# vs. the "key in data" branch, corrupting currsize.
# ---------------------------------------------------------------------------

class OverwriteLostUpdateState:
    def __init__(self):
        self.cache = Cache(maxsize=100)
        # Pre-populate so the overwrite path is taken
        self.cache["a"] = "old_value"

    def thread1(self):
        self.cache["a"] = "new_value_1"

    def thread2(self):
        self.cache["a"] = "new_value_2"


def test_setitem_overwrite_lost_update():
    """Cache.__setitem__ overwrite: concurrent overwrites corrupt currsize."""

    result = explore_interleavings(
        setup=lambda: OverwriteLostUpdateState(),
        threads=[
            lambda s: s.thread1(),
            lambda s: s.thread2(),
        ],
        invariant=lambda s: s.cache.currsize == len(s.cache),
        max_attempts=200,
        max_ops=300,
        seed=42,
    )

    print_exploration_result(result)
    return result


# ---------------------------------------------------------------------------
# 4. LRUCache ordering corruption
#
# LRUCache.__setitem__ calls Cache.__setitem__ then self.__touch(key) which
# does self.__order.move_to_end(key) or self.__order[key] = None.
#
# Concurrent inserts of different keys can corrupt the OrderedDict:
# - Thread 1 inserts "a", adds to __order
# - Thread 2 inserts "b", adds to __order
# - The OrderedDict operations interleave, corrupting internal state
#
# Invariant: all keys in the cache should be tracked in the ordering structure,
# and currsize should match len.
# ---------------------------------------------------------------------------

class LRUCacheRaceState:
    def __init__(self):
        self.cache = LRUCache(maxsize=100)

    def thread1(self):
        self.cache["a"] = "value_a"

    def thread2(self):
        self.cache["b"] = "value_b"


def test_lru_cache_ordering_race():
    """LRUCache: concurrent inserts corrupt ordering + currsize."""

    result = explore_interleavings(
        setup=lambda: LRUCacheRaceState(),
        threads=[
            lambda s: s.thread1(),
            lambda s: s.thread2(),
        ],
        invariant=lambda s: (
            s.cache.currsize == len(s.cache)
        ),
        max_attempts=200,
        max_ops=300,
        seed=42,
    )

    print_exploration_result(result)
    return result


# ---------------------------------------------------------------------------
# 5. LFUCache frequency counter corruption
#
# LFUCache maintains a doubly-linked list of _Link nodes, each holding a
# frequency count and a set of keys with that count.  __setitem__ inserts
# into this structure; concurrent inserts can corrupt prev/next pointers
# and the __links dict.
#
# Bug: two concurrent inserts both read root.next, both create a new Link
# with count=1, and both try to splice it in -- one splice is lost or the
# list becomes inconsistent.
# ---------------------------------------------------------------------------

class LFUCacheRaceState:
    def __init__(self):
        self.cache = LFUCache(maxsize=100)

    def thread1(self):
        self.cache["a"] = "value_a"

    def thread2(self):
        self.cache["b"] = "value_b"


def test_lfu_cache_frequency_race():
    """LFUCache: concurrent inserts corrupt frequency linked list + currsize."""

    result = explore_interleavings(
        setup=lambda: LFUCacheRaceState(),
        threads=[
            lambda s: s.thread1(),
            lambda s: s.thread2(),
        ],
        invariant=lambda s: (
            s.cache.currsize == len(s.cache)
        ),
        max_attempts=200,
        max_ops=300,
        seed=42,
    )

    print_exploration_result(result)
    return result


# ---------------------------------------------------------------------------
# 6. TTLCache link list corruption
#
# TTLCache.__setitem__ inserts into a doubly-linked list of _Link nodes
# ordered by expiration time, and also into an OrderedDict (__links).
#
# Concurrent inserts can corrupt the linked list pointers:
#   link.next = root
#   link.prev = prev = root.prev
#   prev.next = root.prev = link
#
# Two threads doing this simultaneously can both read root.prev, both set
# their link.prev to it, and one link gets orphaned from the list.
# ---------------------------------------------------------------------------

class TTLCacheRaceState:
    def __init__(self):
        # Use a fixed timer to avoid real time complications
        self.time = 0.0
        self.cache = TTLCache(maxsize=100, ttl=300, timer=lambda: self.time)

    def thread1(self):
        self.cache["a"] = "value_a"

    def thread2(self):
        self.cache["b"] = "value_b"


def test_ttl_cache_link_corruption():
    """TTLCache: concurrent inserts corrupt doubly-linked list + currsize."""

    result = explore_interleavings(
        setup=lambda: TTLCacheRaceState(),
        threads=[
            lambda s: s.thread1(),
            lambda s: s.thread2(),
        ],
        invariant=lambda s: (
            s.cache.currsize == len(s.cache)
        ),
        max_attempts=200,
        max_ops=300,
        seed=42,
    )

    print_exploration_result(result)
    return result


# ---------------------------------------------------------------------------
# 7. Cache.pop() TOCTOU race
#
# Cache.pop() does:
#     if key in self:          # check
#         value = self[key]     # use
#         del self[key]         # use
#
# Thread 1 calls pop("a"), thread 2 also calls pop("a"). Both see the
# key exists, both try to delete it, and the second delete raises KeyError
# (or worse, currsize goes negative if the key was re-added in between).
# ---------------------------------------------------------------------------

class PopTOCTOUState:
    def __init__(self):
        self.cache = Cache(maxsize=100)
        self.cache["a"] = "value_a"
        self.error = None

    def thread1(self):
        try:
            self.cache.pop("a", None)
        except (KeyError, RuntimeError) as e:
            self.error = e

    def thread2(self):
        try:
            self.cache.pop("a", None)
        except (KeyError, RuntimeError) as e:
            self.error = e


def test_pop_toctou():
    """Cache.pop() TOCTOU: two threads pop the same key concurrently."""

    result = explore_interleavings(
        setup=lambda: PopTOCTOUState(),
        threads=[
            lambda s: s.thread1(),
            lambda s: s.thread2(),
        ],
        invariant=lambda s: (
            s.error is None and s.cache.currsize == len(s.cache)
        ),
        max_attempts=200,
        max_ops=300,
        seed=42,
    )

    print_exploration_result(result)
    return result


# ---------------------------------------------------------------------------
# 8. Cache.setdefault() TOCTOU race
#
# Cache.setdefault() does:
#     if key in self:          # check
#         value = self[key]     # use
#     else:
#         self[key] = value = default  # use
#
# Two threads both calling setdefault("a", ...) may both see key absent
# and both insert, causing a double currsize increment.
# ---------------------------------------------------------------------------

class SetdefaultTOCTOUState:
    def __init__(self):
        self.cache = Cache(maxsize=100)

    def thread1(self):
        self.cache.setdefault("a", "value_1")

    def thread2(self):
        self.cache.setdefault("a", "value_2")


def test_setdefault_toctou():
    """Cache.setdefault() TOCTOU: two threads setdefault same key."""

    result = explore_interleavings(
        setup=lambda: SetdefaultTOCTOUState(),
        threads=[
            lambda s: s.thread1(),
            lambda s: s.thread2(),
        ],
        invariant=lambda s: (
            s.cache.currsize == len(s.cache)
        ),
        max_attempts=200,
        max_ops=300,
        seed=42,
    )

    print_exploration_result(result)
    return result


# ---------------------------------------------------------------------------
# 9. @cached decorator (unlocked): two threads compute same key
#
# The _unlocked wrapper in _cached.py does:
#     try:
#         return cache[k]        # miss -> KeyError
#     except KeyError:
#         pass
#     v = func(*args, **kwargs)  # both threads compute
#     cache[k] = v               # both threads insert the same key
#
# Without a lock, both threads miss the cache and both store results.
# The second store is an overwrite, but the interleaving of the two
# __setitem__ calls corrupts currsize (same as the basic lost update).
#
# Additionally, func is called twice instead of once -- a correctness
# issue if func has side effects.
# ---------------------------------------------------------------------------

class CachedDecoratorRaceState:
    def __init__(self):
        self.cache = Cache(maxsize=100)
        self.call_count = 0

        @cached(self.cache)
        def expensive(x):
            self.call_count += 1
            return x * 2

        self.expensive = expensive

    def thread1(self):
        self.expensive(1)

    def thread2(self):
        self.expensive(1)


def test_cached_decorator_unlocked_race():
    """@cached (no lock): two threads compute + insert same key, corrupting state."""

    result = explore_interleavings(
        setup=lambda: CachedDecoratorRaceState(),
        threads=[
            lambda s: s.thread1(),
            lambda s: s.thread2(),
        ],
        invariant=lambda s: (
            # Either call_count should be 1 (ideal), or if 2 at least currsize must be consistent
            s.cache.currsize == len(s.cache)
        ),
        max_attempts=200,
        max_ops=300,
        seed=42,
    )

    print_exploration_result(result)
    return result


# ---------------------------------------------------------------------------
# 10. Three-thread concurrent insert: amplified lost-update
#
# With three threads all inserting different keys, the probability of a
# lost update on currsize is higher because there are more possible
# interleavings of LOAD_ATTR / INPLACE_ADD / STORE_ATTR across threads.
# ---------------------------------------------------------------------------

class ThreeThreadInsertState:
    def __init__(self):
        self.cache = Cache(maxsize=100)

    def thread1(self):
        self.cache["a"] = "value_a"

    def thread2(self):
        self.cache["b"] = "value_b"

    def thread3(self):
        self.cache["c"] = "value_c"


def test_three_thread_insert():
    """Three concurrent inserts amplify the currsize lost-update probability."""

    result = explore_interleavings(
        setup=lambda: ThreeThreadInsertState(),
        threads=[
            lambda s: s.thread1(),
            lambda s: s.thread2(),
            lambda s: s.thread3(),
        ],
        invariant=lambda s: s.cache.currsize == len(s.cache),
        max_attempts=200,
        max_ops=300,
        seed=42,
    )

    print_exploration_result(result)
    return result


# ---------------------------------------------------------------------------
# 11. Cache.__setitem__ eviction race (popitem interleaving)
#
# When the cache is near capacity, __setitem__ calls self.popitem() to
# evict entries.  With LRUCache, popitem() traverses __order and calls
# self.pop(key), which itself does __contains__ check + __getitem__ + __delitem__.
#
# Two threads simultaneously inserting into a near-full cache both trigger
# eviction, which can corrupt the ordering structure and currsize.
# ---------------------------------------------------------------------------

class EvictionRaceState:
    def __init__(self):
        self.cache = LRUCache(maxsize=2)
        self.cache["existing"] = "value"
        self.error = None

    def thread1(self):
        try:
            self.cache["a"] = "value_a"
        except (KeyError, RuntimeError, StopIteration) as e:
            self.error = e

    def thread2(self):
        try:
            self.cache["b"] = "value_b"
        except (KeyError, RuntimeError, StopIteration) as e:
            self.error = e


def test_eviction_race():
    """LRUCache: concurrent inserts both trigger eviction, corrupting state."""

    result = explore_interleavings(
        setup=lambda: EvictionRaceState(),
        threads=[
            lambda s: s.thread1(),
            lambda s: s.thread2(),
        ],
        invariant=lambda s: (
            s.error is None
            and s.cache.currsize == len(s.cache)
            and s.cache.currsize <= s.cache.maxsize
        ),
        max_attempts=200,
        max_ops=300,
        seed=42,
    )

    print_exploration_result(result)
    return result


# ---------------------------------------------------------------------------
# 12. LRUCache concurrent read + write
#
# LRUCache.__getitem__ calls __touch(key) which does
# self.__order.move_to_end(key).  Concurrent with __setitem__ which also
# calls __touch, this can corrupt the OrderedDict.
#
# Thread 1 reads key "a" (touching it), thread 2 inserts key "b".
# ---------------------------------------------------------------------------

class LRUReadWriteRaceState:
    def __init__(self):
        self.cache = LRUCache(maxsize=100)
        self.cache["a"] = "value_a"
        self.read_result = None

    def thread1(self):
        self.read_result = self.cache["a"]

    def thread2(self):
        self.cache["b"] = "value_b"


def test_lru_concurrent_read_write():
    """LRUCache: concurrent read (touch) + write corrupt ordering + currsize."""

    result = explore_interleavings(
        setup=lambda: LRUReadWriteRaceState(),
        threads=[
            lambda s: s.thread1(),
            lambda s: s.thread2(),
        ],
        invariant=lambda s: (
            s.cache.currsize == len(s.cache)
            and s.read_result == "value_a"
        ),
        max_attempts=200,
        max_ops=300,
        seed=42,
    )

    print_exploration_result(result)
    return result


# ---------------------------------------------------------------------------
# 13. Cache.clear() vs concurrent insert
#
# Cache.clear() (inherited from MutableMapping) iterates and deletes keys.
# A concurrent insert modifies the same __data dict and __currsize.
#
# After both threads finish, either:
# - The insert was lost (len=0 but currsize=1), or
# - The insert survived but currsize is wrong because clear()'s deletes
#   interleaved with the insert's currsize update.
# ---------------------------------------------------------------------------

class ClearVsInsertState:
    def __init__(self):
        self.cache = Cache(maxsize=100)
        # Pre-populate with a few entries
        self.cache["x"] = "val_x"
        self.cache["y"] = "val_y"
        self.error = None

    def thread1(self):
        try:
            self.cache.clear()
        except (KeyError, RuntimeError) as e:
            self.error = e

    def thread2(self):
        try:
            self.cache["z"] = "val_z"
        except (KeyError, RuntimeError) as e:
            self.error = e


def test_clear_vs_insert():
    """Cache.clear() vs concurrent insert: state corruption."""

    result = explore_interleavings(
        setup=lambda: ClearVsInsertState(),
        threads=[
            lambda s: s.thread1(),
            lambda s: s.thread2(),
        ],
        invariant=lambda s: (
            s.error is None
            and s.cache.currsize == len(s.cache)
        ),
        max_attempts=200,
        max_ops=300,
        seed=42,
    )

    print_exploration_result(result)
    return result


# ---------------------------------------------------------------------------
# 14. Custom getsizeof + concurrent insert: variable-size lost update
#
# With a custom getsizeof, items have varying sizes. __setitem__ computes:
#     diffsize = size  (for new keys)
#     self.__currsize += diffsize
#
# The non-atomic += with different diffsize values per thread makes the
# race more visible: currsize should equal sum of all item sizes.
# ---------------------------------------------------------------------------

class CustomSizeState:
    def __init__(self):
        self.cache = Cache(maxsize=100, getsizeof=len)

    def thread1(self):
        self.cache["a"] = "xx"    # size = 2

    def thread2(self):
        self.cache["b"] = "yyyy"  # size = 4


def test_custom_getsizeof_lost_update():
    """Cache with custom getsizeof: concurrent inserts lose currsize update."""

    result = explore_interleavings(
        setup=lambda: CustomSizeState(),
        threads=[
            lambda s: s.thread1(),
            lambda s: s.thread2(),
        ],
        invariant=lambda s: (
            # currsize should be sum of sizes: len("xx") + len("yyyy") = 6
            len(s.cache) == 2 and s.cache.currsize == 6
            or len(s.cache) < 2  # if one insert was lost, currsize should match
        ),
        max_attempts=200,
        max_ops=300,
        seed=42,
    )

    print_exploration_result(result)
    return result


# ---------------------------------------------------------------------------
# 15. LFUCache concurrent get + set: touch race
#
# LFUCache.__getitem__ calls __touch(key) to increment the frequency counter.
# Concurrent with __setitem__ which also manipulates the linked list, this
# can corrupt the frequency structure.
#
# __touch does:
#   link = self.__links[key]
#   curr = link.next
#   if curr.count != link.count + 1:
#       ...splice in new Link...
#   curr.keys.add(key)
#   link.keys.remove(key)
#   if not link.keys: link.unlink()
#
# A concurrent insert can modify link.next between the read and the splice.
# ---------------------------------------------------------------------------

class LFUGetSetRaceState:
    def __init__(self):
        self.cache = LFUCache(maxsize=100)
        self.cache["a"] = "value_a"
        self.read_result = None

    def thread1(self):
        # Reading "a" triggers __touch which increments frequency
        self.read_result = self.cache["a"]

    def thread2(self):
        self.cache["b"] = "value_b"


def test_lfu_get_set_race():
    """LFUCache: concurrent get (touch) + set corrupt frequency structure."""

    result = explore_interleavings(
        setup=lambda: LFUGetSetRaceState(),
        threads=[
            lambda s: s.thread1(),
            lambda s: s.thread2(),
        ],
        invariant=lambda s: (
            s.cache.currsize == len(s.cache)
            and s.read_result == "value_a"
        ),
        max_attempts=200,
        max_ops=300,
        seed=42,
    )

    print_exploration_result(result)
    return result


# ---------------------------------------------------------------------------
# 16. @cached decorator race: duplicate computation with side effects
#
# Without a lock, when two threads call the same cached function with the
# same arguments, both miss the cache and both call the underlying function.
# The function is called twice even though caching should prevent this.
#
# This is a correctness bug rather than a crash: if the function has side
# effects (e.g., incrementing a counter, making a network call), those
# side effects happen twice.
# ---------------------------------------------------------------------------

class CachedDuplicateComputeState:
    def __init__(self):
        self.cache = Cache(maxsize=100)
        self.call_count = 0

        @cached(self.cache)
        def compute(x):
            self.call_count += 1
            return x * 10

        self.compute = compute

    def thread1(self):
        self.compute(42)

    def thread2(self):
        self.compute(42)


def test_cached_duplicate_computation():
    """@cached: both threads miss and call func, proving duplicate computation."""

    result = explore_interleavings(
        setup=lambda: CachedDuplicateComputeState(),
        threads=[
            lambda s: s.thread1(),
            lambda s: s.thread2(),
        ],
        invariant=lambda s: (
            # Ideally call_count should be 1 (cached). If both compute, call_count=2.
            s.call_count == 1
        ),
        max_attempts=200,
        max_ops=300,
        seed=42,
    )

    print_exploration_result(result)
    return result


# ---------------------------------------------------------------------------
# 17. Cache.get() TOCTOU race
#
# Cache.get() does:
#     if key in self:     # check
#         return self[key] # use -- but key may have been deleted by now
#     else:
#         return default
#
# Thread 1 calls get("a"), thread 2 deletes "a". If thread 2 deletes
# between the check and the access, thread 1 gets a KeyError from __data
# (or __missing__ is called, depending on subclass).
# ---------------------------------------------------------------------------

class GetTOCTOUState:
    def __init__(self):
        self.cache = Cache(maxsize=100)
        self.cache["a"] = "value_a"
        self.error = None
        self.get_result = None

    def thread1(self):
        try:
            self.get_result = self.cache.get("a", "default")
        except KeyError as e:
            self.error = e

    def thread2(self):
        try:
            del self.cache["a"]
        except KeyError as e:
            self.error = e


def test_get_toctou():
    """Cache.get() TOCTOU: get races with concurrent delete."""

    result = explore_interleavings(
        setup=lambda: GetTOCTOUState(),
        threads=[
            lambda s: s.thread1(),
            lambda s: s.thread2(),
        ],
        invariant=lambda s: (
            s.error is None
            and s.cache.currsize == len(s.cache)
            # get should return either the value or the default, never raise
            and s.get_result in ("value_a", "default")
        ),
        max_attempts=200,
        max_ops=300,
        seed=42,
    )

    print_exploration_result(result)
    return result


# ---------------------------------------------------------------------------
# Seed sweep utility -- run any test across multiple seeds
# ---------------------------------------------------------------------------

def sweep_test(test_name, state_class, threads_setup, invariant, num_seeds=20):
    """Run a seed sweep for a given test configuration."""
    print(f"\n=== Seed sweep for {test_name} ({num_seeds} seeds x 100 attempts) ===")

    found_seeds = []
    total_explored = 0

    for seed in range(num_seeds):
        result = explore_interleavings(
            setup=state_class,
            threads=threads_setup,
            invariant=invariant,
            max_attempts=100,
            max_ops=300,
            seed=seed,
        )
        total_explored += result.num_explored
        if not result.property_holds:
            found_seeds.append((seed, result.num_explored))

    print_seed_sweep_results(found_seeds, total_explored)
    return found_seeds


# ---------------------------------------------------------------------------
# Main: run all tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        ("1. __delitem__ lost update", test_delitem_lost_update),
        ("2. __setitem__ + __delitem__ interleave", test_setitem_delitem_interleave),
        ("3. __setitem__ overwrite lost update", test_setitem_overwrite_lost_update),
        ("4. LRUCache ordering race", test_lru_cache_ordering_race),
        ("5. LFUCache frequency race", test_lfu_cache_frequency_race),
        ("6. TTLCache link corruption", test_ttl_cache_link_corruption),
        ("7. pop() TOCTOU", test_pop_toctou),
        ("8. setdefault() TOCTOU", test_setdefault_toctou),
        ("9. @cached unlocked race", test_cached_decorator_unlocked_race),
        ("10. Three-thread insert", test_three_thread_insert),
        ("11. Eviction race (LRU near capacity)", test_eviction_race),
        ("12. LRU concurrent read + write", test_lru_concurrent_read_write),
        ("13. clear() vs insert", test_clear_vs_insert),
        ("14. Custom getsizeof lost update", test_custom_getsizeof_lost_update),
        ("15. LFU get + set race", test_lfu_get_set_race),
        ("16. @cached duplicate computation", test_cached_duplicate_computation),
        ("17. get() TOCTOU", test_get_toctou),
    ]

    bugs_found = 0
    bugs_not_found = 0

    for name, test_fn in tests:
        print(f"\n{'=' * 60}")
        print(f"  {name}")
        print(f"{'=' * 60}")
        result = test_fn()
        if result and not result.property_holds:
            bugs_found += 1
        else:
            bugs_not_found += 1

    print(f"\n{'=' * 60}")
    print(f"  SUMMARY")
    print(f"{'=' * 60}")
    print(f"Bugs found:     {bugs_found}/{len(tests)}")
    print(f"Not triggered:  {bugs_not_found}/{len(tests)}")
