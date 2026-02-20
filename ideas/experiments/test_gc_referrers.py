"""Verify gc.get_referrers() can walk from a low-level socket up to its owning object.

Goal: confirm that the referrer chain socket → Connection → Engine is
traversable, and that the mapping can be cached after a single walk.
"""

import gc
import socket
import types


# --- Simulate a typical DB driver object hierarchy ---
class ConnectionPool:
    def __init__(self):
        self._connections: list["DBConnection"] = []

    def get_connection(self):
        conn = DBConnection(self)
        self._connections.append(conn)
        return conn


class Engine:
    def __init__(self, name):
        self.name = name
        self._pool = ConnectionPool()

    def connect(self):
        return self._pool.get_connection()


class DBConnection:
    def __init__(self, pool):
        self._pool = pool
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def close(self):
        self._sock.close()


def find_owner(obj, max_depth=10, stop_types=None):
    """Walk gc.get_referrers upward to find the high-level resource owner.

    Returns (owner, chain) where chain is the list of non-dict objects traversed.

    Key insight: Python stores instance attributes in __dict__ (a plain dict).
    gc.get_referrers(socket_obj) returns the DBConnection.__dict__, not
    DBConnection itself. We must walk *through* dicts to their owning objects.
    """
    if stop_types is None:
        stop_types = (types.ModuleType, type)

    visited = {id(obj)}
    current = obj
    chain = []

    for _ in range(max_depth):
        referrers = gc.get_referrers(current)
        candidates = []
        for r in referrers:
            if isinstance(r, types.FrameType):
                continue
            if id(r) in visited:
                continue
            if r is chain:
                continue
            candidates.append(r)

        if not candidates:
            break

        # If the referrer is a dict, it's likely an instance __dict__.
        # Walk through it to find the owning object.
        best = None
        for c in candidates:
            if isinstance(c, dict):
                # This dict refers to `current`. Find who owns this dict.
                visited.add(id(c))
                dict_referrers = gc.get_referrers(c)
                for dr in dict_referrers:
                    if isinstance(dr, (types.FrameType, dict, list, tuple)):
                        continue
                    if id(dr) in visited:
                        continue
                    # Check this object's __dict__ is the dict we found
                    if getattr(dr, "__dict__", None) is c:
                        best = dr
                        break
                if best is not None:
                    break
            elif isinstance(c, (list, tuple)):
                # Walk through containers too
                visited.add(id(c))
                container_referrers = gc.get_referrers(c)
                for cr in container_referrers:
                    if isinstance(cr, (types.FrameType, dict, list, tuple)):
                        continue
                    if id(cr) in visited:
                        continue
                    best = cr
                    break
                if best is not None:
                    break
            else:
                best = c
                break

        if best is None:
            break

        visited.add(id(best))
        chain.append(best)
        current = best

        # Stop if we reached a "root-like" object
        if isinstance(current, stop_types):
            break

    return current, chain


# --- Test 1: Walk from socket to Engine ---
print("=== Test 1: Walk from socket to Engine ===")

engine = Engine("postgres://localhost:5432/mydb")
conn = engine.connect()
target_sock = conn._sock

owner, chain = find_owner(target_sock)

chain_types = [type(x).__name__ for x in chain]
print(f"  Target: {type(target_sock).__name__}")
print(f"  Chain: {' → '.join(chain_types)}")
print(f"  Owner: {type(owner).__name__}")
print(f"  Found Engine: {isinstance(owner, Engine) or any(isinstance(x, Engine) for x in chain)}")

# Check that Engine is in the chain
engine_found = any(isinstance(x, Engine) for x in chain) or isinstance(owner, Engine)
print(f"  Engine in chain: {engine_found}")
if engine_found:
    eng = next((x for x in chain if isinstance(x, Engine)), owner)
    print(f"  Engine name: {eng.name}")

conn.close()

# --- Test 2: Multiple connections through same pool ---
print("\n=== Test 2: Multiple connections, same pool ===")

engine2 = Engine("mysql://localhost:3306/test")
conn_a = engine2.connect()
conn_b = engine2.connect()

owner_a, chain_a = find_owner(conn_a._sock)
owner_b, chain_b = find_owner(conn_b._sock)

# Both should trace back to the same Engine
engine_a = next((x for x in chain_a if isinstance(x, Engine)), owner_a)
engine_b = next((x for x in chain_b if isinstance(x, Engine)), owner_b)

print(f"  Connection A owner: {type(engine_a).__name__} (id={id(engine_a)})")
print(f"  Connection B owner: {type(engine_b).__name__} (id={id(engine_b)})")
print(f"  Same engine: {engine_a is engine_b}")

conn_a.close()
conn_b.close()

# --- Test 3: Different engines, different identity ---
print("\n=== Test 3: Different engines = different identity ===")

engine_pg = Engine("postgres://localhost:5432/db1")
engine_my = Engine("mysql://localhost:3306/db2")
conn_pg = engine_pg.connect()
conn_my = engine_my.connect()

_, chain_pg = find_owner(conn_pg._sock)
_, chain_my = find_owner(conn_my._sock)

# Use chain search to find the Engine (not raw owner which may overshoot to module)
found_pg = next((x for x in chain_pg if isinstance(x, Engine)), None)
found_my = next((x for x in chain_my if isinstance(x, Engine)), None)

print(f"  Postgres engine: {found_pg.name if found_pg else 'NOT FOUND'} (id={id(found_pg)})")
print(f"  MySQL engine: {found_my.name if found_my else 'NOT FOUND'} (id={id(found_my)})")
print(f"  Different engines: {found_pg is not found_my}")
print(f"  Note: practical usage would stop at a 'resource root' heuristic, not walk to module")

conn_pg.close()
conn_my.close()

# --- Test 4: Caching the mapping ---
print("\n=== Test 4: Cache effectiveness ===")

import time

engine3 = Engine("redis://localhost:6379")
conn3 = engine3.connect()
sock3 = conn3._sock

# First walk (cold)
t0 = time.perf_counter_ns()
owner_cold, _ = find_owner(sock3)
cold_ns = time.perf_counter_ns() - t0

# Simulate caching: map socket id -> engine id
resource_cache: dict[int, int] = {}
resource_cache[id(sock3)] = id(owner_cold)

# Cache lookup
t0 = time.perf_counter_ns()
cached_owner_id = resource_cache.get(id(sock3))
cache_ns = time.perf_counter_ns() - t0

print(f"  Cold walk: {cold_ns / 1000:.1f} µs")
print(f"  Cache lookup: {cache_ns / 1000:.1f} µs")
print(f"  Speedup: {cold_ns / max(cache_ns, 1):.0f}x")

conn3.close()

# --- Test 5: Cost of gc.get_referrers on realistic object count ---
print("\n=== Test 5: gc.get_referrers cost with many objects ===")

# Create a bunch of objects to make gc.get_objects() realistic
garbage = [{"key": i, "data": list(range(100))} for i in range(10000)]

engine4 = Engine("test://cost-check")
conn4 = engine4.connect()

t0 = time.perf_counter_ns()
owner, chain = find_owner(conn4._sock)
walk_ns = time.perf_counter_ns() - t0

print(f"  Objects in gc: {len(gc.get_objects())}")
print(f"  Walk time: {walk_ns / 1000:.1f} µs")
print(f"  Chain depth: {len(chain)}")
print(f"  Acceptable for one-shot: {walk_ns < 50_000_000}")  # < 50ms

del garbage
conn4.close()

print("\n=== All tests completed ===")
