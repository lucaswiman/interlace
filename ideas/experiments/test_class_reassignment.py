"""Verify __class__ reassignment for resource instrumentation.

Goal: confirm that swapping an object's __class__ to an instrumented subclass
works for common resource-like objects, preserves isinstance(), and can be
triggered from sys.settrace return events.
"""

import io
import socket
import sqlite3
import sys
import threading
import types

accesses: list[tuple[str, str, str]] = []


def _make_instrumented_subclass(cls):
    """Dynamically create a subclass that reports method calls."""
    if hasattr(cls, "_frontrun_instrumented"):
        return cls

    resource_methods = {"execute", "executemany", "send", "sendall", "recv", "write", "read", "flush", "commit", "close"}
    namespace = {"_frontrun_instrumented": True}

    for method_name in resource_methods:
        original = getattr(cls, method_name, None)
        if original is None:
            continue

        def make_wrapper(name, orig):
            def wrapper(self, *args, **kwargs):
                accesses.append((type(self).__bases__[0].__name__, name, threading.current_thread().name))
                return orig(self, *args, **kwargs)
            return wrapper

        namespace[method_name] = make_wrapper(method_name, original)

    return type(f"Instrumented{cls.__name__}", (cls,), namespace)


# --- Test 1: Pure-Python class ---
print("=== Test 1: __class__ swap on pure-Python object ===")


class Connection:
    def __init__(self, name):
        self.name = name
        self._data = []

    def execute(self, query):
        self._data.append(query)
        return len(self._data)

    def commit(self):
        pass


conn = Connection("test-db")
original_type = type(conn)

# Swap
conn.__class__ = _make_instrumented_subclass(Connection)

assert isinstance(conn, Connection), "isinstance check must still work"
assert conn.name == "test-db", "attributes must be preserved"
result = conn.execute("SELECT 1")
assert result == 1
conn.commit()

print(f"  isinstance(conn, Connection): {isinstance(conn, Connection)}")
print(f"  Attributes preserved: name={conn.name}, _data={conn._data}")
print(f"  Accesses detected: {accesses}")
print(f"  Original type preserved in bases: {type(conn).__bases__[0] is Connection}")


# --- Test 2: socket.socket (C extension type) ---
print("\n=== Test 2: __class__ swap on socket.socket (C extension) ===")

accesses.clear()
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    sock.__class__ = _make_instrumented_subclass(socket.socket)
    print(f"  SUCCESS: socket.__class__ swap worked")
    print(f"  isinstance(sock, socket.socket): {isinstance(sock, socket.socket)}")
except TypeError as e:
    print(f"  EXPECTED FAILURE: {e}")
    print(f"  C extension types with incompatible layout can't be swapped")
finally:
    sock.close()


# --- Test 3: sqlite3.Cursor (C extension type) ---
print("\n=== Test 3: __class__ swap on sqlite3.Cursor ===")

accesses.clear()
db = sqlite3.connect(":memory:")
cursor = db.cursor()
try:
    cursor.__class__ = _make_instrumented_subclass(sqlite3.Cursor)
    print(f"  SUCCESS: sqlite3.Cursor.__class__ swap worked")
    cursor.execute("SELECT 1")
    print(f"  Accesses detected: {accesses}")
except TypeError as e:
    print(f"  EXPECTED FAILURE: {e}")
finally:
    cursor.close()
    db.close()


# --- Test 4: io.StringIO (Python wrapper around C) ---
print("\n=== Test 4: __class__ swap on io.StringIO ===")

accesses.clear()
sio = io.StringIO()
try:
    sio.__class__ = _make_instrumented_subclass(io.StringIO)
    sio.write("hello")
    content = sio.getvalue()
    print(f"  SUCCESS: io.StringIO.__class__ swap worked")
    print(f"  Content after write: {content!r}")
    print(f"  Accesses detected: {accesses}")
except TypeError as e:
    print(f"  FAILURE: {e}")


# --- Test 5: Auto-swap from sys.settrace return event ---
print("\n=== Test 5: Auto-swap from sys.settrace 'return' event ===")

accesses.clear()
_already_instrumented = set()


def _is_resource_like(obj):
    """Duck-type check for resource-like objects."""
    return (
        hasattr(obj, "execute")
        or hasattr(obj, "send")
        or hasattr(obj, "write")
    ) and not isinstance(obj, type) and not isinstance(obj, types.ModuleType)


def auto_swap_trace(frame, event, arg):
    if event == "return" and arg is not None:
        if id(arg) not in _already_instrumented and _is_resource_like(arg):
            try:
                arg.__class__ = _make_instrumented_subclass(type(arg))
                _already_instrumented.add(id(arg))
            except TypeError:
                pass  # C extension type, can't swap
    return auto_swap_trace


def create_connection():
    return Connection("auto-detected")


def use_connection():
    conn = create_connection()
    conn.execute("INSERT INTO foo VALUES (1)")
    conn.commit()
    return conn


sys.settrace(auto_swap_trace)
result_conn = use_connection()
sys.settrace(None)

print(f"  Auto-instrumented type: {type(result_conn).__name__}")
print(f"  Accesses detected: {accesses}")
print(f"  Connection was auto-swapped: {'Instrumented' in type(result_conn).__name__}")


# --- Test 6: Thread safety of __class__ swap ---
print("\n=== Test 6: Concurrent __class__ swap safety ===")

accesses.clear()
errors: list[str] = []


def thread_work(i):
    try:
        c = Connection(f"thread-{i}")
        c.__class__ = _make_instrumented_subclass(Connection)
        c.execute(f"query-{i}")
        assert isinstance(c, Connection)
        assert c.name == f"thread-{i}"
    except Exception as e:
        errors.append(f"thread-{i}: {e}")


threads = [threading.Thread(target=thread_work, args=(i,)) for i in range(10)]
for t in threads:
    t.start()
for t in threads:
    t.join()

print(f"  Threads completed: 10")
print(f"  Errors: {len(errors)} ({errors[:3] if errors else 'none'})")
print(f"  Accesses detected: {len(accesses)}")

print("\n=== All tests completed ===")
