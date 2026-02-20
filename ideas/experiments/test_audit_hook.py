"""Verify sys.addaudithook fires for socket and file I/O operations.

Goal: confirm audit hooks provide the event coverage claimed in the design doc,
and that they fire from C extensions too.
"""

import socket
import sys
import tempfile
import threading

audit_events: list[tuple[str, tuple]] = []
_capture_active = False


def audit_hook(event, args):
    if not _capture_active:
        return
    # Only capture I/O related events
    if event.startswith("socket.") or event == "open" or event.startswith("sqlite3"):
        audit_events.append((event, args))


# Install once (cannot be removed)
sys.addaudithook(audit_hook)


# --- Test 1: Socket connect/send/recv ---
print("=== Test 1: Socket audit events ===")

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(("127.0.0.1", 0))
port = server.getsockname()[1]
server.listen(1)

audit_events.clear()
_capture_active = True

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(("127.0.0.1", port))
conn, _ = server.accept()
client.sendall(b"hello from client")
data = conn.recv(1024)
conn.sendall(b"hello from server")
data2 = client.recv(1024)
client.close()
conn.close()

_capture_active = False

event_types = set(e for e, _ in audit_events)
print(f"  Total audit events: {len(audit_events)}")
print(f"  Event types: {sorted(event_types)}")
print(f"  socket.connect present: {'socket.connect' in event_types}")
print(f"  socket.sendmsg present: {'socket.sendmsg' in event_types}")

# Show connect events with address info
connect_events = [(e, a) for e, a in audit_events if e == "socket.connect"]
for e, a in connect_events[:3]:
    print(f"    {e}: {a[1] if len(a) > 1 else a}")

server.close()

# --- Test 2: File I/O audit events ---
print("\n=== Test 2: File I/O audit events ===")

audit_events.clear()
_capture_active = True

with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
    f.write("test data")
    fname = f.name

with open(fname) as f:
    _ = f.read()

_capture_active = False

open_events = [(e, a) for e, a in audit_events if e == "open"]
print(f"  Total audit events: {len(audit_events)}")
print(f"  'open' events: {len(open_events)}")
for e, a in open_events[:5]:
    # args is (path, mode, flags)
    print(f"    open: path={a[0]}, mode={a[1]}")


# --- Test 3: SQLite3 audit events ---
print("\n=== Test 3: SQLite3 audit events ===")

import sqlite3

audit_events.clear()
_capture_active = True

db = sqlite3.connect(":memory:")
cursor = db.cursor()
cursor.execute("CREATE TABLE test (id INTEGER, name TEXT)")
cursor.execute("INSERT INTO test VALUES (1, 'hello')")
cursor.execute("SELECT * FROM test")
_ = cursor.fetchall()
db.close()

_capture_active = False

event_types = set(e for e, _ in audit_events)
print(f"  Total audit events: {len(audit_events)}")
print(f"  Event types: {sorted(event_types)}")

sqlite_events = [(e, a) for e, a in audit_events if e.startswith("sqlite3")]
for e, a in sqlite_events[:5]:
    print(f"    {e}: {a}")


# --- Test 4: Threaded audit events ---
print("\n=== Test 4: Audit events from threads ===")

audit_events.clear()
_capture_active = True

thread_events: dict[str, int] = {}


def worker(name):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.close()


threads = [threading.Thread(target=worker, args=(f"t{i}",), name=f"t{i}") for i in range(3)]
for t in threads:
    t.start()
for t in threads:
    t.join()

_capture_active = False

print(f"  Audit events from threaded socket ops: {len(audit_events)}")
print(f"  Note: audit hooks fire globally, no per-thread filtering built in")
print(f"  Would need threading.current_thread() in hook to attribute to thread")

# --- Test 5: Can we get the socket object from audit args? ---
print("\n=== Test 5: Socket object identity in audit args ===")

audit_events.clear()
_capture_active = True

test_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
test_sock.bind(("127.0.0.1", 0))

_capture_active = False

bind_events = [(e, a) for e, a in audit_events if e == "socket.bind"]
if bind_events:
    _, args = bind_events[0]
    audit_sock = args[0]
    print(f"  Socket object in audit args: {type(audit_sock).__name__}")
    print(f"  Same object as test_sock: {audit_sock is test_sock}")
    print(f"  Can get id for gc.get_referrers: id={id(audit_sock)}")
else:
    print(f"  No socket.bind events captured")
    print(f"  Events captured: {[(e, type(a)) for e, a in audit_events[:5]]}")

test_sock.close()

print("\n=== All tests completed ===")
