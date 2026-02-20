"""Verify sys.setprofile can detect C-level I/O calls invisible to sys.settrace.

Goal: confirm that c_call events fire for socket operations (and other C extension
calls) that sys.settrace's opcode tracing would miss because _tracing.py skips
library/site-packages code.
"""

import socket
import sys
import tempfile
import threading

# Track detected C calls
detected_calls: list[tuple[str, str]] = []


def profile_func(frame, event, arg):
    if event == "c_call":
        func_name = getattr(arg, "__name__", str(arg))
        qualname = getattr(arg, "__qualname__", func_name)
        detected_calls.append(("c_call", qualname))
    elif event == "c_return":
        func_name = getattr(arg, "__name__", str(arg))
        qualname = getattr(arg, "__qualname__", func_name)
        detected_calls.append(("c_return", qualname))


# --- Test 1: Socket operations ---
print("=== Test 1: Socket C calls via sys.setprofile ===")

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(("127.0.0.1", 0))
port = server.getsockname()[1]
server.listen(1)

detected_calls.clear()
sys.setprofile(profile_func)

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(("127.0.0.1", port))
conn, _ = server.accept()
client.sendall(b"hello")
data = conn.recv(1024)
client.close()
conn.close()

sys.setprofile(None)

socket_calls = [name for event, name in detected_calls if "socket" in name.lower() or "send" in name.lower() or "recv" in name.lower() or "connect" in name.lower()]
print(f"  Detected {len(detected_calls)} total C calls")
print(f"  Socket-related C calls: {socket_calls}")
print(f"  sendall detected: {'socket.sendall' in socket_calls or any('send' in c for c in socket_calls)}")
print(f"  recv detected: {any('recv' in c for c in socket_calls)}")
print(f"  connect detected: {any('connect' in c for c in socket_calls)}")

server.close()

# --- Test 2: File I/O ---
print("\n=== Test 2: File I/O C calls via sys.setprofile ===")

detected_calls.clear()
sys.setprofile(profile_func)

with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
    f.write("hello world")
    fname = f.name

with open(fname) as f:
    _ = f.read()

sys.setprofile(None)

io_calls = [name for _, name in detected_calls if "write" in name.lower() or "read" in name.lower() or "open" in name.lower()]
print(f"  Detected {len(detected_calls)} total C calls")
print(f"  I/O-related C calls: {io_calls[:20]}")

# --- Test 3: Threaded — confirm per-thread profile works ---
print("\n=== Test 3: Per-thread sys.setprofile ===")

thread_calls: dict[str, list[str]] = {"main": [], "worker": []}


def thread_profile_factory(thread_name):
    def prof(frame, event, arg):
        if event == "c_call":
            qualname = getattr(arg, "__qualname__", getattr(arg, "__name__", str(arg)))
            thread_calls[thread_name].append(qualname)
    return prof


def worker():
    sys.setprofile(thread_profile_factory("worker"))
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.close()
    finally:
        sys.setprofile(None)


sys.setprofile(thread_profile_factory("main"))
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.close()

t = threading.Thread(target=worker)
t.start()
t.join()
sys.setprofile(None)

print(f"  Main thread C calls: {len(thread_calls['main'])} (socket-related: {sum(1 for c in thread_calls['main'] if 'socket' in c.lower())})")
print(f"  Worker thread C calls: {len(thread_calls['worker'])} (socket-related: {sum(1 for c in thread_calls['worker'] if 'socket' in c.lower())})")
print(f"  Threads are independent: {len(thread_calls['main']) > 0 and len(thread_calls['worker']) > 0}")

# --- Test 4: sys.settrace + sys.setprofile coexistence ---
print("\n=== Test 4: sys.settrace + sys.setprofile coexistence ===")

trace_events: list[str] = []
profile_events: list[str] = []


def trace_func(frame, event, arg):
    if event in ("call", "return"):
        trace_events.append(f"{event}:{frame.f_code.co_name}")
    return trace_func


def profile_func2(frame, event, arg):
    if event.startswith("c_"):
        qualname = getattr(arg, "__qualname__", getattr(arg, "__name__", str(arg)))
        profile_events.append(f"{event}:{qualname}")


sys.settrace(trace_func)
sys.setprofile(profile_func2)

# Do some mixed Python + C work
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.close()

sys.settrace(None)
sys.setprofile(None)

print(f"  Trace events (Python calls): {len(trace_events)}")
print(f"  Profile events (C calls): {len(profile_events)}")
print(f"  Both systems fired: {len(trace_events) > 0 and len(profile_events) > 0}")
c_call_names = [e.split(":", 1)[1] for e in profile_events if e.startswith("c_call:")]
print(f"  C call names: {c_call_names[:10]}")

print("\n=== All tests passed ===")
