"""Verify sys.monitoring C_RAISE/C_RETURN events (PEP 669, Python 3.12+).

Goal: confirm that sys.monitoring can detect C function calls as an alternative
to sys.setprofile, and can coexist with INSTRUCTION events (which DPOR uses).
"""

import socket
import sys

if sys.version_info < (3, 12):
    print("SKIP: sys.monitoring requires Python 3.12+")
    sys.exit(0)

mon = sys.monitoring

# --- Test 1: Basic C_CALL detection ---
print("=== Test 1: sys.monitoring C_RETURN events for socket ops ===")

detected: list[tuple[str, str]] = []

tool_id = mon.PROFILER_ID
mon.use_tool_id(tool_id, "frontrun-experiment")

# Note: PEP 669 doesn't have C_CALL. It has C_RAISE and C_RETURN (3.12+).
# But CALL fires for all calls including C functions on 3.12+.
available_events = dir(mon.events)
print(f"  Available event types: {[e for e in available_events if not e.startswith('_')]}")

# Try CALL event — fires for both Python and C calls
mon.set_events(tool_id, mon.events.CALL)


def handle_call(code, instruction_offset, callable_obj, arg0):
    name = getattr(callable_obj, "__qualname__", getattr(callable_obj, "__name__", str(callable_obj)))
    detected.append(("call", name))
    return mon.DISABLE  # Don't keep firing for this code object (perf)


mon.register_callback(tool_id, mon.events.CALL, handle_call)

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.close()

mon.set_events(tool_id, 0)
mon.register_callback(tool_id, mon.events.CALL, None)
mon.free_tool_id(tool_id)

socket_calls = [name for _, name in detected if "socket" in name.lower() or "close" in name.lower()]
print(f"  Total calls detected: {len(detected)}")
print(f"  Socket-related: {socket_calls[:10]}")

# --- Test 2: INSTRUCTION + CALL coexistence ---
print("\n=== Test 2: INSTRUCTION + CALL coexistence ===")

instruction_count = 0
call_count = 0

tool_id = mon.OPTIMIZER_ID  # Use different tool to avoid conflict
mon.use_tool_id(tool_id, "frontrun-coexist")


def handle_instruction(code, offset):
    global instruction_count
    instruction_count += 1


def handle_call2(code, instruction_offset, callable_obj, arg0):
    global call_count
    call_count += 1


mon.set_events(tool_id, mon.events.INSTRUCTION | mon.events.CALL)
mon.register_callback(tool_id, mon.events.INSTRUCTION, handle_instruction)
mon.register_callback(tool_id, mon.events.CALL, handle_call2)


def test_func():
    x = 1 + 2
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.close()
    return x


test_func()

mon.set_events(tool_id, 0)
mon.register_callback(tool_id, mon.events.INSTRUCTION, None)
mon.register_callback(tool_id, mon.events.CALL, None)
mon.free_tool_id(tool_id)

print(f"  Instructions: {instruction_count}")
print(f"  Calls: {call_count}")
print(f"  Both fired: {instruction_count > 0 and call_count > 0}")

# --- Test 3: Can we distinguish Python calls from C calls? ---
print("\n=== Test 3: Python vs C call distinction ===")

python_calls = []
c_calls = []

tool_id = mon.PROFILER_ID
mon.use_tool_id(tool_id, "frontrun-distinguish")


def handle_call3(code, instruction_offset, callable_obj, arg0):
    name = getattr(callable_obj, "__qualname__", getattr(callable_obj, "__name__", str(callable_obj)))
    if isinstance(callable_obj, type):
        c_calls.append(f"type:{name}")
    elif hasattr(callable_obj, "__code__"):
        python_calls.append(name)
    else:
        c_calls.append(name)


mon.set_events(tool_id, mon.events.CALL)
mon.register_callback(tool_id, mon.events.CALL, handle_call3)


def python_function():
    return 42


python_function()
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.close()
_ = len([1, 2, 3])

mon.set_events(tool_id, 0)
mon.register_callback(tool_id, mon.events.CALL, None)
mon.free_tool_id(tool_id)

print(f"  Python calls: {python_calls[:10]}")
print(f"  C/builtin calls: {c_calls[:10]}")

print("\n=== All tests completed ===")
