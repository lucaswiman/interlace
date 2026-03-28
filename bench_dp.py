"""Benchmark: search strategies across complex race scenarios."""
from __future__ import annotations

import threading
import time

from frontrun.dpor import explore_dpor

STRATEGIES = [
    "dfs",
    "bit-reversal",
    "bit-reversal:42",
    "round-robin",
    "round-robin:7",
    "stride",
    "stride:3",
    "conflict-first",
]


def make_dining(n: int):
    """N dining philosophers with circular fork ordering (deadlock)."""
    class State:
        def __init__(self) -> None:
            self.forks = [threading.Lock() for _ in range(n)]

    def make_phil(i: int):
        def phil(s: State) -> None:
            left = i
            right = (i + 1) % n
            with s.forks[left]:
                with s.forks[right]:
                    pass
        return phil

    return State, [make_phil(i) for i in range(n)], lambda s: True


def make_producer_consumer_lost():
    """3 producers, 1 consumer. Non-atomic counter tracks items.
    Invariant: consumed count matches what was produced."""
    class State:
        def __init__(self) -> None:
            self.produced = 0
            self.consumed = 0
            self.buffer: list[int] = []

    def producer(s: State) -> None:
        item = s.produced
        s.produced = item + 1
        s.buffer.append(item)

    def consumer(s: State) -> None:
        if s.buffer:
            s.buffer.pop(0)
            c = s.consumed
            s.consumed = c + 1

    return State, [producer, producer, producer, consumer], lambda s: s.produced >= s.consumed


def make_bank_multi_account():
    """4 threads transferring between 3 accounts. Conservation invariant."""
    class State:
        def __init__(self) -> None:
            self.a = 100
            self.b = 100
            self.c = 100

    def transfer_ab(s: State) -> None:
        ta = s.a
        tb = s.b
        s.a = ta - 10
        s.b = tb + 10

    def transfer_bc(s: State) -> None:
        tb = s.b
        tc = s.c
        s.b = tb - 10
        s.c = tc + 10

    def transfer_ca(s: State) -> None:
        tc = s.c
        ta = s.a
        s.c = tc - 10
        s.a = ta + 10

    def transfer_ba(s: State) -> None:
        tb = s.b
        ta = s.a
        s.b = tb - 10
        s.a = ta + 10

    return State, [transfer_ab, transfer_bc, transfer_ca, transfer_ba], lambda s: s.a + s.b + s.c == 300


def make_shared_map_race():
    """4 threads racing to read-modify-write different keys of a shared dict.
    Some keys overlap, creating a complex conflict graph."""
    class State:
        def __init__(self) -> None:
            self.data: dict[str, int] = {"x": 0, "y": 0, "z": 0}

    def t0(s: State) -> None:
        v = s.data["x"]
        s.data["x"] = v + 1
        w = s.data["y"]
        s.data["y"] = w + 1

    def t1(s: State) -> None:
        v = s.data["y"]
        s.data["y"] = v + 1
        w = s.data["z"]
        s.data["z"] = w + 1

    def t2(s: State) -> None:
        v = s.data["z"]
        s.data["z"] = v + 1
        w = s.data["x"]
        s.data["x"] = w + 1

    def t3(s: State) -> None:
        v = s.data["x"]
        s.data["x"] = v + 1

    return State, [t0, t1, t2, t3], lambda s: s.data["x"] == 3 and s.data["y"] == 2 and s.data["z"] == 2


def make_lock_convoy():
    """4 threads acquiring 2 locks in different orders.
    Thread 0,1 take lock_a then lock_b. Thread 2,3 take lock_b then lock_a.
    Classic lock-ordering deadlock."""
    class State:
        def __init__(self) -> None:
            self.lock_a = threading.Lock()
            self.lock_b = threading.Lock()
            self.value = 0

    def ab_worker(s: State) -> None:
        with s.lock_a:
            with s.lock_b:
                v = s.value
                s.value = v + 1

    def ba_worker(s: State) -> None:
        with s.lock_b:
            with s.lock_a:
                v = s.value
                s.value = v + 1

    return State, [ab_worker, ab_worker, ba_worker, ba_worker], lambda s: True


SCENARIOS = [
    ("dining_phil_4", *make_dining(4)),
    ("dining_phil_5", *make_dining(5)),
    ("producer_consumer", *make_producer_consumer_lost()),
    ("bank_4_account", *make_bank_multi_account()),
    ("shared_map_4t", *make_shared_map_race()),
    ("lock_convoy_4t", *make_lock_convoy()),
]


def run_all() -> None:
    col_width = 18
    header = f"{'scenario':<22}" + "".join(f"{s:>{col_width}}" for s in STRATEGIES)
    sep = "-" * len(header)

    print()
    print("=" * len(header))
    print("EXPLORATIONS TO FIRST FAILURE (lower is better)")
    print("=" * len(header))
    print(header)
    print(sep)

    for name, setup, threads, invariant in SCENARIOS:
        is_deadlock = "phil" in name or "lock" in name
        row = f"{name:<22}"

        for strategy in STRATEGIES:
            kwargs: dict = {
                "setup": setup,
                "threads": threads,
                "invariant": invariant,
                "max_executions": 50_000,
                "preemption_bound": 2,
                "stop_on_first": True,
                "search": strategy,
                "detect_io": False,
            }
            if is_deadlock:
                kwargs["deadlock_timeout"] = 2.0

            result = explore_dpor(**kwargs)
            val = str(result.num_explored)
            if result.property_holds:
                val += "*"
            row += f"{val:>{col_width}}"

        print(row)

    print(sep)
    print("* = bug not found within max_executions")
    print()


if __name__ == "__main__":
    run_all()
