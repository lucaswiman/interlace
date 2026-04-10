"""Tests for DPOR randomized search strategy (Proposal A).

TDD red-phase: these tests FAIL because ``search="random"`` is not yet
recognised by the DPOR engine.  Once ``random`` (and ``random:<seed>``) are
implemented in ``crates/dpor/src/lib.rs``, all tests here should turn green.
"""

from __future__ import annotations

import threading

import pytest

from frontrun.dpor import explore_dpor

# ---------------------------------------------------------------------------
# Scenario helpers (mirrors test_search_strategies.py)
# ---------------------------------------------------------------------------


class Counter:
    def __init__(self) -> None:
        self.value = 0


class BankAccount:
    def __init__(self, balance: int) -> None:
        self.balance = balance


class Accounts:
    def __init__(self) -> None:
        self.a = BankAccount(100)
        self.b = BankAccount(100)


def scenario_lost_update():  # noqa: ANN202
    """Two threads doing non-atomic read-modify-write on a shared counter."""

    def t(s: Counter) -> None:
        temp = s.value
        s.value = temp + 1

    return Counter, [t, t], lambda s: s.value == 2, "lost_update"


def scenario_bank_transfer():  # noqa: ANN202
    """Two non-atomic transfers — total balance should stay 200."""

    def transfer(s: Accounts) -> None:
        temp_a = s.a.balance
        temp_b = s.b.balance
        s.a.balance = temp_a - 10
        s.b.balance = temp_b + 10

    return Accounts, [transfer, transfer], lambda s: s.a.balance + s.b.balance == 200, "bank_transfer"


def scenario_three_thread_counter():  # noqa: ANN202
    """Three threads doing non-atomic increment."""

    def t(s: Counter) -> None:
        temp = s.value
        s.value = temp + 1

    return Counter, [t, t, t], lambda s: s.value == 3, "three_thread_counter"


def scenario_dining_philosophers_3():  # noqa: ANN202
    """Three dining philosophers — deadlock scenario."""
    n = 3

    class State:
        def __init__(self) -> None:
            self.forks = [threading.Lock() for _ in range(n)]

    def make_phil(i: int):  # noqa: ANN202
        def phil(s: State) -> None:
            left = i
            right = (i + 1) % n
            with s.forks[left]:
                with s.forks[right]:
                    pass

        return phil

    return State, [make_phil(i) for i in range(n)], lambda s: True, "dining_philosophers_3"


SCENARIOS = [
    scenario_lost_update,
    scenario_bank_transfer,
    scenario_three_thread_counter,
    scenario_dining_philosophers_3,
]

# ---------------------------------------------------------------------------
# 1. Basic acceptance — search="random" is accepted without error
# ---------------------------------------------------------------------------


def test_random_strategy_accepted() -> None:
    """search='random' is accepted and finds the lost-update bug."""
    setup, threads, invariant, _ = scenario_lost_update()
    result = explore_dpor(
        setup=setup,
        threads=threads,
        invariant=invariant,
        max_executions=500,
        stop_on_first=True,
        search="random",
        detect_io=False,
        reproduce_on_failure=0,
    )
    assert not result.property_holds, "search='random' should find lost update bug"
    assert result.num_explored >= 1


# ---------------------------------------------------------------------------
# 2. Seeded variant — search="random:<seed>" is accepted without error
# ---------------------------------------------------------------------------


def test_random_seeded_strategy_accepted() -> None:
    """search='random:42' is accepted and finds the lost-update bug."""
    setup, threads, invariant, _ = scenario_lost_update()
    result = explore_dpor(
        setup=setup,
        threads=threads,
        invariant=invariant,
        max_executions=500,
        stop_on_first=True,
        search="random:42",
        detect_io=False,
        reproduce_on_failure=0,
    )
    assert not result.property_holds, "search='random:42' should find lost update bug"
    assert result.num_explored >= 1


# ---------------------------------------------------------------------------
# 3. Different seeds produce different exploration counts
# ---------------------------------------------------------------------------


def test_different_seeds_explore_differently() -> None:
    """Two distinct seeds should traverse the wakeup tree in different orders.

    With stop_on_first=True the number of executions explored before the first
    failure is a proxy for the traversal order.  We run several seed pairs and
    assert that at least one pair differs — this is extremely unlikely to be
    violated by any correct random implementation.
    """
    setup, threads, invariant, _ = scenario_lost_update()

    common_kwargs = {
        "setup": setup,
        "threads": threads,
        "invariant": invariant,
        "max_executions": 500,
        "stop_on_first": True,
        "detect_io": False,
        "reproduce_on_failure": 0,
    }

    counts: list[int] = []
    for seed in (1, 2, 3, 7, 99):
        r = explore_dpor(**common_kwargs, search=f"random:{seed}")  # type: ignore[arg-type]
        counts.append(r.num_explored)

    # At least two seeds must produce a different exploration count.
    assert len(set(counts)) > 1, (
        f"All seeds produced the same num_explored={counts[0]}; "
        "random strategy must vary traversal order across seeds"
    )


# ---------------------------------------------------------------------------
# 4. Same seed → same exploration count (deterministic replay)
# ---------------------------------------------------------------------------


def test_same_seed_is_deterministic() -> None:
    """Replaying with the same seed must produce the identical exploration count."""
    setup, threads, invariant, _ = scenario_lost_update()

    common_kwargs = {
        "setup": setup,
        "threads": threads,
        "invariant": invariant,
        "max_executions": 500,
        "stop_on_first": True,
        "detect_io": False,
        "reproduce_on_failure": 0,
        "search": "random:42",
    }

    r1 = explore_dpor(**common_kwargs)  # type: ignore[arg-type]
    r2 = explore_dpor(**common_kwargs)  # type: ignore[arg-type]

    assert r1.num_explored == r2.num_explored, (
        f"Same seed gave different exploration counts: {r1.num_explored} vs {r2.num_explored}"
    )
    assert r1.property_holds == r2.property_holds


# ---------------------------------------------------------------------------
# 5. Parametrized: random strategy finds bugs in all standard scenarios
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scenario_fn", SCENARIOS, ids=[fn.__name__ for fn in SCENARIOS])
def test_random_finds_bug_in_scenario(scenario_fn) -> None:  # noqa: ANN001
    """search='random' finds the race condition / deadlock in every scenario."""
    setup, threads, invariant, name = scenario_fn()

    kwargs: dict[str, object] = {
        "setup": setup,
        "threads": threads,
        "invariant": invariant,
        "max_executions": 5000,
        "stop_on_first": True,
        "search": "random",
        "detect_io": False,
        "reproduce_on_failure": 0,
    }
    if name == "dining_philosophers_3":
        kwargs["deadlock_timeout"] = 2.0
        kwargs["preemption_bound"] = 2

    result = explore_dpor(**kwargs)  # type: ignore[arg-type]
    assert not result.property_holds, f"search='random' should find bug in scenario '{name}'"


# ---------------------------------------------------------------------------
# 6. Seeded random also finds bugs in standard scenarios
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scenario_fn", SCENARIOS, ids=[fn.__name__ for fn in SCENARIOS])
def test_random_seeded_finds_bug_in_scenario(scenario_fn) -> None:  # noqa: ANN001
    """search='random:7' finds the race condition / deadlock in every scenario."""
    setup, threads, invariant, name = scenario_fn()

    kwargs: dict[str, object] = {
        "setup": setup,
        "threads": threads,
        "invariant": invariant,
        "max_executions": 5000,
        "stop_on_first": True,
        "search": "random:7",
        "detect_io": False,
        "reproduce_on_failure": 0,
    }
    if name == "dining_philosophers_3":
        kwargs["deadlock_timeout"] = 2.0
        kwargs["preemption_bound"] = 2

    result = explore_dpor(**kwargs)  # type: ignore[arg-type]
    assert not result.property_holds, f"search='random:7' should find bug in scenario '{name}'"


# ---------------------------------------------------------------------------
# 7. Invalid seed string is rejected with a clear error
# ---------------------------------------------------------------------------


def test_random_invalid_seed_rejected() -> None:
    """search='random:notanumber' must raise a ValueError-like exception."""
    setup, threads, invariant, _ = scenario_lost_update()
    with pytest.raises(Exception):
        explore_dpor(
            setup=setup,
            threads=threads,
            invariant=invariant,
            max_executions=10,
            search="random:notanumber",
            detect_io=False,
            reproduce_on_failure=0,
        )


# ---------------------------------------------------------------------------
# 8. search="random" and search="random:0" are both valid (edge-case seed)
# ---------------------------------------------------------------------------


def test_random_seed_zero_accepted() -> None:
    """Seed 0 is a valid, distinct seed — must not be treated as 'no seed'."""
    setup, threads, invariant, _ = scenario_lost_update()
    result = explore_dpor(
        setup=setup,
        threads=threads,
        invariant=invariant,
        max_executions=500,
        stop_on_first=True,
        search="random:0",
        detect_io=False,
        reproduce_on_failure=0,
    )
    assert not result.property_holds, "search='random:0' should find lost update bug"
