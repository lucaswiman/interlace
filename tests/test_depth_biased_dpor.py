"""Tests for Proposal D: Depth-biased backtrack selection.

Verifies that ``search="depth-biased"`` and ``search="depth-biased:<seed>"``
are accepted and work correctly.  These tests are the TDD red-phase for the
feature — they FAIL until the Rust engine recognises the new strategy string.

Design notes
------------
- Depth-biased selection uses weighted random sampling over the backtrack
  stack, where shallower entries receive higher weight when alpha < 1.
- alpha > 1 biases toward deep (DFS-like); alpha < 1 biases toward shallow
  (more globally different traces); alpha = 1 is uniform random.
- Seed controls the PRNG used for weighted selection, making results
  deterministic for a given seed.
"""

from __future__ import annotations

import threading

import pytest

from frontrun.dpor import explore_dpor


# ---------------------------------------------------------------------------
# Scenario helpers (identical to test_search_strategies.py)
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


def scenario_lost_update():
    """Two threads doing non-atomic read-modify-write on a shared counter."""

    def t(s: Counter) -> None:
        temp = s.value
        s.value = temp + 1

    return Counter, [t, t], lambda s: s.value == 2


def scenario_bank_transfer():
    """Two non-atomic transfers from account a to account b."""

    def transfer(s: Accounts) -> None:
        temp_a = s.a.balance
        temp_b = s.b.balance
        s.a.balance = temp_a - 10
        s.b.balance = temp_b + 10

    return Accounts, [transfer, transfer], lambda s: s.a.balance + s.b.balance == 200


def scenario_three_thread_counter():
    """Three threads doing non-atomic increment."""

    def t(s: Counter) -> None:
        temp = s.value
        s.value = temp + 1

    return Counter, [t, t, t], lambda s: s.value == 3


def scenario_dining_philosophers_3():
    """Three dining philosophers — deadlock detection."""
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

    return State, [make_phil(i) for i in range(n)], lambda s: True


# ---------------------------------------------------------------------------
# 1. Basic acceptance: "depth-biased" is recognised without error
# ---------------------------------------------------------------------------


def test_depth_biased_accepted_no_seed() -> None:
    """search="depth-biased" is accepted without raising ValueError."""
    setup, threads, invariant = scenario_lost_update()
    result = explore_dpor(
        setup=setup,
        threads=threads,
        invariant=invariant,
        max_executions=500,
        stop_on_first=True,
        search="depth-biased",
        detect_io=False,
        reproduce_on_failure=0,
    )
    assert not result.property_holds, "depth-biased should find the lost update bug"
    assert result.num_explored >= 1


def test_depth_biased_accepted_with_seed() -> None:
    """search="depth-biased:42" with explicit seed is accepted without error."""
    setup, threads, invariant = scenario_lost_update()
    result = explore_dpor(
        setup=setup,
        threads=threads,
        invariant=invariant,
        max_executions=500,
        stop_on_first=True,
        search="depth-biased:42",
        detect_io=False,
        reproduce_on_failure=0,
    )
    assert not result.property_holds, "depth-biased:42 should find the lost update bug"
    assert result.num_explored >= 1


def test_depth_biased_seed_zero() -> None:
    """search="depth-biased:0" (explicit seed 0) is accepted."""
    setup, threads, invariant = scenario_lost_update()
    result = explore_dpor(
        setup=setup,
        threads=threads,
        invariant=invariant,
        max_executions=500,
        stop_on_first=True,
        search="depth-biased:0",
        detect_io=False,
        reproduce_on_failure=0,
    )
    assert not result.property_holds, "depth-biased:0 should find the lost update bug"


# ---------------------------------------------------------------------------
# 2. Different seeds produce different exploration orders
# ---------------------------------------------------------------------------


def test_different_seeds_produce_different_orders() -> None:
    """Different seeds lead to different num_explored when stop_on_first=True.

    With stop_on_first=True the engine stops at the first failing execution,
    so the choice of which trace to visit first determines how many executions
    are needed.  Two different seeds should (with high probability across the
    full trace space) produce different counts at least once across the
    scenarios tried here.
    """
    setups_and_threads_and_invariants = [
        scenario_lost_update(),
        scenario_three_thread_counter(),
        scenario_bank_transfer(),
    ]

    counts_seed_1 = []
    counts_seed_99 = []

    for setup, threads, invariant in setups_and_threads_and_invariants:
        r1 = explore_dpor(
            setup=setup,
            threads=threads,
            invariant=invariant,
            max_executions=1000,
            stop_on_first=True,
            search="depth-biased:1",
            detect_io=False,
            reproduce_on_failure=0,
        )
        r99 = explore_dpor(
            setup=setup,
            threads=threads,
            invariant=invariant,
            max_executions=1000,
            stop_on_first=True,
            search="depth-biased:99",
            detect_io=False,
            reproduce_on_failure=0,
        )
        counts_seed_1.append(r1.num_explored)
        counts_seed_99.append(r99.num_explored)

    # At least one scenario should differ between the two seeds.
    assert counts_seed_1 != counts_seed_99, (
        f"seeds 1 and 99 produced identical num_explored across all scenarios: "
        f"seed_1={counts_seed_1}, seed_99={counts_seed_99}"
    )


# ---------------------------------------------------------------------------
# 3. Same seed produces deterministic results
# ---------------------------------------------------------------------------


def test_same_seed_is_deterministic() -> None:
    """Running depth-biased with the same seed twice yields the same num_explored."""
    setup, threads, invariant = scenario_three_thread_counter()

    kwargs = {
        "setup": setup,
        "threads": threads,
        "invariant": invariant,
        "max_executions": 1000,
        "stop_on_first": True,
        "search": "depth-biased:7",
        "detect_io": False,
        "reproduce_on_failure": 0,
    }

    result_a = explore_dpor(**kwargs)  # type: ignore[arg-type]
    result_b = explore_dpor(**kwargs)  # type: ignore[arg-type]

    assert result_a.num_explored == result_b.num_explored, (
        f"depth-biased:7 is non-deterministic: first run={result_a.num_explored}, "
        f"second run={result_b.num_explored}"
    )
    assert not result_a.property_holds
    assert not result_b.property_holds


def test_default_depth_biased_is_deterministic() -> None:
    """search="depth-biased" (no seed) is deterministic (uses seed=0 by default)."""
    setup, threads, invariant = scenario_lost_update()

    kwargs = {
        "setup": setup,
        "threads": threads,
        "invariant": invariant,
        "max_executions": 500,
        "stop_on_first": True,
        "search": "depth-biased",
        "detect_io": False,
        "reproduce_on_failure": 0,
    }

    result_a = explore_dpor(**kwargs)  # type: ignore[arg-type]
    result_b = explore_dpor(**kwargs)  # type: ignore[arg-type]

    assert result_a.num_explored == result_b.num_explored, (
        "depth-biased with default seed is non-deterministic: "
        f"first={result_a.num_explored}, second={result_b.num_explored}"
    )


# ---------------------------------------------------------------------------
# 4. All standard scenarios are found
# ---------------------------------------------------------------------------


def test_finds_lost_update() -> None:
    """depth-biased finds the classic lost update race."""
    setup, threads, invariant = scenario_lost_update()
    result = explore_dpor(
        setup=setup,
        threads=threads,
        invariant=invariant,
        max_executions=500,
        stop_on_first=True,
        search="depth-biased",
        detect_io=False,
        reproduce_on_failure=0,
    )
    assert not result.property_holds


def test_finds_bank_transfer_bug() -> None:
    """depth-biased finds the bank transfer race."""
    setup, threads, invariant = scenario_bank_transfer()
    result = explore_dpor(
        setup=setup,
        threads=threads,
        invariant=invariant,
        max_executions=500,
        stop_on_first=True,
        search="depth-biased",
        detect_io=False,
        reproduce_on_failure=0,
    )
    assert not result.property_holds


def test_finds_three_thread_counter_bug() -> None:
    """depth-biased finds the three-thread counter race."""
    setup, threads, invariant = scenario_three_thread_counter()
    result = explore_dpor(
        setup=setup,
        threads=threads,
        invariant=invariant,
        max_executions=1000,
        stop_on_first=True,
        search="depth-biased",
        detect_io=False,
        reproduce_on_failure=0,
    )
    assert not result.property_holds


def test_finds_dining_philosophers_deadlock() -> None:
    """depth-biased finds the dining philosophers deadlock."""
    setup, threads, invariant = scenario_dining_philosophers_3()
    result = explore_dpor(
        setup=setup,
        threads=threads,
        invariant=invariant,
        max_executions=5000,
        preemption_bound=2,
        stop_on_first=True,
        search="depth-biased",
        detect_io=False,
        reproduce_on_failure=0,
        deadlock_timeout=2.0,
    )
    assert not result.property_holds


# ---------------------------------------------------------------------------
# 5. Seeded variant also finds all standard bugs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 42, 123, 999])
def test_seeded_depth_biased_finds_lost_update(seed: int) -> None:
    """depth-biased:<seed> finds the lost update bug for various seeds."""
    setup, threads, invariant = scenario_lost_update()
    result = explore_dpor(
        setup=setup,
        threads=threads,
        invariant=invariant,
        max_executions=500,
        stop_on_first=True,
        search=f"depth-biased:{seed}",
        detect_io=False,
        reproduce_on_failure=0,
    )
    assert not result.property_holds, f"depth-biased:{seed} failed to find lost update bug"


@pytest.mark.parametrize("seed", [0, 7, 42])
def test_seeded_depth_biased_finds_bank_transfer_bug(seed: int) -> None:
    """depth-biased:<seed> finds the bank transfer bug."""
    setup, threads, invariant = scenario_bank_transfer()
    result = explore_dpor(
        setup=setup,
        threads=threads,
        invariant=invariant,
        max_executions=500,
        stop_on_first=True,
        search=f"depth-biased:{seed}",
        detect_io=False,
        reproduce_on_failure=0,
    )
    assert not result.property_holds, f"depth-biased:{seed} failed to find bank transfer bug"


# ---------------------------------------------------------------------------
# 6. Bug found in reasonable number of executions (not worse than 2x DFS)
# ---------------------------------------------------------------------------


def test_depth_biased_efficiency_vs_dfs_lost_update() -> None:
    """depth-biased finds the lost update bug within 2x the DFS execution count."""
    setup, threads, invariant = scenario_lost_update()

    dfs_result = explore_dpor(
        setup=setup,
        threads=threads,
        invariant=invariant,
        max_executions=500,
        stop_on_first=True,
        search="dfs",
        detect_io=False,
        reproduce_on_failure=0,
    )
    assert not dfs_result.property_holds, "DFS baseline should find the bug"
    dfs_count = dfs_result.num_explored

    depth_biased_result = explore_dpor(
        setup=setup,
        threads=threads,
        invariant=invariant,
        max_executions=500,
        stop_on_first=True,
        search="depth-biased",
        detect_io=False,
        reproduce_on_failure=0,
    )
    assert not depth_biased_result.property_holds, "depth-biased should find the bug"
    depth_biased_count = depth_biased_result.num_explored

    assert depth_biased_count <= dfs_count * 2, (
        f"depth-biased took {depth_biased_count} executions to find the bug, "
        f"more than 2x DFS ({dfs_count})"
    )


def test_depth_biased_efficiency_vs_dfs_bank_transfer() -> None:
    """depth-biased finds the bank transfer bug within 2x the DFS execution count."""
    setup, threads, invariant = scenario_bank_transfer()

    dfs_result = explore_dpor(
        setup=setup,
        threads=threads,
        invariant=invariant,
        max_executions=500,
        stop_on_first=True,
        search="dfs",
        detect_io=False,
        reproduce_on_failure=0,
    )
    assert not dfs_result.property_holds, "DFS baseline should find the bug"
    dfs_count = dfs_result.num_explored

    depth_biased_result = explore_dpor(
        setup=setup,
        threads=threads,
        invariant=invariant,
        max_executions=500,
        stop_on_first=True,
        search="depth-biased",
        detect_io=False,
        reproduce_on_failure=0,
    )
    assert not depth_biased_result.property_holds, "depth-biased should find the bug"
    depth_biased_count = depth_biased_result.num_explored

    assert depth_biased_count <= dfs_count * 2, (
        f"depth-biased took {depth_biased_count} executions to find the bug, "
        f"more than 2x DFS ({dfs_count})"
    )


def test_depth_biased_efficiency_vs_dfs_three_threads() -> None:
    """depth-biased finds the three-thread counter bug within 2x the DFS execution count."""
    setup, threads, invariant = scenario_three_thread_counter()

    dfs_result = explore_dpor(
        setup=setup,
        threads=threads,
        invariant=invariant,
        max_executions=1000,
        stop_on_first=True,
        search="dfs",
        detect_io=False,
        reproduce_on_failure=0,
    )
    assert not dfs_result.property_holds, "DFS baseline should find the bug"
    dfs_count = dfs_result.num_explored

    depth_biased_result = explore_dpor(
        setup=setup,
        threads=threads,
        invariant=invariant,
        max_executions=1000,
        stop_on_first=True,
        search="depth-biased",
        detect_io=False,
        reproduce_on_failure=0,
    )
    assert not depth_biased_result.property_holds, "depth-biased should find the bug"
    depth_biased_count = depth_biased_result.num_explored

    assert depth_biased_count <= dfs_count * 2, (
        f"depth-biased took {depth_biased_count} executions to find the bug, "
        f"more than 2x DFS ({dfs_count})"
    )


# ---------------------------------------------------------------------------
# 7. Invalid seed format is rejected
# ---------------------------------------------------------------------------


def test_depth_biased_invalid_seed_rejected() -> None:
    """search="depth-biased:notanumber" raises ValueError."""
    setup, threads, invariant = scenario_lost_update()
    with pytest.raises(Exception):
        explore_dpor(
            setup=setup,
            threads=threads,
            invariant=invariant,
            max_executions=10,
            search="depth-biased:notanumber",
            detect_io=False,
            reproduce_on_failure=0,
        )
