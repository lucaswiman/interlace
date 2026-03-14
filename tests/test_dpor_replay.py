from __future__ import annotations

import pytest

from frontrun.dpor import explore_dpor


class TestReplayHarness:
    def test_reproduction_uses_dpor_runner_not_bytecode_replay(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """DPOR counterexample replay should not depend on bytecode.run_with_schedule."""
        import frontrun.bytecode as bytecode

        def _unexpected_bytecode_replay(*args: object, **kwargs: object) -> object:
            raise AssertionError("explore_dpor replay should not call frontrun.bytecode.run_with_schedule")

        monkeypatch.setattr(bytecode, "run_with_schedule", _unexpected_bytecode_replay)

        class State:
            def __init__(self) -> None:
                self.value = 0

        def thread_fn(state: State) -> None:
            temp = state.value
            state.value = temp + 1

        result = explore_dpor(
            setup=State,
            threads=[thread_fn, thread_fn],
            invariant=lambda s: s.value == 2,
            detect_io=False,
            reproduce_on_failure=3,
            max_executions=50,
            preemption_bound=2,
        )

        assert not result.property_holds, "DPOR should find the lost-update race"
        assert result.reproduction_attempts == 3
        assert result.reproduction_successes == 3, (
            f"Expected DPOR-native replay to reproduce 3/3 times, got "
            f"{result.reproduction_successes}/{result.reproduction_attempts}"
        )
