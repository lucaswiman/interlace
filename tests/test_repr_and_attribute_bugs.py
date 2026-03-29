"""Tests for InterleavingResult repr with Schedule counterexample.

Bug: InterleavingResult.__repr__ calls len() on counterexample, which
crashes with TypeError when counterexample is a Schedule object (as set
by explore_marker_interleavings).
"""

from frontrun.common import InterleavingResult, Schedule, Step
from frontrun.trace_markers import explore_marker_interleavings


class TestInterleavingResultReprWithSchedule:
    """InterleavingResult.__repr__ must handle Schedule counterexamples."""

    def test_repr_with_short_schedule(self):
        """repr() should not crash when counterexample is a Schedule."""
        schedule = Schedule([Step("t1", "a"), Step("t2", "b")])
        result = InterleavingResult(
            property_holds=False,
            counterexample=schedule,  # type: ignore[arg-type]
            num_explored=3,
        )
        r = repr(result)
        assert "property_holds=False" in r
        assert "num_explored=3" in r

    def test_repr_with_long_schedule(self):
        """repr() should truncate long Schedule counterexamples gracefully."""
        steps = [Step(f"t{i % 2}", f"m{i}") for i in range(20)]
        schedule = Schedule(steps)
        result = InterleavingResult(
            property_holds=False,
            counterexample=schedule,  # type: ignore[arg-type]
            num_explored=5,
        )
        r = repr(result)
        assert "property_holds=False" in r

    def test_repr_with_none_counterexample(self):
        """repr() still works with None counterexample."""
        result = InterleavingResult(property_holds=True, num_explored=10)
        r = repr(result)
        assert "property_holds=True" in r
        assert "None" in r

    def test_repr_with_list_counterexample(self):
        """repr() still works with list[int] counterexample (original behavior)."""
        result = InterleavingResult(
            property_holds=False,
            counterexample=[0, 1, 0, 1, 0],
            num_explored=5,
        )
        r = repr(result)
        assert "[0, 1, 0, 1, 0]" in r

    def test_repr_with_long_list_counterexample(self):
        """repr() truncates long list counterexamples."""
        result = InterleavingResult(
            property_holds=False,
            counterexample=list(range(20)),
            num_explored=5,
        )
        r = repr(result)
        assert "20 steps" in r

    def test_explore_marker_interleavings_result_repr(self):
        """End-to-end: repr of a real explore_marker_interleavings result."""

        class Counter:
            def __init__(self):
                self.value = 0

            def inc(self):
                temp = self.value  # frontrun: read
                self.value = temp + 1  # frontrun: write

        result = explore_marker_interleavings(
            setup=Counter,
            threads={
                "t1": (lambda s: s.inc(), ["read", "write"]),
                "t2": (lambda s: s.inc(), ["read", "write"]),
            },
            invariant=lambda s: s.value == 2,
        )
        assert not result.property_holds
        # This should not raise TypeError
        r = repr(result)
        assert "property_holds=False" in r
