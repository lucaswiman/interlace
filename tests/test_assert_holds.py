"""Tests for InterleavingResult.assert_holds()."""

from __future__ import annotations

import pytest

from frontrun.common import InterleavingResult


class TestAssertHolds:
    """InterleavingResult.assert_holds() raises or returns None appropriately."""

    def test_raises_when_property_fails(self) -> None:
        """assert_holds() raises AssertionError when property_holds is False."""
        result = InterleavingResult(
            property_holds=False,
            explanation="Thread A read stale value written by Thread B.",
        )
        with pytest.raises(AssertionError, match="Thread A read stale value written by Thread B."):
            result.assert_holds()

    def test_does_not_raise_when_property_holds(self) -> None:
        """assert_holds() returns None silently when property_holds is True."""
        result = InterleavingResult(property_holds=True, num_explored=5)
        assert result.assert_holds() is None

    def test_msg_prefix_prepended_on_failure(self) -> None:
        """assert_holds() prepends msg_prefix to the explanation on failure."""
        result = InterleavingResult(
            property_holds=False,
            explanation="Race on counter.",
        )
        with pytest.raises(AssertionError, match="counter_test: Race on counter."):
            result.assert_holds(msg_prefix="counter_test: ")

    def test_msg_prefix_ignored_on_success(self) -> None:
        """assert_holds() returns None even when msg_prefix is provided and property holds."""
        result = InterleavingResult(property_holds=True, num_explored=3)
        assert result.assert_holds(msg_prefix="ignored: ") is None

    def test_empty_msg_prefix_uses_explanation_directly(self) -> None:
        """assert_holds() with empty msg_prefix uses explanation as-is."""
        result = InterleavingResult(
            property_holds=False,
            explanation="Dirty read detected.",
        )
        with pytest.raises(AssertionError, match="Dirty read detected."):
            result.assert_holds(msg_prefix="")

    def test_raises_with_none_explanation(self) -> None:
        """assert_holds() raises AssertionError even when explanation is None."""
        result = InterleavingResult(property_holds=False, explanation=None)
        with pytest.raises(AssertionError):
            result.assert_holds()
