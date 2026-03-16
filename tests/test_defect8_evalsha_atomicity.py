"""Defect #8: DPOR does not model Redis Lua script (EVALSHA) atomicity.

DPOR reports false positive races for Redis operations protected by Lua
scripts because it treats EVALSHA key accesses as interleaving operations.
Redis Lua scripts execute atomically — no other commands can run between
the script's Redis calls.

The fix: treat EVAL/EVALSHA as atomic (transaction control) so DPOR
doesn't create false key-level conflict arcs.
"""

from __future__ import annotations

from frontrun._redis_parsing import parse_redis_access


class TestEvalshaAtomicity:
    """EVAL/EVALSHA should be treated as atomic operations."""

    def test_evalsha_no_key_conflicts(self) -> None:
        """EVALSHA should not report individual key-level accesses.

        Without this fix, DPOR creates false write-write conflicts between
        two threads' EVALSHA calls on the same keys, leading to false
        positive race reports.
        """
        result = parse_redis_access("EVALSHA", ("abc123", "2", "key1", "key2", "arg1"))
        # EVALSHA should be treated as atomic — no individual key conflicts
        assert result.read_keys == [], f"EVALSHA should not report read keys, got {result.read_keys}"
        assert result.write_keys == [], f"EVALSHA should not report write keys, got {result.write_keys}"
        assert result.is_transaction_control, "EVALSHA should be marked as transaction control (atomic)"

    def test_eval_no_key_conflicts(self) -> None:
        """EVAL should also be treated as atomic."""
        result = parse_redis_access("EVAL", ("return 1", "1", "key1"))
        assert result.read_keys == []
        assert result.write_keys == []
        assert result.is_transaction_control

    def test_evalsha_ro_no_key_conflicts(self) -> None:
        """EVALSHA_RO should also be treated as atomic."""
        result = parse_redis_access("EVALSHA_RO", ("abc123", "1", "key1"))
        assert result.read_keys == []
        assert result.write_keys == []
        assert result.is_transaction_control

    def test_eval_ro_no_key_conflicts(self) -> None:
        """EVAL_RO should also be treated as atomic."""
        result = parse_redis_access("EVAL_RO", ("return 1", "1", "key1"))
        assert result.read_keys == []
        assert result.write_keys == []
        assert result.is_transaction_control
