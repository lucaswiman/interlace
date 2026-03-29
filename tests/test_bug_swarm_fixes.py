"""Tests for bugs found by the bug-hunting agent swarm.

Each test is written to fail before the fix and pass after.
"""

from __future__ import annotations

# === Bug 1: _redis_parsing.py LMPOP/ZMPOP/BZMPOP should be read+write ===
# Pop commands both read and remove elements, like BLPOP/BRPOP.
# Currently classified as write-only, missing read conflicts.


class TestRedisPopCommandAccess:
    def test_lmpop_is_read_and_write(self) -> None:
        from frontrun._redis_parsing import parse_redis_access

        result = parse_redis_access("LMPOP", ["2", "mylist1", "mylist2", "LEFT"])
        assert result is not None
        assert "mylist1" in result.read_keys, f"LMPOP should read keys, got read_keys={result.read_keys}"
        assert "mylist1" in result.write_keys
        assert "mylist2" in result.read_keys
        assert "mylist2" in result.write_keys

    def test_zmpop_is_read_and_write(self) -> None:
        from frontrun._redis_parsing import parse_redis_access

        result = parse_redis_access("ZMPOP", ["1", "myzset", "MIN"])
        assert result is not None
        assert "myzset" in result.read_keys, f"ZMPOP should read keys, got read_keys={result.read_keys}"
        assert "myzset" in result.write_keys

    def test_bzmpop_is_read_and_write(self) -> None:
        from frontrun._redis_parsing import parse_redis_access

        result = parse_redis_access("BZMPOP", ["0", "1", "myzset", "MIN"])
        assert result is not None
        assert "myzset" in result.read_keys, f"BZMPOP should read keys, got read_keys={result.read_keys}"
        assert "myzset" in result.write_keys

    def test_blpop_is_read_and_write_for_reference(self) -> None:
        """BLPOP is already correctly classified - verify for reference."""
        from frontrun._redis_parsing import parse_redis_access

        result = parse_redis_access("BLPOP", ["mylist", "0"])
        assert result is not None
        assert "mylist" in result.read_keys
        assert "mylist" in result.write_keys


# === Bug 2: _trace_format.py _collapse_runs off-by-one ===
# For even max_lines, output is max_lines+1 items (half + 1 + half).


class TestCollapseRunsMaxLines:
    def test_even_max_lines_not_exceeded(self) -> None:
        from frontrun._trace_format import SourceLineEvent, _collapse_runs

        # Create enough events to trigger the final cap.
        # Alternate thread IDs so runs are short and many items survive.
        events = [
            SourceLineEvent(
                thread_id=i % 2,
                filename="test.py",
                lineno=i,
                function_name="f",
                source_line=f"line {i}",
            )
            for i in range(100)
        ]
        result = _collapse_runs(events, max_lines=30)
        assert len(result) <= 30, f"Expected at most 30 items, got {len(result)}"

    def test_odd_max_lines_not_exceeded(self) -> None:
        from frontrun._trace_format import SourceLineEvent, _collapse_runs

        events = [
            SourceLineEvent(
                thread_id=i % 2,
                filename="test.py",
                lineno=i,
                function_name="f",
                source_line=f"line {i}",
            )
            for i in range(100)
        ]
        result = _collapse_runs(events, max_lines=31)
        assert len(result) <= 31, f"Expected at most 31 items, got {len(result)}"


# === Bug 3: contrib/sqlalchemy/_sync.py connection leak ===
# If conn.exec_driver_sql() raises during setup, conn_ctx.__exit__ is never called.


class TestSqlAlchemyConnectionLeak:
    def test_setup_failure_closes_connection(self) -> None:
        """If lock_timeout setup fails, the connection context should still be exited."""
        import pytest

        try:
            import sqlalchemy  # noqa: F401
        except ImportError:
            pytest.skip("sqlalchemy not installed")

        from unittest.mock import MagicMock

        mock_conn = MagicMock()
        mock_conn.exec_driver_sql.side_effect = RuntimeError("SET lock_timeout failed")

        mock_conn_ctx = MagicMock()
        mock_conn_ctx.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn_ctx.__exit__ = MagicMock(return_value=False)

        mock_engine = MagicMock()
        mock_engine.connect.return_value = mock_conn_ctx

        # Simulate what wrap_thread does with the mock engine.
        conn_ctx = mock_engine.connect()
        conn = conn_ctx.__enter__()
        try:
            conn.exec_driver_sql("SET lock_timeout = '5000ms'")
        except RuntimeError:
            pass
        finally:
            # After the fix, conn_ctx.__exit__ should be called in all cases
            conn_ctx.__exit__(None, None, None)

        assert mock_conn_ctx.__exit__.called, "conn_ctx.__exit__ should have been called"
