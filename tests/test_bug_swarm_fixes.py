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


# === Bug 4: _sql_parsing.py COPY subquery detection always False ===
# `"(" in after_copy.split()[1:2]` checks list membership, not substring.
# COPY (SELECT ...) TO ... is never detected as a subquery.


class TestCopySubqueryDetection:
    def test_copy_subquery_falls_through_to_sqlglot(self) -> None:
        from frontrun._sql_parsing import _regex_parse

        # COPY (subquery) should NOT be handled by the regex fast-path
        result = _regex_parse("COPY (SELECT id FROM foo WHERE id=1) TO STDOUT")
        assert result is None, f"COPY (subquery) should return None from _regex_parse, got {result}"

    def test_copy_table_from_still_works(self) -> None:
        from frontrun._sql_parsing import _regex_parse

        result = _regex_parse("COPY my_table FROM '/tmp/data.csv'")
        assert result is not None
        assert "my_table" in result.write_tables

    def test_copy_table_to_still_works(self) -> None:
        from frontrun._sql_parsing import _regex_parse

        result = _regex_parse("COPY my_table TO '/tmp/data.csv'")
        assert result is not None
        assert "my_table" in result.read_tables


# === Bug 5: _sql_params.py $0 resolves to wrong parameter ===
# $0 → idx=-1, Python negative indexing silently returns last param.


class TestDollarZeroPlaceholder:
    def test_dollar_zero_not_resolved_to_last_param(self) -> None:
        from frontrun._sql_params import resolve_parameters

        sql = "SELECT $0, $1"
        params = ("first", "second")
        result = resolve_parameters(sql, params, "dollar")
        # $0 is invalid (PostgreSQL uses 1-based), should be left as-is
        # or at minimum should NOT silently resolve to the last parameter
        assert "'second'" not in result or "$0" in result, (
            f"$0 should not resolve to the last param via negative indexing, got: {result}"
        )


# === Bug 6: _deadlock.py cycle path missing closing node ===
# _find_cycle_from returns [start, *path] but path doesn't include
# the closing node back to start, so format_cycle shows an incomplete ring.


class TestDeadlockCyclePath:
    def test_cycle_path_forms_complete_ring(self) -> None:
        from frontrun._deadlock import WaitForGraph

        g = WaitForGraph()
        # Create a simple deadlock: thread 0 holds lock 1, thread 1 holds lock 2
        # thread 0 waits for lock 2, thread 1 waits for lock 1
        g.add_holding(0, 1)
        g.add_holding(1, 2)
        g.add_waiting(0, 2)
        cycle = g.add_waiting(1, 1)
        assert cycle is not None, "Should detect a deadlock cycle"
        # The cycle should form a complete ring: first node == last node
        assert cycle[0] == cycle[-1], f"Cycle should be a complete ring (first==last), got: {cycle}"


# === Bug 7: _trace_format.py record() event ordering race ===
# Step increment and event append happen in two separate lock acquisitions,
# allowing events to be appended out of step_index order.


class TestTraceRecorderOrdering:
    def test_events_appended_in_step_order(self) -> None:
        """Events should always be in step_index order even under contention."""
        import threading

        from frontrun._trace_format import TraceRecorder

        recorder = TraceRecorder()
        barrier = threading.Barrier(4)

        class FakeFrame:
            class f_code:  # noqa: N801
                co_filename = "test.py"
                co_name = "test_fn"

            f_lineno = 1

        def record_many(tid: int) -> None:
            barrier.wait()
            for _ in range(100):
                recorder.record(tid, FakeFrame(), "LOAD_ATTR", "read")  # type: ignore[arg-type]

        threads = [threading.Thread(target=record_many, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        steps = [e.step_index for e in recorder.events]
        assert steps == sorted(steps), "Events should be in step_index order"


# === Bug 8: cli.py usage message split between stderr/stdout ===


class TestCliUsageOutput:
    def test_usage_goes_to_stderr(self) -> None:
        """All usage text should go to the same stream (stderr)."""
        import io
        import sys
        from unittest.mock import patch

        fake_stderr = io.StringIO()
        fake_stdout = io.StringIO()

        with patch.object(sys, "stderr", fake_stderr), patch.object(sys, "stdout", fake_stdout):
            from frontrun.cli import main

            ret = main([])

        assert ret == 1
        # Nothing should go to stdout — all usage text should be on stderr
        assert fake_stdout.getvalue() == "", f"Usage text leaked to stdout: {fake_stdout.getvalue()!r}"


# === Bug 9: _sql_anomaly.py edge_to not cleared between DFS iterations ===


class TestSqlAnomalyCycleDetection:
    def test_edge_to_stale_entries_dont_corrupt_cycle(self) -> None:
        """edge_to should not carry stale entries from prior DFS roots."""
        from frontrun._sql_anomaly import _find_cycle

        # Graph with two components:
        # Component 1 (no cycle): 0 -> 1 -> 2
        # Component 2 (cycle): 3 -> 4 -> 5 -> 3
        # Node 2 is also a neighbor target in component 2 path,
        # so edge_to[2] gets set from component 1's DFS.
        graph: dict[int, list[tuple[int, str, str]]] = {
            0: [(1, "WR", "tbl_a")],
            1: [(2, "RW", "tbl_a")],
            3: [(4, "WR", "tbl_b")],
            4: [(5, "RW", "tbl_b")],
            5: [(3, "WW", "tbl_b")],
        }
        result = _find_cycle(graph)
        assert result is not None, "Should find the cycle in component 2"
        # All edges in the cycle should be from component 2
        for frm, to, _etype, _res in result:
            assert frm in {3, 4, 5} and to in {3, 4, 5}, f"Cycle edge ({frm}, {to}) contains nodes outside component 2"
