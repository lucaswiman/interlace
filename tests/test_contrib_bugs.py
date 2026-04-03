"""Tests for bugs in contrib SQLAlchemy wrappers."""

from __future__ import annotations


class TestSqlalchemyAsyncLockTimeoutForwarding:
    """async_sqlalchemy_dpor should forward lock_timeout to explore_async_dpor."""

    def test_lock_timeout_forwarded(self) -> None:
        """lock_timeout should be passed through to explore_async_dpor.

        Regression: lock_timeout is consumed by the function signature and
        NOT present in **kwargs, so it was never forwarded (unlike the sync
        version which explicitly passes lock_timeout=lock_timeout).
        """
        import inspect

        from frontrun.contrib.sqlalchemy._async import async_sqlalchemy_dpor

        source = inspect.getsource(async_sqlalchemy_dpor)
        assert "lock_timeout=lock_timeout" in source, (
            "async_sqlalchemy_dpor does not forward lock_timeout to explore_async_dpor. "
            "The sync version (sqlalchemy_dpor) passes lock_timeout=lock_timeout explicitly."
        )


class TestSqlalchemySyncExitExceptionInfo:
    """SQLAlchemy sync wrapper should pass exception info to __exit__."""

    def test_exit_receives_exception_info(self) -> None:
        """conn_ctx.__exit__ should receive actual exception info, not always (None, None, None).

        Regression: the finally block always calls conn_ctx.__exit__(None, None, None),
        even when fn(state) raises. SQLAlchemy uses the exception info to decide
        whether to rollback. With (None, None, None), it may try to commit instead.
        """
        import inspect

        from frontrun.contrib.sqlalchemy._sync import sqlalchemy_dpor

        source = inspect.getsource(sqlalchemy_dpor)

        # The source should NOT have unconditional __exit__(None, None, None) in the
        # finally block of the main fn(state) call path. It's OK in the lock_timeout
        # error path since that's before fn(state) runs.
        #
        # Count occurrences of __exit__(None, None, None)
        exit_none_count = source.count("__exit__(None, None, None)")

        # There should be at most 1 occurrence (the lock_timeout error path).
        # The main fn(state) path should use sys.exc_info() or a with statement.
        assert exit_none_count <= 1, (
            f"Found {exit_none_count} occurrences of __exit__(None, None, None) in sqlalchemy_dpor. "
            "The finally block after fn(state) should pass actual exception info to __exit__, "
            "not (None, None, None), so SQLAlchemy can decide whether to rollback."
        )
