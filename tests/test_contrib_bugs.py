"""Tests for bugs in contrib Django and SQLAlchemy wrappers.

Uses mock objects to test without requiring Django or SQLAlchemy installed.
"""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock, patch


class TestDjangoAsyncLockTimeoutForwarding:
    """async_django_dpor should forward lock_timeout to explore_async_dpor."""

    def test_lock_timeout_forwarded(self) -> None:
        """lock_timeout should be passed through to explore_async_dpor.

        Regression: lock_timeout is consumed by the function signature and
        NOT present in **kwargs, so it was never forwarded (unlike the sync
        version which explicitly passes lock_timeout=lock_timeout).
        """
        captured_kwargs: dict[str, Any] = {}

        async def fake_explore_async_dpor(**kwargs: Any) -> MagicMock:
            captured_kwargs.update(kwargs)
            result = MagicMock()
            result.property_holds = True
            return result

        # Mock Django connections
        mock_connections = MagicMock()
        mock_conn = MagicMock()
        mock_connections.__getitem__ = MagicMock(return_value=mock_conn)
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        with (
            patch.dict(sys.modules, {"django": MagicMock(), "django.db": MagicMock()}),
            patch("frontrun.async_dpor.explore_async_dpor", fake_explore_async_dpor),
        ):
            # Force re-import to pick up mocks
            import importlib

            import frontrun.contrib.django._async as mod

            importlib.reload(mod)

            async def _run() -> None:
                await mod.async_django_dpor(
                    setup=lambda: None,
                    tasks=[],
                    invariant=lambda s: True,
                    lock_timeout=500,
                )

            # Patch the import inside the function
            with patch.object(mod, "__builtins__", {}):
                pass

            # Simpler approach: directly inspect the source code
            import inspect

            source = inspect.getsource(mod.async_django_dpor)
            # The call to explore_async_dpor should include lock_timeout=lock_timeout
            assert "lock_timeout=lock_timeout" in source, (
                "async_django_dpor does not forward lock_timeout to explore_async_dpor. "
                "The sync version (django_dpor) passes lock_timeout=lock_timeout explicitly."
            )


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


class TestDjangoAsyncCursorLeak:
    """Django async wrapper should use context manager for cursor."""

    def test_cursor_uses_context_manager(self) -> None:
        """The cursor should be managed with 'with' to prevent leaks on exception.

        Regression: cursor.execute() could raise, leaving cursor.close() uncalled.
        The sync version correctly uses 'with conn.cursor() as cursor:'.
        """
        import inspect

        from frontrun.contrib.django._async import async_django_dpor

        source = inspect.getsource(async_django_dpor)

        # The cursor should use a context manager like: with conn.cursor() as cursor:
        has_context_manager = "with conn.cursor()" in source

        # Buggy pattern: bare assignment without context manager
        has_bare_cursor = "cursor = conn.cursor()" in source

        assert not has_bare_cursor or has_context_manager, (
            "Django async wrapper uses bare 'cursor = conn.cursor()' without a context manager. "
            "If cursor.execute() raises, cursor.close() is never called (cursor leak). "
            "The sync version correctly uses 'with conn.cursor() as cursor:'."
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
