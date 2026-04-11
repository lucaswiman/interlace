"""Shared helpers for Django contrib adapters."""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from typing import Any, TypeVar

T = TypeVar("T")

#: Default ``trace_packages`` for Django projects.  Traces code inside any
#: ``django_*`` third-party app (e.g. ``django_filters``, ``django_rest_framework``)
#: and ``django.contrib.sites`` submodules, since these commonly participate
#: in race conditions visible during integration tests.
DJANGO_TRACE_PACKAGES: list[str] = ["django_*", "django.contrib.sites.*"]


def wrap_setup(setup: Callable[[], T], close_all: Callable[[], None]) -> Callable[[], T]:
    """Return a setup wrapper that clears all Django connections first."""

    def wrapped_setup() -> T:
        close_all()
        return setup()

    return wrapped_setup


def wrap_sync_thread(
    fn: Callable[[T], None],
    *,
    connections: Any,
    db_alias: str,
    lock_timeout: int | None,
) -> Callable[[T], None]:
    """Return a thread wrapper that opens a fresh Django connection."""

    def wrapper(state: T) -> None:
        conn = connections[db_alias]
        conn.close()
        conn.ensure_connection()
        if lock_timeout is not None:
            with conn.cursor() as cursor:
                cursor.execute(f"SET lock_timeout = '{int(lock_timeout)}ms'")
        try:
            fn(state)
        finally:
            conn.close()

    return wrapper


def wrap_async_task(
    fn: Callable[[T], Coroutine[Any, Any, None]],
    *,
    connections: Any,
    db_alias: str,
    lock_timeout: int | None,
) -> Callable[[T], Coroutine[Any, Any, None]]:
    """Return a task wrapper that opens a fresh Django connection."""

    async def wrapper(state: T) -> None:
        conn = connections[db_alias]
        conn.close()
        conn.ensure_connection()
        if lock_timeout is not None:
            with conn.cursor() as cursor:
                cursor.execute(f"SET lock_timeout = '{int(lock_timeout)}ms'")
        try:
            await fn(state)
        finally:
            conn.close()

    return wrapper
