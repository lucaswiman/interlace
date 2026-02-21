"""Pytest plugin that patches threading primitives before test collection.

Register via the ``pytest11`` entry point so that ``patch_locks()`` runs
in ``pytest_configure`` — *before* pytest imports any test modules or
conftest files.  This ensures that libraries which create module-level
``threading.Lock()`` instances at import time get cooperative versions
instead of real ones.

Patching is **on by default**.  To disable::

    pytest --no-frontrun-patch-locks

On free-threaded Python (3.13t+), global patching is **off by default**
because zombie daemon threads left by intentional-deadlock tests can
hold cooperative locks that spin in Python, triggering ``sys.monitoring``
callbacks during tool-ID teardown/re-registration and causing CPython
internal mutex deadlocks.  Per-test patching via ``BytecodeShuffler`` /
``DporBytecodeRunner`` is unaffected because those runners join their
threads before returning.
"""

from __future__ import annotations

import sysconfig
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest

_FREE_THREADED = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))


def _should_patch(config: pytest.Config) -> bool:
    """Return True if global cooperative lock patching should be active."""
    if config.getoption("--no-frontrun-patch-locks", default=False):
        return False
    if _FREE_THREADED:
        return False
    return True


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("frontrun", "Frontrun concurrency testing")
    group.addoption(
        "--no-frontrun-patch-locks",
        action="store_true",
        default=False,
        help="Disable automatic cooperative lock patching (enabled by default).",
    )


def pytest_configure(config: pytest.Config) -> None:
    if not _should_patch(config):
        return

    from frontrun._cooperative import patch_locks

    patch_locks()


def pytest_unconfigure(config: pytest.Config) -> None:
    if not _should_patch(config):
        return

    from frontrun._cooperative import unpatch_locks

    unpatch_locks()
