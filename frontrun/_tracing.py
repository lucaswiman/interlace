"""Shared file-filtering logic for bytecode tracing.

Both ``bytecode.py`` (random exploration) and ``dpor.py`` (systematic DPOR)
need to distinguish user code from library/stdlib code.  This module
provides a single implementation so the filtering rules stay in sync.

The default behaviour skips stdlib, site-packages, and frontrun internals.
Users can widen the filter to include specific installed packages by
passing ``trace_packages`` patterns to the exploration APIs.  Patterns
use :func:`fnmatch.fnmatch` syntax (e.g. ``"django_*"``) and are matched
against the **module name** derived from the file path.
"""

from __future__ import annotations

import fnmatch
import os
import re
import sys
import threading
from collections.abc import Sequence

# Directories to never trace into (stdlib, site-packages)
_SKIP_DIRS: frozenset[str] = frozenset(p for p in sys.path if "lib/python" in p or "site-packages" in p)

_THREADING_FILE = threading.__file__

# Skip the entire frontrun package directory
_FRONTRUN_DIR = os.path.dirname(os.path.abspath(__file__)) + os.sep


def _filename_to_module(filename: str) -> str | None:
    """Convert a filename to a dotted module name, or None if not determinable.

    Strips the site-packages prefix and ``.py`` suffix, then converts
    path separators to dots.  Returns ``None`` for non-``.py`` files or
    files outside site-packages.
    """
    for skip_dir in _SKIP_DIRS:
        if filename.startswith(skip_dir):
            rel = filename[len(skip_dir) :]
            if rel.startswith(os.sep):
                rel = rel[1:]
            if rel.endswith(os.sep + "__init__.py"):
                rel = rel[: -(len(os.sep) + len("__init__.py"))]
            elif rel.endswith(".py"):
                rel = rel[:-3]
            else:
                # .so / .pyd etc — strip the extension
                dot = rel.rfind(".")
                if dot != -1:
                    rel = rel[:dot]
            return rel.replace(os.sep, ".")
    return None


class TraceFilter:
    """Configurable filter deciding which files should be traced.

    The default filter (``trace_packages=None``) traces only user code:
    files outside stdlib, site-packages, and frontrun internals.

    When ``trace_packages`` is provided, files in site-packages whose
    module names match any of the given patterns are **also** traced.
    Patterns use :func:`fnmatch.fnmatch` syntax (e.g. ``"django_*"``,
    ``"mylib.*"``).

    Example::

        # Trace user code + any django_* package + myapp.utils
        filt = TraceFilter(trace_packages=["django_*", "myapp.*"])
        filt.should_trace_file("/path/to/site-packages/django_filters/views.py")
        # -> True
    """

    def __init__(self, trace_packages: Sequence[str] | None = None) -> None:
        if trace_packages:
            # Compile fnmatch patterns to regexes for speed
            combined = "|".join(fnmatch.translate(p) for p in trace_packages)
            self._pattern: re.Pattern[str] | None = re.compile(combined)
        else:
            self._pattern = None

    def should_trace_file(self, filename: str) -> bool:
        """Check whether *filename* should be traced."""
        if filename == _THREADING_FILE:
            return False
        if filename.startswith("<frozen"):
            return False
        if filename.startswith(_FRONTRUN_DIR):
            return False

        # Check against skip dirs (stdlib / site-packages)
        in_skip_dir = False
        for skip_dir in _SKIP_DIRS:
            if filename.startswith(skip_dir):
                in_skip_dir = True
                break

        if not in_skip_dir:
            return True  # user code — always trace

        # File is in a skip dir.  Allow it only if it matches a trace_packages pattern.
        if self._pattern is not None:
            module_name = _filename_to_module(filename)
            if module_name is not None and self._pattern.match(module_name):
                return True

        return False


_DEFAULT_FILTER = TraceFilter()

# ---- active filter (context-managed) ----

_active_filter: threading.local = threading.local()


def set_active_trace_filter(filt: TraceFilter | None) -> None:
    """Set the active trace filter for the current thread.

    Pass ``None`` to reset to the default filter.
    """
    _active_filter.filter = filt


def get_active_trace_filter() -> TraceFilter:
    """Return the currently active trace filter."""
    return getattr(_active_filter, "filter", None) or _DEFAULT_FILTER


def should_trace_file(filename: str) -> bool:
    """Check whether a file is user code that should be traced.

    Delegates to the currently active :class:`TraceFilter`.
    """
    return get_active_trace_filter().should_trace_file(filename)


def is_dynamic_code(filename: str) -> bool:
    """Check whether a filename indicates dynamically generated code.

    Returns True for filenames like ``<string>``, ``<generated>``, ``<stdin>``
    that are produced by ``exec()``, ``compile()``, or interactive mode.
    Does NOT match ``<frozen ...>`` (already excluded by ``should_trace_file``).
    """
    return filename.startswith("<")
