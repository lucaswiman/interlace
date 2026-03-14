"""Tests for configurable trace filtering (TraceFilter, trace_packages)."""

from __future__ import annotations

import os

import pytest

from frontrun._tracing import (
    _SKIP_DIRS,
    TraceFilter,
    _filename_to_module,
    get_active_trace_filter,
    set_active_trace_filter,
    should_trace_file,
)

# ---------------------------------------------------------------------------
# Unit tests for _filename_to_module
# ---------------------------------------------------------------------------


class TestFilenameToModule:
    def test_returns_none_for_user_code(self, tmp_path: object) -> None:
        assert _filename_to_module("/home/user/myproject/app.py") is None

    def test_converts_site_packages_path(self) -> None:
        # Pick a real site-packages dir from _SKIP_DIRS
        sp_dir = next((d for d in _SKIP_DIRS if "site-packages" in d), None)
        if sp_dir is None:
            pytest.skip("no site-packages in _SKIP_DIRS")
        fake = os.path.join(sp_dir, "django_filters", "views.py")
        assert _filename_to_module(fake) == "django_filters.views"

    def test_converts_init_py(self) -> None:
        sp_dir = next((d for d in _SKIP_DIRS if "site-packages" in d), None)
        if sp_dir is None:
            pytest.skip("no site-packages in _SKIP_DIRS")
        fake = os.path.join(sp_dir, "django_filters", "__init__.py")
        assert _filename_to_module(fake) == "django_filters"

    def test_converts_nested_module(self) -> None:
        sp_dir = next((d for d in _SKIP_DIRS if "site-packages" in d), None)
        if sp_dir is None:
            pytest.skip("no site-packages in _SKIP_DIRS")
        fake = os.path.join(sp_dir, "django", "contrib", "sites", "models.py")
        assert _filename_to_module(fake) == "django.contrib.sites.models"


# ---------------------------------------------------------------------------
# Unit tests for TraceFilter
# ---------------------------------------------------------------------------


class TestTraceFilter:
    def test_default_traces_user_code(self) -> None:
        filt = TraceFilter()
        assert filt.should_trace_file("/home/user/myproject/app.py") is True

    def test_default_skips_site_packages(self) -> None:
        filt = TraceFilter()
        sp_dir = next((d for d in _SKIP_DIRS if "site-packages" in d), None)
        if sp_dir is None:
            pytest.skip("no site-packages in _SKIP_DIRS")
        fake = os.path.join(sp_dir, "django_filters", "views.py")
        assert filt.should_trace_file(fake) is False

    def test_trace_packages_allows_matching_site_packages(self) -> None:
        filt = TraceFilter(trace_packages=["django_*"])
        sp_dir = next((d for d in _SKIP_DIRS if "site-packages" in d), None)
        if sp_dir is None:
            pytest.skip("no site-packages in _SKIP_DIRS")
        fake = os.path.join(sp_dir, "django_filters", "views.py")
        assert filt.should_trace_file(fake) is True

    def test_trace_packages_rejects_non_matching(self) -> None:
        filt = TraceFilter(trace_packages=["django_*"])
        sp_dir = next((d for d in _SKIP_DIRS if "site-packages" in d), None)
        if sp_dir is None:
            pytest.skip("no site-packages in _SKIP_DIRS")
        fake = os.path.join(sp_dir, "requests", "api.py")
        assert filt.should_trace_file(fake) is False

    def test_trace_packages_dot_star_pattern(self) -> None:
        filt = TraceFilter(trace_packages=["django.contrib.sites.*"])
        sp_dir = next((d for d in _SKIP_DIRS if "site-packages" in d), None)
        if sp_dir is None:
            pytest.skip("no site-packages in _SKIP_DIRS")
        fake = os.path.join(sp_dir, "django", "contrib", "sites", "models.py")
        assert filt.should_trace_file(fake) is True
        # But django.contrib.auth should NOT match
        fake2 = os.path.join(sp_dir, "django", "contrib", "auth", "models.py")
        assert filt.should_trace_file(fake2) is False

    def test_always_skips_threading(self) -> None:
        import threading

        filt = TraceFilter(trace_packages=["*"])
        assert filt.should_trace_file(threading.__file__) is False

    def test_always_skips_frozen(self) -> None:
        filt = TraceFilter(trace_packages=["*"])
        assert filt.should_trace_file("<frozen importlib._bootstrap>") is False

    def test_always_skips_frontrun(self) -> None:
        from frontrun._tracing import _FRONTRUN_DIR

        filt = TraceFilter(trace_packages=["*"])
        assert filt.should_trace_file(os.path.join(_FRONTRUN_DIR, "dpor.py")) is False

    def test_multiple_patterns(self) -> None:
        filt = TraceFilter(trace_packages=["django_*", "requests.*"])
        sp_dir = next((d for d in _SKIP_DIRS if "site-packages" in d), None)
        if sp_dir is None:
            pytest.skip("no site-packages in _SKIP_DIRS")
        assert filt.should_trace_file(os.path.join(sp_dir, "django_filters", "views.py")) is True
        assert filt.should_trace_file(os.path.join(sp_dir, "requests", "api.py")) is True
        assert filt.should_trace_file(os.path.join(sp_dir, "flask", "app.py")) is False


# ---------------------------------------------------------------------------
# Unit tests for active filter context management
# ---------------------------------------------------------------------------


class TestActiveFilter:
    def test_default_active_filter(self) -> None:
        filt = get_active_trace_filter()
        assert isinstance(filt, TraceFilter)

    def test_set_and_reset(self) -> None:
        custom = TraceFilter(trace_packages=["mylib.*"])
        set_active_trace_filter(custom)
        try:
            assert get_active_trace_filter() is custom
        finally:
            set_active_trace_filter(None)
        # After reset, should be the default again
        assert get_active_trace_filter() is not custom

    def test_should_trace_file_uses_active_filter(self) -> None:
        sp_dir = next((d for d in _SKIP_DIRS if "site-packages" in d), None)
        if sp_dir is None:
            pytest.skip("no site-packages in _SKIP_DIRS")
        fake = os.path.join(sp_dir, "mylib", "core.py")

        # Default: should NOT trace
        assert should_trace_file(fake) is False

        # With custom filter: should trace
        set_active_trace_filter(TraceFilter(trace_packages=["mylib.*"]))
        try:
            assert should_trace_file(fake) is True
        finally:
            set_active_trace_filter(None)

        # After reset: should NOT trace again
        assert should_trace_file(fake) is False
