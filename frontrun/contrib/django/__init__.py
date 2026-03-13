"""Django helpers for DPOR integration testing (sync and async)."""

from frontrun.contrib.django._async import async_django_dpor
from frontrun.contrib.django._sync import django_dpor

__all__ = ["async_django_dpor", "django_dpor"]
