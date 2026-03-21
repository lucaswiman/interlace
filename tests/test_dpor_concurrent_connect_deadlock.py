"""Regression test for Defect #15: DPOR deadlock when tracing django-reversion code.

django-reversion DPOR tests deadlock because tracing reversion code creates
a massive number of false conflict points.  Both threads execute
reversion.revisions code (signal handlers, version creation), and every
LOAD_GLOBAL / LOAD_ATTR in that code reports a READ on the *same* module
globals dict and class __dict__ objects.  DPOR treats these as real
conflicts, creating exponentially many backtrack points that drown out
the actual database-level race.

Without tracing reversion code, DPOR finds the ignore_duplicates race in
2 interleavings (~10 s).  With trace_packages=["reversion.*"], DPOR
explores 50+ interleavings in 60 s without ever reaching the critical
interleaving — effectively a deadlock from the user's perspective.
"""

from __future__ import annotations

import os

import pytest

try:
    import django
    from django.conf import settings as django_settings
except ImportError:
    pytest.skip("django not installed", allow_module_level=True)

try:
    import psycopg2  # noqa: F401
except ImportError:
    pytest.skip("psycopg2 not installed", allow_module_level=True)

try:
    import reversion  # noqa: F401
except ImportError:
    pytest.skip("django-reversion not installed", allow_module_level=True)

from frontrun.cli import require_active

_DB_NAME = os.environ.get("FRONTRUN_TEST_DB", "frontrun_test")

if not django_settings.configured:
    django_settings.configure(
        DATABASES={
            "default": {
                "ENGINE": "django.db.backends.postgresql",
                "NAME": _DB_NAME,
            }
        },
        INSTALLED_APPS=[
            "django.contrib.contenttypes",
            "django.contrib.auth",
            "reversion",
        ],
        DEFAULT_AUTO_FIELD="django.db.models.BigAutoField",
        SECRET_KEY="frontrun-test-secret-key-not-for-production",
    )
    django.setup()

import reversion as reversion_mod  # noqa: E402
from django.apps import apps  # noqa: E402
from django.db import connection, models  # noqa: E402
from reversion.models import Revision, Version  # noqa: E402

from frontrun.contrib.django import django_dpor  # noqa: E402


# ---------------------------------------------------------------------------
# Test model
# ---------------------------------------------------------------------------
class Article(models.Model):
    class Meta:
        app_label = "reversion"
        db_table = "defect15_article"

    title = models.CharField(max_length=200)
    content = models.TextField(default="")


app_config = apps.get_app_config("reversion")
app_config.models["article"] = Article
reversion_mod.register(Article, ignore_duplicates=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
_TABLES = [
    "reversion_version",
    "reversion_revision",
    "defect15_article",
    "auth_user_groups",
    "auth_user_user_permissions",
    "auth_user",
    "auth_group_permissions",
    "auth_group",
    "auth_permission",
    "django_content_type",
]


@pytest.fixture(scope="module")
def _pg_available():
    """Ensure Postgres is available and set up test tables."""
    try:
        connection.ensure_connection()
    except Exception:
        pytest.skip(f"Postgres not available at {_DB_NAME}")

    with connection.cursor() as cur:
        for tbl in _TABLES:
            cur.execute(f"DROP TABLE IF EXISTS {tbl} CASCADE")

    with connection.schema_editor() as editor:
        from django.contrib.auth import get_user_model
        from django.contrib.auth.models import Group, Permission
        from django.contrib.contenttypes.models import ContentType

        editor.create_model(ContentType)
        editor.create_model(Permission)
        editor.create_model(Group)
        editor.create_model(get_user_model())
        editor.create_model(Revision)
        editor.create_model(Version)
        editor.create_model(Article)

    yield

    with connection.cursor() as cur:
        for tbl in _TABLES:
            cur.execute(f"DROP TABLE IF EXISTS {tbl} CASCADE")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
class _State:
    def __init__(self) -> None:
        Article.objects.all().delete()
        Revision.objects.all().delete()
        Version.objects.all().delete()
        with reversion_mod.create_revision():
            self.article = Article.objects.create(title="Test Article", content="initial")
            reversion_mod.set_comment("Initial version")
        self.results: list[str | None] = [None, None]


def _make_thread_fn(idx: int):
    def _thread_fn(state: _State) -> None:
        try:
            with reversion_mod.create_revision():
                article = Article.objects.get(pk=state.article.pk)
                article.content = "updated"
                article.save()
                reversion_mod.set_comment(f"Edit by thread {idx}")
            state.results[idx] = "saved"
        except Exception as exc:
            state.results[idx] = f"error: {type(exc).__name__}: {exc}"

    return _thread_fn


def _invariant(state: _State) -> bool:
    if not (state.results[0] == "saved" and state.results[1] == "saved"):
        return True
    version_count = Version.objects.filter(object_id=str(state.article.pk)).count()
    return version_count <= 2


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestDporReversionDeadlock:
    """Defect #15: DPOR cannot find django-reversion races when tracing reversion code."""

    def test_dpor_finds_race_with_reversion_traced(self, _pg_available) -> None:
        """DPOR should find the ignore_duplicates TOCTOU race even when
        tracing reversion code.

        With trace_packages=["reversion.*"], both threads execute
        reversion.revisions code that touches the same module globals
        and class __dict__ objects.  Every LOAD_GLOBAL / LOAD_ATTR is
        reported as a shared-memory access, creating false conflicts
        that explode the backtrack tree.

        Without trace_packages, DPOR finds this race in 2 interleavings.
        With trace_packages, DPOR explores 50+ interleavings in 60 s
        and never reaches the critical schedule.
        """
        require_active("test_dpor_finds_race_with_reversion_traced")

        result = django_dpor(
            setup=_State,
            threads=[_make_thread_fn(0), _make_thread_fn(1)],
            invariant=_invariant,
            deadlock_timeout=15.0,
            timeout_per_run=30.0,
            total_timeout=60.0,
            trace_packages=["reversion.*"],
        )

        assert not result.property_holds, (
            f"DPOR should find the duplicate-version race, but false "
            f"conflict points from tracing reversion module globals / "
            f"class __dict__ prevent it from reaching the critical "
            f"interleaving (Defect #15).\n"
            f"num_explored={result.num_explored}"
        )
