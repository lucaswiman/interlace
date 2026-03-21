"""Regression test for Defect #15: DPOR deadlock on concurrent psycopg2.connect().

When DPOR threads each need to establish a new PostgreSQL connection (via
psycopg2.connect() or Django's ensure_connection() / get_new_connection()),
both threads can deadlock inside the C-level psycopg2.connect() call.
The DPOR scheduler cannot make progress because psycopg2.connect() is a
single C function call — there are no Python-level scheduling points
inside it, and the IO interception layer cannot properly interleave two
concurrent connection handshakes.

This test reproduces the deadlock using django-reversion's create_revision()
pattern, where each thread's ORM operations trigger per-thread connection
establishment and concurrent SQL execution during DPOR exploration.

The deadlock manifests when:
1. Using explore_dpor() directly (without django_dpor's per-thread
   connection pre-establishment)
2. trace_packages includes reversion code, creating scheduling points
   inside the django-reversion signal handlers
3. Both threads call create_revision() with ORM operations that
   trigger per-thread connection establishment

With these conditions, DPOR deadlocks — both threads get stuck inside
C-level code (psycopg2.connect() or cursor.execute()), and the DPOR
scheduler cannot make progress.
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
from django.db import connection, connections, models  # noqa: E402
from reversion.models import Revision, Version  # noqa: E402

from frontrun.dpor import explore_dpor  # noqa: E402


# ---------------------------------------------------------------------------
# Test model — registered with ignore_duplicates=True
# Use 'reversion' as app_label so ContentType.model_class() can resolve it
# back to this class via Django's app registry.
# ---------------------------------------------------------------------------
class Article(models.Model):
    class Meta:
        app_label = "reversion"
        db_table = "defect15_article"

    title = models.CharField(max_length=200)
    content = models.TextField(default="")


# Register the model in Django's app registry so ContentType.model_class()
# can find it (needed by django-reversion's ignore_duplicates feature).
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
    """Verify Postgres is reachable and create the required tables."""
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

        user_model = get_user_model()
        editor.create_model(ContentType)
        editor.create_model(Permission)
        editor.create_model(Group)
        editor.create_model(user_model)
        editor.create_model(Revision)
        editor.create_model(Version)
        editor.create_model(Article)

    yield

    with connection.cursor() as cur:
        for tbl in _TABLES:
            cur.execute(f"DROP TABLE IF EXISTS {tbl} CASCADE")


# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------
def _make_setup_cls():
    class _State:
        def __init__(self) -> None:
            Article.objects.all().delete()
            Revision.objects.all().delete()
            Version.objects.all().delete()
            with reversion_mod.create_revision():
                self.article = Article.objects.create(title="Test Article", content="initial")
                reversion_mod.set_comment("Initial version")
            self.results: list[str | None] = [None, None]

    return _State


def _make_thread_fn(idx: int):
    def _thread_fn(state):  # type: ignore[no-untyped-def]
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


def _invariant(state) -> bool:  # type: ignore[no-untyped-def]
    both_saved = state.results[0] == "saved" and state.results[1] == "saved"
    if not both_saved:
        return True
    version_count = Version.objects.filter(
        object_id=str(state.article.pk),
    ).count()
    return version_count <= 2


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestDporConcurrentConnectDeadlock:
    """Regression test for Defect #15: DPOR deadlock on concurrent psycopg2.connect().

    The bug manifests when DPOR threads need to establish PostgreSQL
    connections during exploration. The threads get stuck inside C-level
    psycopg2.connect() or cursor.execute() calls, and the DPOR scheduler
    cannot make progress.
    """

    def test_dpor_deadlocks_without_preestablished_connections(self, _pg_available) -> None:
        """DPOR deadlocks when threads must establish their own connections.

        Uses explore_dpor() directly (NOT django_dpor()) so threads don't
        get pre-established connections, and trace_packages includes
        reversion code. Both factors combine to trigger the deadlock:
        - Without pre-established connections, psycopg2.connect() is called
          during DPOR exploration
        - Tracing reversion code creates scheduling points inside
          django-reversion's signal handlers

        Expected (once fixed): DPOR should explore interleavings and detect
        the race condition where ignore_duplicates is bypassed.

        Bug (Defect #15): DPOR deadlocks — both threads get stuck inside
        C-level code, and DPOR cannot complete exploration or find the race
        within the timeout.
        """
        require_active("test_dpor_deadlocks_without_preestablished_connections")

        setup_cls = _make_setup_cls()

        # Wrap setup to close connections — forces threads to establish
        # their own connections via psycopg2.connect() during exploration.
        _orig_init = setup_cls.__init__

        def _init_then_close(self):  # type: ignore[no-untyped-def]
            connections.close_all()
            connection.ensure_connection()
            _orig_init(self)
            connections.close_all()

        setup_cls.__init__ = _init_then_close  # type: ignore[method-assign]

        def _make_thread_fn_with_close(idx: int):
            def _thread_fn(state) -> None:  # type: ignore[no-untyped-def]
                conn = connections["default"]
                conn.close()
                try:
                    with reversion_mod.create_revision():
                        article = Article.objects.get(pk=state.article.pk)
                        article.content = "updated"
                        article.save()
                        reversion_mod.set_comment(f"Edit by thread {idx}")
                    state.results[idx] = "saved"
                except Exception as exc:
                    state.results[idx] = f"error: {type(exc).__name__}: {exc}"
                finally:
                    conn.close()

            return _thread_fn

        result = explore_dpor(
            setup=setup_cls,
            threads=[_make_thread_fn_with_close(0), _make_thread_fn_with_close(1)],
            invariant=_invariant,
            detect_io=True,
            deadlock_timeout=15.0,
            timeout_per_run=30.0,
            total_timeout=60.0,
            trace_packages=["reversion.*"],
        )

        # With the deadlock bug, DPOR cannot effectively explore
        # interleavings. Even if some executions complete (by timing
        # out the deadlock), the race condition is never found because
        # the critical interleavings are blocked by the deadlock.
        assert not result.property_holds, (
            f"DPOR should find the duplicate-version race condition, but "
            f"the concurrent psycopg2.connect() deadlock (Defect #15) "
            f"prevents DPOR from reaching the critical interleaving.\n"
            f"num_explored={result.num_explored}\n"
            f"property_holds={result.property_holds}"
        )
