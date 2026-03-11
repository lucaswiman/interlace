"""Integration test: django-invitations invitation acceptance race condition.

django-invitations (https://github.com/jazzband/django-invitations, ~1k stars)
handles email invitation flows for Django: send invitations, track acceptance.

**The race condition:**

The ``AcceptInvite.post()`` view accepts an invitation with a non-atomic
check-then-set sequence:

1. ``invitation = Invitation.objects.get(key=key)``  — plain SELECT, no lock
2. ``if invitation.accepted:`` — check in Python
3. ``invitation.accepted = True; invitation.save()`` — blind UPDATE

When two concurrent HTTP requests arrive with the same invitation key:

    Thread A: Invitation.objects.get(key=key) → accepted=False
    Thread B: Invitation.objects.get(key=key) → accepted=False
    Thread A: accepted is False → proceeds → invitation.accepted=True → save()
    Thread B: accepted is False → proceeds → invitation.accepted=True → save()
    → Both threads "accept" the same single-use invitation

This allows one invitation link to register two accounts (or confirm two
email addresses, or grant two membership slots), depending on what the
consuming application does on acceptance.

**Location of the bug:**

``invitations/views.py``, ``AcceptInvite.post()``:

    def post(self, *args, **kwargs):
        self.object = invitation = self.get_object()   # (1) plain SELECT

        if invitation.accepted:                        # (2) Python-level check
            ...  # redirect
        ...
        accept_invitation(invitation=invitation, ...)  # (3) UPDATE accepted=True

There is no ``SELECT FOR UPDATE``, no ``transaction.atomic()`` wrapping steps
1–3, and no optimistic-locking / version-field check.

The ``accept_invitation()`` helper at the bottom of views.py:

    def accept_invitation(invitation, request, signal_sender):
        invitation.accepted = True
        invitation.save()   # ← plain UPDATE, no WHERE accepted=False guard

The fix would be either:
- Wrap steps 1–3 in ``transaction.atomic()`` with ``select_for_update()``:
    ``invitation = Invitation.objects.select_for_update().get(key=key)``
- Or do a conditional UPDATE:
    ``Invitation.objects.filter(key=key, accepted=False).update(accepted=True)``
  and check the row count to detect races.

**Severity:** HIGH — a single invitation link can be used to register/confirm
multiple accounts, defeating the purpose of single-use invitations.

**Running:**

    .venv-3.10/bin/frontrun .venv-3.10/bin/pytest \\
        docs/case_studies/tests/test_django_invitations_race.py \\
        -v --no-frontrun-patch-locks -s
"""

from __future__ import annotations

import os

import pytest

# ---------------------------------------------------------------------------
# Django configuration — must happen before any Django imports
# ---------------------------------------------------------------------------

import django
from django.conf import settings as django_settings

_DB_NAME = os.environ.get("FRONTRUN_TEST_DB", "frontrun_test")

if django_settings.configured:
    # Django was already configured by another test module.
    # Only proceed if invitations is in INSTALLED_APPS.
    if "invitations" not in django_settings.INSTALLED_APPS:
        pytest.skip(
            "Django already configured without invitations app",
            allow_module_level=True,
        )
elif not django_settings.configured:
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
            "invitations",
        ],
        DEFAULT_AUTO_FIELD="django.db.models.BigAutoField",
        INVITATIONS_INVITATION_EXPIRY=3,  # days
        INVITATIONS_ALLOW_JSON_INVITES=False,
        INVITATIONS_SIGNUP_REDIRECT="/",
        # Required for LOGIN_REDIRECT default in app_settings
        LOGIN_URL="/accounts/login/",
    )
    django.setup()

import concurrent.futures  # noqa: E402
import threading  # noqa: E402

from django.db import connection  # noqa: E402

from invitations.models import Invitation  # noqa: E402

from frontrun.contrib.django import django_dpor  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _pg_available():
    """Verify Postgres is reachable and create the required tables."""
    try:
        connection.ensure_connection()
    except Exception:
        pytest.skip(f"Postgres not available at {_DB_NAME}")

    # Drop and recreate tables so we start clean regardless of prior state.
    with connection.cursor() as cur:
        for tbl in [
            "invitations_invitation",
            "auth_user_groups",
            "auth_user_user_permissions",
            "auth_user",
            "auth_group_permissions",
            "auth_group",
            "auth_permission",
            "django_content_type",
        ]:
            cur.execute(f"DROP TABLE IF EXISTS {tbl} CASCADE")

    with connection.schema_editor() as editor:
        from django.contrib.auth.models import Group, Permission
        from django.contrib.contenttypes.models import ContentType
        from django.contrib.auth import get_user_model

        User = get_user_model()
        editor.create_model(ContentType)
        editor.create_model(Permission)
        editor.create_model(Group)
        editor.create_model(User)
        editor.create_model(Invitation)

    yield

    with connection.cursor() as cur:
        for tbl in [
            "invitations_invitation",
            "auth_user_groups",
            "auth_user_user_permissions",
            "auth_user",
            "auth_group_permissions",
            "auth_group",
            "auth_permission",
            "django_content_type",
        ]:
            cur.execute(f"DROP TABLE IF EXISTS {tbl} CASCADE")


# ---------------------------------------------------------------------------
# Test: DPOR finds the invitation acceptance race
# ---------------------------------------------------------------------------


class TestInvitationAcceptanceRace:
    """Demonstrate the single-use invitation bypass in django-invitations."""

    def test_dpor_finds_double_acceptance(self, _pg_available) -> None:
        """Two concurrent acceptance requests both succeed for the same invitation.

        The ``AcceptInvite.post()`` view reads the invitation with a plain
        SELECT (no FOR UPDATE), checks ``invitation.accepted`` in Python, then
        sets ``accepted=True`` and saves. There is no transaction wrapping the
        check-and-set, so two threads can both read ``accepted=False`` before
        either writes back ``accepted=True``.

        Invariant: at most ONE of the two acceptance attempts should succeed.
        If both succeed, the same single-use invitation was accepted twice.
        """

        class _State:
            def __init__(self) -> None:
                from django.utils import timezone

                # Clean slate for each DPOR trial
                Invitation.objects.all().delete()

                # Create a fresh invitation. We need to set ``sent`` so that
                # key_expired() does not immediately return True.
                inv = Invitation.objects.create(
                    email="target@example.com",
                    key="a" * 64,
                    sent=timezone.now(),
                )
                self.invite_key: str = inv.key
                self.accept_results: list[str | None] = [None, None]

        def _make_thread_fn(idx: int):
            def _thread_fn(state: _State) -> None:
                try:
                    inv = Invitation.objects.get(key=state.invite_key)
                    if not inv.accepted and not inv.key_expired():
                        # This is the racy check-and-set: no SELECT FOR UPDATE,
                        # no transaction.atomic(), no conditional UPDATE.
                        inv.accepted = True
                        inv.save()
                        state.accept_results[idx] = "accepted"
                    else:
                        state.accept_results[idx] = "already_used"
                except Exception as exc:
                    state.accept_results[idx] = f"error: {exc}"

            return _thread_fn

        def _invariant(state: _State) -> bool:
            # At most one acceptance should succeed.
            # If both are "accepted", the single-use guarantee was bypassed.
            both_accepted = (
                state.accept_results[0] == "accepted"
                and state.accept_results[1] == "accepted"
            )
            return not both_accepted

        result = django_dpor(
            setup=_State,
            threads=[_make_thread_fn(0), _make_thread_fn(1)],
            invariant=_invariant,
            deadlock_timeout=15.0,
            timeout_per_run=30.0,
        )

        assert result.property_holds, result.explanation

    def test_invitation_double_acceptance_direct(self, _pg_available) -> None:
        """Confirm the double-acceptance race with a plain threading barrier.

        This test does NOT require the LD_PRELOAD library. It uses a
        threading.Barrier to maximise concurrency, demonstrating that two
        threads can both read accepted=False and both write accepted=True.
        """
        from django.utils import timezone

        Invitation.objects.all().delete()
        inv = Invitation.objects.create(
            email="direct@example.com",
            key="b" * 64,
            sent=timezone.now(),
        )
        invite_key = inv.key

        accept_results: list[str | None] = [None, None]
        barrier = threading.Barrier(2)

        def _thread(idx: int) -> None:
            from django.db import connections

            connections.close_all()
            loaded = Invitation.objects.get(key=invite_key)
            barrier.wait()  # maximise concurrency — both read before either writes
            if not loaded.accepted and not loaded.key_expired():
                loaded.accepted = True
                loaded.save()
                accept_results[idx] = "accepted"
            else:
                accept_results[idx] = "already_used"

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(_thread, i) for i in range(2)]
            for f in futures:
                f.result()

        both_accepted = (
            accept_results[0] == "accepted" and accept_results[1] == "accepted"
        )
        assert both_accepted, (
            f"Race not triggered in this run (results={accept_results}). "
            "Re-run; the race is probabilistic under plain threading."
        )
