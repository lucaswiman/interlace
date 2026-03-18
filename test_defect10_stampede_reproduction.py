"""Defect #10: Stampede reproduction failure — partially resolved by defect #9 fix.

After defect #9 added ``patch_redis_for_replay``, simple GET→increment→SET
races reproduce 10/10.  Most "stampede" patterns now also reproduce correctly.

However, one residual failure pattern remains: Flask-Caching's ``@memoize``
decorator still shows 0/10 reproduction.  The ``@memoize`` decorator differs
from ``@cached`` because it calls ``_memoize_version()`` which issues
``get_many()`` / ``set_many()`` (via cachelib's Redis backend) to manage
version keys.  This creates a data-dependent code path: on first call the
version doesn't exist, so ``set_many`` runs (extra Redis pipeline + bytecodes);
on subsequent calls, the version exists and ``set_many`` is skipped.

During DPOR exploration (with ``detect_io=True``), the schedule records
scheduling points at I/O boundaries (Redis commands).  During replay (with
``detect_io=False``), the bytecode tracer adds scheduling points at every
opcode.  The short I/O-level schedule is consumed by the first few bytecodes
of the library overhead (Flask app context, decorator logic, version check),
and the ``_extend_schedule`` round-robin padding does not enforce the critical
interleaving at the Redis command level.

The @cached decorator (which does NOT have version checking) reproduces 10/10,
confirming the issue is specific to memoize's ``_memoize_version`` pattern.

This test verifies:
1. Simple stampede patterns reproduce correctly (defect #9 fix works)
2. The defect #10 memoize pattern still fails to reproduce (0/10)

Running::

    REDIS_PORT=16399 frontrun python -m pytest \\
      frontrun-bugs/tests/test_defect10_stampede_reproduction.py -v
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time

import pytest

try:
    import redis as redis_lib
except ImportError:
    pytest.skip("redis package not installed", allow_module_level=True)

from frontrun.dpor import explore_dpor

_REDIS_PORT = int(os.environ.get("REDIS_PORT", "16399"))


@pytest.fixture(scope="module")
def redis_port():
    """Provide a Redis port, starting a server if needed."""
    r = redis_lib.Redis(port=_REDIS_PORT)
    try:
        r.ping()
        r.close()
        yield _REDIS_PORT
        return
    except redis_lib.ConnectionError:
        r.close()

    if not shutil.which("redis-server"):
        pytest.skip("redis-server not available")

    proc = subprocess.Popen(
        ["redis-server", "--port", str(_REDIS_PORT), "--save", "", "--appendonly", "no", "--loglevel", "warning"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(0.5)
    r = redis_lib.Redis(port=_REDIS_PORT)
    try:
        r.ping()
    except redis_lib.ConnectionError:
        proc.kill()
        pytest.skip("Could not start redis-server")
    finally:
        r.close()
    yield _REDIS_PORT
    proc.terminate()
    proc.wait(timeout=5)


class TestStampedeReproduction:
    """Defect #10: stampede reproduction after defect #9 fix.

    Simple stampede (GET miss → compute → SET) now reproduces 10/10.
    The memoize pattern (with version checking) still fails 0/10.
    """

    def test_simple_stampede_reproduces(self, redis_port: int) -> None:
        """Simple GET→compute→SET stampede reproduces (defect #9 fix works)."""
        port = redis_port

        class State:
            def __init__(self) -> None:
                r = redis_lib.Redis(port=port, decode_responses=True)
                r.delete("defect10:cache", "defect10:compute_count")
                r.set("defect10:compute_count", "0")
                r.close()

        def get_or_compute(state: State) -> None:
            r = redis_lib.Redis(port=port, decode_responses=True)
            cached = r.get("defect10:cache")
            if cached is None:
                r.incr("defect10:compute_count")
                # Some computation between GET and SET
                result = sum(i * 3 for i in range(50))
                r.set("defect10:cache", str(result))
            r.close()

        def invariant(state: State) -> bool:
            r = redis_lib.Redis(port=port, decode_responses=True)
            count = int(r.get("defect10:compute_count"))  # type: ignore[arg-type]
            r.close()
            return count == 1

        result = explore_dpor(
            setup=State,
            threads=[get_or_compute, get_or_compute],
            invariant=invariant,
            detect_io=True,
            max_executions=50,
            deadlock_timeout=15.0,
            reproduce_on_failure=10,
        )

        # DPOR detects the stampede.
        assert not result.property_holds, (
            "DPOR should detect cache stampede race "
            f"(explored {result.interleavings_explored} interleavings)"
        )
        # Defect #9 fix: reproduction now succeeds.
        assert result.reproduction_successes > 0, (
            f"Simple stampede should reproduce (defect #9 fix) but got "
            f"{result.reproduction_successes}/{result.reproduction_attempts}"
        )

    def test_simple_counter_reproduces(self, redis_port: int) -> None:
        """Control: simple GET→increment→SET reproduces (defect #9 regression)."""
        port = redis_port

        class State:
            def __init__(self) -> None:
                r = redis_lib.Redis(port=port, decode_responses=True)
                r.set("defect10:counter", "0")
                r.close()

        def increment(state: State) -> None:
            r = redis_lib.Redis(port=port, decode_responses=True)
            val = int(r.get("defect10:counter"))  # type: ignore[arg-type]
            r.set("defect10:counter", str(val + 1))
            r.close()

        def invariant(state: State) -> bool:
            r = redis_lib.Redis(port=port, decode_responses=True)
            result = int(r.get("defect10:counter"))  # type: ignore[arg-type]
            r.close()
            return result == 2

        result = explore_dpor(
            setup=State,
            threads=[increment, increment],
            invariant=invariant,
            detect_io=True,
            max_executions=50,
            deadlock_timeout=15.0,
            reproduce_on_failure=10,
        )

        assert not result.property_holds, "DPOR should detect lost-update on Redis counter"
        assert result.reproduction_successes > 0, (
            f"Simple counter should reproduce (defect #9 fix) but got "
            f"{result.reproduction_successes}/{result.reproduction_attempts}"
        )

    def test_memoize_stampede_fails_to_reproduce(self, redis_port: int) -> None:
        """Defect #10 residual: Flask-Caching @memoize stampede fails 0/10.

        The @memoize decorator's _memoize_version() issues get_many/set_many
        via cachelib, creating a data-dependent code path that causes the
        replay schedule to misalign with the actual bytecode execution.
        """
        port = redis_port

        try:
            from flask import Flask
            from flask_caching import Cache
        except ImportError:
            pytest.skip("flask-caching not installed")

        class _State:
            def __init__(self) -> None:
                r = redis_lib.Redis(port=port, decode_responses=True)
                r.flushdb()
                r.close()
                self.app = Flask(__name__)
                self.app.config["CACHE_TYPE"] = "RedisCache"
                self.app.config["CACHE_REDIS_PORT"] = port
                self.app.config["CACHE_DEFAULT_TIMEOUT"] = 60
                self.app.config["CACHE_KEY_PREFIX"] = "test_d10_memo_"
                self.cache = Cache(self.app)

                @self.cache.memoize(timeout=60)
                def _expensive_compute(x: int) -> int:
                    r = redis_lib.Redis(port=port, decode_responses=True)
                    r.incr("d10_memo_count")
                    r.close()
                    return x * 2

                self.expensive_compute = _expensive_compute

        def _make_fn(i: int):
            def fn(s: _State) -> None:
                with s.app.app_context():
                    s.expensive_compute(42)
            return fn

        def _inv(s: _State) -> bool:
            r = redis_lib.Redis(port=port, decode_responses=True)
            count = int(r.get("d10_memo_count"))  # type: ignore[arg-type]
            r.close()
            return count <= 1

        result = explore_dpor(
            setup=_State,
            threads=[_make_fn(0), _make_fn(1)],
            invariant=_inv,
            detect_io=True,
            max_executions=50,
            deadlock_timeout=15.0,
            reproduce_on_failure=10,
        )

        # DPOR should detect the stampede.
        assert not result.property_holds, (
            "DPOR should detect memoize stampede "
            f"(explored {result.interleavings_explored} interleavings)"
        )

        # Defect #10: reproduction fails for the memoize pattern.
        assert result.reproduction_successes == 0, (
            f"Expected 0/10 reproduction (defect #10) but got "
            f"{result.reproduction_successes}/{result.reproduction_attempts}. "
            f"If this passes, defect #10 may be fixed!"
        )
