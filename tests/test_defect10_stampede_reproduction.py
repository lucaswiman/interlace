"""Defect #10: Stampede reproduction failure for memoize-style patterns.

After defect #9 added ``patch_redis_for_replay``, simple GET→SET races
reproduce 10/10.  However, Flask-Caching's ``@memoize`` decorator (and
similar patterns with heavy library overhead between function entry and
the first Redis command) still shows 0/10 reproduction.

Root cause: during DPOR exploration (``detect_io=True``), certain code is
filtered differently than during replay (``detect_io=False``).  The
``_is_dynamic_code`` filter skips dynamic code unconditionally with
``detect_io=True`` but conditionally with ``detect_io=False``, causing a
schedule length mismatch that misaligns the replay.

This test asserts that the memoize stampede pattern SHOULD reproduce.
Before the fix, it fails (0/10 reproduction).  After the fix, it
should pass (>0/10 reproduction).

Running::

    make test-integration-3.14 PYTEST_ARGS="-v -k test_defect10"
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
    """Defect #10: memoize-style stampede should reproduce after fix."""

    def test_simple_stampede_reproduces(self, redis_port: int) -> None:
        """Control: simple GET→compute→SET stampede reproduces (defect #9 fix)."""
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

        assert not result.property_holds, (
            "DPOR should detect cache stampede race "
            f"(explored {result.interleavings_explored} interleavings)"
        )
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

    def test_memoize_stampede_reproduces(self, redis_port: int) -> None:
        """Defect #10: Flask-Caching @memoize stampede SHOULD reproduce after fix.

        The @memoize decorator's _memoize_version() issues get_many/set_many
        via cachelib, creating a data-dependent code path.  Before the fix,
        this causes 0/10 reproduction.  After the fix, it should reproduce.
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

        # After defect #10 fix: reproduction should succeed.
        assert result.reproduction_successes > 0, (
            f"Memoize stampede should reproduce after defect #10 fix but got "
            f"{result.reproduction_successes}/{result.reproduction_attempts}"
        )
