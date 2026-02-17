================================================================================
Interlace Case Studies (Round 2): Concurrency Bug Detection in External Libraries
================================================================================

**A note on intent:** These case studies continue the work of the original
``docs/CASE_STUDIES.rst``.  As before, the goal is not to criticise these
libraries — concurrency is hard.  We picked codebases that are either
explicitly not thread-safe or that have specific hot-paths where thread safety
is not required, precisely because they provide crisp, reproducible examples of
what interlace can find.  The bugs shown here exist in real production code.

This document presents five additional case studies demonstrating how
**interlace** can find, reproduce, and test concurrency bugs by running
bytecode exploration directly against **unmodified library code**.

**All tests import directly from local source checkouts** (not from
site-packages) so that interlace's opcode-level tracer can reach the code.

**Total: 15 passing PoC tests across 5 libraries (3 per library).**

Run individual tests from ``docs/case_studies/tests/``::

    PYTHONPATH=interlace python docs/case_studies/tests/test_pybreaker_real.py
    PYTHONPATH=interlace python docs/case_studies/tests/test_urllib3_real.py
    PYTHONPATH=interlace python docs/case_studies/tests/test_sqlalchemy_pool_real.py
    PYTHONPATH=interlace python docs/case_studies/tests/test_amqtt_real.py
    PYTHONPATH=interlace python docs/case_studies/tests/test_pykka_real.py

----

Table of Contents
==================

1. `pybreaker`_ — Circuit breaker with unprotected fail counter
2. `urllib3`_ — Connection pool with unprotected connection counter
3. `SQLAlchemy (pool)`_ — Unlocked overflow counter in unlimited-overflow mode
4. `amqtt`_ — MQTT session with unprotected packet-ID generator
5. `pykka`_ — Actor framework with TOCTOU in ``tell()``

----

1. pybreaker
============

**Repository:** `pybreaker on GitHub <https://github.com/danielfm/pybreaker>`_

**Version tested:** 1.4.1

**What it does:** Python implementation of the Circuit Breaker pattern.
``CircuitMemoryStorage`` tracks failure and success counters for a breaker
stored entirely in process memory.

Bug: ``CircuitMemoryStorage.increment_counter()`` Lost Update (Critical)
------------------------------------------------------------------------

.. code-block:: python

    # pybreaker/__init__.py
    def increment_counter(self) -> None:
        self._fail_counter += 1   # ← not atomic!

The ``+=`` operator compiles to three separate bytecode instructions
(``LOAD_ATTR`` / ``BINARY_OP`` / ``STORE_ATTR`` in Python 3.11+).
Two threads calling ``increment_counter()`` concurrently can both read
the same ``_fail_counter`` value, both compute ``value + 1``, and one
write overwrites the other.

**Impact:** The circuit breaker under-counts failures.  A service that
is actually failing fast may never trip the breaker because its fail
counter stays permanently lower than the threshold.

Exploration Results
~~~~~~~~~~~~~~~~~~~

Bytecode exploration was run directly against the real
``CircuitMemoryStorage.increment_counter()`` method.

==============================  ===============================
Metric                          Result
==============================  ===============================
Seeds that found bug            **20 / 20**
Avg. attempts to find           **~2**
Deterministic reproduction      **10/10**
==============================  ===============================

Example output::

    Run 1: counter=1 [BUG]   ← should be 2
    ...
    Run 10: counter=1 [BUG]

**Test file:** ``docs/case_studies/tests/test_pybreaker_real.py``

----

2. urllib3
==========

**Repository:** `urllib3 on GitHub <https://github.com/urllib3/urllib3>`_

**Version tested:** main branch (post-2.x)

**What it does:** The foundational HTTP client library used by Requests and
many other Python projects.  ``HTTPConnectionPool`` manages a pool of
persistent connections to a single host.

Bug: ``HTTPConnectionPool._new_conn()`` Lost Update (Critical)
--------------------------------------------------------------

.. code-block:: python

    # urllib3/connectionpool.py
    def _new_conn(self) -> BaseHTTPConnection:
        self.num_connections += 1   # ← not atomic!
        conn = self.ConnectionCls(
            host=self._proxy_host or self.host,
            port=self.port,
            ...
        )
        return conn

``num_connections`` is a plain integer attribute incremented with ``+=``,
which is not atomic.  Two threads each calling ``_new_conn()`` (e.g. both
finding the pool empty and creating a new connection) can both read the same
counter, both write ``value + 1``, and one increment is lost.

**Impact:** ``num_connections`` is used for logging and debugging pool
behaviour.  An under-counted value makes it impossible to diagnose
connection-exhaustion issues.  If the counter is ever used for pool
limits, under-counting could allow more connections than intended.

**Note:** ``_new_conn()`` only instantiates the connection object; it does
not perform a TCP handshake.  No real server is needed for the test.

Exploration Results
~~~~~~~~~~~~~~~~~~~

==============================  ===============================
Metric                          Result
==============================  ===============================
Seeds that found bug            **20 / 20**
Avg. attempts to find           **~1.5**
Deterministic reproduction      **10/10**
==============================  ===============================

Example output::

    Run 1: num_connections=1 [BUG]   ← should be 2
    ...
    Run 10: num_connections=1 [BUG]

**Test file:** ``docs/case_studies/tests/test_urllib3_real.py``

----

3. SQLAlchemy (pool)
====================

**Repository:** `SQLAlchemy on GitHub <https://github.com/sqlalchemy/sqlalchemy>`_

**Version tested:** 2.x main branch

**What it does:** The most widely-used Python SQL toolkit and ORM.
``QueuePool`` (the default pool) maintains a fixed number of idle
connections plus an optional "overflow" of temporary connections.

Finding: ``QueuePool._inc_overflow()`` skips the lock in unlimited mode
-----------------------------------------------------------------------

.. note::

    After closer analysis this is **intentional behaviour**, not a bug.
    It is included here as a case study in how interlace can surface
    deliberate design decisions that look like races — and as a reminder
    to verify *impact* before filing an issue.

.. code-block:: python

    # sqlalchemy/pool/impl.py
    def _do_get(self) -> ConnectionPoolEntry:
        use_overflow = self._max_overflow > -1        # False when unlimited
        wait = use_overflow and self._overflow >= self._max_overflow
        ...
        if self._inc_overflow():
            return self._create_connection()

    def _inc_overflow(self) -> bool:
        if self._max_overflow == -1:
            self._overflow += 1   # ← no lock
            return True
        with self._overflow_lock:          # lock only needed for finite limit
            if self._overflow < self._max_overflow:
                self._overflow += 1
                return True
            return False

When ``max_overflow == -1`` (unlimited overflow), the method skips
``_overflow_lock`` and increments the counter bare.  Interlace detects this
and flags a lost-update race.

**Why the lock is deliberately omitted:**

* ``use_overflow = self._max_overflow > -1`` evaluates to ``False``, so
  ``_overflow`` is **never consulted** for pool admission control in unlimited
  mode.  The lock exists solely to protect the ``_overflow < _max_overflow``
  comparison; without a finite limit there is nothing to compare against.
* ``max_overflow`` cannot be changed at runtime (no setter; ``recreate()``
  constructs a new pool from original params), so the mode is fixed for the
  pool's lifetime.
* The only downstream consumers of ``_overflow`` in unlimited mode are the
  diagnostic methods ``overflow()`` and ``checkedout()``.  A transiently
  incorrect diagnostic counter is an acceptable trade-off for avoiding a
  contended lock on every connection checkout.

**What interlace tells us here:**

Interlace correctly identifies that the ``+=`` is non-atomic at the bytecode
level and that a concurrent schedule can produce a wrong count.  The
follow-up question — *does this wrong count affect correctness?* — requires
reading the call-site logic, which shows it does not.  This is a good
example of a **true race / no-impact** finding: interlace is right that the
memory model is unsound, but the library is also right that the impact is
bounded to a slightly stale diagnostic counter.

Exploration Results
~~~~~~~~~~~~~~~~~~~

==============================  ===============================
Metric                          Result
==============================  ===============================
Seeds that found race           **20 / 20**
Avg. attempts to find           **~2.5**
Deterministic reproduction      **10/10**
==============================  ===============================

Example output::

    Run 1: _overflow=-4 (expected -3) [BUG]
    ...
    Run 10: _overflow=-4 (expected -3) [BUG]

**Test file:** ``docs/case_studies/tests/test_sqlalchemy_pool_real.py``

----

4. amqtt
=========

**Repository:** `amqtt on GitHub <https://github.com/Yakifo/amqtt>`_

**Version tested:** main branch

**What it does:** An MQTT 3.1.1 broker and client library built on asyncio.
``Session`` represents a single MQTT client session and provides
``next_packet_id`` to allocate unique IDs for in-flight messages.

Bug: ``Session.next_packet_id`` Duplicate Packet IDs (Critical)
---------------------------------------------------------------

.. code-block:: python

    # amqtt/session.py
    @property
    def next_packet_id(self) -> int:
        self._packet_id = (self._packet_id % 65535) + 1   # WRITE
        limit = self._packet_id                             # READ (shared!)
        while self._packet_id in self.inflight_in or \
              self._packet_id in self.inflight_out:
            self._packet_id = (self._packet_id % 65535) + 1
            if self._packet_id == limit:
                raise AMQTTError("More than 65535 messages pending")
        return self._packet_id

No lock protects ``_packet_id``.  Two threads/tasks calling this property
concurrently on the same ``Session`` interleave their writes and reads of
``_packet_id``, and can both return the same integer.

**Race scenario (T1 starts from _packet_id = 0):**

::

    T1: _packet_id = 1                 (writes 1)
    T2: _packet_id = 2  (reads T1's 1, writes 2)
    T1: limit = self._packet_id → 2!  (reads T2's 2)
    T1: returns 2
    T2: limit = self._packet_id → 2
    T2: returns 2
    ⇒ Both threads receive packet ID 2 — a duplicate!

**Impact:** The MQTT protocol requires in-flight QoS 1/2 messages to have
unique packet IDs.  Duplicate IDs cause ``PUBACK`` / ``PUBREC`` messages
to be associated with the wrong publish, silently corrupting message
delivery.  This is a protocol-level data-corruption bug.

Exploration Results
~~~~~~~~~~~~~~~~~~~

==============================  ============================================
Metric                          Result
==============================  ============================================
Seeds that found bug            **20 / 20**
Avg. attempts to find           **1** (every seed, every first attempt!)
Deterministic reproduction      **10/10**
==============================  ============================================

Example output::

    Run 1: thread1_id=2, thread2_id=2 [BUG]   ← both got the same ID!
    ...
    Run 10: thread1_id=2, thread2_id=2 [BUG]

The property's code path is so short and the race window so wide that the
**very first random schedule tried always triggers it** across all seeds.

**Test file:** ``docs/case_studies/tests/test_amqtt_real.py``

----

5. pykka
=========

**Repository:** `pykka on GitHub <https://github.com/jodal/pykka>`_

**Version tested:** 4.4.1

**What it does:** A Python actor framework providing thread-based actors.
Each actor runs in its own thread and communicates via message-passing.
``ActorRef.tell()`` sends a fire-and-forget message to an actor.

Bug: ``ActorRef.tell()`` TOCTOU — Ghost Messages (High)
-------------------------------------------------------

.. code-block:: python

    # pykka/_ref.py
    def tell(self, message: Any) -> None:
        if not self.is_alive():                             # CHECK
            raise ActorDeadError(f"{self} is dead")
        self.actor_inbox.put(                               # ACT
            Envelope(message=message, reply_to=None),
            block=False,
        )

There is no lock between the ``is_alive()`` check (which reads
``actor_stopped.is_set()``) and the ``actor_inbox.put()``.  A concurrent
``stop()`` call can set ``actor_stopped`` in the window between the check
and the put.

When this race fires:

1. ``tell()`` checks ``is_alive()`` — returns **True** (actor not yet stopped).
2. Another thread calls ``stop()`` — actor loop processes ``_ActorStop``,
   calls ``actor_stopped.set()``, and exits its loop.
3. ``tell()`` puts the message into the inbox — succeeds without raising
   ``ActorDeadError``.
4. **Nobody reads the message.** The actor loop has exited; the inbox is
   never drained again.  The message is silently lost — a *ghost message*.

**Impact:** From the caller's perspective ``tell()`` returned successfully,
implying the message was delivered.  In reality the message is silently
dropped.  This violates the implicit contract that a non-raising ``tell()``
guarantees delivery to a live actor.

Test Design
~~~~~~~~~~~

We use a shared ``received`` list populated by the actor's ``on_receive``
hook, plus a ``tell_successes`` counter.  Thread 2 calls
``stop(block=True)`` which blocks until the actor loop has fully exited,
making ``received`` a final, stable snapshot.  The invariant:

.. code-block:: python

    tell_successes == len(received)

A violation (``tell_successes=1, received=[]``) means ``tell()`` returned
without an error but the actor never processed the message.

Exploration Results
~~~~~~~~~~~~~~~~~~~

==============================  ===============================
Metric                          Result
==============================  ===============================
Seeds that found bug            **20 / 20**
Avg. attempts to find           **~17**
Deterministic reproduction      **10/10**
==============================  ===============================

Example output::

    Run 1: tell_successes=1, received=0 [BUG]
    ...
    Run 10: tell_successes=1, received=0 [BUG]

The pykka test requires more attempts on average (~17 vs ~2 for simple
lost-update bugs) because the race window is narrower: the stop() must
complete its actor-loop shutdown *between* the ``is_alive()`` check and the
``actor_inbox.put()`` call inside ``tell()``.

**Test file:** ``docs/case_studies/tests/test_pykka_real.py``

----

Summary
=======

=================  ====================================  =======  ==================  ==============  =========  =========
Library            Finding                               Version  Seeds Found (/ 20)  Avg. Attempts   Reproduce  Is a bug?
=================  ====================================  =======  ==================  ==============  =========  =========
pybreaker          ``increment_counter`` lost update     1.4.1    **20 / 20**         ~2              10/10      **Yes**
urllib3            ``_new_conn`` lost update             2.x      **20 / 20**         ~1.5            10/10      **Yes**
SQLAlchemy pool    ``_inc_overflow`` race (diagnostic)   2.x      **20 / 20**         ~2.5            10/10      Intentional
amqtt              ``next_packet_id`` duplicate IDs      main     **20 / 20**         **1**           10/10      **Yes**
pykka              ``tell()`` TOCTOU ghost messages      4.4.1    **20 / 20**         ~17             10/10      **Yes**
=================  ====================================  =======  ==================  ==============  =========  =========

Key Findings
============

1. **Lost-update bugs in plain ``+=`` are universal and easy to find.**
   pybreaker, urllib3, and SQLAlchemy all share the same root cause: a
   ``self.x += 1`` in a hot path without a lock.  interlace finds all three
   on literally the first or second attempt.  Any Python library that
   maintains a counter without a lock is vulnerable.

2. **Protocol-level bugs are the most severe and the easiest to detect.**
   amqtt's ``next_packet_id`` duplicates an MQTT packet ID — a
   protocol-correctness violation — and interlace finds it on **every single
   seed on the very first attempt**.  The race window spans the entire
   property body.

3. **TOCTOU in higher-level APIs is harder to detect but still reliably
   found.**  pykka's ``tell()`` ghost-message bug requires a specific
   three-phase interleaving (check → stop → put).  interlace still finds it
   in 20/20 seeds, taking ~17 attempts on average — more than the simple
   counter bugs, but well within practical reach.

4. **Deterministic reproduction is 100% reliable across all five libraries.**
   Once a counterexample schedule is found, ``run_with_schedule`` reproduces
   the bug 10/10 times for every library tested.

5. **Testing unmodified library code works without any source modifications.**
   All five tests run against production library code cloned from GitHub.
   The only setup required is adding the library's source directory to
   ``sys.path`` so interlace's tracer can reach it.

6. **Interlace finds races correctly; impact assessment requires call-site
   reading.**  The SQLAlchemy ``_inc_overflow()`` finding is a real
   non-atomic operation, but the counter it corrupts is only ever used for
   diagnostic reporting in unlimited-overflow mode — the pool's admission
   logic explicitly bypasses it.  Interlace does its job; the follow-up
   question of *whether the race matters* still requires human analysis.
   A useful heuristic: if the racy counter only feeds logging or monitoring
   endpoints and cannot be changed at runtime, a lost update is probably
   a deliberate performance trade-off rather than a bug.
