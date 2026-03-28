Changelog
=========

All releases: https://github.com/lucaswiman/frontrun/releases

Unreleased
----------

* **Search strategies for DPOR** — ``explore_dpor()`` accepts a new ``search``
  parameter (``SearchStrategy`` enum) to control the order in which wakeup tree
  branches are explored.  Five strategies are available: **DFS** (default,
  optimal for exhaustive runs), **bit-reversal** (van der Corput
  low-discrepancy sequence for maximal early spread), **round-robin**,
  **stride** (coprime-stride permutation), and **conflict-first** (prefers
  threads added by race reversals).  All strategies visit the same set of
  Mazurkiewicz trace equivalence classes; only the exploration order differs.
  Non-DFS strategies can find bugs 30–35% faster when using
  ``stop_on_first=True``.
* **Marker-level exhaustive exploration** — new ``explore_marker_interleavings()``, ``all_marker_schedules()``, and ``marker_schedule_strategy()`` provide completeness guarantees at trace-marker granularity.
* Fixed multiple DPOR correctness bugs: ``BoundedSemaphore.release()`` missing ``_report()``, ``Condition.notify(1)`` waking all waiters instead of one, false deadlock detection in async DPOR, lock over-exploration, and ``record_access()`` unconditionally upgrading ``AccessKind`` to ``Write``.
* Fixed SQL parsing of quoted schema-qualified table names and Redis blocking-pop classification and reproduction issues.

0.3.0 (2026-03-23)
-------------------

* **Interactive HTML report** — ``--frontrun-report=path.html`` pytest flag
  generates a self-contained HTML file visualising the full DPOR exploration:
  SVG timelines, clickable switch points, side-by-side race views with source
  context and values, keyboard/swipe navigation.

* **``track_dunder_dict_accesses`` parameter** — ``explore_dpor()`` no longer
  reports ``obj.__dict__`` accesses by default.  The duplicate conflict points
  doubled wakeup tree insertions for every attribute race with negligible
  benefit (only catches the rare ``self.x`` vs ``self.__dict__['x']`` cross-path).
  Pass ``track_dunder_dict_accesses=True`` to restore the old behaviour.

* **Improved Free-threading support** — There were some bugs on freethreaded python
  (3.14t) which lead to an explosion of spurious conflict points, where the scheduler
  was depending on GIL-synchronization to avoid writing an incorrect conflict index.

* **Missed traces** — Fixed bug where all traces had to start with the first thread as the first operation.
  All distinct Mazurkiewicz traces should now be explored.


0.2.0 (2026-03-20)
-------------------

**Redis and SQL conflict detection**

DPOR now understands Redis and SQL.  Instead of treating all traffic to the same
``host:port`` as a single conflict point, frontrun intercepts execute methods
on the db drivers in common sql and redis clients. This means that DPOR can
analyze whether two SQL queries can conflict (e.g. read or update the same row)
and not explore all interleavings of independent SQL queries.

This means that Frontrun can detect complex race conditions involving interactions
between threading primitives, sql databases and redis while keeping the exploration
space manageably small.

**Async DPOR**

DPOR now automatically treats await points in coroutines as possible conficts,
using tracing/io interception/sql+redis parsing to identify which interleavings
of awaits might could lead to resource conflicts and race conditions.

**Many bugfixes**

A large number of bugfixes from having Claude run frontrun against dozens of open
source libraries. DPOR should now be more accurate about identifying conflict
points and succeed in identifying more complex races.

**Optimal DPOR**

Switched DPOR backend to use wakeup trees and source sets, meaning that each equivalence class of trace should be explored exactly once.

0.1.0 (2026-02-27)
-------------------

**DPOR (Dynamic Partial Order Reduction)**

Systematic interleaving exploration via a Rust engine (``frontrun._dpor``,
built with PyO3/maturin).  Unlike the bytecode explorer which samples random
schedules, DPOR guarantees that every meaningfully distinct interleaving is
tried exactly once.  Shared-memory conflicts are detected automatically at the
bytecode level using a shadow stack; vector clocks prune redundant orderings.

**Automatic I/O detection**

Both the bytecode explorer and DPOR now detect socket and file I/O
automatically (``detect_io=True`` by default).  When two threads touch the same
network endpoint or file path the scheduler treats the operations as
conflicting and explores their reorderings.

**C-level I/O interception**

A new ``frontrun`` CLI wraps any command with an ``LD_PRELOAD`` library
(``libfrontrun_io.so``, built from ``crates/io/``) that intercepts libc I/O
functions (``connect``, ``send``, ``recv``, ``read``, ``write``, etc.).  This
covers opaque C extensions such as database drivers, Redis clients, and HTTP
libraries.

**Interpretable error messages**

When a race condition is found, ``result.explanation`` now contains a
human-readable trace showing interleaved source lines, the conflict pattern
(lost update, write–write, etc.), and reproduction statistics.

**LD_PRELOAD events wired into DPOR**

When run under the ``frontrun`` CLI with ``detect_io=True`` (the default),
``explore_dpor()`` now consumes C-level I/O events from the ``LD_PRELOAD``
library via ``IOEventDispatcher`` → ``_PreloadBridge``.  This means DPOR
detects races involving opaque C extensions (e.g. psycopg2/libpq calling
libc ``send()``/``recv()`` directly) that previously went unnoticed.

**Improved DPOR race detection**

- Global variable and module-level attribute accesses are now tracked as
  shared-memory conflicts.
- C-level container mutations (``list.append``, ``dict.__setitem__``, etc.)
  detected via ``sys.setprofile``.
- Closure variable (``LOAD_DEREF`` / ``STORE_DEREF``) accesses tracked.
- Builtin function calls that mutate containers (e.g. ``sorted()``,
  ``list()``) treated as reads on their arguments.
- Container iteration (``GET_ITER`` / ``FOR_ITER``) tracked as reads.
- Known limitation: C-level iteration interleaving (e.g.
  ``list(od.keys())`` vs ``OrderedDict.move_to_end()``, or ``itertools``
  combinators racing with mutations) is undetectable at the opcode level.
  See ``PEP-703-REPORT.md``.

**Other changes**

- Cooperative threading primitives (``Lock``, ``RLock``, ``Semaphore``,
  ``Event``, ``Condition``, ``Queue``, etc.) extracted to a shared module so
  both the bytecode explorer and DPOR use the same wrappers.
- Deadlock detection via wait-for graph cycle detection.
- ``--frontrun-patch-locks`` pytest plugin for early cooperative patching;
  tests that need the frontrun environment are auto-skipped when it is absent.
- Free-threaded Python (3.13t, 3.14t) support, including a fix for a PyO3
  "Already borrowed" panic and intermittent hangs in cooperative lock
  patching on 3.14t.
- Multi-version test matrix: Python 3.10, 3.14, 3.14t.
- ``DporResult`` merged into ``InterleavingResult``; all three ``explore_*``
  functions return the same type.
- Improved ``_dpor`` import error with build instructions.

0.0.2 (2026-02-17)
-------------------

Rename library from interlace to frontrun.

0.0.1 (2026-02-17)
-------------------

Initial release (as "interlace").  Includes trace markers, bytecode
exploration, and async variants.
