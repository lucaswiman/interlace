DPOR: Dynamic Partial Order Reduction
======================================

Frontrun includes a built-in `DPOR
<https://en.wikipedia.org/wiki/Partial_order_reduction>`_ engine for
*systematic* concurrency testing. Where the bytecode explorer samples random
interleavings hoping to hit a bug, DPOR guarantees that every meaningfully
different interleaving is explored exactly once --- and nothing redundant is
ever re-run.

The engine is written in Rust for performance and exposed to Python via PyO3.
Its design is inspired by the `loom <https://github.com/tokio-rs/loom>`_
library for Rust, which pioneered the idea of embedding a DPOR-based model
checker directly into a language's test harness.


Why DPOR?
---------

Consider two threads, each executing *n* shared-memory operations. The total
number of possible interleavings is:

.. math::

   \binom{2n}{n} = \frac{(2n)!}{(n!)^2}

For *n* = 3 that is 20 interleavings; for *n* = 10 it is 184,756. The number
grows exponentially --- yet most of these interleavings produce the same
observable outcome. If the threads access disjoint variables, *every* ordering
is equivalent and only one execution is needed.

DPOR exploits this insight. It tracks which operations actually *conflict*
(access the same object with at least one write) and only explores alternative
orderings at those conflict points. For programs with mostly thread-local work
this collapses an exponential search space down to a handful of executions.


Algorithm overview
------------------

Frontrun implements a hybrid DPOR algorithm combining ideas from three
papers:

- **Classic DPOR** --- `Flanagan and Godefroid, POPL'05
  <https://dl.acm.org/doi/10.1145/1040305.1040315>`_. Introduces dynamic
  partial-order reduction with backtrack sets.
- **Optimal DPOR with wakeup trees** --- `Abdulla, Aronis, Jonsson, and
  Sagonas, POPL'14
  <https://dl.acm.org/doi/10.1145/2535838.2535845>`_. Replaces backtrack
  sets with wakeup trees to achieve exactly one execution per Mazurkiewicz
  trace.
- **Source sets and deferred race detection** --- `Abdulla et al., JACM'17
  <https://doi.org/10.1145/3073408>`_. The full journal version, which
  Frontrun follows most closely. See :ref:`dpor-wakeup-trees` below for
  the wake-up tree machinery from Algorithm 2.

Optional preemption bounding follows `Musuvathi and Qadeer, PLDI'07
<https://www.microsoft.com/en-us/research/publication/iterative-context-bounding-for-systematic-testing-of-multithreaded-programs/>`_.

The core loop repeats until no unexplored interleavings remain:

1. **Execute** the program under a deterministic schedule, recording every
   shared-memory access and synchronization event.
2. **Detect races** --- pairs of concurrent accesses to the same object where
   at least one is a write.  Races are collected during execution and processed
   at the end (deferred race detection, Algorithm 2 of the JACM'17 paper).
3. **Compute wakeup sequences** --- for each race, compute the ``notdep``
   sequence (independent events between the two racing accesses) and insert it
   into the **wakeup tree** at the appropriate branch.
4. **Propagate sleep sets** --- threads whose next actions are independent of
   the chosen thread's action are kept asleep, preventing redundant
   exploration of equivalent interleavings.
5. **Advance** to the next unexplored path by picking the next branch from the
   wakeup tree in depth-first order.


Key concepts
~~~~~~~~~~~~

Before diving into the details, here are the core ideas in abstract form.

A **trace** is a sequence of operations (events) performed by all threads
during one execution:

.. math::

   \sigma = e_1, e_2, \ldots, e_k

Two operations :math:`e_i` and :math:`e_j` are **dependent** if they access
the same shared object and at least one is a write. Operations that are
independent can be freely reordered without changing the outcome.

Two operations are **co-enabled** if both threads could have been scheduled at
that point. The DPOR algorithm identifies pairs of operations that are both
dependent *and* co-enabled --- these are the only points where exploring a
different scheduling order could produce a different result.

For example, consider a trace of four events on two threads accessing objects
*x* and *y*:

.. math::

   \sigma = \underbrace{W_0(x)}_{e_1},\;
            \underbrace{R_0(y)}_{e_2},\;
            \underbrace{W_1(x)}_{e_3},\;
            \underbrace{R_1(y)}_{e_4}

Here :math:`W_t(o)` means thread *t* writes object *o* and :math:`R_t(o)`
means thread *t* reads object *o*. Events :math:`e_1` and :math:`e_3` are
dependent (both access *x*, one is a write). Events :math:`e_2` and
:math:`e_4` are independent (both are reads of *y*). DPOR would insert a
backtrack point to try scheduling thread 1 before :math:`e_1`, producing a
trace where :math:`W_1(x)` precedes :math:`W_0(x)`.


.. _dpor-happens-before:

Happens-before and vector clocks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. seealso::

   :doc:`vector-clocks` for a comprehensive treatment including the three
   per-thread clocks (``dpor_vv``, ``io_vv``, ``causality``), the
   ``VersionVec`` implementation, and worked examples with and without
   synchronization.

Two operations are *concurrent* if neither `happens before
<https://en.wikipedia.org/wiki/Happened-before>`_ the other. The happens-before
relation :math:`\to` is the smallest partial order satisfying:

.. math::

   \text{(program order)} \quad & e_i \to e_j
     \text{ if } e_i \text{ and } e_j \text{ are by the same thread and }
     i < j \\
   \text{(synchronization)} \quad & \text{release}(l) \to \text{acquire}(l)
     \text{ for lock } l \\
   \text{(spawn)} \quad & \text{spawn}(t) \to e
     \text{ for any first event } e \text{ of thread } t \\
   \text{(join)} \quad & e \to \text{join}(t)
     \text{ for any last event } e \text{ of thread } t \\
   \text{(transitivity)} \quad & e_i \to e_k
     \text{ if } e_i \to e_j \text{ and } e_j \to e_k

Two events :math:`e_i` and :math:`e_j` are **concurrent** (written
:math:`e_i \| e_j`) when :math:`e_i \not\to e_j` and :math:`e_j \not\to e_i`.

Frontrun tracks happens-before using `vector clocks
<https://en.wikipedia.org/wiki/Vector_clock>`_ (``VersionVec``). A vector clock
is an array of counters with one entry per thread (stored as a contiguous
``Vec<u32>`` indexed by thread ID):

.. math::

   V : \text{ThreadId} \to \mathbb{N}

The key operations are:

``increment(thread_id)``
    Advance the local component: :math:`V[t] \leftarrow V[t] + 1`. Called each
    time a thread is scheduled.

``join(other)``
    Point-wise maximum: :math:`V[i] \leftarrow \max(V[i], V'[i])` for all *i*.
    Used when synchronization transfers causal knowledge between threads.

``partial_le(other)``
    Returns true if :math:`V[i] \le V'[i]` for every component. If
    ``a.partial_le(b)`` then *a* happens before *b*:

    .. math::

       V \le V' \iff \forall\, i : V[i] \le V'[i]

``concurrent_with(other)``
    True when neither clock dominates the other:

    .. math::

       V \| V' \iff V \not\le V' \;\wedge\; V' \not\le V

Each thread carries *two* vector clocks:

``causality``
    Tracks the program's semantic happens-before relation. Updated when
    synchronization primitives (locks, joins, spawns) transfer ordering
    information between threads.

``dpor_vv``
    Tracks the scheduler's branch decisions. Incremented each time the thread
    is scheduled, and merged (via the same component-wise-max ``join``
    operation) on synchronization events just like ``causality``. This is the
    clock used for DPOR dependency detection --- it tells us whether two
    scheduling decisions were causally ordered or concurrent.


Conflict detection
~~~~~~~~~~~~~~~~~~~

Every shared-memory access is reported to the engine with a thread ID, an
object ID, and a kind (read or write). The engine maintains an ``ObjectState``
for each object, recording the last access of any kind and the last write
access separately.

When a new access arrives the engine asks: *what was the last dependent access
to this object?*  The dependency (or **conflict**) relation is:

.. math::

   \text{dep}(e_i, e_j) \iff
     \text{obj}(e_i) = \text{obj}(e_j) \;\wedge\;
     \bigl(\text{kind}(e_i) = W \;\lor\; \text{kind}(e_j) = W\bigr)

In the implementation, there are four access kinds with the following
conflict rules:

- **Read** conflicts with **Write** and **WeakWrite** (but not other Reads).
- **Write** conflicts with all other access kinds.
- **WeakWrite** conflicts with **Read** and **Write** (but not other
  WeakWrites or WeakReads). This models container subscript writes where
  different keys on the same container are independent.
- **WeakRead** conflicts only with **Write**.

If such a prior access :math:`e_p` exists and its ``dpor_vv`` is *not*
``partial_le`` of the current thread's ``dpor_vv``, the two accesses are
concurrent and could race:

.. math::

   \text{dep}(e_p, e_c) \;\wedge\; e_p \| e_c
   \implies \text{insert wakeup sequence at branch}(e_p)

The engine collects the race as a ``PendingRace`` and, at the end of the
execution, computes a ``notdep`` wakeup sequence and inserts it into the
wakeup tree at the branch where the prior access was made. It also performs
immediate inline backtracking (inserting a single-thread sequence) for
responsiveness. This ensures a future execution will try scheduling the
racing thread at that earlier point, reversing the order of the two
conflicting operations.


Synchronization events
~~~~~~~~~~~~~~~~~~~~~~~

Synchronization primitives update the ``causality`` (and ``dpor_vv``) clocks so
that accesses ordered by proper synchronization are not flagged as conflicts:

**Lock acquire**
    The acquiring thread joins the vector clock that was stored when the lock
    was last released:

    .. math::

       V_t \leftarrow V_t \sqcup V_{\text{lock}}

    This establishes that the acquire happens after the previous release.

**Lock release**
    The releasing thread's current causality clock is stored on the lock for
    future acquirers:

    .. math::

       V_{\text{lock}} \leftarrow V_t

**Thread join**
    The joining thread joins both the causality and DPOR clocks of the joined
    thread. All of the joined thread's operations now happen before the
    joiner's subsequent operations:

    .. math::

       V_t \leftarrow V_t \sqcup V_{t'}

**Thread spawn**
    The child thread inherits the parent's causality and DPOR clocks. The
    parent's operations before the spawn happen before the child's operations:

    .. math::

       V_{\text{child}} \leftarrow V_{\text{child}} \sqcup V_{\text{parent}}


The exploration tree
--------------------

The engine maintains a ``Path`` --- a sequence of ``Branch`` nodes, one per
scheduling decision. Each branch records:

- The **status** of every thread at that point (disabled, pending, active,
  visited, blocked, or yielded).
- Which thread was **chosen** (the ``active_thread``).
- The cumulative **preemption count** (how many times a runnable thread was
  preempted in favor of a different thread up to this point).
- A **wakeup tree** --- an ordered tree of thread-ID sequences representing
  interleavings still to be explored at this branch (Definition 6.1 of the
  JACM'17 paper, p.21--22).
- A **sleep set** --- threads whose next actions are independent of the
  chosen thread's action and therefore need not be explored at this branch.
- **Access tracking** --- per-thread records of which objects were accessed
  and with what kind (read, write, weak-write, weak-read), used for
  independence checks during sleep set propagation.

To illustrate, consider a two-thread program where DPOR finds one conflict.
The exploration tree looks like:

.. code-block:: text

   Branch 0              Branch 1            Branch 2            Branch 3
   ┌─────────────┐       ┌─────────────┐     ┌─────────────┐     ┌──────────────┐
   │ T0: Active  │  -->  │ T0: Active  │ --> │ T1: Active  │ --> │ T1: Active   │
   │ T1: Pending │       │ T1: Pending │     │ T0: Pending │     │ T0: Disabled │
   │ wut: {}     │       │ wut: {}     │     │ wut: {}     │     │ wut: {}      │
   └─────────────┘       └─────────────┘     └─────────────┘     └──────────────┘
                            ^
                            │
                       CONFLICT DETECTED between
                       T0's write here and T1's
                       later write: insert [T1]
                       into wakeup tree at Branch 1

After the first execution completes, ``step()`` walks backward, finds Branch 1
has a non-empty wakeup tree, and picks the next thread from it:

.. code-block:: text

   Branch 0              Branch 1 (replayed, different choice)
   ┌─────────────┐       ┌─────────────┐
   │ T0: Active  │  -->  │ T1: Active  │ --> ...
   │ T1: Pending │       │ T0: Visited │
   │ wut: {}     │       │ wut: {}     │
   └─────────────┘       └─────────────┘

The prefix up to Branch 0 is replayed identically; only the decision at
Branch 1 changes.

.. _dpor-wakeup-trees:

Wakeup trees
~~~~~~~~~~~~~~

A wakeup tree is the engine's memory of which interleavings still need
to be explored at a given branch. It was introduced by `Abdulla et al.
in POPL'14
<https://dl.acm.org/doi/10.1145/2535838.2535845>`_ and given its full
treatment in the `JACM'17 journal version
<https://doi.org/10.1145/3073408>`_ (Definition 6.1, p.21--22).
``WakeupTree`` lives in ``crates/dpor/src/wakeup_tree.rs``.

Definition
^^^^^^^^^^^

For an execution prefix :math:`E` with sleep set :math:`P`, a wakeup tree
:math:`\langle B, \prec \rangle` is an ordered, rooted tree whose nodes
are labeled by thread IDs. The children of each node are totally ordered
by the exploration-order relation :math:`\prec`. Each path from the root
to any node spells out a **wakeup sequence** --- an initial fragment of
a future execution that will reverse a detected race.

The tree must satisfy two properties (JACM'17 Def 6.1):

*Property 1 (no blocked leaf).*
   For every leaf :math:`w`, the set of weak initials
   :math:`WI_{[E]}(w)` is disjoint from the sleep set :math:`P`. Every
   sequence in the tree leads somewhere that is not already blocked by
   sleep.

*Property 2 (sleep-set-removing siblings).*
   For siblings :math:`u.p \prec u.q` where :math:`u.q` is a leaf,
   :math:`p \notin WI_{[E.u]}(q)`. Exploring :math:`u.p` first adds it to
   the sleep set, but exploring :math:`u.q` afterwards is guaranteed to
   remove :math:`p` before :math:`q` is reached. This is what lets
   Optimal DPOR reach every Mazurkiewicz class without redundant work.

Internal shape
^^^^^^^^^^^^^^^

Each ``WakeupTree`` is an ordered list of top-level children; each child
is a ``WakeupNode`` holding a ``thread_id`` and its own ordered list of
children. Sequences that share a prefix are merged into the same
subtree, which is both more compact and essential for the
:math:`\prec`-order to line up with Property 2.

Example: insert evolution
^^^^^^^^^^^^^^^^^^^^^^^^^^

Suppose three races are detected at one branch and the algorithm inserts
three wakeup sequences, in order:

1. ``[T1]``
2. ``[T1, T2]``
3. ``[T2, T0, T1]``

.. code-block:: text

   After insert [T1]:        After insert [T1, T2]:      After insert [T2, T0, T1]:

      (root)                    (root)                      (root)
        │                         │                        ╱      ╲
        T1                        T1                      T1       T2
                                  │                       │        │
                                  T2                      T2       T0
                                                                   │
                                                                   T1

Insert 1 creates a new top-level branch ``T1``. Insert 2 matches ``T1``
among the existing children, descends into it, and appends ``T2``.
Insert 3 does not match ``T1`` (its first element is ``T2``), so a new
top-level branch is appended at the rightmost position and the remaining
chain ``T0, T1`` hangs off it as a linear path.

Example: subtree extraction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Continuing from the tree above: if the scheduler picks ``T1`` to explore
first (via ``min_thread()``), then ``subtree(T1)`` returns the children
below the ``T1`` node, which become the wakeup tree guiding the *next*
branch. The parent's ``T1`` branch is removed (``remove_branch(T1)``),
leaving the sibling ``T2`` branch behind:

.. code-block:: text

   Before choosing T1:           After choosing T1:

   Parent wut:                   Parent wut (T1 removed):

      (root)                         (root)
      ╱    ╲                            │
     T1    T2                           T2
     │     │                            │
     T2    T0                           T0
           │                            │
           T1                           T1

                                 Child wut (for next branch)
                                 = subtree(T1) = just "T2":

                                     (root)
                                        │
                                        T2

This propagation is what lets a *multi-step* wakeup sequence drive
exploration correctly: insert 2, ``[T1, T2]``, requires choosing T1 at
depth :math:`d` and then T2 at depth :math:`d+1`. Without carrying the
subtree down, the engine would lose track of the required second step.

Operations and Algorithm 2 line references
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``insert(sequence)`` (JACM'17 Algorithm 2 line 6, Def 6.2 p.22)
    Add a scheduling sequence to the tree. The implementation walks down
    existing children by prefix matching; the first divergence becomes a
    new rightmost sibling. **Simplification vs. the paper:** Definition
    6.2 uses the equivalence relation :math:`v \sim_{[E]} w` from Lemma
    6.2 to skip inserting sequences already covered by an existing
    branch. Frontrun uses exact prefix matching --- sound but occasionally
    retains redundant branches (see ``ideas/optimal_dpor.md`` Phase 4c).

``min_thread()`` (Algorithm 2 line 15)
    Return the minimum thread ID among the root-level children.
    Frontrun uses the minimum thread ID as a deterministic proxy for the
    paper's :math:`\prec` order.

``subtree(p)`` (Algorithm 2 line 17: :math:`WuT' = \text{subtree}(wut(E), p)`)
    When thread *p* is chosen at this branch, the subtree rooted at *p*
    becomes the wakeup guidance for the subsequent branch.

``remove_branch(p)`` (Algorithm 2 line 19: "remove all sequences of form p.w")
    Remove the top-level branch starting with *p* after it has been
    explored.

Scheduling
~~~~~~~~~~~

When the engine needs to pick the next thread:

1. If we are **replaying** a previously recorded path (``pos < branches.len``),
   return the same choice as before and propagate sleep sets forward using
   independence checks. This is how the engine deterministically re-executes
   the shared prefix leading to a backtrack point.

2. Otherwise this is a **new** scheduling decision. If a wakeup subtree
   provides guidance (from a multi-step wakeup sequence), the guided thread
   is chosen. Otherwise the engine prefers the currently active thread (to
   minimize preemptions). A new ``Branch`` is created recording the decision,
   thread statuses, and access information.

Backtracking and wakeup sequence insertion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Race detection uses a **hybrid approach**: immediate inline backtracking for
single-step races (Algorithm 1 style) plus deferred ``notdep`` sequence
computation for multi-step wakeup sequences (Algorithm 2 style).

When ``process_access()`` detects a race between events *e* and *e'*:

1. **Inline**: Insert ``[thread_id]`` into the wakeup tree at the branch
   where *e* occurred (unless the thread is already in the sleep set).
2. **Deferred**: Collect a ``PendingRace`` record for later processing.

At the end of each execution, ``next_execution()`` processes all deferred
races:

1. For each race, compute ``notdep(e, E).e'`` --- the sequence of events
   between *e* and *e'* that are independent of *e*, followed by *e'*.
   Same-thread events and events with conflicting accesses are excluded.
2. Insert the ``notdep`` sequence into the wakeup tree at branch *e*.
   This enables exploration of interleavings where the independent events
   run before the racing thread, producing better state-space coverage
   than single-thread backtracking alone.

When the current execution finishes, ``step()`` walks backward through the
branch list:

1. Save the active thread's access information in the trace cache
   (``prev_thread_all_accesses``) for sleep set propagation in future
   executions.
2. Mark the current branch's active thread as ``Visited`` and add it to
   the sleep set.
3. Remove the current thread's branch from the wakeup tree.
4. Look for a non-empty wakeup tree in this branch. Pick the next thread
   via ``min_thread()`` and extract the subtree for wakeup guidance.
5. If the wakeup tree is empty, pop the branch and continue walking backward.
6. If all branches are exhausted, exploration is complete.

Sleep set propagation
~~~~~~~~~~~~~~~~~~~~~~

Sleep sets prevent redundant exploration of equivalent interleavings. A thread
is **asleep** at a branch if its next action is independent of all the actions
that will be taken before it next runs.

The propagation works in two stages:

1. **During replay**: At each replayed position, ``propagate_sleep()`` carries
   sleeping threads forward. For each sleeping thread *q*, the engine checks
   whether *q*'s recorded accesses are independent of the active thread *p*'s
   accesses (using ``accesses_are_independent()``). If independent, *q* stays
   asleep; if conflicting, *q* is woken.

2. **At new branches**: Beyond the replay prefix, sleep set propagation uses
   the **trace cache** (``prev_thread_all_accesses``) --- the union of all
   accesses each thread performed in the previous execution. This conservative
   approximation considers ALL of a sleeping thread's future accesses, not
   just one position's snapshot. If any future access would conflict, the
   thread is woken.

The independence check (``access_kinds_conflict()``) mirrors the engine's
conflict semantics:

- **Read + Read**: independent (reads commute)
- **WeakWrite + WeakWrite**: independent (different keys on same container)
- **Write + anything**: dependent (writes conflict with all access kinds)

This is particularly effective for *writer-readers* patterns: when a writer
and multiple readers access the same object, the readers' accesses are
independent of each other, so DPOR explores only :math:`2^N` interleavings
for *N* readers (matching the theoretical optimum from JACM'17 Table 1)
instead of :math:`(N+1)!`.

This is a depth-first search over the tree of scheduling choices, pruned by
DPOR so that only branches with genuine conflicts are explored, and further
reduced by sleep sets so that equivalent orderings of independent actions are
not re-explored.


Search strategies
~~~~~~~~~~~~~~~~~~

The default exploration order is DFS (always pick the lowest thread ID at each
wakeup-tree node).  This matches the paper's Algorithm 2 and produces the
**optimal** number of executions --- exactly one per Mazurkiewicz trace
equivalence class --- because the sleep-set pruning is maximally effective
under this ordering.

When the trace space is very large and you have a limited execution budget
(``stop_on_first=True``, or a low ``max_executions``), alternative strategies
can find bugs significantly faster by spreading exploration across diverse
conflict points early.  These strategies --- **bit-reversal**, **round-robin**,
**stride**, and **conflict-first** --- visit the same set of trace equivalence
classes but in a different order.  The trade-off is that non-DFS orderings may
explore a small number of redundant trace classes (typically ~5% on complex
lock patterns) because changing the sibling ordering can reduce sleep-set
effectiveness.

**Rule of thumb:**

- **Exhaustive exploration** (``stop_on_first=False``): use **DFS** (the default)
  to minimize total executions.
- **Bug-finding with a budget** (``stop_on_first=True`` or low
  ``max_executions``): use **bit-reversal** or **round-robin** to maximize
  diversity early.

See :doc:`search` for a detailed comparison of all strategies and the
theoretical basis for the optimality trade-off.


Preemption bounding
--------------------

Real programs often have far more conflicts than can feasibly be explored.
*Preemption bounding* limits exploration to executions with at most *k*
preemptions (context switches away from a runnable thread). Empirical research
has shown that most concurrency bugs surface with very few preemptions ---
`Musuvathi and Qadeer (2007)
<https://www.microsoft.com/en-us/research/publication/iterative-context-bounding-for-systematic-testing-of-multithreaded-programs/>`_
found that a bound of 2 is sufficient to catch the vast majority of bugs in
practice.

When a backtrack point would create a preemption that exceeds the bound, the
engine falls back to ``add_conservative_backtrack``: it walks backward through
earlier branches looking for a point where the same thread can be explored
without exceeding the preemption budget. This maintains soundness within the
bounded exploration --- every execution with at most *k* preemptions that
differs in a dependent operation will still be explored.

With preemption bounding, the number of explored executions is polynomial in
the program length for a fixed bound *k*:

.. math::

   O\!\left(\binom{n}{k}\right)

where *n* is the number of scheduling points, versus the exponential
:math:`O(t^n)` for *t* threads without bounding.


Abstract reduction example
---------------------------

To see how DPOR prunes the search space, consider three events by two threads:

.. math::

   e_1 = W_0(x), \quad e_2 = R_1(y), \quad e_3 = W_1(x)

Events :math:`e_1` and :math:`e_3` are **dependent** (both access *x*, at
least one is a write). Events :math:`e_1` and :math:`e_2` are **independent**
(different objects). Events :math:`e_2` and :math:`e_3` are **independent**
(same thread --- program-ordered).

A naive scheduler would explore all interleavings:

.. math::

   &(1)\; e_1, e_2, e_3 \qquad
    (2)\; e_1, e_3, e_2 \qquad
    (3)\; e_2, e_1, e_3 \\
   &(4)\; e_2, e_3, e_1 \qquad
    (5)\; e_3, e_1, e_2 \qquad
    (6)\; e_3, e_2, e_1

But interleavings that only swap independent events produce the same result.
For example, (1) :math:`e_1, e_2, e_3` and (3) :math:`e_2, e_1, e_3` differ
only in the order of :math:`e_1` and :math:`e_2`, which are independent --- so
they are equivalent. The equivalence classes (called `Mazurkiewicz traces
<https://en.wikipedia.org/wiki/Trace_theory>`_) are:

.. math::

   \{(1), (3)\} \quad \text{and} \quad \{(4), (5), (6), (2)\}

DPOR explores **one representative per class** --- here just 2 executions
instead of 6. The two classes can be visualized as follows:

.. code-block:: text

   6 linearizations                        2 Mazurkiewicz classes
   ─────────────────                       ──────────────────────
   (1) e1, e2, e3  ┐
                   ├── equivalent  ──►   Class A:  e1 precedes e3
   (3) e2, e1, e3  ┘                     (representative: e1, e2, e3)

   (2) e1, e3, e2  ┐
   (4) e2, e3, e1  │
                   ├── equivalent  ──►   Class B:  e3 precedes e1
   (5) e3, e1, e2  │                     (representative: e3, e2, e1)
   (6) e3, e2, e1  ┘

Within each class the only variation is the position of :math:`e_2`
relative to the two writes. Since :math:`e_2` is independent of both, it
can slide freely and every ordering produces the same final state. DPOR
runs one linearization per class.

The reduction is even more dramatic with more threads and more
thread-local work.


Worked example: concurrent counter
------------------------------------

Consider the classic lost-update bug: two threads each read a shared counter,
increment locally, and write back.

.. code-block:: python

   counter = 0

   def thread_0():
       local = counter    # R_0(counter)
       counter = local+1  # W_0(counter)

   def thread_1():
       local = counter    # R_1(counter)
       counter = local+1  # W_1(counter)

There are four shared-memory operations, giving
:math:`\binom{4}{2} = 6` possible interleavings in total. Let's trace
how DPOR explores them.

Execution 1: T0 runs to completion, then T1
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The engine's default scheduling preference is to keep running the current
thread, producing:

.. code-block:: text

   Step   Thread   Operation        Object    Kind
   ──────────────────────────────────────────────────
    0      T0      R_0(counter)     counter   read
    1      T0      W_0(counter)     counter   write
    2      T1      R_1(counter)     counter   read
    3      T1      W_1(counter)     counter   write

   Result: counter = 2  (correct)

Vector clocks evolve as follows (showing ``dpor_vv`` as ``[T0, T1]``):

.. code-block:: text

   Step 0:  T0 scheduled  →  T0.dpor_vv = [1, 0]
            R_0(counter): no prior access → no conflict

   Step 1:  T0 scheduled  →  T0.dpor_vv = [2, 0]
            W_0(counter): last dependent = R_0 at step 0 with vv [1, 0]
            [1, 0] ≤ [2, 0]?  YES → happens-before, no backtrack

   Step 2:  T1 scheduled  →  T1.dpor_vv = [0, 1]
            R_1(counter): last dependent = W_0 at step 1 with vv [2, 0]
            [2, 0] ≤ [0, 1]?  NO → CONCURRENT!
            ⟹ backtrack: mark T1 for exploration at branch 1

   Step 3:  T1 scheduled  →  T1.dpor_vv = [0, 2]
            W_1(counter): last dependent = W_0 at step 1 with vv [2, 0]
            [2, 0] ≤ [0, 2]?  NO → concurrent, but T1 already marked

One backtrack point was inserted at branch 1 (step 1), marking T1 for
exploration there.

Execution 2: T0 does one step, then T1 runs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The engine replays the prefix through branch 0 (T0 at step 0), then at
branch 1 schedules T1 instead of T0:

.. code-block:: text

   Step   Thread   Operation        Object    Kind
   ──────────────────────────────────────────────────
    0      T0      R_0(counter)     counter   read     (replayed)
    1      T1      R_1(counter)     counter   read     ← different choice
    2      T0      W_0(counter)     counter   write
    3      T1      W_1(counter)     counter   write

   Result: counter = 1  (BUG! lost update)

Both threads read ``counter = 0``, then both write ``1``. The engine has
found the bug in just 2 executions out of the 6 possible interleavings.

Why not more executions?
~~~~~~~~~~~~~~~~~~~~~~~~~

DPOR did not need to explore orderings like T1-first-then-T0 because they
would be reached by the same backtrack mechanism from the other direction. The
key insight is that only the *relative order of dependent operations* matters.
Swapping two independent events (like :math:`R_0(x)` and :math:`R_1(y)`)
produces an equivalent trace, so DPOR skips it.


Reduction in practice
~~~~~~~~~~~~~~~~~~~~~~

The following table shows how DPOR reduces the exploration space as programs
grow:

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 20

   * - Scenario
     - Naive
     - DPOR
     - Reduction
   * - 2 threads, 2 ops each, same object
     - 6
     - 2
     - 3x
   * - 2 threads, 5 ops each, same object
     - 252
     - 2
     - 126x
   * - 2 threads, 2 ops each, disjoint objects
     - 6
     - 1
     - 6x
   * - 3 threads, 2 ops each, one shared object
     - 90
     - 6
     - 15x

For threads accessing disjoint objects the reduction is total (one execution
regardless of operation count). When threads share objects, the explored set
corresponds to the `Mazurkiewicz traces
<https://en.wikipedia.org/wiki/Trace_theory>`_ --- equivalence classes of
interleavings under independent-operation reordering.

Sleep set propagation further reduces exploration by recognizing fine-grained
independence. For example, a writer with *N* readers on the same object
requires :math:`2^N` interleavings (each reader can go before or after the
writer), not :math:`(N+1)!` --- the reader orderings are equivalent.


Data structures
---------------

The implementation is split across seven Rust modules in ``crates/dpor/src/``:

``vv.rs`` --- Vector clocks
    ``VersionVec``: a contiguous ``Vec<u32>`` indexed by thread ID with
    ``increment``, ``join``, ``partial_le``, and ``concurrent_with`` operations.

``access.rs`` --- Access records
    ``AccessKind`` (``Read``, ``Write``, ``WeakWrite``, ``WeakRead``) and
    ``Access``, which stores the ``path_id`` (branch index where the access
    occurred), the thread's ``dpor_vv`` at that moment, and the ``thread_id``.
    ``WeakWrite`` is used for container subscript writes (different keys on the
    same container are independent); ``WeakRead`` is used for container loads
    before subscripting.

``object.rs`` --- Shared object state
    ``ObjectState`` tracks per-thread access history for each access kind.
    ``dependent_accesses(kind)`` returns all prior accesses that conflict with
    the new access for race detection.

``thread.rs`` --- Thread state
    ``Thread`` holds two vector clocks (``causality`` and ``dpor_vv``) plus
    ``io_vv`` (for I/O operations, which omit lock-based happens-before so
    that I/O always appears concurrent). ``ThreadStatus`` is the per-branch
    status enum used by the exploration tree.

``wakeup_tree.rs`` --- Wakeup trees
    ``WakeupTree`` and ``WakeupNode``: an ordered tree of thread-ID sequences
    for exploring alternative interleavings (Definition 6.1, JACM'17 p.21--22).
    Supports insert, remove, subtree extraction, and min-thread queries.

``path.rs`` --- Exploration tree
    ``Branch`` and ``Path``. Each ``Branch`` holds thread statuses, a wakeup
    tree, a sleep set, and per-thread access records. ``Path`` drives
    scheduling (with sleep set propagation and wakeup subtree guidance),
    backtracking (with ``notdep`` sequence computation), and depth-first
    advancement. The ``prev_thread_all_accesses`` trace cache enables sleep
    set propagation to new branches beyond the replay prefix.

``engine.rs`` --- Orchestration
    ``DporEngine`` ties everything together. ``Execution`` holds per-run state
    (threads, objects, lock release clocks, schedule trace). The engine detects
    races during execution, collects them as ``PendingRace`` entries, processes
    deferred races at the end of each execution to compute ``notdep`` wakeup
    sequences, and advances to the next execution.


Python API
----------

The Rust engine is exposed to Python via PyO3 as the ``frontrun._dpor`` native
module. The two Python-visible classes are ``PyDporEngine`` and
``PyExecution``.

.. code-block:: python

   from frontrun._dpor import PyDporEngine, PyExecution

   engine = PyDporEngine(
       num_threads=2,
       preemption_bound=2,       # optional; None = unbounded
       max_branches=100_000,     # safety limit per execution
       max_executions=None,      # optional cap on total executions
   )

   while True:
       execution = engine.begin_execution()

       while True:
           thread_id = engine.schedule(execution)
           if thread_id is None:
               break  # deadlock or branch limit

           # ... run thread_id until it performs a shared access ...

           engine.report_access(execution, thread_id, object_id, "write")
           # or: engine.report_sync(execution, thread_id, "lock_acquire", lock_id)

           execution.finish_thread(thread_id)

       # check invariants on this execution's final state ...

       if not engine.next_execution():
           break  # all interleavings explored

``report_access(execution, thread_id, object_id, kind)``
    Report a shared-memory access. ``kind`` is one of ``"read"``, ``"write"``,
    ``"weak_write"`` (container subscript writes --- different keys on the
    same container are independent), or ``"weak_read"`` (container loads
    before subscripting). ``object_id`` is an opaque ``u64`` that uniquely
    identifies the shared object. Frontrun uses stable monotonic IDs
    (``StableObjectIds``) rather than ``id(obj)`` to ensure object keys are
    consistent across executions.

``report_sync(execution, thread_id, event_type, sync_id)``
    Report a synchronization event. ``event_type`` is one of
    ``"lock_acquire"``, ``"lock_release"``, ``"thread_join"``,
    ``"thread_spawn"``. ``sync_id`` identifies the lock or thread.

``next_execution()``
    Advance to the next unexplored path. Returns ``False`` when exploration
    is complete.

``execution.finish_thread(thread_id)``
    Mark a thread as finished (no more operations).

``execution.block_thread(thread_id)`` / ``execution.unblock_thread(thread_id)``
    Mark a thread as blocked or unblocked (e.g., waiting on a lock).

Properties: ``engine.executions_completed``, ``engine.tree_depth``,
``engine.num_threads``, ``execution.schedule_trace``, ``execution.aborted``.


Complexity
----------

**Per access:** :math:`O(T)` where :math:`T` is the number of threads,
dominated by the vector-clock comparison.

**Space:** :math:`O(D \cdot T + O)` where :math:`D` is the exploration tree
depth and :math:`O` is the number of unique shared objects. Only the last two
accesses per object are retained.

**Executions:** In the worst case exponential in the number of dependent
operations, but in practice DPOR prunes the vast majority of redundant
interleavings. With preemption bounding at bound :math:`k`, the explored
subset is :math:`O\!\left(\binom{n}{k}\right)` which is polynomial in program length *n*
for fixed *k*.


Further reading
---------------

- Cormac Flanagan and Patrice Godefroid, `"Dynamic Partial-Order Reduction for
  Model Checking Software"
  <https://dl.acm.org/doi/10.1145/1040305.1040315>`_, POPL 2005 --- the
  original DPOR paper.
- Parosh Aziz Abdulla, Stavros Aronis, Bengt Jonsson, and Konstantinos Sagonas,
  `"Optimal Dynamic Partial Order Reduction"
  <https://dl.acm.org/doi/10.1145/2535838.2535845>`_, POPL 2014 --- source
  sets and optimal exploration.
- Parosh Aziz Abdulla, Stavros Aronis, Bengt Jonsson, and Konstantinos Sagonas,
  `"Source Sets: A Foundation for Optimal Dynamic Partial Order Reduction"
  <https://doi.org/10.1145/3073408>`_, JACM 2017 --- the full journal version
  with wakeup trees, deferred race detection, ``notdep`` sequences, and
  algorithms for locks/disabling. Frontrun's DPOR engine implements key ideas
  from Algorithms 1 and 2 of this paper.
- Madanlal Musuvathi and Shaz Qadeer, `"Iterative Context Bounding for
  Systematic Testing of Multithreaded Programs"
  <https://www.microsoft.com/en-us/research/publication/iterative-context-bounding-for-systematic-testing-of-multithreaded-programs/>`_,
  PLDI 2007 --- preemption bounding.
- `Partial order reduction (Wikipedia)
  <https://en.wikipedia.org/wiki/Partial_order_reduction>`_
- `Vector clock (Wikipedia)
  <https://en.wikipedia.org/wiki/Vector_clock>`_
- `Mazurkiewicz trace theory (Wikipedia)
  <https://en.wikipedia.org/wiki/Trace_theory>`_
- `loom <https://github.com/tokio-rs/loom>`_ --- Rust concurrency testing
  library that inspired this implementation.
