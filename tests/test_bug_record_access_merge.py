"""Test that record_access uses AccessKind::merge semantics.

Bug: path.rs record_access() unconditionally upgrades to Write when the
same object is accessed with two different AccessKinds at the same
scheduling step.  The correct behavior is to use AccessKind::merge()
which handles Read+WeakRead as Read (not Write).

This bug causes overly conservative sleep-set independence checks:
a thread with merged Read+WeakRead (which should be Read, conflicting
only with Write/WeakWrite) is treated as if it performed a Write
(which conflicts with everything).  The result: threads that only
read the same object are falsely woken from the sleep set, causing
redundant interleaving exploration.

The Rust code has a merge() function with exactly the right semantics
but record_access() does not use it.
"""

from frontrun._dpor import PyDporEngine


def test_record_access_merge_read_weakread():
    """Verify that Read + WeakRead on the same object at one step uses merge.

    This test drives the Rust DPOR engine directly (without Python-level
    opcode tracing) to isolate the record_access/sleep-set behavior.

    Setup: 3 threads, all accessing the same object.
      Thread 0: Read + WeakRead on object X (at step 0)
      Thread 1: WeakRead on object X (at step 1)
      Thread 2: does nothing (finishes immediately)

    After thread 0's step, object X has:
      - per_thread_read[0] at path_id=0  (from Read)
      - per_thread_weak_read[0] at path_id=0  (from WeakRead)

    At the sleep-set level, thread 0's step recorded both Read and
    WeakRead for object X.  The merged kind should be Read (not Write).

    When thread 1 runs, it does WeakRead on object X.  Is thread 0's
    merged access independent of thread 1's WeakRead?

    - If merged to Read: Read vs WeakRead is NOT a conflict (per
      access_kinds_conflict), so thread 0's exploration could remain
      in the sleep set.
    - If merged to Write: Write vs WeakRead IS a conflict, so thread 0
      would be unnecessarily woken from the sleep set.

    We test this by counting interleavings: with correct merge, DPOR
    should explore fewer interleavings.
    """
    # 2 threads, both doing only reads on the same object.
    # Thread 0: reports Read then WeakRead on object 42
    # Thread 1: reports WeakRead on object 42
    #
    # These are all non-conflicting access pairs:
    # - Read vs WeakRead: not a conflict
    # - WeakRead vs WeakRead: not a conflict
    #
    # So DPOR should find NO races and explore exactly 1 interleaving.
    # But if record_access merges Read+WeakRead to Write, the sleep set
    # may see a spurious conflict and explore 2.
    engine = PyDporEngine(num_threads=2, preemption_bound=None, max_branches=10000)

    exec_count = 0
    while True:
        execution = engine.begin_execution()

        t = engine.schedule(execution)
        assert t is not None
        # Thread t: Read then WeakRead on same object
        engine.report_access(execution, t, 42, "read")
        engine.report_access(execution, t, 42, "weak_read")
        execution.finish_thread(t)

        t2 = engine.schedule(execution)
        assert t2 is not None
        # Other thread: WeakRead on same object
        engine.report_access(execution, t2, 42, "weak_read")
        execution.finish_thread(t2)

        exec_count += 1
        if not engine.next_execution():
            break

    # With correct merge (Read+WeakRead -> Read), the sleep set
    # should recognize that Read is independent of WeakRead, keeping
    # the first-explored thread asleep and avoiding redundant exploration.
    # Either way, no races exist (Read/WeakRead never conflict with
    # WeakRead), so exactly 1 interleaving should be explored.
    assert exec_count == 1, (
        f"Expected 1 interleaving (no conflicts between Read/WeakRead "
        f"and WeakRead), but got {exec_count}. This may indicate "
        f"record_access is merging Read+WeakRead to Write, causing "
        f"false sleep-set wakeups."
    )
