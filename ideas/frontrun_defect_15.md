# Defect #15: DPOR cannot find races in complex check-then-insert patterns

## Summary

DPOR finds simple check-then-insert TOCTOU races easily (SELECT +
INSERT, ~2 ops per thread).  But when intermediate SQL operations
separate the check from the insert — as happens in real ORMs like
django-reversion — the additional conflict points between threads
prevent DPOR from exploring the critical interleaving within its
search tree.

## The pattern

django-reversion's `create_revision()` generates ~5 SQL operations per
thread:

```
Thread A                              Thread B
──────────────────                    ──────────────────
1. SELECT article                     1. SELECT article
2. UPDATE article                     2. UPDATE article
3. SELECT version (dedup check) → ∅   3. SELECT version (dedup check) → ∅
4. INSERT revision                    4. INSERT revision
5. INSERT version                     5. INSERT version  ← duplicate!
   COMMIT                                COMMIT
```

The race: both dedup checks (step 3) see no matching version because
neither thread has inserted yet.  Both proceed to INSERT, creating
duplicate versions.

## Why DPOR can't find it

DPOR finds this race trivially when it's just SELECT + INSERT (2 ops
per thread) — confirmed by `test_simple_check_then_insert_is_found`.

But adding the intermediate SELECT article + UPDATE article (steps 1-2)
creates additional cross-thread conflict points:

- `articles` row: READ (SELECT) vs WRITE (UPDATE) — 2 conflicts
- `articles:seq`: READ (UPDATE phantom) vs READ — more conflicts
- `revisions` table: WRITE vs WRITE — conflict even though rows differ
- `versions:seq`: READ (SELECT) vs WRITE (INSERT) — the real one
- `versions` table: READ vs WRITE — another real one

DPOR explores all interleavings in its backtrack tree.  With only the
version operations, the tree is small enough (2-4 interleavings) to
quickly reach the critical one where both SELECTs precede both INSERTs.

With the article operations added, the tree grows because:
1. The article UPDATE-UPDATE conflict on the same row creates backtrack
   points that DPOR must explore
2. The article SELECT-UPDATE conflicts create more branches
3. These branches multiply with the version conflicts

DPOR exhausts its tree (~30 interleavings) but the critical ordering
— both threads complete steps 1-3 before either does step 5 — is not
reachable within the explored paths.  The article conflicts "distract"
DPOR into exploring orderings of the article operations rather than the
version operations.

## What works vs what doesn't

| Pattern | Ops/thread | DPOR finds it? | Why |
|---------|-----------|----------------|-----|
| SELECT + INSERT | 2 | ✅ Yes | Small tree, critical interleaving in first few paths |
| SELECT + INSERT + INSERT (with FK) | 3 | ✅ Yes | Still small enough |
| SELECT + UPDATE + SELECT + INSERT + INSERT | 5 | ❌ No | Article conflicts dominate the tree |
| django-reversion `create_revision()` | 5-7 | ❌ No | Same issue, plus ORM overhead |

## Possible approaches

1. **Conflict prioritization**: Prioritize backtrack points involving
   `:seq` resources (phantom-read conflicts) over same-row conflicts.
   The `:seq` conflicts are more likely to reveal TOCTOU races because
   they represent check-vs-mutate dependencies.

2. **Operation coalescing for unrelated tables**: If the article
   operations (SELECT + UPDATE) are on a different table than the
   version operations (SELECT + INSERT), DPOR could recognize they're
   independent and avoid exploring orderings of article ops against
   version ops from the same thread.

3. **Targeted exploration**: Allow users to annotate which resources
   are "interesting" for race detection.  DPOR would only create
   backtrack points for those resources, ignoring conflicts on other
   tables.

4. **Hybrid approach**: Use DPOR for conflict detection but fall back
   to random scheduling when the tree gets too large.  Random
   exploration might stumble on the critical interleaving faster than
   systematic search through an enormous tree.

Approach 2 seems most promising for the general case — many real-world
races involve operations on table X interspersed with operations on
table Y, and DPOR shouldn't let Y-conflicts prevent exploration of
X-interleavings.

## Test cases

- `test_simple_check_then_insert_is_found`: Baseline — DPOR finds the
  race with a simple pattern (PASSES).
- `test_orm_style_check_then_insert_race`: ORM-style pattern with
  intermediate operations — DPOR cannot find the race (XFAIL).
