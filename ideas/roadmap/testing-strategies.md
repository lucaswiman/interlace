# Remaining Testing Strategy Extensions

## Overview

The core property-based marker schedule testing infrastructure is **fully implemented**:

- `marker_schedule_strategy()` ✓ — Hypothesis strategy for marker-level schedule generation
- `all_marker_schedules()` ✓ — Exhaustive enumeration of all valid interleavings
- `explore_marker_interleavings()` ✓ — Exhaustive exploration with invariant checking
- Bytecode-level exploration (`explore_interleavings`) ✓ — Random bytecode schedules
- DPOR systematic exploration (`explore_dpor`) ✓ — Rust-backed conflict analysis

This document captures extensions and improvements to these existing implementations.

---

## Extension 1: Adaptive Marker Placement

**Status:** Not implemented
**Complexity:** Medium

Automatically identify which lines in source code are good candidates for marker placement based on static analysis of lock/variable access patterns.

### Idea
Instead of requiring users to manually add `# frontrun:` comments, analyze the code to:
1. Detect potential race windows (e.g., between unlocking one lock and acquiring another)
2. Suggest marker placement at those locations
3. Optionally auto-insert markers with user approval

### Implementation approach
- Parse AST to identify lock acquire/release patterns
- Identify gaps between lock operations (potential race windows)
- Generate marker suggestions at those locations
- Provide a CLI tool or decorator to apply suggestions

### Value
- Reduces annotation burden for developers
- Makes marker placement more systematic and less error-prone
- Could integrate with linters to flag suspicious race windows

---

## Extension 2: Hypothesis Integration for Marker Schedules

**Status:** Partially done
**Complexity:** Low

The current `marker_schedule_strategy()` works with Hypothesis, but we could add convenience decorators and profiles.

### Remaining work
1. **Pre-defined profiles** for common patterns:
   - `simple_two_thread()` — Quick setup for 2-thread tests
   - `pipeline()` — Producer-consumer patterns
   - `barrier()` — Multi-thread synchronization

2. **Automatic schedule shrinking visualization**:
   - When Hypothesis finds a counterexample, pretty-print the minimal schedule
   - Show timeline of events for easier debugging

3. **Integration with `@frontrun` decorator**:
   ```python
   @frontrun(
       threads={"producer": ["read", "write"], "consumer": ["read"]},
       exhaustive=True,
   )
   def test_queue_safety(schedule):
       ...
   ```

### Value
- Makes marker-based testing more accessible to users unfamiliar with Hypothesis
- Better error messages and debugging output

---

## Extension 3: Hybrid Marker + Bytecode Exploration

**Status:** Not implemented
**Complexity:** Medium

Combine marker-level and bytecode-level exploration for cases where markers bracket a race but the exact bytecode interleaving within that window matters.

### Idea
1. User places markers at high-level race windows (e.g., lock acquire/release boundaries)
2. Tool explores all marker-level interleavings
3. For each marker-level schedule, run bytecode-level exploration *within* that schedule
4. This creates a two-level search: coarse (marker) + fine (bytecode)

### Implementation approach
```python
def explore_hybrid_interleavings(
    setup: Callable,
    threads: dict[str, tuple[Callable, list[str]]],
    invariant: Callable,
    bytecode_per_marker: bool = True,  # fine-grained search within each marker schedule
    bytecode_attempts: int = 100,
):
    """Explore marker schedules, then bytecode-level within each."""
```

### Value
- Captures bugs that require both marker-level ordering AND specific bytecode interleaving
- Reduces search space vs. pure bytecode exploration
- Guarantees coverage of marker-level interleavings

---

## Extension 4: Schedule Filtering and Constraints

**Status:** Not implemented
**Complexity:** Low to Medium

Allow users to filter generated schedules by custom constraints (e.g., "thread A must reach marker X before thread B reaches marker Y").

### Idea
```python
schedules = all_marker_schedules(
    threads={"t1": ["a", "b"], "t2": ["x", "y"]},
    constraints=[
        ("t1", "b", "<", "t2", "y"),  # t1's "b" before t2's "y"
    ],
)
```

Or with Hypothesis:
```python
@given(schedule=marker_schedule_strategy(
    threads={...},
    assume=lambda s: must_order(s, ("t1", "read"), ">", ("t2", "write")),
))
```

### Implementation approach
- Add optional `constraints` parameter to `all_marker_schedules()`
- Filter schedules post-generation based on constraint predicates
- For Hypothesis, use `.filter()` on the strategy

### Value
- Lets users focus on semantically interesting interleavings
- Reduces search space for large marker sets
- Useful for tests where certain orderings are physically impossible or uninteresting

---

## Extension 5: Distribution Analysis of Marker Interleavings

**Status:** Not implemented
**Complexity:** Low

Provide statistics on how marker interleavings are distributed and whether the random strategy is uniform.

### Idea
When using `marker_schedule_strategy()` with Hypothesis, we could measure:
- How often each marker ordering appears across generated examples
- Whether some interleavings are under-represented
- Visualization of the search space

### Implementation approach
- Add optional stats collection to `marker_schedule_strategy()`
- Log frequency of each unique schedule
- Provide helper function to analyze Hypothesis example database

### Value
- Helps users understand test coverage
- Can detect if shrinking is biasing toward certain interleavings
- Useful for CI reporting

---

## Extension 6: Async/Await Marker Support

**Status:** Not implemented
**Complexity:** Medium

Extend marker-based scheduling to async code (coroutines, tasks, etc.).

### Current state
- Only works with threading (synchronous code with `sys.settrace`)
- `explore_interleavings` has async variants, but markers don't

### Idea
1. Add marker detection for async code
2. Use async-aware tracing to intercept at marker points
3. Create async equivalents of `TraceExecutor` and `explore_marker_interleavings`

### Implementation approach
```python
async def async_marker_executor(...):
    """Async version of TraceExecutor."""

async def explore_async_marker_interleavings(...):
    """Exhaustive async marker exploration."""
```

### Value
- Covers an increasingly important concurrency model
- Could find bugs in async concurrent code (e.g., race conditions in asyncio event loop interactions)

---

## Extension 7: Multi-Level Markers

**Status:** Not implemented
**Complexity:** Medium

Allow markers at different granularity levels (coarse, medium, fine) and explore at multiple levels.

### Idea
```python
# Coarse markers (logical phases)
# frontrun: phase:1
# frontrun: phase:2

# Medium markers (lock operations)
# frontrun: acquire_lock
# frontrun: release_lock

# Fine markers (critical sections)
# frontrun: read_shared_data
# frontrun: write_shared_data
```

Tool could explore at any granularity or combine them.

### Implementation approach
- Extend marker syntax to include optional level/category
- Parse and group markers by level
- Generate strategies for subsets of markers at chosen granularities

### Value
- Reduces marker count by consolidating into logical phases
- Allows progressive refinement of test design
- Easier to understand and maintain large marker sets

---

## Extension 8: Marker Coverage and Regression Testing

**Status:** Not implemented
**Complexity:** Low

Track which marker combinations have been tested and flag gaps.

### Idea
After running marker-based tests, generate a report of:
- Which marker-level interleavings were actually executed
- Which were missed or underexplored
- Recommendations for additional test cases

### Implementation approach
- Instrument `explore_marker_interleavings` to log executed schedules
- Compare against `all_marker_schedules()` to identify gaps
- Generate human-readable coverage report

### Value
- Helps teams know if they've covered all interesting cases
- Detects regressions (e.g., if a bug fix is only tested on one marker interleaving)

---

## Extension 9: Integration with Existing Test Frameworks

**Status:** Not implemented
**Complexity:** Low

Provide pytest plugins and integrations to make marker-based testing feel native to the test suite.

### Idea
```python
# pytest plugin
import pytest
from frontrun.pytest_plugin import frontrun_markers

@pytest.mark.frontrun_markers(
    threads={"t1": ["a", "b"], "t2": ["x", "y"]},
)
def test_something():
    ...

# Runs exhaustively by default, respects -m, --markers, etc.
```

### Implementation approach
- Create pytest plugin that auto-discovers marker declarations
- Hook into parametrization to generate schedule variants
- Add summary output to pytest reports

### Value
- Seamless integration into existing test infrastructure
- Familiar test discovery and execution patterns
- Better IDE support (markers recognized by pytest plugins)

---

## Extension 10: Comparative Benchmarking

**Status:** Not implemented
**Complexity:** Low

Compare effectiveness and performance of marker vs. bytecode exploration on the same code.

### Idea
```python
result_markers = explore_marker_interleavings(...)
result_bytecode = explore_interleavings(...)

compare(result_markers, result_bytecode)
# → Marker strategy found bug in X schedules
# → Bytecode strategy found bug in Y schedules
# → Marker strategy was Z% faster
```

### Implementation approach
- Add benchmark harness
- Run both strategies on the same test case
- Generate comparison report (time, coverage, bugs found)

### Value
- Empirical data on when each strategy is best
- Helps users choose the right approach for their code
- Could feed back into adaptive strategy selection

---

## Summary Table

| Extension | Status | Complexity | Priority |
|-----------|--------|-----------|----------|
| Adaptive marker placement | Not done | Medium | High |
| Hypothesis convenience features | Partial | Low | Medium |
| Hybrid marker + bytecode | Not done | Medium | Medium |
| Schedule filtering/constraints | Not done | Low-Medium | Low |
| Distribution analysis | Not done | Low | Low |
| Async/await support | Not done | Medium | Medium |
| Multi-level markers | Not done | Medium | Low |
| Coverage & regression tracking | Not done | Low | High |
| pytest plugin integration | Not done | Low | High |
| Comparative benchmarking | Not done | Low | Low |

---

## Recommended Next Steps

**High priority (most user-facing value):**
1. Pytest plugin integration
2. Coverage & regression tracking
3. Adaptive marker placement

**Medium priority (fills capability gaps):**
1. Async/await support
2. Hybrid marker + bytecode exploration

**Nice-to-have (better UX):**
1. Hypothesis convenience profiles
2. Schedule constraints
3. Comparative benchmarking
