# Search Strategies for DPOR Wakeup Tree Exploration

**Status:** ✅ Implemented. All 5 strategies (DFS, bit-reversal, round-robin, stride, conflict-first)
are in `crates/dpor/src/path.rs`, exposed via `DporEngine(..., search="bit-reversal:42")`.
Tests in `tests/test_search_strategies.py`.

This document originally contained the design and benchmarking rationale. The implementation
is complete; see the code and tests for details.
