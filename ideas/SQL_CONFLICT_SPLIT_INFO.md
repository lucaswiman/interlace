# SQL_CONFLICT.md Split Into sql_conflict/ Directory

This file has been reorganized for better readability and maintainability.

## Where to Find the Content

The original `SQL_CONFLICT.md` has been split into **18 organized markdown files** in the `sql_conflict/` directory:

### Start Here
- **[sql_conflict/00_overview_and_sections.md](sql_conflict/00_overview_and_sections.md)** — Master table of contents, status tracking, and dependency graph

### File Organization (Lexicographically Ordered)
1. **01_problem.md** — Problem statement
2. **02_architecture_overview.md** — Design overview
3. **03_algorithm_1_sql_parsing.md** — SQL parsing (regex + sqlglot)
4. **04_algorithm_1_5_parameter_resolution.md** — Parameter resolution
5. **05_algorithm_2_cursor_patching.md** — DBAPI monkey-patching
6. **06_algorithm_3_endpoint_suppression.md** — Endpoint-level suppression
7. **07_algorithm_4_objectid_derivation.md** — ObjectId mapping
8. **08_algorithm_5_row_level_predicates.md** — Row-level detection (Phase 2)
9. **09_algorithm_6_wire_protocol.md** — Wire protocol parsing (Phase 3)
10. **10_integration_points.md** — Integration with dpor.py
11. **11_correctness_argument.md** — Soundness & completeness
12. **12_test_plan.md** — Unit & integration tests
13. **13_phased_implementation.md** — Implementation roadmap
14. **14_decisions_resolved.md** — Design decisions
15. **15_formal_verification.md** — TLA+ specs & verification
16. **16_file_by_file_changes.md** — File modification summary
17. **17_references.md** — Academic papers & tools

### Archive
- **SQL_CONFLICT.md.archive** — Original 1327-line file (kept for reference)

## Why Split?

- **Readability**: Each algorithm can be read independently
- **Maintainability**: Changes to specific stages don't require editing the full document
- **Navigation**: Master index (00) provides quick jumping to relevant sections
- **Phasing**: Easy to track which stages are completed/pending
- **Dependencies**: Explicit diagram of algorithm dependencies

## Key Improvements

✓ Files are numbered 00-17 to ensure lexicographic sorting matches logical flow
✓ Master overview (00_overview_and_sections.md) serves as entry point
✓ Quick-start guide and implementation checklist included
✓ All content preserved exactly as in original
✓ Links between files for easy cross-reference

## For Developers

1. **Start**: Read the master overview: `sql_conflict/00_overview_and_sections.md`
2. **Understand**: Jump to algorithms that interest you
3. **Implement**: Follow phases 1-4 in `13_phased_implementation.md`
4. **Verify**: Check correctness proofs in `15_formal_verification.md`
5. **Test**: Implement tests from `12_test_plan.md`

---

Created: 2026-03-03 | Status: Ready for Phase 1 implementation
