# CLAUDE.md

## Project overview

frontrun is a Python library for deterministic concurrency testing. It helps reliably reproduce and test race conditions using two approaches: random bytecode exploration and systematic DPOR (Dynamic Partial Order Reduction) with a Rust engine.

## Commands

- `make check` — Run lint + type-check (the full pre-commit check)
- `make lint` — Run `ruff check` and `ruff format --check` on `frontrun/` and `tests/`
- `make type-check` — Run `pyright` (strict mode, Python 3.13 target)
- `make test` — Run tests across all Python versions (3.10, 3.14, 3.14t); requires building the Rust DPOR extension
- `make test-3.10` — Run tests for a specific Python version
- `make build-dpor-3.10` — Build the Rust DPOR extension for a specific Python version
- `make clean` — Remove all build artifacts, virtualenvs, and caches

## Project structure

- `frontrun/` — Main Python package
  - `bytecode.py` — Random bytecode exploration engine
  - `dpor.py` — Systematic DPOR exploration engine (uses Rust extension)
  - `_cooperative.py` — Shared cooperative threading primitives
- `frontrun-dpor/` — Rust PyO3 extension for DPOR engine
- `tests/` — Test suite (pytest + hypothesis)
- `docs/` — Sphinx documentation

## Key conventions

- Python >=3.10, line length 120
- Linting: ruff with select rules E, F, W, I, N, UP, A, C4, ISC, PIE, Q, LOG, PERF
- Type checking: pyright in strict mode
- Auto-fix lint issues: `ruff check --fix frontrun tests` then `ruff format frontrun tests`
- Tests use pytest with hypothesis for property-based testing
- Rust extension built with maturin (PyO3)
