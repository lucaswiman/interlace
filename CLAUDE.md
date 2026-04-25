# CLAUDE.md

> `AGENTS.md` and `GEMINI.md` are symlinks to this file — edits here apply to all three.

Deterministic concurrency testing library. Four exploration approaches:

- **Trace markers** — `# frontrun: name` comments + line-level `sys.settrace` (`frontrun/trace_markers.py`, async: `async_trace_markers.py`)
- **Marker schedule exploration** — exhaustive interleaving at the marker level (also in `trace_markers.py`)
- **Random bytecode exploration** — opcode-level fuzzing with Hypothesis (`frontrun/bytecode.py`, async: `async_shuffler.py`)
- **Systematic DPOR** — Rust engine, vector clocks, wakeup tree (sync: `frontrun/_dpor_runtime/` subpackage; async: `frontrun/async_dpor.py`, currently a single ~1.3 kLOC module)

Unified entry point is `frontrun.explore(strategy=...)`. Older names (`explore_dpor`, `explore_interleavings`, …) resolve via a `__getattr__` deprecation shim in `frontrun/__init__.py` (`DEPRECATION_MESSAGES`). C-level I/O interception is handled by the `LD_PRELOAD` library in `crates/io/`.

## Project layout

- `frontrun/` — Python package (pure Python + compiled `_dpor` extension)
  - `_dpor_runtime/` — sync DPOR internals (`explore.py`, `scheduler.py`, `runner.py`, `replay.py`, `preload_bridge.py`)
  - `contrib/django/`, `contrib/sqlalchemy/` — framework-specific helpers (`_sync.py`, `_async.py`, `_shared.py`)
  - Large modules worth knowing about before editing:
    - `_opcode_observer.py` — `sys.settrace` vs `sys.monitoring` abstraction
    - `_cooperative.py` — lock/queue/event patching (sync only)
    - `_sql_cursor.py` + `_sql_parsing.py` + `_sql_cursor_async.py` — SQL access tracking
    - `_redis_client.py` / `_redis_client_async.py` — Redis access tracking
    - `async_dpor.py` — monolithic async DPOR (no subpackage, unlike sync)
- `crates/dpor/` — Rust PyO3 DPOR extension (built by maturin → `frontrun/_dpor.so`)
- `crates/io/` — Rust LD_PRELOAD I/O interception library (built by cargo → `frontrun/libfrontrun_io.so`)
- `tests/` — organized by feature: `test_dpor.py`, `test_bytecode.py`, `test_async_*.py`, `test_sql_*.py`, plus numbered `test_defect*.py` regressions
- `specs/` — TLA+ specifications for DPOR correctness
- `examples/`, `docs/`, `benchmarks/` — as named
- `Cargo.toml` — workspace root (`crates/dpor`, `crates/io`)
- `pyproject.toml` — maturin build backend (builds DPOR extension as `frontrun._dpor`)

## Environment setup

Use the Makefile to build virtualenvs. Prefer working in the **3.14** virtualenv:

- `make build-dpor-3.14` — build the 3.14 venv with Rust DPOR extension (preferred)
- `make build-dpor-3.10` / `make build-dpor-3.14t` — other versions
- `make build-io` — build the LD_PRELOAD I/O interception library (libfrontrun_io.so)
- Virtualenvs live at `.venv-3.14t/`, `.venv-3.10/`, `.venv-3.14/`
- Run tools via e.g. `.venv-3.14/bin/pytest`, `.venv-3.14/bin/python`
- Use `frontrun` CLI to run commands with I/O interception: `frontrun pytest -v tests/`

## Running tests

Always use `make test-<version>` to run tests. This builds the DPOR extension and I/O library, then runs pytest through the `frontrun` CLI wrapper (which sets up `LD_PRELOAD` for C-level I/O interception). Do **not** run `.venv-3.14/bin/pytest` directly — tests that use `explore_dpor()` will be skipped or misconfigured without the `frontrun` wrapper.

- `make test-3.14` / `make test-3.10` / `make test-3.14t` — single version
- `make test` — all Python versions (3.10, 3.11, 3.12, 3.13, 3.14, 3.14t)
- Override pytest args: `make test-3.14 PYTEST_ARGS="-v --timeout=120 -k test_name"`
- Fast inner loop: scope with `-k` (e.g. `PYTEST_ARGS="-k test_dpor"`); the full suite is slow.

### Integration tests (Redis, HTTP, ORM)

Integration tests require additional packages and services:

- `make build-integration-3.14` / `make build-integration-3.10` — install redis, requests, sqlalchemy, psycopg2-binary
- `make test-integration-3.14` — run integration tests only
- Redis and Postgres are available but not running by default. Start them with:
  - `redis-server --daemonize yes`
  - `sudo pg_ctlcluster 16 main start`

## Commands

- `make check` — lint + type-check
- `make lint` / `make type-check` — run separately
- `make clean` — remove artifacts
- Auto-fix: `ruff check --fix frontrun tests && ruff format frontrun tests`

## Conventions

- Python >=3.10, line length 120
- ruff (E, F, W, I, N, UP, A, C4, ISC, PIE, Q, LOG, PERF), pyright strict
- Tests: pytest + hypothesis
- Rust extensions: maturin (PyO3) for DPOR (`crates/dpor/`), cargo for LD_PRELOAD library (`crates/io/`)

## Gotchas

- 3.14t (free-threaded) routes opcode tracing through `sys.monitoring` rather than `sys.settrace` (CPython #118415). Both paths are abstracted in `_opcode_observer.py` — touch it carefully.
- `_cooperative.py` is sync-only by design: asyncio is already cooperative, so async paths use stock primitives. Don't try to add an async mirror.
- The async DPOR pipeline lives in a single `async_dpor.py` instead of mirroring `_dpor_runtime/` — when extending DPOR, expect to make changes in two shapes (subpackage + monolith).
- Old API names continue to work via `frontrun.__getattr__` (PEP 562); when adding/renaming public APIs, update `DEPRECATION_MESSAGES` and the alias table in `frontrun/__init__.py`.
- `bench_dp.py` (root), `tags`, and `_startdocker.sh` are stale top-level files unrelated to the active build — ignore them; the canonical benchmarks live under `benchmarks/`.

## Development workflow

* Start out a session with `make rebuild` — cleans and rebuilds everything, only outputs on failure or success. This is important since sometimes development is done on MacOS and sometimes in a docker container so incompatible binaries may be present.
* Always use red/green TDD: create a failing test case (or failing manual test procedure), then verify that it fails for the right reason. Commit the failing test. Then fix the failing test and commit that.
* Make sure that `make check` passes at the very end before declaring a task done. If possible, use a cheap subagent with a fresh context.
