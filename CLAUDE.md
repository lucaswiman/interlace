# CLAUDE.md

Deterministic concurrency testing library for Python, with a Rust DPOR
engine and an `LD_PRELOAD`-based C-level I/O interception library. See
`README.md` for the library surface.

## Project layout

- `frontrun/` — Python package (pure Python + compiled `_dpor` extension)
- `crates/dpor/` — Rust PyO3 DPOR extension, built by maturin to
  `frontrun/_dpor.so`
- `crates/io/` — Rust `LD_PRELOAD` library, built by cargo to
  `frontrun/libfrontrun_io.so`

## Environment

Prefer the 3.14 virtualenv at `.venv-3.14/`. Other envs live at
`.venv-3.10/` and `.venv-3.14t/`. Build with `make build-dpor-3.14` (or
`-3.10` / `-3.14t`); `make build-io` builds the C-level I/O library.

Run tools via the venv (e.g. `.venv-3.14/bin/python`). For anything that
exercises I/O interception, use the `frontrun` CLI wrapper, which sets
up `LD_PRELOAD`: `frontrun pytest -v tests/`.

## Running tests

**Always use `make test-<version>`.** It builds the DPOR extension and
I/O library, then runs pytest via the `frontrun` wrapper. Running
`.venv-3.14/bin/pytest` directly causes tests that use `explore_dpor()`
to be skipped or misconfigured.

- `make test-3.14` (or `-3.10` / `-3.14t`) — single version
- `make test` — all versions (3.10, 3.11, 3.12, 3.13, 3.14, 3.14t)
- `make test-3.14 PYTEST_ARGS="-v -k test_name"` — override pytest args

### Integration tests (Redis, HTTP, ORM)

`make build-integration-3.14` installs redis, requests, sqlalchemy, and
psycopg2-binary; `make test-integration-3.14` runs the integration
suite. Redis and Postgres are installed but not running by default:

- `redis-server --daemonize yes`
- `sudo pg_ctlcluster 16 main start`

## Quality gates

- `make check` — lint + type-check. Must pass before declaring a task done.
- Auto-fix with
  `ruff check --fix frontrun tests && ruff format frontrun tests`.
- Python >=3.10, line length 120, pyright strict, pytest + hypothesis.

## Development workflow

- Start sessions with `make rebuild` (silent on success). Development
  happens on both macOS and Linux containers, so stale cross-platform
  binaries are a recurring gotcha.
- Use red/green TDD: commit a failing test first, verify it fails for
  the right reason, then commit the fix.
- Run `make check` before declaring done. A cheap subagent with a fresh
  context is a good way to verify cleanly.
