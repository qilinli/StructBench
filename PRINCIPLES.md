# PRINCIPLES.md

*Coding conventions and dependency policy for StructBench. Read at session start; consulted whenever code is written, reviewed, or a dependency is proposed.*

---

## Purpose

This document governs *how* code is written in StructBench: language version, style, typing, testing, documentation, logging, dependencies, and git conventions. It is the designated home for these rules — they are not duplicated in code comments, ADRs, or `CLAUDE.md`.

It sits alongside two siblings. `ARCHITECTURE.md` governs *what* the code is — module structure, interfaces, the case schema. `HARNESS.md` governs *why* the project is run the way it is. Where a convention here touches structure (the private-symbol boundary, the `src/`-layout) it cross-references `ARCHITECTURE.md` rather than restating it.

Conventions here are durable defaults, not laws of physics. A genuinely bad fit is flagged and the document revised — not overridden silently. Decisions that add or change a dependency are additionally recorded as ADRs (see *Dependency policy*).

---

## Language & runtime

- **Python 3.11+** is the floor. Code may use any feature available in 3.11 (`tomllib`, exception groups, finer-grained typing) and must not require 3.12+ without raising the floor here first.
- The package uses the **`src/`-layout** (`src/structbench/`); see `ARCHITECTURE.md` for the rationale and module map.
- Standard-library solutions are preferred over a new dependency when the difference is marginal. Adding a dependency is a deliberate act governed by *Dependency policy* below.

---

## Style & formatting

- **Ruff** is the single formatter and linter (it replaces black, flake8, and isort). Its configuration lives in `pyproject.toml`. Formatting is not a matter of taste once Ruff has run — match its output.
- **Line length: 88 columns.**
- **Imports** are ordered by Ruff's isort rules: standard library, third-party, first-party (`structbench`), each group separated by a blank line. Absolute imports within the package; relative imports only within a module.
- **Naming**: `snake_case` for functions, variables, and modules; `PascalCase` for classes; `UPPER_SNAKE_CASE` for constants. Names communicate intent over brevity.
- **The `_`-prefix is a hard boundary, not a hint.** A symbol whose name starts with `_` is private to its module and must not be imported across module boundaries, per the interface discipline in `ARCHITECTURE.md`.

---

## Type annotations

- **Type hints are required on every public API** — anything re-exported from a module's `__init__.py`. This includes function/method signatures (parameters and return) and public attributes.
- Internal and private code is **encouraged but not required** to be annotated; annotate it where it aids clarity, especially around the case schema and array shapes.
- **mypy** is the type-checker, configured in `pyproject.toml` and run in CI. Public APIs must pass; the configuration may relax checks for internal code but should not silence errors on the public surface.
- Prefer precise types: `numpy.typing.NDArray` for arrays, `TypedDict`/dataclasses for structured records, explicit `| None` over bare `Optional` where it reads cleanly. Array-shape conventions for schema fields are documented at their definition, not enforced by the type system.

---

## Testing

- **pytest** is the test framework. Tests live in a top-level `tests/` directory mirroring the package layout (`tests/core/`, `tests/datasets/`, …), not inside `src/`.
- Tests are first-class: writing and running them is a unilateral action (see `CLAUDE.md` authority tiers). A bug fix lands with a test that would have caught it.
- **What must be tested**: the case-schema readers/writers and validators, every public API in `core/`, and any non-trivial transform. Round-trip tests (write → read → compare) are the expected pattern for I/O.
- **Coverage** is a diagnostic, not a target to game. New code in `core/` and `datasets/` is expected to be well-covered; there is no blanket percentage gate.
- Tests must be deterministic and must not require a solver install, network access, or large data files. Fixtures use small synthetic cases; the `Taylor.k` reference deck is a development aid, not a test dependency.

---

## Documentation

- **Docstrings use NumPy style** (`Parameters`, `Returns`, `Raises`, `Examples` sections). This is the scientific-Python convention and renders cleanly under most doc tooling.
- **Every public API carries a docstring.** Private helpers are documented when their purpose is non-obvious.
- Each module's `__init__.py` carries a module-level docstring stating the module's responsibility — the one-sentence version of its `ARCHITECTURE.md` entry.
- Docstrings describe contract and intent (what, why, units, shapes), not implementation line-by-line. Code comments explain the non-obvious *why*; they do not narrate the *what*.
- Schema-facing code states **units and array shapes** in the docstring. Units are SI throughout (per ADR-0012); shapes follow the conventions in that ADR.

---

## Logging

- The package uses the standard-library **`logging`** module. **`print` is not used in library code** — only, sparingly, in `cli/` for user-facing output.
- Each module obtains its logger with `logging.getLogger(__name__)`. The library never configures the root logger or adds handlers at import time; configuration is the application's (or CLI's) responsibility.
- Levels: `DEBUG` for developer detail, `INFO` for normal progress, `WARNING` for recoverable surprises, `ERROR` for failures. Long-running data-generation and training steps log progress at `INFO`.
- **No secrets, credentials, or full file paths containing user identifiers** in log output. (Handling secrets at all is forbidden per `CLAUDE.md`.)

---

## Dependency policy

A dependency is a long-term commitment, not a convenience. The bar to add one is deliberately high.

**Runtime dependencies** — anything `structbench` imports at runtime — are **flag-first**: proposed with reasoning (what it does, why a stdlib solution won't serve, maintenance health) and **recorded as an ADR** once accepted. The approved list below is the authoritative record; if a package is not on it, it is not approved.

**Development dependencies** — tooling not shipped to consumers (formatter, type-checker, test runner, lockfile tool) — are established by this document and listed below. Adding a new dev tool is flag-first but does not require an ADR; it is recorded by editing this list.

**Version management**: the distributable library declares **loose lower bounds** (`>=`) in `pyproject.toml`, so downstream users can co-install it alongside their own stack. Reproducible development and CI environments are pinned exactly via a **`uv` lockfile** committed to the repo. Data-generation and training runs that need bit-level reproducibility pin against the lockfile.

### Approved dependencies

**Runtime** *(each added via flag-first proposal + ADR)*:

| Package | Purpose | ADR |
|---------|---------|-----|
| numpy | Array core; the case schema's arrays are NumPy arrays | ADR-0013 |
| h5py | Canonical reader/writer for the HDF5 case format | ADR-0013 |

**Development** *(established by this document)*:

| Tool | Purpose |
|------|---------|
| ruff | Formatting + linting |
| mypy | Static type-checking |
| pytest | Test framework |
| uv | Environment and lockfile management |

---

## Git conventions

- **Branches** never receive direct commits to `main`; work happens on feature branches. Branch names follow `type/short-description` (e.g. `init/foundation`, `feat/hdf5-io`, `fix/connectivity-indexing`).
- **Commits follow Conventional Commits**: a `type: summary` subject line (`feat`, `fix`, `docs`, `chore`, `test`, `refactor`), imperative mood, with a body explaining *why* when the change is non-trivial. Unfinished work is committed with a `WIP:` prefix (see `CLAUDE.md`).
- Agent-authored commits end with the `Co-Authored-By:` trailer for Claude Code.
- **Pushing, tagging, releasing, merging PRs, and rewriting shared history are out-of-session human actions** — forbidden within a coding session per `CLAUDE.md`. This document does not loosen that boundary.
