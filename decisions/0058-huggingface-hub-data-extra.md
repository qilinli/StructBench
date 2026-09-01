# 0058 — `huggingface_hub` as an optional `data` extra

**Status**: Proposed
**Type**: Durable
**Date**: 2026-08-31

## Context

Since ADR-0040's 2026-08-28 amendment, the public Hugging Face mirror is the
live distribution channel for the three LS-DYNA canonical archives, and all
four benchmark cards say so in their `data_access` text: *"Fetch one case with
`hf_hub_download` or the whole archive with `snapshot_download` and point
`--data-root` at it."*

`huggingface_hub` was in neither `pyproject.toml` nor `uv.lock`. A clean
install therefore fails at the card's own first instruction with an
`ImportError`, and `tools/build_hf_bundle.py:265` writes `pip install
huggingface_hub` into every generated archive README — treating the gap as the
reader's problem rather than the platform's.

The audience this hits hardest is the one VISION.md most wants to serve:
structural engineers and research groups who are not programmers, for whom an
unexplained import error at step one is where the evaluation ends.

## Decision

Add **`huggingface_hub>=1.0`** as an **optional extra named `data`**, not a
core runtime dependency:

```toml
[project.optional-dependencies]
data = ["huggingface_hub>=1.0"]
```

Installed with `uv sync --extra data` or `pip install structbench[data]`. The
lower bound matches the `hf` CLI generation that `tools/build_hf_bundle.py`
already documents for publishing.

It is an extra rather than a core dependency because **the package never
imports it** — `src/structbench/` has zero occurrences. It is a user-facing
fetch tool and a maintainer-tool dependency, in exactly the position
`matplotlib` occupies for `viz` (ADR-0022).

## Alternatives considered

- **Core runtime dependency.** Rejected: `structbench` does not import it, and
  it pulls eight transitive packages (`anyio`, `httpx`, `httpcore`, `h11`,
  `hf-xet`, `click`, `pyyaml`, plus itself) into every install — including DUG
  training jobs that read a local `--data-root` and never touch the network.
- **Status quo: leave it to the user.** Rejected: the documented first step
  fails on a clean install, and the affected audience is the least equipped to
  diagnose it.
- **Vendor a small downloader over the HF resolve URLs with `urllib`.**
  Rejected: loses resumability, the `HF_HOME` cache, and revision pinning (the
  cards pin the dataset tag `v0.1.0`), and reimplements a maintained client
  badly for no dependency saving that matters at the extra level.

## Consequences

- `uv.lock` grows 84 → 92 packages. Verified additive only: no existing pin
  changed, and the `nvidia-*` lines in the diff are a reordering inside the
  `torch` entry (10 removed, 10 re-added).
- The four cards' `data_access` text and the generated archive READMEs should
  offer `pip install structbench[data]` in place of `pip install
  huggingface_hub` — a follow-up, since both are generated views.
- `viz` / `data` / `dev` now form one consistent pattern: optional capability
  groups, none of them imported by core.
- Core install stays lean for the training path, which was the reason to keep
  the runtime dependency list short in the first place.
