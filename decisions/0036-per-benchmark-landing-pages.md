# 0036 — Per-benchmark landing pages: one generated docs page per benchmark (extends 0027)

**Status**: Accepted
**Type**: Durable
**Date**: 2026-07-09

## Context

A prospective user landing on the repo has no single place that answers
"what is this benchmark, what data does it use, how well do the reference
models do, and how do I start?" for one benchmark. The reference point is
NVIDIA PhysicsNeMo's `structural_mechanics/crash` example page, which puts
the problem, the dataset, a baseline-comparison table, and a one-command
quickstart on one browsable page. We want that per-benchmark landing page,
starting with Taylor 2D.

Most of it already exists as a *generated* view. `render_archive_readme`
(ADR-0027) already emits Task, Evaluation criteria, "Numbers to beat"
(ADR-0033 baselines), and a usage/quickstart section — from the card and
the results registry. What it lacks is (a) a home in the repo tree (it is
generated on demand to ship with the hosted dataset, ADR-0021/0031, not
committed as a browsable page) and (b) the parts a card cannot derive: a
physics problem narrative and figures.

Two constraints bound the design. ADR-0027 rejected hand-written per-folder
READMEs and hand-maintained index tables as guaranteed drift, and ADR-0033
requires baseline numbers to render only through generated views. So the
page cannot be hand-typed markdown: its structured content must render from
`card.py` + `results.py`. Separately, an `examples/<benchmark>/` code folder
(the literal PhysicsNeMo shape) does not fit — StructBench deliberately
centralises training into one `structbench-train` + grouped TOML configs
(ADR-0017, ADR-0032), so such a folder would hold no code, only a pointer to
`configs/<benchmark>/cgn.toml` that would drift against it.

## Decision

1. **One generated landing page per benchmark at
   `docs/benchmarks/<registry-name>.md`.** `docs/benchmarks.md` stays the
   cross-benchmark index and gains a link to each page. A new
   `render_benchmark_page(spec, name)` in `benchmarks/render.py` builds the
   page (`name` is the registry key — page filename and grouped-config path),
   reusing the existing archive-README section bodies (Task, Evaluation
   criteria, Numbers to beat) so the numbers have exactly one source; the page
   adds a problem-narrative section at the top, a figures section, and its own
   quickstart. `tools/gen_benchmark_docs.py` writes all pages, and a
   drift test asserts each committed page equals its render — the same
   no-drift guarantee `docs/benchmarks.md` already carries.

2. **The non-derivable content lives in the typed card, not a hand-file.**
   `BenchmarkCard` gains two optional fields:
   - `overview: str = ""` — a multi-paragraph problem/physics narrative
     (markdown), rendered as the page's lead section; empty renders no
     section.
   - `figures: tuple[BenchmarkFigure, ...] = ()` — an ordered list of a new
     frozen `BenchmarkFigure(path, caption, alt)` record, each pointing at a
     committed repo asset (e.g. `assets/taylor_rollout.gif`). Rendered as
     the figures section in order.
   This keeps the card the single typed source (ADR-0027), so `card.json`
   and every generated view stay consistent by construction. Both fields are
   optional, so the other three cards are unchanged and their pages render
   without a narrative or figures until authored.

3. **Figures are committed repo assets, promoted deliberately from run
   artifacts.** The rollout hero GIF is already committed
   (`assets/taylor_rollout.gif`); the prediction-vs-truth and error-vs-time
   figures a page references are copied from the (gitignored)
   `runs/<run>/plots/` into `assets/` and referenced by path. Which figures
   to promote for Taylor is a content choice made when the page is built,
   not fixed here. An environment-independent test validates that every
   `figure.path` on every card exists in the repo.

4. **This never bumps a benchmark version.** Pages, narrative, and figures
   are presentation; only protocol changes bump the version (ADR-0032 §4),
   exactly as ADR-0033 established for adding baseline results.

## Alternatives considered

- **`examples/<benchmark>/` folder mirroring PhysicsNeMo.** Rejected: our
  pipeline is centralised, so the folder holds no code, and its README
  numbers must still be generated (ADR-0033) — it would be the generated
  page with a misleading `examples/` label implying runnable per-example
  code we deliberately do not have.
- **Commit the archive README as-is under the data-archive path.** The
  smallest step, but it defers the narrative and figures indefinitely and
  splits "the page that ships with data" from "the page users browse"
  without giving the second one its missing content. Kept the single
  renderer but added the narrative/figure surface instead.
- **Narrative + figures as a hand-authored markdown fragment the generator
  includes**, rather than card fields. Workable and keeps prose as prose,
  but reintroduces a hand-file next to the card as a second source; the
  typed-card route stays consistent with ADR-0027 and keeps `card.json`
  self-complete. Chosen the card route; this remains a fallback if a
  narrative outgrows a comfortable Python string literal.
- **Hand-written landing page.** Rejected directly by ADR-0027/0033.

## Consequences

- `BenchmarkCard` gains two optional fields and a `BenchmarkFigure` record;
  the public card API and `card.json` schema grow (additive, back-compatible).
- `docs/benchmarks/` enters the tree; `tools/gen_benchmark_docs.py` emits N+1
  files (index + one page per benchmark) and the drift check / CI gate covers
  them. Regenerating pages joins the benchmark-addition checklist.
- Authoring a landing page for a benchmark is: write `overview`, list
  `figures` (committing the referenced assets), regenerate. The Taylor page
  ships first; the other three get pages as their narratives and baselines land.
- `assets/` grows by a few committed figures per benchmark; the promotion
  from `runs/**/plots` is manual and deliberate, keeping gitignored run
  artifacts out of history except where a page needs them.
