# Design: ROADMAP.md becomes a living todo list in README.md

**Date**: 2026-07-05
**Status**: Approved (maintainer, in-session)
**Scope**: docs-only change — README.md, ROADMAP.md (deleted), CLAUDE.md,
decisions/0009 (dated amendment), docs/ARCHITECTURE.md, docs/WORKFLOW.md.

---

## Problem

ROADMAP.md is the planning home but has drifted from reality: several v0.2
definition-of-done items are complete yet unchecked (benchmark cards,
benchmark-selection registry), while genuinely new work (ADR-0030
follow-through, the ADR-0028 retrain) appears nowhere. The maintainer wants
**one plan home**, rendered on the repo's GitHub landing page, that supports
ad-hoc additions with zero tooling: done items crossed out with dates, new
items appended freely.

## Decisions (settled in-session)

1. **One plan home** — no TODO.md/ROADMAP.md coexistence.
2. **The home is a `## Roadmap` section in README.md**; ROADMAP.md is deleted
   outright (no stub — git history is the record). Chosen over a generated or
   linked view because ad-hoc additions must be one edit to one file,
   including from the GitHub web UI.
3. **Milestone-grouped shape** (v0.1 / v0.2 / Inbox / Later), not
   status-grouped or flat: release progress stays readable and ADR references
   stay attached. An **Inbox** subsection receives untriaged ad-hoc items.
4. **Sub-task granularity for pending items** (indented checkboxes); done
   work stays compressed at one line.

## Conventions for the living list

Recorded as an HTML comment at the top of the section (invisible in the
GitHub render):

- Done = `[x]` + strikethrough + `(YYYY-MM-DD)`.
- Ad-hoc additions land in **Inbox** and are triaged into a milestone during
  planning sessions.
- When a milestone ships, its crossed-out block may be compressed to one line.
- Rationale lives in `decisions/`, never in the list (HARNESS tenet 2).
- Substrate-layer work only, per ADR-0014's litmus test.
- A `*Last revised: date*` line is updated with each edit.

## Section content — verified against the repo (2026-07-05 sweep)

A five-agent verification workflow checked every ROADMAP claim against the
codebase, ADR log, and git history. Corrections it produced:

- **Done, previously unchecked**: benchmark cards (ADR-0027) fully
  implemented including Taylor retrofit and generated views; benchmark
  selection is a lazy registry (`src/structbench/benchmarks/registry.py`) —
  `cli/train.py` no longer hard-imports Taylor.
- **Reframed**: the v0.1 "trained baseline" is a *retrain* — the first full
  DUG run (2026-07-03) exposed five recipe defects; ADR-0028's recipe is in
  code but the run with it hasn't happened.
- **New tasks (ADR-0030 follow-through)**: the ADR file was never written or
  indexed though commit 84d84eb and `patch_units.py` cite it;
  `SOURCE_UNITS = "g-mm-ms"` is still hardcoded in
  `1DWavePropagation/convert.py` and `2DNotchBeam/convert.py`; three
  benchmark cards still declare `g-mm-ms`; patch execution over the 237
  canonical HDF5s is unconfirmed (script is idempotent).
- **Inbox seeds**: `lr_init` code default 1e-3 vs ADR-0028's 1e-4
  (TOML-only); Taylor-deck unit sanity check; ADR-0012 Voigt-prose
  reconciliation (from CORRECTIONS.md).
- **From ARCHITECTURE.md promises**: leaderboard submission validator,
  cross-benchmark eval utilities, published checkpoint per model → Later.

The full section text (approved draft) ships in the implementation commit;
its structure: `### v0.1` (7 crossed-out + retrain & release pending),
`### v0.2` (5 crossed-out + ADR-0030 follow-through, three baselines,
cracked_fraction validation, dataset hosting, archive packaging),
`### Inbox`, `### Later` (compact bullets: v0.3 RC beam/erosion, segmented
beam, MS-GNS, training robustness, eval growth, checkpoint publishing,
data-generation autonomy, scale, other solvers, SHM, deployment, packaging,
interop).

## ROADMAP.md retirement — content rehoming

| ROADMAP content | New home |
|---|---|
| Milestone DoD lists, near/later horizons, parked rows | README `## Roadmap` (parked rows become annotations on their items) |
| Scope rule (ADR-0014 litmus) | One line in the section's HTML comment |
| Open question: dataset hosting | v0.2 checklist item (it gates both releases) |
| Release-history note (RESEARCH-PROGRAM.md untracked; clean-cut republish caution) | `docs/WORKFLOW.md`, short "Publication note" — git-process guidance, not landing-page material |
| Resolved-question footnotes | Dropped — recorded in ADR-0021 already |

## Reference updates (grep-verified complete list)

| File | Change |
|---|---|
| README.md:12 | status-blurb link → `#roadmap` anchor |
| README.md:173–183 | section body replaced by the todo list; line 183 keeps only the `decisions/` pointer |
| README.md:170, 188 | drive-by: stale "21 ADRs" counts → count-free phrasing |
| CLAUDE.md:19, 42, 132 | → "the Roadmap section of README.md" |
| decisions/0009 (line 27) | dated revision-log amendment: conditional read is now README § Roadmap |
| docs/ARCHITECTURE.md:73 | "see ROADMAP.md" → the README section |
| docs/WORKFLOW.md | + Publication note (from ROADMAP's release-history note) |
| decisions/0001, 0021, 0024; docs/plans/2026-07-03-* | **untouched** — historical records of their time (ADR-0009 precedent) |

## Out of scope

Executing any discovered task (ADR-0030 drafting, SOURCE_UNITS fixes, card
regeneration, retraining). Those are list *entries*, not part of this change.
No README restyling beyond the section and the two stale-count fixes.

## Error handling / verification

- `tools/gen_benchmark_docs.py --check` still passes (no generated file is
  touched by this change).
- All grep hits for `ROADMAP` after the change are either historical
  documents (allowed) or the deleted file itself (gone).
- README renders correctly on GitHub (checkbox + strikethrough combination).

## Process

Feature branch `docs/readme-roadmap-todo`; conventional commits; merge to
`main` only on explicit in-session instruction (ADR-0023). CLAUDE.md and
ADR-0009 edits are flag-first — covered by the in-session approval of this
design; the human finalises the ADR amendment at merge time.
