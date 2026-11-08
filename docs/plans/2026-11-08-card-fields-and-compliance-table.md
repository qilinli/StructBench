# Plan: the benchmark card's V&V fields and the data-standards compliance table

*Build plan for ADR-0065 follow-up 2. Written 2026-11-08 before the work;
the full suite is the gate. This is a plan, not a record: every judgment it
rests on is already made, and it cites them. The maintainer applied the
decision log's bar (`docs/decisions/README.md`, 2026-10-01) and ruled that
follow-up 2 is a plan plus a card change: the standard's items are fixed by
ADR-0073 D14, grandfathering by ADR-0065 (Consequences and the review note of
2026-10-01), the blind-evaluation row by ADR-0074 (clauses 1 and 2, the last
Consequence), the field names by ADR-0065's Consequences. The card as the
single typed source with generated views and drift tests is ADR-0027 and
ADR-0036; this plan extends ADR-0027's field list, which is a card-schema
change and so flag-first: the commit names it.*

## What is built

1. **New optional card fields** (`BenchmarkCard`, `src/structbench/benchmarks/card.py`),
   every one defaulting to empty so an unchanged card renders byte-identically:
   - `structural_scale` — the tier of the structure, one of `material`,
     `specimen`, `component`, `assembly` (ADR-0065's roadmap vocabulary:
     "component-tier", "assembly-tier"); validated when set.
   - `load_type` — the loading in a phrase (impact, stress wave, quasi-static
     contact, …). Free text: the existing `loading` field holds the
     parameters, this one names the kind.
   - `autonomy` — what the surrogate is given during the scored rollout
     beyond the seed frames and the case parameters: nothing (autonomous), or
     bodies driven from ground truth (notch-impact's impactor and supports,
     DeformingPlate's actuator and handle nodes). Free text.
   - `parameter_space: ParameterSpace | None` — the training design as
     ADR-0074 clause 3 asks the card to declare it: `design` (grid levels or
     Sobol count, or how the source authors sampled), `box` (the declared box
     of the *training* design per numeric parameter, so a scored case
     beyond it reads as *beyond* under clause 2), `margin` (the declared
     margin blind cases may lie within, clause 1; empty means none declared),
     `categorical` (levels of categorical parameters, whose unseen level is
     *novelty*).
   - `ground_truth: GroundTruth | None` — `regime` (the solver, formulation
     and what one case is) and `uncertainty` (what is measured about the
     reference's own error: a noise floor, a refinement difference; or that
     nothing is).
   - `compliance: tuple[ComplianceRow, ...]` — the table below.

2. **The compliance table.** One `ComplianceRow(item, status, reason)` per
   standard item. The items, in render order, are a fixed vocabulary
   (`STANDARD_ITEMS`): `convergence_run`, `noise_floor`, `survivor_log`,
   `constitutive_sensitivity` (required of a benchmark admitted after
   2026-10-01, ADR-0073 D14); `physical_test_anchor`,
   `second_solver_reproduction` (recorded where they exist, not required,
   D14); `blind_evaluation` (ADR-0074). The status is one of `pass` (the
   item exists and is recorded), `missing` (it does not exist yet),
   `not_applicable` (the item has no meaning for this data), `never` (it
   cannot be produced); the reason is one short sentence and is never
   blank. Validation rejects an unknown item, a duplicate, an unknown status
   and a blank reason. A card may carry no rows (the default); a *registered*
   benchmark must cover every item, enforced by a test rather than by the
   dataclass so that a test fixture or a definition loaded by path
   (ADR-0032 note, 2026-11-06) is not forced to declare what it has not
   assessed.

3. **Rendering** (`render.py`). A `## Data standard` section: one paragraph
   saying what the standard is, which items are required, that the shipped
   benchmarks are grandfathered, and what each status means; then the table
   `| Item | Status | Reason |`. On the landing page it sits after
   `## Evaluation criteria` and before the results (`## Leaderboard`, the
   section ADR-0036 called *Numbers to beat*); in the archive README, which
   carries no results, it sits before `## Benchmark protocol`, the pointer
   to them. The new scalar fields render as bullets in *Data at a glance*
   (page) and *Dataset summary* (README), each only when declared. A card
   without compliance rows renders no section. `card.json` carries the new
   fields through `asdict`.

4. **The four shipped cards, filled honestly.** Reasons name the evidence:
   - *Taylor2D-Impact*: convergence run `pass` (the held-aside
     `T-20-80-Convergence` case, ADR-0019 clause 4; the 4x-finer re-run
     differs by ~0.27 relative L2 on von Mises, one refinement level, recorded
     on the card's overview; the verification engine's two-grid row is still
     owed, README Roadmap); noise floor `pass` (the particle-scale von Mises
     floor ~0.12 relative L2, measured 2026-08 and recorded on the card);
     survivor log and constitutive sensitivity `missing`; physical-test
     anchor `missing` (the `taylor_copper` reference set exists, ADR-0072,
     but no record against this dataset: runs at the experiments' conditions
     are owed); second-solver reproduction `missing`; blind `missing`
     (when solver access allows, ADR-0074).
   - *Wave1D-Propagation* and *NotchBeam2D-Impact*: the first six `missing`;
     blind `missing` (when solver access allows).
   - *DeformingPlate*: the first six `missing` (published data; the source
     authors published none of them); blind `never` (no generator: the
     platform cannot produce withheld cases, ADR-0074).
   The scalar fields: all four are specimen-tier; load types and autonomy as
   the protocols define them (ADR-0019, 0025, 0026, 0043); parameter spaces
   are the published factor grids (Taylor 3 lengths x 7 training velocities,
   wave 4 x 4 interior holdout, notch 3 x 3 x 3 x 4 with shape and notch
   position categorical) and, for DeformingPlate, the source authors'
   sampling with the parameters unpublished; ground-truth uncertainty is
   measured on Taylor alone.

5. **Tests.** `tests/benchmarks/test_card.py`: defaults, validation of the
   new records, JSON serialisation. `tests/benchmarks/test_render.py`: a card
   without the new fields renders exactly as before (no section, no bullets);
   with them, the section's position in both views. New
   `tests/benchmarks/test_compliance.py`: every registered card's rows cover
   `STANDARD_ITEMS`, the DeformingPlate blind row reads `never`, the
   LS-DYNA blind rows `missing`. The committed pages are regenerated; the
   drift tests pass.

6. **Closing the thread.** Dated notes after a `---` rule on ADR-0065
   (follow-up 2 built by this plan, not an ADR) and ADR-0074 (its last
   Consequence's "the next record drafted" reads "the plan of 2026-11-08");
   a one-line note on ADR-0027 (its field list is extended by this plan);
   this plan's row in `docs/plans/README.md`; the README Roadmap item and
   the `CLAUDE.md` snapshot sentence that said follow-up 2 was undrafted.

## Not in this plan

Coverage computation, the `blind` set, registry fields and tiers
(ADR-0074's own plan); the properties block (ADR-0075's plan); a card
field for the validation record (README Roadmap, ADR-0072); the
`leaderboard` → `results registry` wording sweep (ADR-0074 Consequences).
The compliance table shows a status; it does not gate anything, and a
benchmark's status changes by editing its card and regenerating.

## Gate

Full suite green (`python -m pytest -q`, with `set -o pipefail`); the
generated pages byte-match their renders; a card constructed with none of
the new fields renders the same text as before this plan.
