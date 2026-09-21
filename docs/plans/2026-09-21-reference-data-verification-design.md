# Design: reference-data verification (`verification/`) and the Taylor slice

**Date**: 2026-09-21
**Status**: Draft for maintainer review (decisions below settled in-session)
**Scope**: new module `src/structbench/verification/`; two new `core` files
(`core/evidence.py`, `core/io/lsdyna_run.py`); `cli/datacheck.py`; per-dataset
glue under `data_generation/`; `docs/datachecks/`; `docs/ARCHITECTURE.md`;
README Roadmap. Governing decision: ADR-0066 (Proposed).

---

## Problem

ADR-0065 asks the platform to ship reference data "whose uncertainty is
declared" and says none of the four benchmarks would be admitted today. The
repository cannot currently say *why* for any single run: no code measures
termination, time-step behaviour, energy balance, particle health,
constitutive consistency, or data integrity; the only such checks are asserts
in exploratory `tools/state_probe/`; provenance says `"unknown"`.

The goal is to encode that judgment once, as deterministic checks, so that
(a) what varies by solver, by formulation, by material class, and by
benchmark each has exactly one home; (b) the same kernels later serve
model-side verification (ADR-0065 follow-up 1); and (c) the footprint is the
smallest that works.

## Decisions (settled in-session, 2026-09-21)

1. **New top-level module, named `verification/`**, layered between
   `datasets/` and the peer modules.
2. **Acceptance criteria are a priori and platform-level.** The maintainer
   verifies an external source and commits the criterion before the cases it
   judges are measured. Until then the row is
   `not_assessable / no_ratified_criterion`. Instrument tolerances are the
   only numbers calibrated from measurement.
3. **The published record is measurements, not verdicts**:
   `docs/datachecks/<benchmark>.json` plus a generated `.md`, with a
   data-free drift test.
4. **Stage 1 covers the full public-tier catalogue** (about 28 quantities),
   not a lean port.
5. **Start with Taylor as a vertical slice**; widen to wave-1D, then
   notch-impact.

## How the design was reached

Four read-only surveys of the repository; three independent module designs
(subtraction-first, layered-portability, consumer-first); three adversarial
reviews (governance, computational mechanics, buildability). No design won
outright: governance ranked subtraction first, mechanics ranked layered
first, buildability ranked consumer first. The synthesis takes the
measure/judge split and the machine contract from *consumer*, the verdict
ownership order, per-term energy ledger and criteria records from *layered*,
and the sequencing and "no declaration until a fact needs it" discipline
from *subtraction*.

## The model

**Two verbs.** `measure`: evidence → threshold-free measurement, or a typed
absence. `judge`: measurements + criteria → verdicts; never touches data.

**Four verdicts, fixed ownership and order.**

| Order | Owner | Verdict | Example |
|---|---|---|---|
| 1 | formulation traits | `not_applicable` | hourglass energy on an SPH part |
| 2 | evidence absence | `not_assessable` + reason | energy decomposition when the deck never requested it |
| 3 | measurement vs criterion | `pass` / `fail` | yield ratio against its tolerance |

The test between rows 1 and 2: *could better evidence change the answer?*
`not_applicable` is neutral; `not_assessable` is a gap whose reason names
how to close it. Reasons: `not_requested`, `not_computed`, `not_ingested`,
`source_missing`, `unparsable`, `source_unreadable`, `unsupported`, and
(judge-side) `no_ratified_criterion`. Unknown traits give `not_assessable`.
On Taylor the deck has a `*HOURGLASS` card and hourglass-energy computation
switched off; evaluating traits first gives the truthful `not_applicable`
(SPH), where the same switches on an under-integrated FE model give
`not_assessable / not_computed` — a real gap, because the hourglass work then
silently leaves the energy balance.

**Three evidence tiers.**

| Tier | Source | Availability | Feeds |
|---|---|---|---|
| `case` | canonical `.h5`: response fields, global energies | public | constitutive, closures, particle health, integrity |
| `deck` | `metadata/source_deck` in the same file | public | traits, requested outputs, energy switches, mass scaling, tables, deck lint |
| `run` | solver message file, global-statistics file | maintainer-held | termination banner, cycles, time-step series, energy ledger, solver identity |

When tiers disagree the `run` tier is authoritative and both values are
reported. Run-tier measurement needs no `Case`.

## What is shared by what

| Category | Shared by all | Per solver (`core/io`) | Per formulation (traits) | Per material class | Per benchmark | Reusable model-side |
|---|---|---|---|---|---|---|
| Numerical health | verdict logic, criteria | banner, cycles, time step, added mass, solver identity, deck keywords | explicit: time step, mass scaling · under-integrated FE: hourglass · SPH: deactivated particles, neighbour count, smoothing length vs the deck's own bounds, wall penetration | — | — | wall non-penetration |
| Conservation | signed, normalised energy residual; KE and IE closure; active mass | filling the energy ledger and explaining each absent term | which terms exist (rigid wall, contact, erosion); closed-system test; SPH formulation id recorded beside the verdict | EOS closure (later) | — | closure kernels |
| Constitutive | kernels on plain arrays | keyword → class; faithful card reader | plane strain vs axisymmetric | meaning of the internal-variable slot (plastic strain vs damage), bounds, which yield law is assessable | the one yield table (`BenchmarkSpec.hardening_curve`) | fully shared |
| Data integrity | finite values, time axis, artifact frame, declared-vs-derived cross-checks, deck lint | source-units token | alive mask under erosion (later) | table matches deck, is monotone, covers the strain range | `card.fields`, declared traits | finite/shape |
| Cross-source | two-source QoI difference | — | what refinement holds fixed | — | which pairs and anchors exist | reference-uncertainty number |

Material knowledge is deliberately **two tables**: *solver keyword →
material class* is solver-specific and lives with the adapter; *class →
semantics* is solver-neutral and lives in `verification/materials.py`. The
second table is what lets one schema field (`effective_plastic_strain`) be
checked as plastic strain for a J2 metal and as scaled damage in `[0, 2]` for
the K&C concrete model.

Traits are **per part**, derived from the deck's part → material map, never
from a case file's element keys. The distinction is real on Taylor: the deck
defines one SPH part and nothing else, but the canonical file also holds a
four-node shell (4804 nodes against 4800 particles) — visualisation geometry
written into the d3plot, with no part in the deck. Its origin (pre-processor
scaffold or solver-drawn rigid wall) is unverified and does not matter to
the rule: an element block with no deck part is non-structural, excluded
from every check, and counted by an integrity row
(`elements_without_deck_part`). Card declarations are cross-checked against
derived traits (`declared_traits_match_deck`) and are the fallback for
deck-less data.

## Repository layout

```
src/structbench/
  core/evidence.py          Absence, AbsenceReason, PartTraits, DeckFacts, RunEvidence (SI)
  core/io/lsdyna_run.py     deck text → DeckFacts; message / global-statistics TEXT → RunEvidence
  verification/
    kernels.py              array-level measures returning small mergeable records
    materials.py            material class → state-variable meaning, bounds, assessability
    quantities.py           catalogue: name, category, unit, meaning of a violation, definition_version
    measure.py              measure_case(case | None, deck_facts, run_evidence) -> CaseMeasurements
    criteria.py             Criterion records, the platform standard, judge()
    report.py               JSON and markdown
  cli/datacheck.py          python -m structbench.cli.datacheck measure | judge
data_generation/lsdyna/2D-Copper-Bar-Taylor-Impact/collect_run_evidence.py
tests/{core,verification,cli}/
docs/datachecks/<benchmark>.{json,md}
```

Import edges: `verification → {core, datasets}`; `benchmarks`, `eval`, `cli`
may import `verification`. `datasets` exports `n_valid_frames` (today it is
reachable only through `datasets.canonical`).

Sketch of the central types (signatures are pinned in the implementation
plan):

```python
class Verdict(str, Enum): PASS; FAIL; NOT_APPLICABLE; NOT_ASSESSABLE
class Tier(str, Enum): CASE; DECK; RUN
class CriterionKind(str, Enum): ACCEPTANCE; INSTRUMENT

@dataclass(frozen=True)
class Measurement:            # threshold-free
    quantity: str; value: float | None; unit: str
    tier: Tier | None; n_samples: int | None
    absence: Absence | None   # invariant: (value is None) == (absence is not None)
    detail: Mapping[str, float | int | str]

@dataclass(frozen=True)
class Criterion:              # platform standard
    quantity: str; lo: float | None; hi: float | None
    kind: CriterionKind; rationale: str; provisional: bool   # rationale non-blank

def judge(measurements: DatasetMeasurements,
          criteria: Mapping[str, Criterion]) -> DatasetReport: ...
```

Every catalogue quantity carries a one-line **meaning of a violation** (for
example "solver output untrustworthy", "export or post-processing error"),
rendered in the report. PASS on numerical-health and conservation rows is a
necessary solution-verification indicator; the report says so, and says it
is not evidence of accuracy.

## Physics corrections the review forced

1. The value 1.001 in ADR-0064 bounds a *model head's* float32 saturation
   error (~2e-7). The excess measured on reference data, 3.5e-4, comes from a
   single case — the slowest and shortest — and is far above float32
   round-off. A candidate cause is the solver's discretised plastic return
   across a knot of the piecewise-linear hardening table, which would grow
   with impact velocity; this is a hypothesis to test, not an established
   fact. So the yield tolerance is a provisional instrument tolerance,
   calibrated over all cases, and the measurement reports the plastic strain
   at the maximum, the frame, the distance to the nearest table knot, and the
   violation fraction.
2. The yield check needs a **lower-bound companion**: if any sample has
   yielded, some sample must sit near the surface. Without it, stress stored
   a thousand times too small passes silently — the ADR-0030 bug class.
3. Comparing stored density with deck density is **not a units check**; both
   sides pass through the same conversion factor, and the comparison would
   have passed every mislabelled ADR-0030 file. It is kept under its honest
   name, `density_slot_matches_deck`. A real units check needs an anchor
   outside the label and arrives with the notch widening.
4. **Energy**: report signed excursions (largest gain, largest loss, final);
   normalise by the largest of kinetic, internal, and external-work
   magnitudes so driven and from-rest systems are defined; treat the system
   as closed only when every energy-relevant deck keyword is on a known list
   (fail closed); record the SPH formulation id and the energy-accounting
   switches beside the verdict; measure the difference between the canonical
   and run-tier totals, which settles what the public "total energy"
   contains. A ratio above one in a closed system is energy creation, not
   dissipation.
5. Termination and integrity checks read the **raw** time axis and all stored
   frames. Only checks that need a uniform interval apply `n_valid_frames`
   (ADR-0028); on the artifact-dropped axis every healthy Taylor case would
   fail `reached_end_time`.
6. Plane strain is a **deck trait**, tested by zero out-of-plane normal
   strain; zero out-of-plane shear only proves "2D".
7. SPH-specific numerical health — deactivated particles, neighbour
   starvation and clumping, smoothing length against the deck's own bounds,
   rigid-wall penetration — was absent from every design although every
   input is already ingested.
8. Kinetic-energy closure (`½ Σ m |v|²` against the stored global) is the
   cheapest detector of a particle/node index mix-up.
9. Deck lint is evidence: a non-monotone hardening knot is an input defect
   and appears as a recorded finding.
10. Global-statistics extremes are *sampled* extremes (one sample per output
    interval), and the report says so.

## Taylor slice

**Stage 1 — public tiers (`case` + `deck`).** Order: result types and
invariants → kernels → faithful deck reader, `DeckFacts`, `PartTraits` →
measures → criteria and `judge` → report and CLI. The synthetic fixtures
mirror the real traps: a case with a shell block that has no deck part, and a
deck whose material card has a blank second row.

| Category | Quantities |
|---|---|
| Numerical health | `reached_end_time` · `mass_scaling_requested` · `hourglass_energy_ratio` (not applicable per SPH part) · `sph_deactivated_particles` · `sph_neighbors_min` · `sph_neighbors_growth` · `sph_smoothing_length_range` · `rigid_wall_penetration_max` |
| Conservation | `energy_gain_max` · `energy_loss_max` · `energy_residual_final` · `energy_non_mechanical_fraction` · `kinetic_energy_closure` · `internal_energy_closure` · `active_mass_drift` |
| Constitutive | `yield_ratio_max` (+ detail) · `yield_saturation_min` · `state_variable_decrease_max` · `state_variable_min` · `plane_strain_ezz_max` · `out_of_plane_shear_max` · `pressure_trace_residual` (measured, no criterion yet) |
| Data integrity | `nonfinite_count` · `time_axis_monotone` · `terminal_artifact_frames` · `elements_without_deck_part` · `fields_match_card` · `density_slot_matches_deck` · `hardening_table_matches_deck` · `hardening_table_monotone` · `hardening_table_covers_range` · `declared_traits_match_deck` |

*Hand-run acceptance*: on `T-20-60-100` reproduce the probe's recorded
numbers (maximum yield ratio 1.000353; zero monotonicity violations in
720,000 transitions; 151 valid frames; 4800 particles); take one timing on a
hydrated and a dehydrated file; then run the 33 benchmark cases and the
held-aside convergence case.

**Stage 2 — run tier.** Global-statistics and message-file parsers (fixtures
use invented numbers on the real layout, name the solver version, fail closed
on unknown labels); the per-term energy ledger; `terminated_normally`,
`timestep_min_ratio`, `energy_total_tier_mismatch`; glue writing
`run_evidence.json`; a privacy test over every rendered artefact.
*Hand-run acceptance* on the one run already inspected: normal termination,
3918 cycles, time step 7.46e-5 – 7.78e-5 ms, energy ratio peak 1.089 and
final 1.070, rigid-wall energy 311 against initial kinetic energy 4.45e4
(deck units).

**Stage 3 — ratify and publish.** The maintainer verifies external sources
and commits acceptance criteria; instrument tolerances are pinned with dated
notes; `docs/datachecks/taylor_impact_2d.{json,md}` and the drift test land;
the two-grid QoI difference on `T-20-80-Convergence` is reported and labelled
"two-grid, not a grid-convergence index".

## Deliberately not built yet

| Item | Trigger |
|---|---|
| Selective or streamed reader in `core/io`; chunk-merged measures | notch-sized cases (hundreds of MB each) |
| Material classes beyond tabulated J2 and null; the material-class ADR | wave-1D and notch widenings |
| Per-benchmark facts field on `BenchmarkSpec` | first non-derivable fact (units anchor, wave-speed anchor) |
| Waiver records | first ratified criterion a shipped benchmark fails |
| Deck-derived operative yield table | first dataset with no `BenchmarkSpec` |
| Binary solver-output reading | a dependency ADR (it pulls in `pandas`) |
| Re-ingesting discarded d3plot arrays (per-part energies, wall force) | a dated note on ADR-0016 |
| EOS closure; momentum rows | after the Taylor slice |
| Compliance table, card fields, archive artefacts | ADR-0065 follow-up 2 |
| Kernels called from `eval/` | ADR-0065 follow-up 1 |
| Console script; benchmark-free CLI mode; `conftest.py` | first unregistered run; first shared fixture need |

## Open items for the maintainer

1. **External sources for the acceptance criteria.** Candidates were named
   from memory during review and are *unverified*: a roadside-safety
   solution-verification table (energy, added-mass, and hourglass limits) and
   a textbook energy-balance criterion for explicit integration. Nothing
   numeric enters code or an ADR until verified, including whether the
   energy limit is an end-of-run or a maximum-over-time criterion.
2. **Tolerances marked provisional** are pinned only after all Taylor cases
   are measured, each with a dated note.
3. **Re-run policy**: with no CI, the published record is regenerated on a
   dataset re-release or a `definition_version` bump, by hand.

## Verification

- Every stage: `ruff format --check . && ruff check . && mypy src && pytest`.
- Synthetic tests exercise all four verdict branches per quantity, every
  fail-closed deck path, a byte-identical re-run, the privacy scan, and an
  import-boundary test that resolves relative imports.
- Real data is never a test dependency; environment-gated tests assert the
  acceptance facts above and skip when unset.
- `python tools/gen_benchmark_docs.py --check` stays green — nothing
  generated by the existing pipeline is touched.
