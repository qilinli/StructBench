# 0026 — Notch-beam 2D benchmark pair: two benchmarks, tasks, splits, eval

**Status**: Accepted
**Type**: Durable
**Date**: 2026-07-03

## Context

ADR-0024 puts the 2D notched-beam dataset in v0.2. The data
(`../data/Concrete-Beam/2DNotchBeam`) is a three-point-bend family of SPH
simulations of a notched concrete beam — K&C concrete
(`*MAT_CONCRETE_DAMAGE_REL3`) with a plastic-kinematic loader and supports,
**no erosion** — in two loading families, per the collaborators'
`simulation_specification.xlsx`:

- **Constant velocity** (quasi-static bend): H80 beams, span
  {320, 480, 640} × pin velocity {8, 12, 16, 20} mm/s × loading point
  {A, B, C} × notch position {a, b, c} = **108 cases**, 500 ms at 1 ms
  output (500 frames).
- **Initial-velocity impact** (drop weight): span {320, 480, 640} ×
  impact velocity {40, 80, 120, 160} mm/s × impactor shape
  {Bullet, Rectangular, Sphere} × notch position {a, b, c} = **108
  cases** (loading point fixed), same output cadence.

A third folder, `2DGeneralizibility`, holds **5 purpose-built probe
cases** from the prior study: new geometries (60×240, 80×560, 100×800)
and out-of-range velocities, three under constant-velocity loading
(`C_*`) and two under sphere impact (`S_*`).

The raw tree contains some runs beyond the spec sheet's enumeration
(e.g. 72 run folders in one span directory against 36 spec'd); the spec
sheet is authoritative for the benchmark. Deck unit conventions and
per-case particle counts (spec ID tables: ~4.2k–8.3k particles across
concrete, kinematic loader, and supports) are verified at ingestion.

The open design question this ADR settles: one benchmark or two? The
maintainer trained GNS separately per family in the prior work.

## Decision

1. **Two benchmarks, not one**: `NotchBeam2D-Bend` (constant velocity)
   and `NotchBeam2D-Impact` (drop weight). They are different learning
   tasks under ARCHITECTURE.md's definition (one task + one split + one
   protocol per benchmark): different physics regimes (quasi-static
   crack growth vs impact dynamics with inertia), different varied
   factors (loading position vs impactor shape), and separately trained
   baselines, matching prior practice. Versioning and leaderboards stay
   independent.

2. **Flat sibling modules** — `benchmarks/notch_beam_2d_bend/` and
   `benchmarks/notch_beam_2d_impact/`, not a nested family package. This
   preserves the "every child of `benchmarks/` is one benchmark"
   invariant that discovery, the card index (ADR-0027), and the coming
   benchmark-selection mechanism rely on. Shared raw-tree parsing and
   field derivation live in `datasets/` (the ADR-0019 precedent for the
   von Mises derivation), which both modules reference.

3. **Task (both benchmarks)** — the ADR-0019 autoregressive transition
   task. The auxiliary per-particle field is the **K&C scaled damage
   measure** (what `MAT_CONCRETE_DAMAGE_REL3` writes to the d3plot
   effective-plastic-strain slot; the prior study's extracts call it
   "strain"). Canonically named `damage`; it carries the crack pattern,
   which is the scientific point of the benchmark. Particle roles
   (concrete / kinematic loader / support) come from the spec sheet's ID
   tables and feed the model's particle-type embedding, as in Taylor.

4. **Splits.** Per benchmark: **train 88 / val 8 / test-interpolation
   12**, constructed by a fixed stratified rule — held-out cells are
   interior combinations chosen so that every factor level (span,
   velocity, loading point or impactor shape, notch position) still
   appears in train. The exact case-id lists are frozen in each
   benchmark module at ingestion time and are immutable thereafter
   (changing them is a new benchmark version, ADR-0019 precedent).

5. **Generalisation probes instead of synthetic extrapolation.** The
   `2DGeneralizibility` cases are each benchmark's separately-reported
   probe set: the three `C_*` cases for Bend, the two `S_*` cases for
   Impact. They probe new geometry and out-of-range velocity — stronger
   and more honest than a held-out grid corner.

6. **Evaluation protocol.** Seed with the first `L` ground-truth frames,
   roll out to the end. Metrics, per split and probe set:
   - one-step and full-rollout **position RMSE**;
   - **damage RMSE** over the rollout;
   - **quantities of interest**: mid-span deflection error (peak and
     history) and final-frame damage-field error (the crack pattern).
     A load-capacity QoI may be added later if ingestion confirms the
     reaction data supports it.

## Alternatives considered

- **One combined benchmark with two tracks.** Rejected: two tasks under
  one name muddies the leaderboard and couples the versioning of two
  independent contracts. The genuinely interesting combined problem —
  one model mastering both regimes — is a *harder, new* task, parked as
  a possible future "unified" track once both leaderboards exist.

- **A nested family package** (`notch_beam_2d/{bend,impact}`). Rejected:
  breaks the one-benchmark-per-directory invariant for every tool that
  iterates benchmarks; the shared code it would justify belongs in
  `datasets/` anyway; and the RC-beam family (v0.3) would pose the same
  question again — the flat `<family>_<track>` naming answers it once.

- **Synthetic extrapolation splits** (holding out velocity 20 / 160 or
  a span). Unnecessary — purpose-built probe cases exist and are
  stronger; the grid stays fully available for training density.

- **Effective plastic strain or stress as the auxiliary field.**
  The K&C damage measure is what the d3plot slot physically records for
  this material and what the prior study used; stress remains available
  in the canonical data for future benchmark versions.

## Consequences

- Two modules, two cards, two GNS checkpoints, two leaderboard sections;
  a shared docs page presents the family and its shared provenance.

- The exact split lists are produced and frozen at ingestion; until
  then, the split is defined by rule only. The ADR-0024 ingestion pass
  enumerates the spec's 216 cases + 5 probes and flags extras in the
  raw tree without ingesting them into the benchmark.

- The `damage` extraction and its naming (d3plot effective-plastic-
  strain slot → `damage` for this material model) live in `datasets/`
  with tests, so the mislabelling trap the prior extracts fell into is
  fenced off at one tested seam.

- The kinematic loader differs between the two benchmarks (32-particle
  pin vs 112-particle impactor in three shapes); each module documents
  its own role tables, one more reason the contracts stay separate.
