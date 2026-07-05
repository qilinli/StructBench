# 0030 — Concrete-Beam decks are kg-mm-ms; canonical data patched in place

**Status**: Accepted
**Type**: Durable
**Date**: 2026-07-05

*(This ADR documents a decision made and executed on 2026-07-05 — commit
84d84eb introduced `patch_units.py` citing "ADR-0030" before this document
was written. Recorded retroactively the same day to close the gap.)*

## Context

LS-DYNA decks carry no unit system; the adapter is told the convention per
dataset (ADR-0016 §5). The Concrete-Beam family (2DNotchBeam and
1DWavePropagation, one deck lineage) was ingested as **g-mm-ms** — the same
convention as the Taylor deck. Physical checks against the ingested data
showed that is wrong; the decks are **kg-mm-ms**:

- The K&C concrete material's unit conversion factor `UCF = 145000`
  (psi per unit stress) implies the deck stress unit is GPa — which holds
  only under kg-mm-ms (kg·mm/ms² per mm² = GPa).
- Steel reinforcement `E = 200` reads as 200 GPa under kg-mm-ms — the
  textbook value. Under g-mm-ms it would be 200 MPa, three orders too soft.
- Concrete density `RO = 2.4e-6` is 2400 kg/m³ under kg-mm-ms — real
  concrete. (Contrast the Taylor deck: `RO = 0.0089` is 8930 kg/m³ only
  under g-mm-ms.)
- The measured wavefront speed in the ingested wave-1d data ≈ 3100 m/s
  matches concrete's elastic wave speed with time in ms.

Consequence of the wrong label: every **mass-derived** SI quantity in the
237 canonical HDF5s (stress, pressure, density, mass, and the energies) was
stored 1000× too small (a g/kg factor). Kinematic fields (time,
displacement, velocity, acceleration, strain and its derivatives, radius)
depend only on mm and ms and were correct throughout.

## Decision

1. **The Concrete-Beam family's `source_units` is `kg-mm-ms`.** The Taylor
   deck (`2D-Copper-Bar-Taylor-Impact`) remains `g-mm-ms` — verified
   2026-07-05 against `scratch/Taylor.k`: copper `RO = 0.0089` g/mm³,
   `G = 37590` MPa ≈ 37.6 GPa, Gruneisen `C = 3940` mm/ms = copper's bulk
   sound speed. Each is physical only under g-mm-ms.
2. **Patch the canonical HDF5s in place** (`data_generation/lsdyna/
   Concrete-Beam-unit-patch/patch_units.py`): multiply the mass-derived
   datasets by 1000 and set `metadata.attrs["source_units"] = "kg-mm-ms"`,
   gated on that attribute (g-mm-ms → patch; kg-mm-ms → skip; anything
   else → fail loudly). Stale `derived/` normalization caches are deleted.
3. **Correct the sources** so re-conversion cannot reintroduce the bug:
   `SOURCE_UNITS = "kg-mm-ms"` in the wave and notch converters, and the
   three benchmark cards' `source_units` plus mass-unit-dependent material
   constants (wave `E` in GPa; concrete densities in kg/mm³), with
   `docs/benchmarks.md` regenerated.

Execution record: the patch ran on the pre-reorg layout 2026-07-05; after
the ADR-0031 reorganization a full idempotent pass over all 237 files
reported 237 skip / 0 would-patch / 0 fail — every file carries the
corrected attribute and values. Source corrections landed in commit c5df6c8.

## Alternatives considered

- **Re-convert from the raw d3plots with the corrected label**: rejected —
  hydrates tens of GB of OneDrive d3plot families and recomputes everything
  to change a multiplicative factor on eight datasets; the in-place patch
  is exact, idempotent, and auditable.
- **Keep the stored values and re-label the metadata only**: rejected — the
  canonical format's contract is strict SI values (ADR-0012); the stored
  numbers themselves were wrong.
- **Fix converters only and leave existing files wrong until the next full
  re-conversion**: rejected — v0.2 baseline training reads these fields
  (aux targets, normalization); training on 1000×-off stresses corrupts
  the benchmark.

## Consequences

- All 237 Concrete-Beam canonical cases are strict-SI correct; the
  `source_units` attribute now doubles as the patch's idempotency gate.
- Anything derived from pre-patch files (normalization caches — deleted;
  any inspection figures) is invalid; no training runs predate the patch,
  so no run artifacts needed invalidation.
- The wave benchmark's physical description sharpens: deck `E = 0.01` is
  0.01 GPa (10 MPa), not 0.01 MPa; card text corrected accordingly.
- `patch_units.py` stays in the repo as the auditable record and safe
  re-run tool (`--roots`, defaults per the ADR-0031 layout).
- Numbering note: ADR-0031 was drafted and accepted before this document
  was written (the number 0030 was reserved by the in-flight citation in
  commit 84d84eb), which is why the log briefly showed a 0029 → 0031 gap.
