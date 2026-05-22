# 0012 — Case schema field-level structure

**Status**: Accepted
**Type**: Durable
**Date**: 2026-04-28

## Context

ADR-0011 fixed the vocabulary for the case schema (case = specimen + scenario + response, plus frame and transition) and explicitly deferred the field-level structure: which data items the schema represents, where they sit, and what conventions they follow.

This ADR settles that structure. Together with ADR-0011 it constitutes the conceptual and structural definition of the case schema. The remaining stage — concrete persistence (HDF5 group spelling, dtypes, attribute conventions) — is left to a future ADR.

The structure was developed against the v0.1 anchor problems (2D Taylor impact in `Taylor.k`; RC beam impact in ADR-0003) and the constraint set in the prior ADRs (solver-agnostic, ML-pipeline-friendly, extensible to post-v0.1 SHM).

## Decision

### Field set

A case file holds the following kinds of data, each occupying its own namespace within the file. The vocabulary mapping (specimen / scenario / response) is documentation only; the file does not separate them structurally.

**Geometry and topology**
- `nodes` — `coords` of shape `(n_nodes, dim)`, plus `node_id` of shape `(n_nodes,)` preserving the original solver IDs.
- `elements/<type>` — for each element type present (`sph`, `solid`, `beam`, `shell`, `discrete`, …): `connectivity` of shape `(n_elem, n_nodes_per_elem)` using 0-indexed references into `nodes/coords`, plus `element_id` and `part_id`.
- `parts` — first-class grouping linking a set of elements to a section and a material. Mirrors LS-DYNA `*PART` and equivalents in other solvers.
- `sections` — section properties (cross-section for beams, thickness for shells, SPH parameters).

**Materials**
- `materials` — per material entry, hybrid representation:
  - `canonical_model` — name from a small enum (`linear_elastic`, `concrete_damage`, `piecewise_linear_plasticity`, `elastic_plastic_hydro`, …); populated when a clean canonical mapping exists, null otherwise.
  - `source_model` and `source_params` — solver-native model name and parameter dict, always populated.
  - The enum grows additively over time as new mappings are settled.

**Constraints, loading, initial conditions**
- `boundary_conditions` — fixed / prescribed / symmetry constraints.
- `loading` — applied loads, body forces, contacts, rigid walls.
- `initial_conditions` — IC *spec* (kind, target set, value); optional, present only when the source supplies an explicit IC card. The actual t=0 state is at `response/` frame 0, not here.
- `time_curves` — named (t, value) curves referenced by loading / BCs / IC.
- `sets` — named node and element sets referenced by IC / BCs / loading.

**Response**
- `response/time/t` — single time array of length `n_frames`. One global time axis for all of `response/` (with `response/sensor` permitted its own axis when SHM is detailed).
- `response/node` — per-node, per-frame fields: `displacement`, `velocity`, `acceleration` of shape `(n_frames, n_nodes, dim)`.
- `response/element/<type>` — per-element-type, per-frame fields: `stress`, `strain`, `damage`, etc. Tensor fields use Voigt-symmetric components.
- `response/global` — per-frame scalars: total kinetic / internal energy, contact force, reactions.
- `response/sensor` — sensor readings; slot reserved, internal shape deferred to SHM scope.

**Sensors**
- `sensors` — sensor placements; slot reserved, internal shape deferred to SHM scope.

**Metadata**
- `metadata/case_id` — required, unique identifier.
- `metadata/schema_version` — required.
- `metadata/units_convention` — required, set to `"SI"` (see cross-cutting conventions).
- `metadata/dimension` — required, `2` or `3`.
- `metadata/provenance` — required when solver-generated: `solver_name`, `solver_version`, `generation_date`.
- `metadata/source_units` — original convention if not natively SI, for provenance.
- `metadata/source_deck` — verbatim source-deck blob; optional, present when the adapter preserves it.
- `metadata/asset_id`, `metadata/dataset_id` — workflow-level identifiers, optional.

### Cross-cutting conventions

- **Units**: strict SI canonical. Adapters convert solver-native units to SI on write. The original convention is preserved in `metadata/source_units`.
- **Identity / referencing**: connectivity arrays use 0-indexed sequential references into the `nodes` table (analogously for elements, parts, materials). Original solver IDs are preserved as `*_id` columns alongside.
- **Time axis**: single `response/time/t` array of length `n_frames`. Uniformity is implicit. All response fields share this axis (with `response/sensor` allowed its own when SHM lands).
- **t=0 state**: lives at frame 0 of `response/`. The `initial_conditions` group, when present, holds the source deck's IC *spec* for roundtrip — not the t=0 state itself.
- **Tensor components**: Voigt-symmetric. 3D: `(xx, yy, zz, xy, yz, xz)`. 2D: `(xx, yy, zz, xy)`.
- **Component naming**: lowercase, underscore-separated.

### Validity tiers

A schema-valid case file MUST contain:

- `nodes`
- at least one `elements/<type>`
- `materials`
- `metadata` (`case_id`, `schema_version`, `units_convention`, `dimension`)
- `metadata/provenance` when the case is solver-generated.

A case file MUST contain the following when applicable:

- `parts` (when elements reference a non-trivial part).
- `sections` (when an element type needs them — beams, shells).
- `boundary_conditions`, `loading`, `time_curves`, `sets`, `initial_conditions` — each when present in the source.
- `response`, `response/time/t`, `response/node/displacement` — when the case has been simulated. A file without `response` is a valid "stub" (specimen + scenario specified, simulation not yet run).

A case file MAY contain:

- `sensors`, `response/sensor`.
- `response/global`, `response/element/*`.
- `metadata/source_deck`, `metadata/asset_id`, `metadata/dataset_id`.

### Out of scope for this ADR

- The on-disk HDF5 layout (groups vs flat datasets, exact path spellings, dtype choices, attribute conventions). Separate ADR when settled.
- Internal field shape for `sensors` and `response/sensor`. Deferred to when SHM scope is concrete.
- Detailed field shape inside `boundary_conditions` and `loading`. The current decision is that they exist as named groups; the field-level shape is a smaller follow-on decision driven by adapter needs.
- The contents of the canonical-material-name enum. The hybrid mechanism is settled here; the enum grows additively.

## Alternatives considered

- **Specimen / scenario as structural folders.** Rejected. Source decks (LS-DYNA `.k`, ANSYS `.imp`) interleave specimen and scenario freely; forcing them apart in the canonical file would be an artificial cut. The vocabulary stays in ADR-0011 for documentation; the file does not separate them.
- **`initial_conditions` required, `response` derivative.** Rejected. Simulation cases have a full `response`; SHM cases have observed `response` but no IC card. The asymmetry forces `initial_conditions` to be optional and `response` frame 0 to be authoritative.
- **Stable solver IDs as connectivity references, not 0-indexed.** Rejected. ML pipelines need clean tensor indexing; ID-based connectivity forces every consumer to build a lookup. Original IDs are preserved as `*_id` columns for roundtrip.
- **File-declared `units_convention` (mm-g-ms allowed alongside SI).** Rejected in favour of strict SI canonical. The adapter conversion cost is small and one-time; the consumer-side simplification is permanent and prevents cross-convention dataset footguns.
- **Canonical-only or solver-native-only material naming.** Rejected. Canonical-only fights solver diversity and rejects unusual models; solver-native-only forces every consumer to know every solver's material zoo. The hybrid keeps both.
- **Separate `t=0` group for the initial state.** Rejected. The information already lives at frame 0 of `response/`; a separate group would duplicate.
- **Per-Gauss-point tensor fields by default.** Rejected. Most ML uses per-element averages; per-Gauss is exposed only when the source provides it (`stress_gauss` etc.).
- **Multiple time axes by default in `response/`.** Rejected. v0.1 cases share one global axis; sensor-side SHM exception is permitted but not the default.

## Consequences

- **Adapter contract** is bounded: solver-output adapters (e.g., LS-DYNA d3plot → canonical) convert units to SI, re-index node/element references to 0-indexed, populate `materials` with both canonical and solver-native fields where possible, and write the source deck blob into `metadata/source_deck`.
- **Reader contract** is simple: consumers can rely on SI units, sequential indexing, and the required-field tier without per-source branching.
- **Schema versioning**: this structure pins the initial `schema_version`. Future field additions are minor versions; structural changes are major and require a superseding ADR.
- **Validation tooling** can be written from the validity tiers above. It checks group presence per tier and structural shapes.
- **SHM extension is additive**: when SHM scope concretises, `sensors` and `response/sensor` get their internal shape designed; this extension does not affect existing v0.1 cases.
- **HDF5 layout design (Stage 3) is now scoped**: it must faithfully represent the field set above with reasonable HDF5 idioms; the design space is bounded.
- **Real follow-on decisions**, each likely warranting its own ADR: the canonical-material-name enum, BC and loading field-level shape, the HDF5 layout, sensor representation when SHM is on the table.
