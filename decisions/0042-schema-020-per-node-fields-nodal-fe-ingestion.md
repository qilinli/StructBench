# 0042 — Schema 0.2.0 adds per-node fields; nodal-FE ingestion via download-and-convert (deforming_plate)

**Status**: Accepted
**Type**: Durable
**Date**: 2026-08-07

## Context

ADR-0041 makes `DeformingPlate` (MeshGraphNets, Pfaff et al. 2021) the v0.3
benchmark and asserts that ingesting it needs "no schema change" because mesh
cells map onto `elements/<type>/connectivity`. Verifying the dataset's own
`meta.json` (live at
`https://storage.googleapis.com/dm-meshgraphnets/deforming_plate/meta.json`)
against the schema shows that claim is only *partly* true. The dataset carries
three **per-node** quantities that schema 0.1.0 — designed around SPH particles
(single-node elements) and per-*element* response fields — has no home for:

| field | `meta.json` shape | nature | 0.1.0 gap |
|---|---|---|---|
| `node_type` | `[1,-1,1]` int32, static | per-node classification (NORMAL=0 plate, OBSTACLE=1 actuator, HANDLE=3 fixed) | no per-node static slot |
| `stress` (von Mises) | `[400,-1,1]` float32, dynamic | per-node scalar | `response.node` fields must be `dim`-wide `(T,N,dim)`; a `(T,N,1)` scalar is rejected, and `response.element` is per-element |
| `mesh_pos` | `[1,-1,3]` float32, static | per-node reference/material coords (MGN's mesh-space frame) | no per-node static vector slot |

The mesh connectivity itself (`cells` `[1,-1,4]`, linear tetrahedra) does map to
`elements/tetra/connectivity` with no change, as ADR-0041 said. The nodal fields
do not.

Two storage routes were weighed in session:

- **Reuse element blocks (no schema change):** represent nodes as a degenerate
  single-node `"node"` element block (`part_id = node_type`,
  `response.element["node"]["stress"]` = per-node von Mises), plus a `"tetra"`
  block for edges. Keeps ADR-0041's wording intact, but spreads a
  "nodes-as-fake-elements" convention, and has no clean home for `mesh_pos`
  (works only if `mesh_pos == world_pos[0]`).
- **Additive per-node schema fields:** the chosen route.

Per-node nodal fields are not a deforming-plate quirk — they are how *every*
mesh/FE dataset stores output (the v0.4 crash candidate is also nodal-FE). The
schema's versioning was built for exactly this additive step (ADR-0013:
"additive changes bump the minor version").

## Decision

1. **Schema 0.2.0 — additive, backward-compatible.** `SCHEMA_VERSION` bumps
   `"0.1.0"` → `"0.2.0"`. Readers stay compatible with 0.1.0 files (the new
   fields are optional and absent there). Two additions:

   a. **`Nodes` gains two optional per-node static fields:**
      ```python
      node_type: NDArray[np.int64] | None = None        # (n_nodes,)
      reference_coords: NDArray[np.float64] | None = None  # (n_nodes, dim)
      ```
      `node_type` is a per-node classification consumed downstream as the
      conditioning feature and the kinematic-prescription key (see 2d).
      `reference_coords` holds a material/reference configuration (e.g.
      `mesh_pos`) when it differs from `coords`. Validation, when present:
      `node_type.shape == (n_nodes,)`; `reference_coords.shape == coords.shape`.
      A general per-node feature dict is deliberately *not* introduced now
      (YAGNI); it can be added additively later if a dataset needs it.

   b. **`response.node` is relaxed to admit per-node scalar/tensor fields.**
      A `response.node[field]` array must be `(n_frames, n_nodes, k)` with
      `k >= 1` (was: exactly `k == dim`). `displacement` remains required and
      must be `dim`-wide. This lets a per-node scalar such as
      `response.node["von_mises_stress"]` `(T, N, 1)` validate.

2. **Deforming-plate ingestion path.**

   a. **Download-and-convert, not rehost.** The dataset has **no explicit
      redistribution licence** (only the DeepMind *code* is Apache-2.0; the
      hosted data carries no stated terms — verified). StructBench therefore
      ships a converter that pulls `meta.json` + `{train,valid,test}.tfrecord`
      from the source bucket and produces canonical `.h5` **locally**;
      StructBench does not host or redistribute the raw data. This **supersedes
      ADR-0041's "link or rehost"** in favour of download-and-convert, which
      sidesteps the redistribution question entirely.

   b. **Units are measured, not assumed.** `world_pos`/`stress` units are
      undocumented in `meta.json` and the paper. The converter records the
      determined convention in `metadata.source_units` and canonicalises to SI;
      **blessing MGN is gated on empirically SI-verified data** (the
      Taylor/Concrete-Beam precedent, ADR-0030).

   c. **Quasi-static handling.** `deforming_plate` is quasi-static
      (`meta.json` `dt = 0`). Canonical `response/time/t` carries the load-step
      index as a pseudo-time axis; the surrogate task is load-stepping rollout
      (next-configuration prediction). The benchmark-protocol ADR states this.

   d. **Field mapping** (converter): `world_pos[0]` → `nodes.coords`;
      `world_pos[t] - world_pos[0]` → `response.node["displacement"]`;
      `mesh_pos` → `nodes.reference_coords`; `node_type` → `nodes.node_type`
      (feeding the `particle_type` embedding and the `kinematic_types`
      prescription — OBSTACLE/HANDLE prescribed from ground truth, NORMAL
      predicted and scored); `stress` → `response.node["von_mises_stress"]`
      `(T,N,1)`; `cells` → `elements["tetra"].connectivity`. Materials: a single
      synthesised hyperelastic `Material` (non-empty `source_model`), since
      there is no solver deck. TensorFlow is imported lazily inside the read
      function only (never a runtime dependency), mirroring the lasso pattern.

3. **Correction to ADR-0041.** Its "no schema change is required" line is
   amended: `cells` map without change, but the nodal fields require schema
   0.2.0 per this ADR. A dated note is added to ADR-0041.

## Alternatives considered

- **Reuse element blocks (no schema change)** — the "nodes-as-fake-elements"
  route above. Rejected: it overloads the element/part_id mechanism with a
  meaning it was not designed for, has no clean home for `mesh_pos`, and does
  not generalise to the next nodal-FE dataset (crash, v0.4) — it would have to
  be unwound later at higher cost than an additive field now.

- **A general per-node feature dict** (`Nodes.features: dict[str, NDArray]`)
  instead of named fields. Rejected for now (YAGNI): two named fields cover the
  concrete need with simpler validation; a dict can be added additively if a
  future dataset justifies it.

- **Store per-node stress under a new `response` sub-group** rather than
  relaxing `response.node`. Rejected: per-node scalars are conceptually node
  fields; relaxing the trailing-dim rule is the smaller, more honest change and
  keeps one home for all per-node time series.

## Consequences

- **Code:** `core/schema.py` (`Nodes` fields, `SCHEMA_VERSION`),
  `core/validation.py` (per-node field rules, relaxed trailing dim),
  `core/io/` reader+writer (persist/read `node_type`, `reference_coords`, and
  arbitrary-width `response.node` fields). Existing 0.1.0 archives remain
  readable unchanged.
- **Downstream enablement:** `datasets/` can now source `particle_type` from
  `nodes.node_type` (not only element `part_id`) and carry mesh edges from
  `elements["tetra"].connectivity` — the two things `models/mgn` needs.
- **Ingestion:** a new non-importable `data_generation/meshgraphnets/`
  `deforming_plate/convert.py`, run in a throwaway TF environment.
- **Docs:** ARCHITECTURE.md case-schema section and its "implementation status"
  note update to record the 0.2.0 per-node fields; ADR-0041 gets the correction
  note; ADR-0013 (HDF5 layout) takes a dated note for the added node datasets.
- **Scope honesty:** this is the first schema-version bump since 0.1.0. It is
  additive and backward-compatible; no data is rewritten and no consumer of
  0.1.0 data breaks.
