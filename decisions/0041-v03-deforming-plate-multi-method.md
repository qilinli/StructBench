# 0041 — v0.3 pivots to a public multi-method benchmark: DeformingPlate with native MGN/Transolver/GeoFLARE (supersedes ADR-0024's v0.3 scope)

**Status**: Accepted
**Type**: Durable
**Date**: 2026-08-07

## Context

ADR-0024 set v0.3 as the 3D RC beam under drop-weight impact, with element
erosion as the headline problem, and wove that slot into a difficulty ladder
(clause 3). In session on 2026-08-07 the maintainer redirected v0.3 away from
RC beam and erosion entirely, toward two goals:

1. **Turn StructBench into a cross-method comparison platform.** Today the
   platform is multi-benchmark / single-baseline (CGN on each benchmark,
   ADR-0034). The maintainer wants a *method axis* — several reference methods
   compared on a shared task and protocol.

2. **Own durable, reusable method implementations** for the maintainer's own
   research: StructBench-native baselines that future research can build on and
   compare against, rather than one-off external code.

Both land inside the substrate boundary. Under the ADR-0014 litmus test,
reference implementations of published methods — released as known-good
baselines with a number to beat — are substrate, and a cross-method comparison
over existing methods is a substrate output (the comparison, not any single
method, is the contribution). Neither goal is a brain-layer research artefact.

The redirect has two axes, each with a constraint discovered in session:

- **Public data.** Motivated by escaping the LS-DYNA / collaborator
  data-generation bottleneck (ADR-0015) so benchmarks can grow without
  GUI-driven case production. Constraint: public data that is *both*
  rollout-shaped *and* solid mechanics is thin. The MeshGraphNets
  `deforming_plate` set (Pfaff et al. 2021) is available, canonical (five years
  of published baselines), and tractable (~1,271 nodes, 400 steps, ~1,200
  trajectories). Public *crash* rollout data is not downloadable: CarCrashNet
  (arXiv 2605.07098) is unreleased pending peer review and is 6.65 TB with
  high-resolution full-vehicle meshes (likely ≥10⁶ nodes); NVIDIA's crash
  datasets (PhysicsNeMo crash example; arXiv 2510.15201) are bring-your-own or
  proprietary.

- **Methods beyond CGN's GNN family.** The maintainer named Transolver and, via
  the NVIDIA PhysicsNeMo crash example, GeoFLARE — transformer operators
  (physics-attention, geometry-aware slicing), a different architectural family
  from the message-passing GNNs StructBench ships. Constraint: these methods'
  published numbers live on operator-learning benchmarks (Transolver) or crash
  data (GeoFLARE), not on `deforming_plate`.

Task-shape was settled in session: StructBench's entire protocol is
autoregressive time-rollout (ADR-0019/0025/0026/0035). The maintainer chose to
*keep* that family and cross-run the new methods on it, rather than add an
operator-learning (steady-state / parametric) modality where Transolver is
native — accepting that Transolver/GeoFLARE therefore run off-native, with no
published rollout numbers to reproduce.

## Decision

1. **v0.3 is redefined as a single public benchmark, `DeformingPlate`** — the
   MeshGraphNets deformable-plate set (Pfaff et al. 2021, COMSOL ground truth),
   **StructBench's first 3D benchmark**. Autoregressive rollout of node
   positions; auxiliary target = von Mises stress (the existing `aux_field`
   pattern); the canonical 1000/100/100 split reused verbatim for comparability
   with published numbers. The detailed split / metrics / scored-horizon /
   input-window protocol is deferred to a follow-on benchmark ADR (the
   ADR-0019/0025/0026 pattern), respecting the rollout-init rule of ADR-0035.

2. **The headline is cross-method comparison, not a new physics problem.** The
   v0.3 method set is **MGN, Transolver, and GeoFLARE**, all implemented
   **natively** under `models/` — no PhysicsNeMo runtime dependency.
   - **MGN is the blessed baseline**, validated against its published
     `deforming_plate` numbers (this reproduction is what certifies the whole
     ingestion → training → eval pipeline).
   - **Transolver and GeoFLARE ship as provisional native implementations** —
     best-effort ports whose fidelity check against published numbers is
     deferred (see Alternatives). They are flagged `provisional` in the results
     registry and rendered as provisional in the comparison view, so their
     numbers are never read as blessed.
   - CGN sits out this comparison (it is a particle/radius-graph GNN; on a mesh
     task it would run off-native, adding a fourth method without a published
     anchor). This is a deliberate, one-benchmark departure from the
     "CGN on every benchmark" pattern, noted here rather than silently.

3. **Ingestion is an offline `tfrecord` → canonical HDF5 conversion**, run in a
   throwaway environment (mirroring the lasso-python conversion pattern) so
   **TensorFlow is never a runtime dependency**. Output is a standard
   `canonical/deforming_plate/*.h5` archive (ADR-0031). Mesh cells map to
   `elements/<type>/connectivity` — already part of settled, implemented schema
   0.1.0, so **no schema change is required**. `world_pos` maps to node coords +
   `response/node/displacement`, `stress` to element stress, `node_type` to the
   conditioning feature. The adapter and unit handling are detailed in a
   follow-on ingestion ADR.

4. **The cross-method infrastructure is the reusable substrate this release
   delivers.** The results registry (ADR-0033) extends from per-benchmark to
   **per-(benchmark × method)** with a `provisional` flag; the generated landing
   page (ADR-0036) renders a **method-comparison table** distinguishing blessed
   from provisional. `models/` gains a **transformer-operator family** alongside
   the message-passing GNNs, and `datasets/` generalizes to serve **point-set
   inputs** (Transolver/GeoFLARE) alongside graph windows (CGN/MGN).

5. **RC beam is deferred, and crash becomes a v0.4 candidate.** RC beam is
   removed from v0.3 and parked with no scheduled release (as the segmented beam
   is); ADR-0024's erosion analysis remains the gate if it is revived. A crash
   benchmark is a v0.4 candidate, gated on (a) public crash data existing —
   CarCrashNet's release, or maintainer-generated LS-DYNA crash data released
   under an open licence — and (b) the scale infrastructure it needs (the
   roadmap's cell-list `radius_graph` backend and TB-scale hosting).

6. **This supersedes ADR-0024's v0.3 scope** (clause 4, "RC beam moves to
   v0.3") and revises its difficulty-ladder framing (clause 3). ADR-0024's v0.2
   record is historical fact and stands unchanged. The reference-baseline set
   broadens beyond CGN; **ADR-0034's "the reference baseline" language takes a
   light amending note** to read as "the reference GNN baseline / the incumbent
   baseline," which a follow-on edit records.

7. **Build order (internal checkpoints, not a scope cut):**
   ① ingestion + `DeformingPlate` + MGN blessed → ② Transolver provisional →
   ③ GeoFLARE provisional. Each is a checkpoint so partial value lands if ③
   slips.

## Alternatives considered

- **Keep v0.3 = RC beam / erosion (ADR-0024).** Rejected: the maintainer
  redirected. Erosion stays a genuine research problem better carried when a
  modelling approach exists (ADR-0024's own reasoning); the public-data +
  multi-method direction serves the platform's comparison mission and the
  maintainer's research now. RC beam is deferred, not dropped.

- **Add an operator-learning (steady-state / parametric) modality** where
  Transolver and GeoFLARE are native and their published numbers reproduce.
  Rejected for v0.3: it introduces a second task type and a second eval
  protocol. The maintainer chose to keep the autoregressive-rollout family and
  cross-run methods on it, accepting off-native operation as the price of a
  contained release.

- **Run the transformer methods via PhysicsNeMo interop as an optional extra**
  (which ADR-0017 explicitly sanctions, and VISION's "build on existing
  libraries" favours). Rejected in favour of native reimplementation: the
  maintainer wants StructBench-owned, durable implementations to reuse as
  baselines in future research, not a coupling to PhysicsNeMo's API and release
  cycle. Cost accepted: larger implementation effort and a fidelity-validation
  debt, the latter managed by the `provisional` flag.

- **Include CarCrash as a second v0.3 benchmark** (the maintainer's initial
  ask). Rejected: its data is unreleased (peer review pending) and 6.65 TB with
  high-resolution full-vehicle meshes (likely ≥10⁶ nodes) — a hosting and scale
  problem out of proportion to one release, and unbuildable today regardless.
  Deferred to v0.4.

- **Validate Transolver/GeoFLARE now** — either by reproducing their published
  numbers on their native benchmarks (ingesting e.g. Transolver's Elasticity /
  Plasticity purely as a fidelity harness) or by a dev-time cross-check against
  PhysicsNeMo's reference implementations. Rejected for v0.3 in favour of
  best-effort-now / validate-later: it keeps the release shippable, the native
  implementations exist and are reusable immediately, and the `provisional` flag
  keeps the honesty. The fidelity check is scheduled work, not abandoned.

## Consequences

- **New code surface:** a `benchmarks/deforming_plate` module; `models/mgn`,
  `models/transolver`, `models/geoflare`; an offline `tfrecord`→HDF5 conversion
  (in `data_generation/` or a `core/io` adapter invoked offline); a `datasets/`
  generalization to point-set inputs; and results-registry + landing-page
  comparison rendering. Follow-on ADRs are expected for the benchmark protocol,
  the ingestion adapter, the per-method registry schema (extending ADR-0033 /
  ADR-0037), and the transformer-operator family's placement in `models/` and
  `datasets/` (which may touch the ARCHITECTURE.md dependency graph).

- **First 3D benchmark.** The case schema (ADR-0012) and native `radius_graph`
  (ADR-0020) are dimension-agnostic. `viz/` is 2D FEM-convention only
  (ADR-0022): **3D visualization is deferred** — v0.3 ships numbers and the
  comparison table, not 3D fringe figures. VISION's "1D/2D problems only"
  limitation copy updates.

- **Hosting.** `deforming_plate` is public; its dataset *redistribution* terms
  must be confirmed before the ingestion ADR (both the MeshGraphNets code
  licence and the dataset's hosting terms are checked then, not asserted now).
  A redistributable
  licence would let StructBench link or rehost rather than gate on request —
  lighter than the ADR-0040 OneDrive arrangement, which stays in force for the
  maintainer-held datasets.

- **Dependencies.** No PhysicsNeMo. Method ports stay within torch / PyG
  (ADR-0018); any small new dependency (e.g. `einops`) is proposed in its own
  note when it arises. TensorFlow appears only in the offline conversion
  environment.

- **Fidelity debt.** Transolver and GeoFLARE are provisional until validated
  against published numbers. The registry and comparison view must make this
  legible so provisional numbers are never mistaken for blessed baselines.

- **Documentation edits on acceptance:** README roadmap rewritten (RC beam →
  deferred; v0.3 = the deforming-plate multi-method release); ADR-0024's index
  row annotated (v0.3 scope superseded by 0041); ADR-0034 light amendment;
  VISION limitations updated; CLAUDE.md project snapshot updated.

- **Positioning.** The v0.3 story is method comparison on public data — a shift
  from v0.1/v0.2's "ship the maintainer's datasets + a single baseline." It is
  the first release whose contribution is the *comparison infrastructure and a
  set of reusable method implementations*, not a new physics dataset.

---

**Correction (2026-08-07, maintainer).** Two claims in this ADR are revised by
ADR-0042, drafted while implementing ingestion:

1. Decision clause 3 says ingesting deforming_plate requires "no schema change."
   That holds for `cells` → `elements/connectivity`, but the dataset's per-node
   fields (`node_type`, per-node von Mises `stress`, `mesh_pos`) have no home in
   schema 0.1.0. ADR-0042 adds them via an **additive schema 0.2.0** bump
   (per-node `Nodes.node_type`/`reference_coords` and a relaxed `response.node`
   trailing-dim rule).
2. The Consequences "hosting" note offers "link or rehost." The dataset has no
   explicit redistribution licence, so ADR-0042 settles ingestion as
   **download-from-source-and-convert-locally, no rehost**.

Recorded as a dated correction note (not a reversal of the v0.3 scope decision,
which stands) per the index-README convention.
